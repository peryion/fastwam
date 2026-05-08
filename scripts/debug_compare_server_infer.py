import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image

from infer_ego import _build_cfg, _denormalize_action
from fastwam.runtime import build_datasets


def _tensor_frame_to_image(frame_chw: torch.Tensor) -> Image.Image:
    arr = frame_chw.detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr).convert("RGB")


def _image_to_input_tensor(image: Image.Image, size_hw: tuple[int, int]) -> torch.Tensor:
    image = image.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32)
    return (tensor / 127.5 - 1.0).unsqueeze(0)


def _encode_image(image: Image.Image, fmt: str, quality: int) -> str:
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = int(quality)
    image.convert("RGB").save(buf, format=fmt.upper(), **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_numpy(obj: Any) -> np.ndarray:
    raw = base64.b64decode(obj["__numpy__"])
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


def _array_stats(name: str, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "name": name,
        "shape": list(arr.shape),
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
    }


def _diff_stats(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {"name": name, "shape_a": list(a.shape), "shape_b": list(b.shape), "shape_mismatch": True}
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "name": name,
        "shape": list(a.shape),
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else None,
        "max_abs": float(abs_diff.max()) if abs_diff.size else None,
        "rmse": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else None,
        "first_row_a": a[0].tolist() if a.ndim >= 2 and a.shape[0] > 0 else None,
        "first_row_b": b[0].tolist() if b.ndim >= 2 and b.shape[0] > 0 else None,
    }


def _post_server(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("This script needs `requests` to query the server.") from exc
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Server returned status={resp.status_code}: {resp.text}")
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Server returned error: {data['error']}")
    return data


def _resolve_sample_index(dataset, episode_index: int | None, sample_index: int, episode_offset: int) -> int:
    if episode_index is None:
        return int(sample_index)
    ep_from = int(dataset.lerobot_dataset.episode_data_index["from"][int(episode_index)].item())
    ep_to = int(dataset.lerobot_dataset.episode_data_index["to"][int(episode_index)].item())
    resolved = ep_from + int(episode_offset)
    if not (ep_from <= resolved < ep_to):
        raise IndexError(
            f"episode_offset={episode_offset} gives sample_index={resolved}, outside [{ep_from}, {ep_to})"
        )
    return resolved


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare server /act output against local infer_ego-style inference on one sample.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8015/act")
    parser.add_argument("--skip-server", action="store_true", help="Only compare local inference paths; do not query /act.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--image-format", choices=["PNG", "JPEG"], default="PNG")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(args.task)
    train_ds, val_ds = build_datasets(cfg.data)
    dataset = train_ds if args.split == "train" else val_ds
    processor = dataset.lerobot_dataset.processor
    sample_index = _resolve_sample_index(dataset, args.episode_index, args.sample_index, args.episode_offset)
    sample = dataset[sample_index]

    video = sample["video"]
    prompt = sample["prompt"]
    context = sample.get("context")
    context_mask = sample.get("context_mask")
    gt_action = sample.get("action")
    proprio_seq = sample.get("proprio")
    proprio = proprio_seq[0] if proprio_seq is not None else None
    input_image_tensor = video[:, 0].unsqueeze(0)
    num_frames = int(video.shape[1])
    action_horizon = int(gt_action.shape[0]) if gt_action is not None else int(cfg.data.train.num_frames - 1)

    model_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    model.load_checkpoint(args.checkpoint)
    model = model.to(args.device).eval()

    local_kwargs = {
        "prompt": None if context is not None else prompt,
        "input_image": input_image_tensor.to(device=model.device, dtype=model.torch_dtype),
        "num_frames": num_frames,
        "action": None,
        "action_horizon": action_horizon,
        "proprio": None if proprio is None else proprio.to(device=model.device, dtype=model.torch_dtype),
        "context": None if context is None else context.to(device=model.device, dtype=model.torch_dtype),
        "context_mask": None if context_mask is None else context_mask.to(device=model.device, dtype=torch.bool),
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "action_cfg_scale": 1.0,
        "num_inference_steps": int(args.num_inference_steps),
        "sigma_shift": None,
        "seed": int(args.seed),
        "rand_device": args.rand_device,
        "tiled": False,
    }
    local_infer = model.infer(**local_kwargs)
    local_infer_raw = local_infer["action"].detach().cpu()
    local_infer_denorm = _denormalize_action(processor, local_infer_raw, proprio_seq).detach().cpu().numpy()
    local_infer_denorm_no_proprio = _denormalize_action(processor, local_infer_raw, None).detach().cpu().numpy()

    joint_kwargs = dict(local_kwargs)
    joint_kwargs.pop("num_frames")
    joint_kwargs.pop("action_cfg_scale")
    local_joint = model.infer_joint(
        num_video_frames=num_frames,
        test_action_with_infer_action=False,
        **joint_kwargs,
    )
    local_joint_raw = local_joint["action"].detach().cpu()
    local_joint_denorm = _denormalize_action(processor, local_joint_raw, proprio_seq).detach().cpu().numpy()
    local_joint_denorm_no_proprio = _denormalize_action(processor, local_joint_raw, None).detach().cpu().numpy()

    local_action = model.infer_action(
        prompt=None if context is not None else prompt,
        input_image=input_image_tensor.to(device=model.device, dtype=model.torch_dtype),
        action_horizon=action_horizon,
        proprio=None if proprio is None else proprio.to(device=model.device, dtype=model.torch_dtype),
        context=None if context is None else context.to(device=model.device, dtype=model.torch_dtype),
        context_mask=None if context_mask is None else context_mask.to(device=model.device, dtype=torch.bool),
        num_inference_steps=int(args.num_inference_steps),
        sigma_shift=None,
        seed=int(args.seed),
        rand_device=args.rand_device,
        tiled=False,
    )
    local_action_raw = local_action["action"].detach().cpu()
    local_action_denorm = _denormalize_action(processor, local_action_raw, proprio_seq).detach().cpu().numpy()
    local_action_denorm_no_proprio = _denormalize_action(processor, local_action_raw, None).detach().cpu().numpy()

    image = _tensor_frame_to_image(video[:, 0])
    h, w = [int(x) for x in cfg.data.train.video_size]
    roundtrip_tensor = _image_to_input_tensor(image, (h, w))
    image_tensor_diff = _diff_stats(
        "dataset_input_tensor_vs_client_image_tensor",
        input_image_tensor.detach().cpu().numpy(),
        roundtrip_tensor.numpy(),
    )

    server_action = None
    server_meta = None
    if not args.skip_server:
        payload = {
            "image": _encode_image(image, fmt=args.image_format, quality=args.jpeg_quality),
            "return_video": False,
            "seed": int(args.seed),
        }
        if args.instruction is not None:
            payload["instruction"] = args.instruction
        server_data = _post_server(args.server_url, payload, args.timeout)
        server_action = _decode_numpy(server_data["action"]).astype(np.float32, copy=False)
        server_meta = server_data.get("meta", {})

    gt_denorm = None
    if gt_action is not None:
        gt_denorm = _denormalize_action(processor, gt_action.detach().cpu(), proprio_seq).detach().cpu().numpy()

    arrays = {
        "local_infer_denorm": local_infer_denorm,
        "local_infer_denorm_no_proprio": local_infer_denorm_no_proprio,
        "local_joint_denorm": local_joint_denorm,
        "local_joint_denorm_no_proprio": local_joint_denorm_no_proprio,
        "local_action_denorm": local_action_denorm,
        "local_action_denorm_no_proprio": local_action_denorm_no_proprio,
    }
    if server_action is not None:
        arrays["server_action"] = server_action
    if gt_denorm is not None:
        arrays["gt_denorm"] = gt_denorm
    for name, arr in arrays.items():
        np.save(output_dir / f"{name}.npy", arr)

    report = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "server_url": args.server_url,
        "skip_server": bool(args.skip_server),
        "split": args.split,
        "sample_index": int(sample_index),
        "episode_index": None if args.episode_index is None else int(args.episode_index),
        "episode_offset": int(args.episode_offset),
        "frame_index": int(sample.get("frame_index", sample_index)),
        "sample_prompt": prompt,
        "instruction_sent_to_server": args.instruction,
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "rand_device": args.rand_device,
        "image_format": args.image_format,
        "server_meta": server_meta,
        "has_proprio": proprio_seq is not None,
        "state_meta_len": len(processor.shape_meta.get("state", [])),
        "action_horizon": int(action_horizon),
        "num_frames": int(num_frames),
        "image_tensor_diff": image_tensor_diff,
        "array_stats": [_array_stats(name, arr) for name, arr in arrays.items()],
        "diffs": [
            _diff_stats("local_infer_vs_local_joint_test_false", local_infer_denorm, local_joint_denorm),
            _diff_stats("local_joint_test_false_vs_local_action", local_joint_denorm, local_action_denorm),
        ],
    }
    if server_action is not None:
        report["diffs"] = [
            _diff_stats("server_vs_local_infer", server_action, local_infer_denorm),
            _diff_stats("server_vs_local_infer_no_proprio_denorm", server_action, local_infer_denorm_no_proprio),
            _diff_stats("server_vs_local_joint_test_false", server_action, local_joint_denorm),
            _diff_stats("server_vs_local_joint_test_false_no_proprio_denorm", server_action, local_joint_denorm_no_proprio),
            _diff_stats("server_vs_local_action", server_action, local_action_denorm),
            _diff_stats("server_vs_local_action_no_proprio_denorm", server_action, local_action_denorm_no_proprio),
        ] + report["diffs"]
    if gt_denorm is not None:
        if server_action is not None:
            report["diffs"].append(_diff_stats("server_vs_gt", server_action, gt_denorm))
        report["diffs"].append(_diff_stats("local_infer_vs_gt", local_infer_denorm, gt_denorm))

    with open(output_dir / "compare_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[compare] saved report: {output_dir / 'compare_report.json'}")
    print(f"[compare] server_meta={server_meta}")
    print(f"[compare] image max_abs_diff={image_tensor_diff.get('max_abs')}")
    for item in report["diffs"]:
        print(
            f"[diff] {item['name']}: "
            f"mean_abs={item.get('mean_abs')} max_abs={item.get('max_abs')} rmse={item.get('rmse')}"
        )


if __name__ == "__main__":
    main()
