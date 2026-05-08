import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

from infer_ego import (
    _build_cfg,
    _collect_rollout_video_frames,
    _denormalize_action,
    _tensor_video_to_frames,
    _to_uint8_frame,
)
from fastwam.runtime import build_datasets
from fastwam.utils.video_io import save_mp4


def _pil_frame_to_input_tensor(frame) -> torch.Tensor:
    arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32)
    return tensor / 127.5 - 1.0


def _get_dataset_dirs(dataset) -> list[str]:
    lerobot_dataset = getattr(dataset, "lerobot_dataset", None)
    if lerobot_dataset is None:
        return []
    return [str(Path(p).resolve()) for p in getattr(lerobot_dataset, "dataset_dirs", [])]


def _get_episode_data_path(dataset, episode_index: int) -> str | None:
    lerobot_dataset = getattr(dataset, "lerobot_dataset", None)
    multi_dataset = getattr(lerobot_dataset, "multi_dataset", None)
    if multi_dataset is None:
        return None

    local_episode_index = int(episode_index)
    for sub_dataset in getattr(multi_dataset, "_datasets", []):
        if local_episode_index < sub_dataset.num_episodes:
            raw_episode_index = sub_dataset.episodes[local_episode_index]
            return str((sub_dataset.root / sub_dataset.meta.get_data_file_path(raw_episode_index)).resolve())
        local_episode_index -= sub_dataset.num_episodes
    return None


def main():
    parser = argparse.ArgumentParser(description="Autoregressive episode inference for FastWAM using only the first GT frame.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Diffusion inference steps for model.infer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--save-per-step-videos", action="store_true")
    args = parser.parse_args()

    cfg = _build_cfg(args.task)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.mixed_precision]

    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    model.load_checkpoint(args.checkpoint)
    model = model.to(args.device).eval()

    train_ds, val_ds = build_datasets(cfg.data)
    dataset = train_ds if args.split == "train" else val_ds
    processor = dataset.lerobot_dataset.processor

    ep_idx = int(args.episode_index)
    dataset_dirs = _get_dataset_dirs(dataset)
    episode_data_path = _get_episode_data_path(dataset, ep_idx)
    print(f"[infer_ego_autoreg] split={args.split}")
    print(f"[infer_ego_autoreg] dataset_dirs={dataset_dirs}")
    if episode_data_path is not None:
        print(f"[infer_ego_autoreg] episode_data_path={episode_data_path}")
    print(f"[infer_ego_autoreg] num_inference_steps={int(args.num_inference_steps)}")

    ep_from = int(dataset.lerobot_dataset.episode_data_index["from"][ep_idx].item())
    ep_to = int(dataset.lerobot_dataset.episode_data_index["to"][ep_idx].item())
    first_sample = dataset[ep_from]
    prompt = first_sample["prompt"]
    context = first_sample.get("context")
    context_mask = first_sample.get("context_mask")

    sample_video = first_sample["video"]
    num_frames = int(sample_video.shape[1])
    action_horizon = int(first_sample["action"].shape[0])
    chunk_stride = action_horizon

    current_input_image = sample_video[:, 0].unsqueeze(0)
    sample_index = ep_from
    step_id = 0

    pred_video_frames = []
    gt_video_frames = []
    pred_action_chunks = []
    gt_action_chunks = []
    step_records = []

    while sample_index < ep_to:
        sample = dataset[sample_index]
        proprio_seq = sample.get("proprio")
        proprio = proprio_seq[0] if proprio_seq is not None else None

        infer_kwargs = {
            "prompt": None if context is not None else prompt,
            "input_image": current_input_image.to(device=model.device, dtype=model.torch_dtype),
            "num_frames": num_frames,
            "action": None,
            "action_horizon": action_horizon,
            "proprio": None if proprio is None else proprio.to(device=model.device, dtype=model.torch_dtype),
            "context": None if context is None else context.to(device=model.device, dtype=model.torch_dtype),
            "context_mask": None if context_mask is None else context_mask.to(device=model.device, dtype=torch.bool),
            "negative_prompt": "",
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": args.num_inference_steps,
            "sigma_shift": None,
            "seed": args.seed + step_id,
            "rand_device": args.rand_device,
            "tiled": False,
        }

        with torch.no_grad():
            infer_out = model.infer(**infer_kwargs)

        pred_video = infer_out["video"]
        pred_action = infer_out.get("action")
        gt_action = sample.get("action")

        pred_action_denorm = None
        gt_action_denorm = None
        if pred_action is not None:
            pred_action_denorm = _denormalize_action(processor, pred_action.detach().cpu(), proprio_seq)
            pred_action_chunks.append(pred_action_denorm.detach().cpu().numpy())
        if gt_action is not None:
            gt_action_denorm = _denormalize_action(processor, gt_action.detach().cpu(), proprio_seq)
            gt_action_chunks.append(gt_action_denorm.detach().cpu().numpy())

        if step_id == 0:
            pred_video_frames.extend(list(pred_video))
            gt_video_frames.extend(_tensor_video_to_frames(sample["video"]))
        else:
            pred_video_frames.extend(list(pred_video[1:]))
            gt_video_frames.extend(_collect_rollout_video_frames(sample["video"], 1, num_frames - 1))

        last_pred_frame = pred_video[-1]
        current_input_image = _pil_frame_to_input_tensor(last_pred_frame).unsqueeze(0)

        record = {
            "step_id": int(step_id),
            "sample_index": int(sample_index),
            "frame_index": int(sample.get("frame_index", sample_index)),
            "episode_index": int(sample.get("episode_index", ep_idx)),
            "chunk_stride": int(chunk_stride),
            "action_horizon": int(action_horizon),
            "num_frames": int(num_frames),
            "input_source": "gt_first_frame" if step_id == 0 else "prev_pred_last_frame",
        }
        if pred_action_denorm is not None and gt_action_denorm is not None:
            diff = pred_action_denorm - gt_action_denorm
            record["action_l1"] = float(diff.abs().mean().item())
            record["action_l2"] = float(diff.pow(2).mean().item())
        if args.save_per_step_videos:
            step_dir = output_dir / f"step_{step_id:04d}_sample_{sample_index:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_mp4(pred_video, str(step_dir / "pred_video.mp4"), fps=8)
            save_mp4(_tensor_video_to_frames(sample["video"]), str(step_dir / "gt_video.mp4"), fps=8)
            _to_uint8_frame(sample["video"][:, 0]).save(step_dir / "gt_input_frame.png")
            last_pred_frame.save(step_dir / "next_input_frame.png")
            record["step_dir"] = str(step_dir.resolve())
        step_records.append(record)

        sample_index += chunk_stride
        step_id += 1

    if pred_video_frames:
        save_mp4(pred_video_frames, str(output_dir / "pred_video_autoreg.mp4"), fps=8)
    if gt_video_frames:
        save_mp4(gt_video_frames, str(output_dir / "gt_video_autoreg.mp4"), fps=8)

    pred_actions = None
    if pred_action_chunks:
        pred_actions = np.concatenate(pred_action_chunks, axis=0)
        np.save(output_dir / "pred_actions_autoreg.npy", pred_actions)
    gt_actions = None
    if gt_action_chunks:
        gt_actions = np.concatenate(gt_action_chunks, axis=0)
        np.save(output_dir / "gt_actions_autoreg.npy", gt_actions)

    metadata = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "num_inference_steps": int(args.num_inference_steps),
        "episode_index": ep_idx,
        "episode_start_index": ep_from,
        "episode_end_index_exclusive": ep_to,
        "chunk_stride": int(chunk_stride),
        "num_chunks": int(step_id),
        "num_frames_per_chunk": int(num_frames),
        "action_horizon_per_chunk": int(action_horizon),
        "pred_video_autoreg_path": str((output_dir / "pred_video_autoreg.mp4").resolve()) if pred_video_frames else None,
        "gt_video_autoreg_path": str((output_dir / "gt_video_autoreg.mp4").resolve()) if gt_video_frames else None,
        "pred_actions_autoreg_path": str((output_dir / "pred_actions_autoreg.npy").resolve()) if pred_actions is not None else None,
        "gt_actions_autoreg_path": str((output_dir / "gt_actions_autoreg.npy").resolve()) if gt_actions is not None else None,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(output_dir / "step_records.json", "w", encoding="utf-8") as f:
        json.dump(step_records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
