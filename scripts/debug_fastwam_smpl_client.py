import argparse
import base64
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _encode_image(image: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = int(quality)
    image.convert("RGB").save(buf, format=fmt.upper(), **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_image(args) -> Image.Image:
    if args.image is not None:
        return Image.open(Path(args.image).expanduser()).convert("RGB")
    arr = np.zeros((int(args.height), int(args.width), 3), dtype=np.uint8)
    # A tiny nonzero pattern helps catch accidental channel/resize assumptions.
    arr[..., 0] = 24
    arr[:, :, 1] = np.linspace(0, 128, int(args.width), dtype=np.uint8)[None, :]
    arr[:, :, 2] = np.linspace(0, 128, int(args.height), dtype=np.uint8)[:, None]
    return Image.fromarray(arr, mode="RGB")


def _tensor_frame_to_image(frame_chw) -> Image.Image:
    import torch

    if not isinstance(frame_chw, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor frame, got {type(frame_chw)}")
    arr = frame_chw.detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr).convert("RGB")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(max(int(round((len(xs) - 1) * q)), 0), len(xs) - 1)
    return float(xs[idx])


def _decode_numpy(obj: Any) -> np.ndarray:
    raw = base64.b64decode(obj["__numpy__"])
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


def _post_act(requests, url: str, payload: dict[str, Any], timeout: float, request_idx: int) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code != 200:
        raise RuntimeError(f"Request {request_idx} failed with status={resp.status_code}: {resp.text}")
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Request {request_idx} returned error: {data['error']}")
    return data, elapsed_ms


def _request_seed(base_seed: int, request_idx: int, seed_mode: str) -> int:
    if seed_mode == "constant":
        return int(base_seed)
    if seed_mode == "increment":
        return int(base_seed) + int(request_idx)
    raise ValueError(f"Unsupported --seed-mode: {seed_mode}")


def _summarize_latency(latencies_ms: list[float], server_latencies_ms: list[float], warmup: int, action_shape) -> None:
    if not latencies_ms:
        print("[result] no measured requests")
        return
    print(f"action_shape={action_shape}")
    print(
        "client_wall_ms "
        f"mean={statistics.fmean(latencies_ms):.2f} "
        f"p50={_percentile(latencies_ms, 0.50):.2f} "
        f"p95={_percentile(latencies_ms, 0.95):.2f} "
        f"min={min(latencies_ms):.2f} max={max(latencies_ms):.2f} "
        f"warmup={warmup}"
    )
    if server_latencies_ms:
        print(
            "server_meta_ms "
            f"mean={statistics.fmean(server_latencies_ms):.2f} "
            f"p50={_percentile(server_latencies_ms, 0.50):.2f} "
            f"p95={_percentile(server_latencies_ms, 0.95):.2f} "
            f"min={min(server_latencies_ms):.2f} max={max(server_latencies_ms):.2f}"
        )


def _run_dataset_mode(args, requests) -> None:
    from fastwam.runtime import build_datasets
    from infer_ego import _build_cfg, _denormalize_action

    if args.output_dir is None:
        raise ValueError("--output-dir is required when --task is provided.")

    cfg = _build_cfg(args.task)
    train_ds, val_ds = build_datasets(cfg.data)
    dataset = train_ds if args.split == "train" else val_ds
    processor = dataset.lerobot_dataset.processor

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.action_latency_steps is not None:
        action_latency_steps = max(int(args.action_latency_steps), 0)
    else:
        action_latency_steps = max(int(round(float(args.action_latency_ms) * float(args.control_hz) / 1000.0)), 0)

    ep_to = None
    if args.episode_index is None:
        sample_indices = [int(args.sample_index)] * int(args.num_requests)
        episode_meta = {
            "episode_index": None,
            "episode_start_index": None,
            "episode_end_index_exclusive": None,
            "advance_mode": "single_sample",
            "replan_steps": None,
            "control_hz": float(args.control_hz),
            "action_latency_ms": float(args.action_latency_ms),
            "action_latency_steps": int(action_latency_steps),
            "action_start_mode": args.action_start_mode,
            "seed_mode": args.seed_mode,
            "image_format": args.image_format,
        }
    else:
        ep_idx = int(args.episode_index)
        ep_from = int(dataset.lerobot_dataset.episode_data_index["from"][ep_idx].item())
        ep_to = int(dataset.lerobot_dataset.episode_data_index["to"][ep_idx].item())
        sample_indices = [ep_from]
        episode_meta = {
            "episode_index": ep_idx,
            "episode_start_index": ep_from,
            "episode_end_index_exclusive": ep_to,
            "advance_mode": args.advance_mode,
            "replan_steps": max(int(args.replan_steps), 1),
            "control_hz": float(args.control_hz),
            "action_latency_ms": float(args.action_latency_ms),
            "action_latency_steps": int(action_latency_steps),
            "action_start_mode": args.action_start_mode,
            "seed_mode": args.seed_mode,
            "image_format": args.image_format,
        }

    if not sample_indices:
        raise ValueError("No dataset samples selected.")

    first_for_warmup = dataset[sample_indices[0]]
    warmup_image = _tensor_frame_to_image(first_for_warmup["video"][:, 0])
    warmup_payload = {
        "image": _encode_image(warmup_image, fmt=args.image_format, quality=args.jpeg_quality),
        "return_video": bool(args.return_video),
        "seed": _request_seed(args.seed, 0, args.seed_mode),
    }
    if args.instruction is not None:
        warmup_payload["instruction"] = args.instruction

    for i in range(int(args.warmup)):
        _post_act(requests, args.url, warmup_payload, args.timeout, i)

    latencies_ms: list[float] = []
    server_latencies_ms: list[float] = []
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    rollout_pred_chunks: list[np.ndarray] = []
    rollout_gt_chunks: list[np.ndarray] = []
    rollout_source_indices: list[int] = []
    step_records: list[dict[str, Any]] = []
    action_shape = None

    measured_start = time.perf_counter()
    req_id = 0
    sample_index = int(sample_indices[0])
    while True:
        if args.episode_index is None:
            if req_id >= len(sample_indices):
                break
            sample_index = int(sample_indices[req_id])
        else:
            if ep_to is None or sample_index >= int(ep_to):
                break
            if args.max_steps is not None and req_id >= int(args.max_steps):
                break
            if args.num_requests > 0 and args.limit_to_num_requests and req_id >= int(args.num_requests):
                break

        sample = dataset[int(sample_index)]
        image = _tensor_frame_to_image(sample["video"][:, 0])
        payload: dict[str, Any] = {
            "image": _encode_image(image, fmt=args.image_format, quality=args.jpeg_quality),
            "return_video": bool(args.return_video),
            "seed": _request_seed(args.seed, req_id, args.seed_mode),
        }
        if args.instruction is not None:
            payload["instruction"] = args.instruction

        data, elapsed_ms = _post_act(requests, args.url, payload, args.timeout, req_id)
        action = _decode_numpy(data["action"]).astype(np.float32, copy=False)
        pred_chunks.append(action)
        action_shape = tuple(action.shape)
        meta = data.get("meta", {})
        if args.latency_source == "client":
            latency_ms_for_step = float(elapsed_ms)
        else:
            latency_ms_for_step = float(meta.get("latency_ms", elapsed_ms))
        if args.action_latency_steps is not None:
            step_latency_steps = action_latency_steps
        elif args.use_measured_latency:
            step_latency_steps = max(int(round(latency_ms_for_step * float(args.control_hz) / 1000.0)), 0)
        else:
            step_latency_steps = action_latency_steps

        gt_action = sample.get("action")
        if gt_action is not None:
            proprio_seq = sample.get("proprio")
            gt_denorm = _denormalize_action(processor, gt_action.detach().cpu(), proprio_seq)
            gt_np = gt_denorm.detach().cpu().numpy().astype(np.float32, copy=False)
            gt_chunks.append(gt_np)
        else:
            gt_np = None

        if args.episode_index is None:
            take_n = int(action.shape[0])
            action_start = 0
            advance_steps = 0
        else:
            remaining = int(episode_meta["episode_end_index_exclusive"]) - int(sample_index)
            if args.action_start_mode == "latency":
                action_start = min(step_latency_steps, int(action.shape[0]))
            elif args.action_start_mode == "zero":
                action_start = 0
            else:
                raise ValueError(f"Unsupported --action-start-mode: {args.action_start_mode}")
            if args.advance_mode == "latency":
                advance_steps = max(step_latency_steps, 1)
            elif args.advance_mode == "replan":
                advance_steps = max(int(args.replan_steps), 1)
            else:
                raise ValueError(f"Unsupported --advance-mode: {args.advance_mode}")
            take_n = min(
                advance_steps,
                max(remaining - action_start, 0),
                max(int(action.shape[0]) - action_start, 0),
            )
        if take_n > 0:
            action_end = action_start + take_n
            rollout_pred_chunks.append(action[action_start:action_end])
            rollout_source_indices.extend([int(sample_index) + action_start + i for i in range(take_n)])
            if gt_np is not None:
                rollout_gt_chunks.append(gt_np[action_start:action_end])

        latencies_ms.append(elapsed_ms)
        if "latency_ms" in meta:
            server_latencies_ms.append(float(meta["latency_ms"]))

        record = {
            "request_id": int(req_id),
            "sample_index": int(sample_index),
            "frame_index": int(sample.get("frame_index", sample_index)),
            "episode_index": int(sample.get("episode_index", args.episode_index if args.episode_index is not None else -1)),
            "client_wall_ms": float(elapsed_ms),
            "server_latency_ms": float(meta["latency_ms"]) if "latency_ms" in meta else None,
            "latency_ms_used_for_advance": float(latency_ms_for_step),
            "latency_source": str(args.latency_source),
            "action_shape": list(action.shape),
            "action_latency_steps": int(step_latency_steps),
            "action_start_mode": str(args.action_start_mode),
            "action_start_offset": int(action_start),
            "rollout_take_n": int(take_n),
            "advance_steps": int(advance_steps),
            "seed": int(payload["seed"]),
        }
        if gt_np is not None and take_n > 0 and gt_np.shape == action.shape:
            diff = action[action_start:action_start + take_n] - gt_np[action_start:action_start + take_n]
            record["action_l1"] = float(np.mean(np.abs(diff)))
            record["action_l2"] = float(np.mean(np.square(diff)))
        step_records.append(record)

        req_id += 1
        if args.episode_index is not None:
            sample_index = int(sample_index) + max(int(advance_steps), 1)

        progress_mod = max((int(args.max_steps) if args.max_steps is not None else 10) // 10, 1)
        if req_id % progress_mod == 0:
            print(
                f"[dataset] requests={req_id} next_sample={sample_index} "
                f"advance_steps={advance_steps} action_start={action_start} "
                f"latency_used_ms={latency_ms_for_step:.2f} last_wall_ms={elapsed_ms:.2f}"
            )

    measured_end = time.perf_counter()
    wall_s = max(measured_end - measured_start, 1e-9)
    hz = len(pred_chunks) / wall_s

    pred_actions_episode = np.stack(pred_chunks, axis=0)
    np.save(output_dir / "pred_actions_episode.npy", pred_actions_episode)
    if gt_chunks:
        gt_actions_episode = np.stack(gt_chunks, axis=0)
        np.save(output_dir / "gt_actions_episode.npy", gt_actions_episode)
    else:
        gt_actions_episode = None

    pred_actions_rollout = np.concatenate(rollout_pred_chunks, axis=0) if rollout_pred_chunks else None
    gt_actions_rollout = np.concatenate(rollout_gt_chunks, axis=0) if rollout_gt_chunks else None
    if pred_actions_rollout is not None:
        np.save(output_dir / "pred_actions_rollout.npy", pred_actions_rollout)
    if gt_actions_rollout is not None:
        np.save(output_dir / "gt_actions_rollout.npy", gt_actions_rollout)
    if rollout_source_indices:
        np.save(output_dir / "rollout_source_indices.npy", np.asarray(rollout_source_indices, dtype=np.int64))

    np.savez_compressed(
        output_dir / "actions.npz",
        pred_actions_episode=pred_actions_episode,
        gt_actions_episode=gt_actions_episode if gt_actions_episode is not None else np.empty((0,), dtype=np.float32),
        pred_actions_rollout=pred_actions_rollout if pred_actions_rollout is not None else np.empty((0,), dtype=np.float32),
        gt_actions_rollout=gt_actions_rollout if gt_actions_rollout is not None else np.empty((0,), dtype=np.float32),
        sample_indices=np.asarray([r["sample_index"] for r in step_records], dtype=np.int64),
        rollout_source_indices=np.asarray(rollout_source_indices, dtype=np.int64),
        action_latency_steps=np.asarray([action_latency_steps], dtype=np.int64),
    )
    with open(output_dir / "step_records.json", "w", encoding="utf-8") as f:
        json.dump(step_records, f, ensure_ascii=False, indent=2)
    metadata = {
        "task": args.task,
        "split": args.split,
        "url": args.url,
        "num_requests": len(pred_chunks),
        "warmup": int(args.warmup),
        "wall_s": float(wall_s),
        "hz": float(hz),
        "pred_actions_episode_path": str((output_dir / "pred_actions_episode.npy").resolve()),
        "gt_actions_episode_path": str((output_dir / "gt_actions_episode.npy").resolve()) if gt_actions_episode is not None else None,
        "pred_actions_rollout_path": str((output_dir / "pred_actions_rollout.npy").resolve()) if pred_actions_rollout is not None else None,
        "gt_actions_rollout_path": str((output_dir / "gt_actions_rollout.npy").resolve()) if gt_actions_rollout is not None else None,
        "actions_npz_path": str((output_dir / "actions.npz").resolve()),
        "rollout_num_steps": int(pred_actions_rollout.shape[0]) if pred_actions_rollout is not None else 0,
        "mean_client_wall_ms": float(statistics.fmean(latencies_ms)) if latencies_ms else None,
        "mean_server_latency_ms": float(statistics.fmean(server_latencies_ms)) if server_latencies_ms else None,
    }
    metadata.update(episode_meta)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n[result]")
    print(f"requests={len(pred_chunks)} warmup={int(args.warmup)} wall_s={wall_s:.3f} hz={hz:.3f}")
    _summarize_latency(latencies_ms, server_latencies_ms, int(args.warmup), action_shape)
    print(f"saved pred episode: {output_dir / 'pred_actions_episode.npy'}")
    if pred_actions_rollout is not None:
        print(f"saved pred rollout: {output_dir / 'pred_actions_rollout.npy'}")
    if gt_actions_episode is not None:
        print(f"saved gt episode: {output_dir / 'gt_actions_episode.npy'}")
    if gt_actions_rollout is not None:
        print(f"saved gt rollout: {output_dir / 'gt_actions_rollout.npy'}")
    print(f"saved npz: {output_dir / 'actions.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug FastWAM SMPL server latency/Hz via /act.")
    parser.add_argument("--url", default="http://127.0.0.1:8015/act")
    parser.add_argument("--health-url", default=None)
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, sends a dummy RGB image.")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-mode", choices=["increment", "constant"], default="increment", help="Use increment to vary seed per request; use constant to match scripts/infer_ego.py.")
    parser.add_argument("--image-format", choices=["JPEG", "PNG"], default="JPEG", help="Use PNG to avoid JPEG compression when comparing against offline tensor inference.")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--return-video", action="store_true", help="Ask server to return predicted video JPEGs. This measures network serialization too.")
    parser.add_argument("--task", default=None, help="Optional task config. If set, use dataset frames instead of dummy/single image.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--advance-mode", choices=["latency", "replan"], default="latency", help="Dataset episode mode: latency advances by latency steps; replan advances by --replan-steps.")
    parser.add_argument("--replan-steps", type=int, default=1, help="Only used with --advance-mode replan.")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Action control rate used to convert latency ms to skipped action steps.")
    parser.add_argument("--action-latency-ms", type=float, default=0.0, help="Simulated inference/communication latency. Rollout skips this many ms of the predicted chunk.")
    parser.add_argument("--action-latency-steps", type=int, default=None, help="Override latency in action steps. If set, ignores --action-latency-ms.")
    parser.add_argument("--action-start-mode", choices=["latency", "zero"], default="latency", help="latency skips the stale prefix of each chunk; zero always takes action[0:], matching scripts/infer_ego.py rollout.")
    parser.add_argument("--use-measured-latency", action="store_true", help="Use each request's measured server latency to compute skipped/advanced action steps.")
    parser.add_argument("--latency-source", choices=["server", "client"], default="client", help="When --use-measured-latency is set, use server model latency or client wall-clock latency.")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit selected dataset requests when --episode-index is used.")
    parser.add_argument("--limit-to-num-requests", action="store_true", help="Also cap dataset episode mode by --num-requests.")
    parser.add_argument("--output-dir", default=None, help="Directory to save actions.npz and pred/gt action .npy in dataset mode.")
    args = parser.parse_args()

    try:
        import requests
    except ImportError as exc:
        raise ImportError("This debug client requires `requests`. Install it in the same env first.") from exc

    health_url = args.health_url
    if health_url is None and args.url.endswith("/act"):
        health_url = args.url[:-4] + "health"

    if health_url:
        try:
            health = requests.get(health_url, timeout=args.timeout)
            print(f"[health] status={health.status_code} body={health.text[:500]}")
        except Exception as exc:
            print(f"[health] failed: {exc}")

    if args.task is not None:
        _run_dataset_mode(args, requests)
        return

    image_b64 = _encode_image(_make_image(args), fmt=args.image_format, quality=args.jpeg_quality)
    payload: dict[str, Any] = {
        "image": image_b64,
        "return_video": bool(args.return_video),
        "seed": _request_seed(args.seed, 0, args.seed_mode),
    }
    if args.instruction is not None:
        payload["instruction"] = args.instruction

    latencies_ms = []
    server_latencies_ms = []
    action_shape = None

    total = int(args.warmup) + int(args.num_requests)
    measured_start = None
    measured_end = None
    for i in range(total):
        is_warmup = i < int(args.warmup)
        if not is_warmup and measured_start is None:
            measured_start = time.perf_counter()

        data, elapsed_ms = _post_act(requests, args.url, payload, args.timeout, i)

        if not is_warmup:
            latencies_ms.append(elapsed_ms)
            meta = data.get("meta", {})
            if "latency_ms" in meta:
                server_latencies_ms.append(float(meta["latency_ms"]))
            if action_shape is None and "action" in data:
                action_shape = tuple(_decode_numpy(data["action"]).shape)

        if (i + 1) % max(total // 10, 1) == 0:
            phase = "warmup" if is_warmup else "measure"
            print(f"[{phase}] {i + 1}/{total} last_wall_ms={elapsed_ms:.2f}")

    measured_end = time.perf_counter()
    wall_s = max((measured_end - measured_start), 1e-9) if measured_start is not None else 0.0
    hz = len(latencies_ms) / wall_s if wall_s > 0 else 0.0

    print("\n[result]")
    print(f"requests={len(latencies_ms)} warmup={int(args.warmup)} wall_s={wall_s:.3f} hz={hz:.3f}")
    _summarize_latency(latencies_ms, server_latencies_ms, int(args.warmup), action_shape)


if __name__ == "__main__":
    main()
