import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from fastwam.runtime import build_datasets
from fastwam.datasets.lerobot.utils.rotation import rotation_6d_to_matrix
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.video_io import save_mp4


def _to_uint8_frame(frame_chw: torch.Tensor) -> Image.Image:
    arr = frame_chw.detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def _save_tensor_video(video_cthw: torch.Tensor, path: Path, fps: int = 8) -> None:
    frames = [_to_uint8_frame(video_cthw[:, t]) for t in range(video_cthw.shape[1])]
    save_mp4(frames, str(path), fps=fps)


def _tensor_video_to_frames(video_cthw: torch.Tensor):
    return [_to_uint8_frame(video_cthw[:, t]) for t in range(video_cthw.shape[1])]


def _slice_pred_video_frames(pred_video, take_n: int):
    if take_n <= 0:
        return []
    return list(pred_video[:take_n])


def _collect_rollout_video_frames(video_source, start_idx: int, count: int):
    if count <= 0:
        return []
    end_idx = start_idx + count
    if isinstance(video_source, torch.Tensor):
        return _tensor_video_to_frames(video_source[:, start_idx:end_idx])
    return list(video_source[start_idx:end_idx])


def _denormalize_action(processor, action_txd: torch.Tensor, proprio_txd: Optional[torch.Tensor]) -> torch.Tensor:
    if action_txd.ndim != 2:
        raise ValueError(f"`action_txd` must be [T, D], got {tuple(action_txd.shape)}")
    batch = {
        "action": action_txd.unsqueeze(0).to(dtype=torch.float32, device="cpu"),
    }
    if proprio_txd is not None:
        batch["state"] = proprio_txd.unsqueeze(0).to(dtype=torch.float32, device="cpu")

    if "state" in batch:
        batch = processor.action_state_merger.backward(batch)
        batch = processor.normalizer.backward(batch)
        merged_batch = {
            "action": {
                meta["key"]: batch["action"][meta["key"]].squeeze(0)
                for meta in processor.shape_meta["action"]
            },
            "state": {
                meta["key"]: batch["state"][meta["key"]].squeeze(0)
                for meta in processor.shape_meta["state"]
            },
        }
        merged_batch = processor.action_state_merger.forward(merged_batch)
        return merged_batch["action"]

    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged_batch = {
        "action": {
            meta["key"]: batch["action"][meta["key"]].squeeze(0)
            for meta in processor.shape_meta["action"]
        },
        "state": {},
    }
    merged_batch = processor.action_state_merger.forward(merged_batch)
    return merged_batch["action"]


def _project_points(points: np.ndarray, width: int, height: int, view: str) -> np.ndarray:
    if view == "front":
        xy = points[:, [0, 1]]
    elif view == "side":
        xy = points[:, [2, 1]]
    elif view == "top":
        xy = points[:, [0, 2]]
    else:
        raise ValueError(f"Unsupported view: {view}")

    xy = xy.astype(np.float32)
    min_xy = xy.min(axis=0, keepdims=True)
    max_xy = xy.max(axis=0, keepdims=True)
    span = np.maximum(max_xy - min_xy, 1e-6)
    xy = (xy - min_xy) / span
    xy[:, 0] = xy[:, 0] * (width * 0.8) + width * 0.1
    xy[:, 1] = (1.0 - xy[:, 1]) * (height * 0.8) + height * 0.1
    return xy


def _draw_skeleton_frame(
    joints_xyz: np.ndarray,
    title: str,
    width: int = 960,
    height: int = 320,
) -> Image.Image:
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10),
        (2, 5), (5, 8), (8, 11), (3, 6), (6, 9), (9, 12),
        (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        (12, 14), (14, 17), (17, 19), (19, 21), (21, 23), (12, 15),
    ]
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    panel_w = width // 3
    for panel_idx, view in enumerate(["front", "side", "top"]):
        proj = _project_points(joints_xyz, panel_w, height - 30, view=view)
        x_offset = panel_idx * panel_w
        draw.text((x_offset + 8, 8), view, fill=(20, 20, 20))
        for a, b in edges:
            xa, ya = proj[a]
            xb, yb = proj[b]
            draw.line((x_offset + xa, ya + 20, x_offset + xb, yb + 20), fill=(40, 90, 180), width=3)
        for x, y in proj:
            r = 3
            draw.ellipse((x_offset + x - r, y + 20 - r, x_offset + x + r, y + 20 + r), fill=(220, 60, 60))
    draw.text((8, height - 18), title, fill=(20, 20, 20))
    return canvas


def _stack_videos_h(frames_a, frames_b):
    out = []
    for fa, fb in zip(frames_a, frames_b):
        wa, ha = fa.size
        wb, hb = fb.size
        canvas = Image.new("RGB", (wa + wb, max(ha, hb)), (255, 255, 255))
        canvas.paste(fa, (0, 0))
        canvas.paste(fb, (wa, 0))
        out.append(canvas)
    return out


def _infer_smpl_layout(action_dim: int, layout: str) -> str:
    if layout != "auto":
        return layout
    if action_dim == 72:
        return "pose72"
    if action_dim == 75:
        return "trans3_pose72"
    if action_dim == 143:
        return "ego143"
    if action_dim == 69:
        return "body69"
    raise ValueError(
        f"Cannot auto-infer SMPL layout from action_dim={action_dim}. "
        "Please pass --smpl-layout explicitly."
    )


def _rotation_matrix_from_name(name: str) -> np.ndarray:
    if name == "none":
        return np.eye(3, dtype=np.float32)

    angle_map = {
        "x90": ("x", np.pi / 2),
        "xm90": ("x", -np.pi / 2),
        "y90": ("y", np.pi / 2),
        "ym90": ("y", -np.pi / 2),
        "z90": ("z", np.pi / 2),
        "zm90": ("z", -np.pi / 2),
        "x180": ("x", np.pi),
        "y180": ("y", np.pi),
        "z180": ("z", np.pi),
    }
    if name not in angle_map:
        raise ValueError(f"Unsupported rotation fix: {name}")

    axis, angle = angle_map[name]
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    if axis == "x":
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.ndim != 2 or axis_angle.shape[1] != 3:
        raise ValueError(f"Expected axis-angle tensor [N, 3], got {tuple(axis_angle.shape)}")
    theta = torch.linalg.norm(axis_angle, dim=1, keepdim=True)
    axis = axis_angle / theta.clamp(min=1e-8)
    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(axis_angle.shape[0], -1, -1)
    theta_expand = theta.unsqueeze(-1)
    sin_theta = torch.sin(theta_expand)
    cos_theta = torch.cos(theta_expand)
    outer = axis.unsqueeze(-1) @ axis.unsqueeze(1)
    return cos_theta * eye + (1 - cos_theta) * outer + sin_theta * K


def _matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotation matrix tensor [N, 3, 3], got {tuple(matrix.shape)}")
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [
            matrix[:, 2, 1] - matrix[:, 1, 2],
            matrix[:, 0, 2] - matrix[:, 2, 0],
            matrix[:, 1, 0] - matrix[:, 0, 1],
        ],
        dim=1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.empty_like(theta)
    small = theta < 1e-6
    scale[~small] = theta[~small] / (2.0 * sin_theta[~small].clamp(min=1e-8))
    scale[small] = 0.5
    return vee * scale.unsqueeze(1)


def _apply_root_orient_fix(global_orient: torch.Tensor, root_orient_fix: str) -> torch.Tensor:
    if root_orient_fix == "none":
        return global_orient
    fix_rot = torch.from_numpy(_rotation_matrix_from_name(root_orient_fix)).to(device=global_orient.device, dtype=global_orient.dtype)
    current_rot = _axis_angle_to_matrix(global_orient)
    fixed_rot = fix_rot.unsqueeze(0) @ current_rot
    return _matrix_to_axis_angle(fixed_rot)


def _actions_to_smpl_joints(
    action_txd: torch.Tensor,
    smpl_model_path: Path,
    smpl_layout: str,
    gender: str,
    root_orient_fix: str = "none",
    zero_global_orient: bool = False,
) -> np.ndarray:
    legacy_numpy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for alias, value in legacy_numpy_aliases.items():
        if not hasattr(np, alias):
            setattr(np, alias, value)
    if not hasattr(np, "unicode"):
        setattr(np, "unicode", str)

    try:
        import smplx
    except ImportError as exc:
        raise ImportError(
            "SMPL visualization requires `smplx`. Install it in the current environment first."
        ) from exc

    action = action_txd.detach().to(device="cpu", dtype=torch.float32)
    T, D = action.shape
    layout = _infer_smpl_layout(D, smpl_layout)

    transl = torch.zeros((T, 3), dtype=torch.float32)
    global_orient = torch.zeros((T, 3), dtype=torch.float32)
    body_pose = torch.zeros((T, 69), dtype=torch.float32)

    if layout == "pose72":
        global_orient = action[:, :3]
        body_pose = action[:, 3:72]
    elif layout == "trans3_pose72":
        transl = action[:, :3]
        global_orient = action[:, 3:6]
        body_pose = action[:, 6:75]
    elif layout == "pose69_trans3":
        body_pose = action[:, :69]
        transl = action[:, 69:72]
    elif layout == "body69":
        body_pose = action[:, :69]
    elif layout == "ego143":
        smpl_pose63 = action[:, 72:135]
        body_rot6d = action[:, 137:143]
        body_pose[:, :63] = smpl_pose63
        global_orient = _matrix_to_axis_angle(rotation_6d_to_matrix(body_rot6d))
    else:
        raise ValueError(f"Unsupported smpl layout: {layout}")

    if zero_global_orient:
        global_orient = torch.zeros_like(global_orient)
    global_orient = _apply_root_orient_fix(global_orient, root_orient_fix)

    model = smplx.create(
        str(smpl_model_path),
        model_type="smpl",
        gender=gender,
        use_pca=False,
        batch_size=T,
    )
    model = model.to(device="cpu")
    out = model(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        return_verts=False,
    )
    return out.joints[:, :24].detach().cpu().numpy()


def _render_smpl_video(
    action_txd: torch.Tensor,
    output_path: Path,
    smpl_model_path: Path,
    smpl_layout: str,
    gender: str,
    title_prefix: str,
    root_orient_fix: str = "none",
) -> None:
    joints = _actions_to_smpl_joints(
        action_txd=action_txd,
        smpl_model_path=smpl_model_path,
        smpl_layout=smpl_layout,
        gender=gender,
        root_orient_fix=root_orient_fix,
    )
    frames = [
        _draw_skeleton_frame(joints[t], title=f"{title_prefix} frame={t:03d}")
        for t in range(joints.shape[0])
    ]
    save_mp4(frames, str(output_path), fps=8)


def _build_cfg(task_name: str):
    register_default_resolvers()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])
    return cfg


def _resolve_episode_for_sample(dataset, sample_index: int) -> tuple[int, int, int]:
    episode_from = dataset.lerobot_dataset.episode_data_index["from"]
    episode_to = dataset.lerobot_dataset.episode_data_index["to"]
    sample_index = int(sample_index)
    for ep_idx in range(len(episode_from)):
        ep_from = int(episode_from[ep_idx].item())
        ep_to = int(episode_to[ep_idx].item())
        if ep_from <= sample_index < ep_to:
            frame_in_episode = sample_index - ep_from
            return ep_idx, ep_from, frame_in_episode
    raise IndexError(f"Could not resolve sample_index={sample_index} to an episode.")


def _resolve_raw_episode_id(dataset, split_episode_index: int) -> tuple[int, str, int]:
    remaining = int(split_episode_index)
    multi_dataset = dataset.lerobot_dataset.multi_dataset
    for dataset_idx, sub_dataset in enumerate(multi_dataset._datasets):
        if remaining < sub_dataset.num_episodes:
            raw_episode_index = int(sub_dataset.episodes[remaining])
            repo_id = str(sub_dataset.repo_id)
            return dataset_idx, repo_id, raw_episode_index
        remaining -= sub_dataset.num_episodes
    raise IndexError(f"Could not resolve split episode index {split_episode_index} to a raw episode id.")


def _run_single_inference(model, sample, processor, args, output_dir: Path, save_visuals: bool = True):
    video = sample["video"]
    prompt = sample["prompt"]
    context = sample.get("context")
    context_mask = sample.get("context_mask")
    gt_action = sample.get("action")
    proprio_seq = sample.get("proprio")
    proprio = proprio_seq[0] if proprio_seq is not None else None
    input_image = video[:, 0].unsqueeze(0)
    num_frames = int(video.shape[1])
    action_horizon = None if gt_action is None else int(gt_action.shape[0])

    infer_kwargs = {
        "prompt": None if context is not None else prompt,
        "input_image": input_image.to(device=model.device, dtype=model.torch_dtype),
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
        "seed": args.seed,
        "rand_device": args.rand_device,
        "tiled": False,
    }

    with torch.no_grad():
        infer_out = model.infer(**infer_kwargs)

    pred_video = infer_out["video"]
    pred_action = infer_out.get("action")

    pred_action_denorm = None
    gt_action_denorm = None
    if pred_action is not None:
        pred_action_denorm = _denormalize_action(processor, pred_action.detach().cpu(), proprio_seq)
    if gt_action is not None:
        gt_action_denorm = _denormalize_action(processor, gt_action.detach().cpu(), proprio_seq)

    metadata = {
        "prompt": prompt,
        "num_frames": num_frames,
        "action_horizon": action_horizon,
    }

    if pred_action_denorm is not None and gt_action_denorm is not None:
        diff = pred_action_denorm - gt_action_denorm
        metadata["action_l1"] = float(diff.abs().mean().item())
        metadata["action_l2"] = float(diff.pow(2).mean().item())

    if save_visuals:
        output_dir.mkdir(parents=True, exist_ok=True)
        _to_uint8_frame(video[:, 0]).save(output_dir / "input_frame.png")
        _save_tensor_video(video, output_dir / "gt_video.mp4", fps=8)
        save_mp4(pred_video, str(output_dir / "pred_video.mp4"), fps=8)
        if pred_action_denorm is not None:
            np.save(output_dir / "pred_action.npy", pred_action_denorm.detach().cpu().numpy())
        if gt_action_denorm is not None:
            np.save(output_dir / "gt_action.npy", gt_action_denorm.detach().cpu().numpy())

        metadata["pred_video_path"] = str((output_dir / "pred_video.mp4").resolve())
        metadata["gt_video_path"] = str((output_dir / "gt_video.mp4").resolve())
        metadata["input_frame_path"] = str((output_dir / "input_frame.png").resolve())

    return {
        "video": video,
        "pred_video": pred_video,
        "pred_action_denorm": pred_action_denorm,
        "gt_action_denorm": gt_action_denorm,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference + visualization helper for ego FastWAM checkpoints.")
    parser.add_argument("--task", required=True, help="Task config name, e.g. ego_idm_224_smpl75_1e-4")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to save inference outputs")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None, help="Run and save a whole episode instead of a single sample.")
    parser.add_argument("--save-per-step-videos", action="store_true", help="When using --episode-index, also save per-step predicted videos.")
    parser.add_argument("--replan-steps", type=int, default=None, help="When using --episode-index, re-run inference every N steps and stitch the first N predicted actions from each chunk.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--render-smpl", action="store_true")
    parser.add_argument("--smpl-model-path", default=None, help="Path to SMPL model directory/file for `smplx.create`.")
    parser.add_argument("--smpl-layout", default="auto", choices=["auto", "pose72", "trans3_pose72", "pose69_trans3", "body69", "ego143"])
    parser.add_argument("--smpl-gender", default="neutral", choices=["neutral", "male", "female"])
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
    base_metadata = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
    }

    if args.episode_index is None:
        sample = dataset[args.sample_index]
        result = _run_single_inference(model, sample, processor, args, output_dir, save_visuals=True)
        metadata = dict(base_metadata)
        metadata["sample_index"] = int(args.sample_index)
        resolved_episode_index, episode_start_index, frame_in_episode = _resolve_episode_for_sample(dataset, args.sample_index)
        source_dataset_index, source_repo_id, raw_episode_index = _resolve_raw_episode_id(dataset, resolved_episode_index)
        metadata["episode_index"] = int(resolved_episode_index)
        metadata["raw_episode_index"] = int(raw_episode_index)
        metadata["source_dataset_index"] = int(source_dataset_index)
        metadata["source_repo_id"] = source_repo_id
        metadata["episode_start_index"] = int(episode_start_index)
        metadata["frame_index_in_episode"] = int(frame_in_episode)
        metadata.update(result["metadata"])
        print(
            f"[infer_ego] split={args.split} sample_index={int(args.sample_index)} "
            f"-> split_episode_index={resolved_episode_index}, raw_episode_index={raw_episode_index}, "
            f"frame_in_episode={frame_in_episode}"
        )

        pred_action_denorm = result["pred_action_denorm"]
        gt_action_denorm = result["gt_action_denorm"]
        if args.render_smpl:
            if args.smpl_model_path is None:
                raise ValueError("--render-smpl requires --smpl-model-path")
            smpl_model_path = Path(args.smpl_model_path).resolve()
            if pred_action_denorm is not None:
                _render_smpl_video(
                    action_txd=pred_action_denorm,
                    output_path=output_dir / "pred_smpl.mp4",
                    smpl_model_path=smpl_model_path,
                    smpl_layout=args.smpl_layout,
                    gender=args.smpl_gender,
                    title_prefix="pred",
                )
                metadata["pred_smpl_path"] = str((output_dir / "pred_smpl.mp4").resolve())
            if gt_action_denorm is not None:
                _render_smpl_video(
                    action_txd=gt_action_denorm,
                    output_path=output_dir / "gt_smpl.mp4",
                    smpl_model_path=smpl_model_path,
                    smpl_layout=args.smpl_layout,
                    gender=args.smpl_gender,
                    title_prefix="gt",
                )
                metadata["gt_smpl_path"] = str((output_dir / "gt_smpl.mp4").resolve())
            if pred_action_denorm is not None and gt_action_denorm is not None:
                import imageio

                pred_reader = imageio.get_reader(str(output_dir / "pred_smpl.mp4"))
                gt_reader = imageio.get_reader(str(output_dir / "gt_smpl.mp4"))
                compare_frames = []
                for pred_arr, gt_arr in zip(pred_reader, gt_reader):
                    compare_frames.append(
                        _stack_videos_h([Image.fromarray(gt_arr)], [Image.fromarray(pred_arr)])[0]
                    )
                save_mp4(compare_frames, str(output_dir / "smpl_compare.mp4"), fps=8)
                metadata["smpl_compare_path"] = str((output_dir / "smpl_compare.mp4").resolve())

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return

    ep_idx = int(args.episode_index)
    source_dataset_index, source_repo_id, raw_episode_index = _resolve_raw_episode_id(dataset, ep_idx)
    ep_from = int(dataset.lerobot_dataset.episode_data_index["from"][ep_idx].item())
    ep_to = int(dataset.lerobot_dataset.episode_data_index["to"][ep_idx].item())
    episode_dir = output_dir / f"episode_{ep_idx:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[infer_ego] split={args.split} split_episode_index={ep_idx} "
        f"-> raw_episode_index={raw_episode_index} (repo={source_repo_id})"
    )

    step_records = []
    pred_actions = []
    gt_actions = []
    input_frames = []
    action_l1 = []
    action_l2 = []
    rollout_pred_actions = []
    rollout_gt_actions = []
    rollout_source_indices = []
    rollout_pred_video_frames = []
    rollout_gt_video_frames = []

    replan_steps = None if args.replan_steps is None else max(int(args.replan_steps), 1)
    if replan_steps is None:
        replan_steps = 1
    action_video_freq_ratio = int(dataset.action_video_freq_ratio)
    can_stitch_rollout_video = (replan_steps % action_video_freq_ratio == 0)
    video_chunk_advance = replan_steps // action_video_freq_ratio if can_stitch_rollout_video else 0

    sample_index = ep_from
    is_first_chunk = True
    while sample_index < ep_to:
        sample = dataset[sample_index]
        step_dir = episode_dir / f"step_{sample_index:06d}"
        result = _run_single_inference(
            model,
            sample,
            processor,
            args,
            step_dir,
            save_visuals=bool(args.save_per_step_videos),
        )

        input_frames.append(_to_uint8_frame(result["video"][:, 0]))
        if result["pred_action_denorm"] is not None:
            pred_actions.append(result["pred_action_denorm"].detach().cpu().numpy())
        if result["gt_action_denorm"] is not None:
            gt_actions.append(result["gt_action_denorm"].detach().cpu().numpy())

        if result["pred_action_denorm"] is not None:
            remaining = ep_to - sample_index
            take_n = min(
                replan_steps,
                remaining,
                int(result["pred_action_denorm"].shape[0]),
            )
            rollout_pred_actions.append(result["pred_action_denorm"][:take_n].detach().cpu().numpy())
            rollout_source_indices.extend([int(sample_index)] * take_n)
            if can_stitch_rollout_video:
                if is_first_chunk:
                    chunk_video_count = min(video_chunk_advance + 1, int(result["video"].shape[1]))
                    chunk_video_start = 0
                else:
                    chunk_video_count = min(video_chunk_advance, max(int(result["video"].shape[1]) - 1, 0))
                    chunk_video_start = 1
                rollout_pred_video_frames.extend(
                    _collect_rollout_video_frames(result["pred_video"], chunk_video_start, chunk_video_count)
                )
                rollout_gt_video_frames.extend(
                    _collect_rollout_video_frames(result["video"], chunk_video_start, chunk_video_count)
                )
            if result["gt_action_denorm"] is not None:
                rollout_gt_actions.append(result["gt_action_denorm"][:take_n].detach().cpu().numpy())

        record = {
            "sample_index": int(sample_index),
            "frame_index": int(sample.get("frame_index", sample_index)),
            "episode_index": int(sample.get("episode_index", ep_idx)),
        }
        record.update(result["metadata"])
        if "action_l1" in record:
            action_l1.append(record["action_l1"])
        if "action_l2" in record:
            action_l2.append(record["action_l2"])
        if args.save_per_step_videos:
            record["step_dir"] = str(step_dir.resolve())
        record["replan_steps"] = int(replan_steps)
        record["action_video_freq_ratio"] = int(action_video_freq_ratio)
        record["video_rollout_stitched"] = bool(can_stitch_rollout_video)
        step_records.append(record)
        is_first_chunk = False
        sample_index += replan_steps

    if input_frames:
        save_mp4(input_frames, str(episode_dir / "episode_inputs.mp4"), fps=8)
    if rollout_pred_video_frames:
        save_mp4(rollout_pred_video_frames, str(episode_dir / "pred_video_episode.mp4"), fps=8)
    if rollout_gt_video_frames:
        save_mp4(rollout_gt_video_frames, str(episode_dir / "gt_video_episode.mp4"), fps=8)

    if pred_actions:
        np.save(episode_dir / "pred_actions_episode.npy", np.stack(pred_actions, axis=0))
    if gt_actions:
        np.save(episode_dir / "gt_actions_episode.npy", np.stack(gt_actions, axis=0))
    if rollout_pred_actions:
        rollout_pred = np.concatenate(rollout_pred_actions, axis=0)
        np.save(episode_dir / "pred_actions_rollout.npy", rollout_pred)
    else:
        rollout_pred = None
    if rollout_gt_actions:
        rollout_gt = np.concatenate(rollout_gt_actions, axis=0)
        np.save(episode_dir / "gt_actions_rollout.npy", rollout_gt)
    else:
        rollout_gt = None
    if rollout_source_indices:
        np.save(episode_dir / "rollout_source_indices.npy", np.asarray(rollout_source_indices, dtype=np.int64))

    episode_meta = dict(base_metadata)
    episode_meta.update(
        {
            "episode_index": ep_idx,
            "raw_episode_index": int(raw_episode_index),
            "source_dataset_index": int(source_dataset_index),
            "source_repo_id": source_repo_id,
            "episode_start_index": ep_from,
            "episode_end_index_exclusive": ep_to,
            "num_steps": len(step_records),
            "replan_steps": int(replan_steps),
            "action_video_freq_ratio": int(action_video_freq_ratio),
            "video_rollout_stitched": bool(can_stitch_rollout_video),
            "mean_action_l1": float(np.mean(action_l1)) if action_l1 else None,
            "mean_action_l2": float(np.mean(action_l2)) if action_l2 else None,
            "inputs_video_path": str((episode_dir / "episode_inputs.mp4").resolve()) if input_frames else None,
            "pred_video_episode_path": str((episode_dir / "pred_video_episode.mp4").resolve()) if rollout_pred_video_frames else None,
            "gt_video_episode_path": str((episode_dir / "gt_video_episode.mp4").resolve()) if rollout_gt_video_frames else None,
            "pred_actions_path": str((episode_dir / "pred_actions_episode.npy").resolve()) if pred_actions else None,
            "gt_actions_path": str((episode_dir / "gt_actions_episode.npy").resolve()) if gt_actions else None,
            "pred_actions_rollout_path": str((episode_dir / "pred_actions_rollout.npy").resolve()) if rollout_pred is not None else None,
            "gt_actions_rollout_path": str((episode_dir / "gt_actions_rollout.npy").resolve()) if rollout_gt is not None else None,
            "rollout_source_indices_path": str((episode_dir / "rollout_source_indices.npy").resolve()) if rollout_source_indices else None,
            "rollout_num_steps": int(rollout_pred.shape[0]) if rollout_pred is not None else 0,
        }
    )
    with open(episode_dir / "episode_metadata.json", "w", encoding="utf-8") as f:
        json.dump(episode_meta, f, ensure_ascii=False, indent=2)
    with open(episode_dir / "step_records.json", "w", encoding="utf-8") as f:
        json.dump(step_records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
