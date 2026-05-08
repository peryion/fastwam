import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rerun as rr

LAYOUT_143 = "143"
LAYOUT_86 = "86"
LAYOUT_71 = "71"
LAYOUT_72 = "72"
LAYOUT_AUTO = "auto"

_warned_points_fallback = False
_warned_lines_fallback = False


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_episode_actions(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path, columns=["action", "timestamp"])
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float32)
        return actions, timestamps
    except Exception:
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["action", "timestamp"])
        actions = np.asarray(df["action"].tolist(), dtype=np.float32)
        timestamps = np.asarray(df["timestamp"].to_numpy(), dtype=np.float32)
        return actions, timestamps


def _resolve_video_key(info: Dict, camera: str) -> str:
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    if camera in video_keys:
        return camera

    preferred = f"observation.images.{camera}"
    if preferred in video_keys:
        return preferred

    suffix_match = [k for k in video_keys if k.endswith(f".{camera}")]
    if len(suffix_match) == 1:
        return suffix_match[0]

    return camera


def get_video_path(dataset_dir: Path, info: Dict, episode_index: int, camera: str) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    tpl = info["video_path"]
    if "{video_key}" in tpl:
        video_key = _resolve_video_key(info, camera)
        rel = tpl.format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
    elif "{camera}" in tpl:
        rel = tpl.format(episode_chunk=chunk, episode_index=episode_index, camera=camera)
    else:
        rel = tpl.format(episode_chunk=chunk, episode_index=episode_index)
    return dataset_dir / rel


def read_all_frames(video_path: Path) -> List[np.ndarray]:
    import imageio.v3 as iio

    return [f for f in iio.imiter(video_path)]


def parse_dims(text: str, max_dim: int) -> List[int]:
    if text.strip().lower() == "auto":
        return list(range(min(16, max_dim)))
    out = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        i = int(x)
        if 0 <= i < max_dim:
            out.append(i)
    return sorted(set(out))


def infer_layout(actions: np.ndarray, layout_arg: str) -> str:
    if layout_arg in {LAYOUT_71, LAYOUT_72, LAYOUT_86, LAYOUT_143}:
        return layout_arg
    if layout_arg != LAYOUT_AUTO:
        raise ValueError(f"Unknown layout '{layout_arg}', expected one of: auto|71|72|86|143")

    d = actions.shape[1]
    if d >= 143:
        return LAYOUT_143
    if d >= 86:
        return LAYOUT_86
    if d == 71:
        return LAYOUT_71
    if d >= 72:
        return LAYOUT_72
    raise ValueError(f"auto layout cannot infer from action_dim={d}; expected 71 or >=72")


def parse_action_layout(
    actions: np.ndarray,
    layout: str,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if actions.ndim != 2:
        raise ValueError(f"Expected action with shape (T, D), got {actions.shape}")
    t = actions.shape[0]
    d = actions.shape[1]

    smpl_joints: np.ndarray | None = None
    smpl_pose: np.ndarray | None = None
    left_trigger: np.ndarray | None = None
    right_trigger: np.ndarray | None = None
    body_rotmat: np.ndarray | None = None

    if layout == LAYOUT_143:
        if d < 143:
            raise ValueError(f"layout=143 requires action_dim>=143, got {d}")
        smpl_joints = actions[:, 0:72].reshape(t, 24, 3)
        smpl_pose = actions[:, 72:135]
        left_trigger = actions[:, 135]
        right_trigger = actions[:, 136]
        body_rot6d = actions[:, 137:143]
        body_rotmat = np.stack([rot6d_to_rotmat(x) for x in body_rot6d], axis=0).astype(np.float32, copy=False)
        return smpl_joints, smpl_pose, left_trigger, right_trigger, body_rotmat

    if layout == LAYOUT_86:
        if d < 86:
            raise ValueError(f"layout=86 requires action_dim>=86, got {d}")
        smpl_joints = actions[:, 0:72].reshape(t, 24, 3)
        left_trigger = actions[:, 78]
        right_trigger = actions[:, 79]
        body_rot6d = actions[:, 80:86]
        body_rotmat = np.stack([rot6d_to_rotmat(x) for x in body_rot6d], axis=0).astype(np.float32, copy=False)
        return smpl_joints, smpl_pose, left_trigger, right_trigger, body_rotmat

    if layout == LAYOUT_72:
        if d < 72:
            raise ValueError(f"layout=72 requires action_dim>=72, got {d}")
        smpl_pose = actions[:, 3:66]
        left_trigger = np.mean(actions[:, 66:69], axis=1)
        right_trigger = np.mean(actions[:, 69:72], axis=1)
        return smpl_joints, smpl_pose, left_trigger, right_trigger, body_rotmat

    if layout == LAYOUT_71:
        if d < 71:
            raise ValueError(f"layout=71 requires action_dim>=71, got {d}")
        # [0:6]=global_rot6d, [6:69]=smpl_pose(63), [69]=left_trigger, [70]=right_trigger
        body_rot6d = actions[:, 0:6]
        body_rotmat = np.stack([rot6d_to_rotmat(x) for x in body_rot6d], axis=0).astype(np.float32, copy=False)
        smpl_pose = actions[:, 6:69]
        left_trigger = actions[:, 69]
        right_trigger = actions[:, 70]
        return smpl_joints, smpl_pose, left_trigger, right_trigger, body_rotmat

    raise ValueError(f"Unhandled layout: {layout}")


def get_smpl_edges() -> List[Tuple[int, int]]:
    # Common SMPL kinematic tree for 24 joints.
    return [
        (0, 1), (1, 4), (4, 7), (7, 10),
        (0, 2), (2, 5), (5, 8), (8, 11),
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
    ]


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    r6 = np.asarray(rot6d, dtype=np.float32).reshape(6)
    # Matches flatten order from exporter: [r00, r01, r10, r11, r20, r21].
    a1 = np.asarray([r6[0], r6[2], r6[4]], dtype=np.float32)
    a2 = np.asarray([r6[1], r6[3], r6[5]], dtype=np.float32)
    n1 = float(np.linalg.norm(a1))
    if n1 < 1e-8:
        return np.eye(3, dtype=np.float32)
    b1 = a1 / n1
    a2_proj = a2 - np.dot(b1, a2) * b1
    n2 = float(np.linalg.norm(a2_proj))
    if n2 < 1e-8:
        return np.eye(3, dtype=np.float32)
    b2 = a2_proj / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32, copy=False)


class SMPL71Forward:
    """SMPL forward helper matching visualize_all_z_up_71.py for 71D actions."""

    def __init__(self, model_folder: str, gender: str = "neutral", ext: str = "", device: str = "cuda"):
        self.model_folder = str(Path(model_folder).expanduser())
        self.gender = gender.lower()
        self.ext = ext.strip().lower()
        self.device = device
        self._model = None
        self._torch = None
        self._scipy_rot = None

    def _detect_ext(self) -> str:
        smpl_dir = Path(self.model_folder) / "smpl"
        g = self.gender.upper()
        if (smpl_dir / f"SMPL_{g}.npz").is_file():
            return "npz"
        if (smpl_dir / f"SMPL_{g}.pkl").is_file():
            return "pkl"
        return "npz"

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from smplx import SMPL
        from smplx.utils import Struct

        self._torch = torch
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        dev = torch.device("cuda" if use_cuda else "cpu")
        smpl_dir = Path(self.model_folder) / "smpl"
        ext = self.ext or self._detect_ext()
        g = self.gender.upper()

        if ext == "npz":
            model_path = smpl_dir / f"SMPL_{g}.npz"
            if not model_path.is_file():
                raise FileNotFoundError(f"SMPL npz not found: {model_path}")
            z = np.load(model_path, allow_pickle=True)
            model = SMPL(model_path=str(model_path), data_struct=Struct(**z), gender=self.gender, batch_size=1)
        elif ext == "pkl":
            model_path = smpl_dir / f"SMPL_{g}.pkl"
            if not model_path.is_file():
                raise FileNotFoundError(f"SMPL pkl not found: {model_path}")
            model = SMPL(model_path=str(smpl_dir), gender=self.gender, batch_size=1)
        else:
            raise ValueError(f"Unsupported SMPL ext: {ext}")

        self._model = model.eval().to(dev)
        self._device = dev
        from scipy.spatial.transform import Rotation as _R

        self._scipy_rot = _R

    def forward_joints(self, root_rot6d: np.ndarray, pose63: np.ndarray) -> np.ndarray:
        self._ensure_model()
        assert self._model is not None
        assert self._torch is not None
        assert self._scipy_rot is not None

        root_rotmat = rot6d_to_rotmat(root_rot6d)
        global_orient = self._scipy_rot.from_matrix(root_rotmat).as_rotvec().astype(np.float32)  # (3,)
        body_pose = np.concatenate([np.asarray(pose63, dtype=np.float32).reshape(63), np.zeros(6, dtype=np.float32)], axis=0)
        betas = np.zeros((10,), dtype=np.float32)
        transl = np.zeros((3,), dtype=np.float32)

        torch = self._torch
        with torch.no_grad():
            out = self._model(
                betas=torch.from_numpy(betas[None]).to(self._device),
                global_orient=torch.from_numpy(global_orient[None]).to(self._device),
                body_pose=torch.from_numpy(body_pose[None]).to(self._device),
                transl=torch.from_numpy(transl[None]).to(self._device),
                return_verts=False,
                pose2rot=True,
            )
        joints = out.joints[0, :24].detach().float().cpu().numpy().astype(np.float32, copy=False)
        return joints


def apply_root_rotation(joints_xyz: np.ndarray, root_rot: np.ndarray) -> np.ndarray:
    root = joints_xyz[0:1, :]
    centered = joints_xyz - root
    rotated = centered @ root_rot.T
    return rotated + root


def trigger_to_rgb(trigger_value: float) -> np.ndarray:
    t = float(np.clip(trigger_value, 0.0, 1.0))
    r = int(255 * t)
    g = int(40)
    b = int(255 * (1.0 - t))
    return np.asarray([r, g, b], dtype=np.uint8)


def set_time_compat(frame_idx: int, ts_seconds: float) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_idx)
    elif hasattr(rr, "set_time"):
        try:
            rr.set_time("frame", sequence=frame_idx)
        except TypeError:
            try:
                rr.set_time("frame", frame_idx)
            except Exception:
                pass

    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("time", ts_seconds)
    elif hasattr(rr, "set_time"):
        try:
            rr.set_time("time", seconds=ts_seconds)
        except TypeError:
            try:
                rr.set_time("time", ts_seconds)
            except Exception:
                pass


def log_scalar_compat(path: str, value: float) -> None:
    if hasattr(rr, "Scalar"):
        rr.log(path, rr.Scalar(value))
    elif hasattr(rr, "Scalars"):
        rr.log(path, rr.Scalars([value]))
    else:
        rr.log(path, value)


def log_points3d_compat(path: str, points_xyz: np.ndarray) -> None:
    global _warned_points_fallback
    if hasattr(rr, "Points3D"):
        rr.log(path, rr.Points3D(points_xyz))
        return
    if not _warned_points_fallback:
        print("Warning: rerun.Points3D not available, fallback to per-joint scalar logging.")
        _warned_points_fallback = True
    for j in range(points_xyz.shape[0]):
        log_scalar_compat(f"{path}/j{j:02d}/x", float(points_xyz[j, 0]))
        log_scalar_compat(f"{path}/j{j:02d}/y", float(points_xyz[j, 1]))
        log_scalar_compat(f"{path}/j{j:02d}/z", float(points_xyz[j, 2]))


def log_colored_points3d_compat(path: str, points_xyz: np.ndarray, colors_rgb: np.ndarray, radius: float = 0.02) -> None:
    if hasattr(rr, "Points3D"):
        try:
            rr.log(path, rr.Points3D(points_xyz, colors=colors_rgb, radii=[radius] * len(points_xyz)))
            return
        except TypeError:
            rr.log(path, rr.Points3D(points_xyz))
            return
    for j in range(points_xyz.shape[0]):
        log_scalar_compat(f"{path}/j{j:02d}/x", float(points_xyz[j, 0]))
        log_scalar_compat(f"{path}/j{j:02d}/y", float(points_xyz[j, 1]))
        log_scalar_compat(f"{path}/j{j:02d}/z", float(points_xyz[j, 2]))
        log_scalar_compat(f"{path}/j{j:02d}/r", float(colors_rgb[j, 0]))
        log_scalar_compat(f"{path}/j{j:02d}/g", float(colors_rgb[j, 1]))
        log_scalar_compat(f"{path}/j{j:02d}/b", float(colors_rgb[j, 2]))


def log_lines3d_compat(path: str, points_xyz: np.ndarray, edges: List[Tuple[int, int]]) -> None:
    global _warned_lines_fallback
    if hasattr(rr, "LineStrips3D"):
        strips = [np.asarray([points_xyz[i], points_xyz[j]], dtype=np.float32) for i, j in edges]
        rr.log(path, rr.LineStrips3D(strips))
        return
    if not _warned_lines_fallback:
        print("Warning: rerun.LineStrips3D not available, skipping bone line logging.")
        _warned_lines_fallback = True


def log_line_segment_compat(path: str, p0: np.ndarray, p1: np.ndarray) -> None:
    seg = np.asarray([p0, p1], dtype=np.float32)
    if hasattr(rr, "LineStrips3D"):
        rr.log(path, rr.LineStrips3D([seg]))
        return
    # scalar fallback
    log_scalar_compat(f"{path}/p0/x", float(seg[0, 0]))
    log_scalar_compat(f"{path}/p0/y", float(seg[0, 1]))
    log_scalar_compat(f"{path}/p0/z", float(seg[0, 2]))
    log_scalar_compat(f"{path}/p1/x", float(seg[1, 0]))
    log_scalar_compat(f"{path}/p1/y", float(seg[1, 1]))
    log_scalar_compat(f"{path}/p1/z", float(seg[1, 2]))


def log_axes_compat(path: str, origin: np.ndarray, rot: np.ndarray, scale: float) -> None:
    # x/y/z axes expressed in world coordinates
    x_tip = origin + rot @ np.asarray([scale, 0.0, 0.0], dtype=np.float32)
    y_tip = origin + rot @ np.asarray([0.0, scale, 0.0], dtype=np.float32)
    z_tip = origin + rot @ np.asarray([0.0, 0.0, scale], dtype=np.float32)
    log_line_segment_compat(f"{path}/x", origin, x_tip)
    log_line_segment_compat(f"{path}/y", origin, y_tip)
    log_line_segment_compat(f"{path}/z", origin, z_tip)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LeRobot episodes in Rerun.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="e.g. _lerobot_build_own")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        help="action layout: auto|71|72|86|143",
    )
    parser.add_argument("--action-dims", type=str, default="auto", help="auto or comma list, e.g. 0,1,2")
    parser.add_argument(
        "--smpl-viz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable SMPL 3D skeleton + SMPL curves logging",
    )
    parser.add_argument(
        "--smpl-pose-dims",
        type=str,
        default="auto",
        help="SMPL pose dims (0..62), e.g. auto or 0,1,2,3",
    )
    parser.add_argument(
        "--use-body-quat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply body_rot6d to rotate SMPL joints around root when present",
    )
    parser.add_argument(
        "--show-world-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show world origin + world xyz axes",
    )
    parser.add_argument(
        "--show-root-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show root local xyz axes rotated by body_rot6d each frame",
    )
    parser.add_argument("--world-frame-scale", type=float, default=0.25, help="World axis length")
    parser.add_argument("--root-frame-scale", type=float, default=0.18, help="Root frame axis length")
    parser.add_argument("--spawn", action="store_true", help="Spawn local Rerun viewer")
    parser.add_argument(
        "--reuse-app-id",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse deterministic app_id by episode. Default False uses fresh timestamped app_id to avoid stale layout.",
    )
    parser.add_argument("--save", type=str, default="", help="Optional .rrd output path")
    parser.add_argument("--smpl-model-folder", type=str, default="/home/zhipy/Documents/smpl/SMPL_python_v.1.1.0/", help="Optional SMPL model folder. If set, 71D pose uses real SMPL forward.")
    parser.add_argument("--smpl-gender", type=str, default="neutral")
    parser.add_argument("--smpl-ext", type=str, default="npz", help="npz|pkl (auto if empty)")
    parser.add_argument("--smpl-device", type=str, default="cuda")
    parser.add_argument(
        "--linger-seconds",
        type=float,
        default=1.5,
        help="Extra wait time after logging when --spawn is used (helps viewer receive full stream)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"Missing meta files under {dataset_dir / 'meta'}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = read_jsonl(episodes_path)
    if not episodes:
        raise ValueError("No episodes found in episodes.jsonl")
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise ValueError(f"episode-index out of range: 0..{len(episodes)-1}")

    ep = episodes[args.episode_index]
    episode_index = int(ep["episode_index"])
    chunk = episode_index // int(info["chunks_size"])
    parquet_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    actions, timestamps = load_episode_actions(parquet_path)
    if actions.ndim != 2:
        raise ValueError(f"Unexpected action shape: {actions.shape}")

    dims = parse_dims(args.action_dims, actions.shape[1])
    if not dims:
        dims = list(range(min(8, actions.shape[1])))

    smpl_joints = None
    smpl_pose = None
    left_trigger = None
    right_trigger = None
    body_rotmat = None
    smpl_pose_dims: List[int] = []
    smpl_edges: List[Tuple[int, int]] = []
    smpl71_model = None
    if args.smpl_viz:
        layout = infer_layout(actions, str(args.layout).strip().lower())
        smpl_joints, smpl_pose, left_trigger, right_trigger, body_rotmat = parse_action_layout(actions, layout)
        if smpl_pose is not None:
            smpl_pose_dims = parse_dims(args.smpl_pose_dims, smpl_pose.shape[1])
            if not smpl_pose_dims:
                smpl_pose_dims = list(range(min(12, smpl_pose.shape[1])))
        else:
            smpl_pose_dims = []
        smpl_edges = get_smpl_edges()
        if layout == LAYOUT_71 and args.smpl_model_folder.strip():
            try:
                smpl71_model = SMPL71Forward(
                    model_folder=args.smpl_model_folder.strip(),
                    gender=str(args.smpl_gender).strip(),
                    ext=str(args.smpl_ext).strip(),
                    device=str(args.smpl_device).strip(),
                )
                # warmup instantiate early to fail fast if model path invalid
                smpl71_model._ensure_model()
                print(f"[SMPL71] Using SMPL model from: {args.smpl_model_folder}")
            except Exception as e:
                raise RuntimeError(
                    f"layout=71 requires valid SMPL model. Failed to init from "
                    f"--smpl-model-folder={args.smpl_model_folder!r}. error={e}"
                ) from e
        if layout == LAYOUT_71 and smpl_pose is not None and smpl_pose.shape[1] == 63 and smpl71_model is None:
            raise ValueError(
                "layout=71 requires SMPL model forward for skeleton visualization. "
                "Please pass --smpl-model-folder (and optional --smpl-gender/--smpl-ext/--smpl-device)."
            )

    # Camera stream auto-discovery: support stereo, ego-only, or arbitrary single camera.
    stream_candidates = [
        ("left", "camera/left"),
        ("right", "camera/right"),
        ("egocentric", "camera/egocentric"),
        ("third", "camera/third"),
    ]
    stream_frames: List[Tuple[str, List[np.ndarray]]] = []
    for cam_name, log_path in stream_candidates:
        vp = get_video_path(dataset_dir, info, episode_index, cam_name)
        if vp.exists():
            stream_frames.append((log_path, read_all_frames(vp)))

    if not stream_frames:
        # fallback: try all video keys from meta
        features = info.get("features", {})
        all_video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
        for vk in all_video_keys:
            vp = get_video_path(dataset_dir, info, episode_index, vk)
            if vp.exists():
                short_name = vk.split(".")[-1]
                stream_frames.append((f"camera/{short_name}", read_all_frames(vp)))

    if not stream_frames:
        raise FileNotFoundError(f"No video found for episode {episode_index} under dataset {dataset_dir}")

    lengths = [len(actions)] + [len(frames) for _, frames in stream_frames]
    n = int(min(lengths))
    if n <= 0:
        raise ValueError("No synchronized frames to visualize.")

    if args.reuse_app_id:
        app_id = f"lerobot_ep_{episode_index:06d}"
    else:
        app_id = f"lerobot_ep_{episode_index:06d}_{int(time.time() * 1000)}"
    rr.init(app_id, spawn=False)
    if args.save:
        save_path = Path(args.save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_path))
    if args.spawn:
        if hasattr(rr, "spawn"):
            rr.spawn(connect=True, hide_welcome_screen=True)
        else:
            # Fallback for older SDKs.
            rr.init(app_id, spawn=True)

    rr.log(
        "episode/info",
        rr.TextDocument(
            "\n".join(
                [
                    f"episode={episode_index}",
                    f"frames={n}",
                    f"action_dim={actions.shape[1]}",
                    f"layout={infer_layout(actions, str(args.layout).strip().lower())}",
                    "layout_143=[0:72]=smpl_joints, [72:135]=smpl_pose, [135]=left_trigger, [136]=right_trigger, [137:143]=body_rot6d",
                    "layout_86=[0:72]=smpl_joints, [72:78]=joint_pos_last6, [78]=left_trigger, [79]=right_trigger, [80:86]=body_rot6d",
                    "layout_71=[0:6]=global_rot6d, [6:69]=smpl_pose, [69]=left_trigger, [70]=right_trigger",
                    "layout_72=[0:3]=root_axis_angle, [3:66]=smpl_pose, [66:69]=left_trigger_repeat3, [69:72]=right_trigger_repeat3",
                    f"smpl_viz={args.smpl_viz}",
                    f"use_body_quat={args.use_body_quat}",
                    f"show_world_frame={args.show_world_frame}",
                    f"show_root_frame={args.show_root_frame}",
                    f"camera_streams={[p for p, _ in stream_frames]}",
                ]
            )
        ),
    )
    if args.show_world_frame:
        world_origin = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        world_color = np.asarray([[255, 255, 255]], dtype=np.uint8)
        log_colored_points3d_compat("world/origin", world_origin, world_color, radius=0.03)
        log_axes_compat(
            "world/axes",
            origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            rot=np.eye(3, dtype=np.float32),
            scale=float(args.world_frame_scale),
        )

    for i in range(n):
        ts = float(timestamps[i]) if i < len(timestamps) else float(i / max(info.get("fps", 30), 1))
        set_time_compat(i, ts)

        for log_path, frames in stream_frames:
            rr.log(log_path, rr.Image(frames[i]))

        for d in dims:
            log_scalar_compat(f"action/dim_{d:03d}", float(actions[i, d]))
        if args.smpl_viz and smpl_joints is not None:
            joints_i = smpl_joints[i].copy()
            if args.use_body_quat and body_rotmat is not None:
                joints_i = apply_root_rotation(joints_i, body_rotmat[i])
            log_points3d_compat("smpl/joints", joints_i)
            log_lines3d_compat("smpl/bones", joints_i, smpl_edges)
            if args.show_root_frame and body_rotmat is not None:
                root_rot = body_rotmat[i]
                log_axes_compat(
                    "smpl/root_frame",
                    origin=joints_i[0].astype(np.float32),
                    rot=root_rot.astype(np.float32),
                    scale=float(args.root_frame_scale),
                )
            # End-effectors (SMPL hand tips) with trigger-encoded color.
            ee_points = np.asarray([joints_i[22], joints_i[23]], dtype=np.float32)
            if left_trigger is not None and right_trigger is not None:
                ee_colors = np.asarray(
                    [trigger_to_rgb(float(left_trigger[i])), trigger_to_rgb(float(right_trigger[i]))],
                    dtype=np.uint8,
                )
            else:
                ee_colors = np.asarray([[200, 200, 200], [200, 200, 200]], dtype=np.uint8)
            log_colored_points3d_compat("smpl/end_effectors", ee_points, ee_colors, radius=0.03)
            if smpl_pose is not None:
                for d in smpl_pose_dims:
                    log_scalar_compat(f"smpl/pose/dim_{d:03d}", float(smpl_pose[i, d]))
        elif args.smpl_viz and smpl_pose is not None and smpl_pose.shape[1] == 63:
            root_rot = body_rotmat[i] if body_rotmat is not None else np.eye(3, dtype=np.float32)
            if smpl71_model is None:
                raise RuntimeError(
                    "SMPL71 model is not initialized. Please provide --smpl-model-folder for layout=71."
                )
            joints_i = smpl71_model.forward_joints(root_rot6d=actions[i, 0:6], pose63=smpl_pose[i])
            log_points3d_compat("smpl/joints_fk", joints_i)
            log_lines3d_compat("smpl/bones_fk", joints_i, smpl_edges)
            if args.show_root_frame:
                rr_root = root_rot if (args.use_body_quat and root_rot is not None) else np.eye(3, dtype=np.float32)
                log_axes_compat(
                    "smpl/root_frame_fk",
                    origin=joints_i[0].astype(np.float32),
                    rot=np.asarray(rr_root, dtype=np.float32),
                    scale=float(args.root_frame_scale),
                )
            ee_points = np.asarray([joints_i[22], joints_i[23]], dtype=np.float32)
            if left_trigger is not None and right_trigger is not None:
                ee_colors = np.asarray(
                    [trigger_to_rgb(float(left_trigger[i])), trigger_to_rgb(float(right_trigger[i]))],
                    dtype=np.uint8,
                )
            else:
                ee_colors = np.asarray([[200, 200, 200], [200, 200, 200]], dtype=np.uint8)
            log_colored_points3d_compat("smpl/end_effectors_fk", ee_points, ee_colors, radius=0.03)
            for d in smpl_pose_dims:
                log_scalar_compat(f"smpl/pose/dim_{d:03d}", float(smpl_pose[i, d]))
        if args.smpl_viz and left_trigger is not None and right_trigger is not None:
            log_scalar_compat("smpl/trigger/left", float(left_trigger[i]))
            log_scalar_compat("smpl/trigger/right", float(right_trigger[i]))

    print("Rerun log complete.")
    print(f"Episode: {episode_index}")
    print(f"Frames logged: {n}")
    print(f"Action dims logged: {dims}")
    if args.smpl_viz:
        print(f"SMPL pose dims logged: {smpl_pose_dims}")
        print("SMPL paths: smpl/joints or smpl/joints_fk, smpl/bones or smpl/bones_fk, smpl/pose/*, smpl/trigger/*")
    if args.save:
        print(f"Saved: {Path(args.save).expanduser().resolve()}")
    if args.spawn:
        if args.linger_seconds > 0:
            time.sleep(float(args.linger_seconds))
        if hasattr(rr, "disconnect"):
            rr.disconnect()
    else:
        print("Tip: add --spawn to open the Rerun viewer directly.")


if __name__ == "__main__":
    main()
