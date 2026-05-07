import argparse
import json
import logging
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value

CODE_VERSION = "v2.1"
DEFAULT_FPS = 30
DEFAULT_ACTION_PATTERNS = [
    r"smpl_pose$",
    r"left_trigger$",
    r"right_trigger$",
    r"body_quat_w$",
]
RAW_BODY_QUAT_KEY = "body_quat_w"
EXPORTED_BODY_ROT_NAME = "smpl_root_axis_angle"
RAW_SMPL_POSE_KEY = "smpl_pose"
RAW_LEFT_TRIGGER_KEY = "left_trigger"
RAW_RIGHT_TRIGGER_KEY = "right_trigger"

logging.getLogger("pyarrow").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: Dict[str, Any]


def append_jsonl_line_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def normalize_series(arr: np.ndarray) -> Optional[np.ndarray]:
    if not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def load_h5_arrays(h5_path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        def walk(group: h5py.Group, prefix: str = "") -> None:
            for key, value in group.items():
                name = f"{prefix}/{key}" if prefix else key
                if isinstance(value, h5py.Dataset):
                    arr = normalize_series(value[()])
                    if arr is not None:
                        out[name] = arr
                else:
                    walk(value, name)

        walk(f)
    return out


def pick_keys(all_keys: List[str], patterns: List[str]) -> List[str]:
    chosen: List[str] = []
    for pat in patterns:
        for key in all_keys:
            if re.search(pat, key):
                chosen.append(key)
    # keep order, remove duplicates
    uniq: List[str] = []
    seen = set()
    for k in chosen:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def build_action_vector(data_map: Dict[str, np.ndarray], t: int) -> List[float]:
    root_aa = transform_action_value(RAW_BODY_QUAT_KEY, require_key(data_map, RAW_BODY_QUAT_KEY, Path("action.h5"))[t].reshape(-1))
    smpl_pose = require_key(data_map, RAW_SMPL_POSE_KEY, Path("action.h5"))[t].reshape(-1).astype(np.float32, copy=False)
    if smpl_pose.shape[0] != 63:
        raise ValueError(f"Expected 63D '{RAW_SMPL_POSE_KEY}', got shape {smpl_pose.shape}")

    left_trigger = require_key(data_map, RAW_LEFT_TRIGGER_KEY, Path("action.h5"))[t].reshape(-1)
    right_trigger = require_key(data_map, RAW_RIGHT_TRIGGER_KEY, Path("action.h5"))[t].reshape(-1)
    if left_trigger.shape[0] != 1 or right_trigger.shape[0] != 1:
        raise ValueError(
            f"Expected scalar triggers, got {RAW_LEFT_TRIGGER_KEY}={left_trigger.shape}, "
            f"{RAW_RIGHT_TRIGGER_KEY}={right_trigger.shape}"
        )

    left_trigger3 = np.repeat(left_trigger.astype(np.float32, copy=False), 3)
    right_trigger3 = np.repeat(right_trigger.astype(np.float32, copy=False), 3)
    action = np.concatenate([root_aa, smpl_pose, left_trigger3, right_trigger3]).astype(np.float32, copy=False)
    if action.shape[0] != 72:
        raise RuntimeError(f"Expected 72D action, got {action.shape[0]}")
    return action.tolist()


def transform_action_value(key: str, value: np.ndarray) -> np.ndarray:
    # The raw HDF5 action key is `body_quat_w`, but we export the recovered raw
    # SMPL root orientation in axis-angle form.
    if key.endswith(RAW_BODY_QUAT_KEY):
        if value.shape[0] != 4:
            raise ValueError(f"Expected 4D quaternion for '{key}', got shape {value.shape}")
        return recover_raw_smpl_root_axis_angle(value.astype(np.float64, copy=False))
    return value


def quat_normalize_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        raise ValueError(f"Received near-zero quaternion for {RAW_BODY_QUAT_KEY}")
    return quat / norm


def quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1.tolist()
    w2, x2, y2, z2 = q2.tolist()
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def angle_axis_to_quaternion_wxyz(angle_axis: np.ndarray) -> np.ndarray:
    angle_axis = np.asarray(angle_axis, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(angle_axis)
    if angle <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = angle_axis / angle
    half = 0.5 * angle
    sin_half = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )


def quaternion_to_angle_axis_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat_normalize_wxyz(quat)
    if quat[0] < 0:
        quat = -quat

    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:]
    sin_half = np.linalg.norm(xyz)
    if sin_half <= 1e-12:
        return np.zeros(3, dtype=np.float32)

    axis = xyz / sin_half
    angle = 2.0 * math.atan2(sin_half, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return (axis * angle).astype(np.float32, copy=False)


def recover_raw_smpl_root_axis_angle(body_quat_w: np.ndarray) -> np.ndarray:
    q_buf = quat_normalize_wxyz(body_quat_w)
    q_x90 = angle_axis_to_quaternion_wxyz(np.array([math.pi / 2.0, 0.0, 0.0], dtype=np.float64))
    q_y180 = angle_axis_to_quaternion_wxyz(np.array([0.0, math.pi, 0.0], dtype=np.float64))
    q_base = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    q_raw = quat_mul_wxyz(
        quat_conj_wxyz(q_x90),
        quat_mul_wxyz(
            quat_mul_wxyz(q_buf, q_base),
            quat_conj_wxyz(q_y180),
        ),
    )
    return quaternion_to_angle_axis_wxyz(q_raw)


def require_key(data_map: Dict[str, np.ndarray], key: str, source: Path) -> np.ndarray:
    value = data_map.get(key)
    if value is None:
        raise ValueError(f"Missing required key '{key}' in {source}")
    return value


def build_state_map(state_map: Dict[str, np.ndarray], state_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lowstate = require_key(state_map, "rt_lowstate", state_path)
    lowstate_ts = require_key(state_map, "rt_lowstate_timestamp", state_path)
    left_state = require_key(state_map, "inspire_left_state", state_path)
    right_state = require_key(state_map, "inspire_right_state", state_path)

    if lowstate.shape[0] == left_state.shape[0] + 1 and lowstate.shape[0] == right_state.shape[0] + 1:
        lowstate = lowstate[:-1]
        lowstate_ts = lowstate_ts[:-1]

    state_len = min(lowstate.shape[0], lowstate_ts.shape[0], left_state.shape[0], right_state.shape[0])
    if state_len <= 0:
        raise ValueError(
            "State streams are empty after normalization: "
            f"rt_lowstate={lowstate.shape[0]}, "
            f"rt_lowstate_timestamp={lowstate_ts.shape[0]}, "
            f"inspire_left_state={left_state.shape[0]}, "
            f"inspire_right_state={right_state.shape[0]} in {state_path}"
        )

    lowstate = lowstate[-state_len:]
    lowstate_ts = lowstate_ts[-state_len:]
    left_state = left_state[-state_len:]
    right_state = right_state[-state_len:]

    state_vecs = np.concatenate([lowstate, left_state, right_state], axis=1).astype(np.float32, copy=False)
    state_ts = lowstate_ts[:, 0] if lowstate_ts.ndim == 2 else lowstate_ts
    return state_vecs, state_ts.astype(np.float64, copy=False)


def align_state_indices(action_ts: np.ndarray, state_ts: np.ndarray) -> np.ndarray:
    if action_ts.ndim != 1 or state_ts.ndim != 1:
        raise ValueError("align_state_indices expects 1D timestamp arrays")
    if len(action_ts) == 0 or len(state_ts) == 0:
        raise ValueError("align_state_indices received an empty timestamp array")

    action_rel = to_relative_seconds(action_ts)
    state_rel = to_relative_seconds(state_ts)
    indices = np.searchsorted(state_rel, action_rel, side="right") - 1
    return np.clip(indices, 0, len(state_rel) - 1)


def to_relative_seconds(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(np.float64, copy=False)
    rel = ts - float(ts[0])
    if len(rel) <= 1:
        return rel

    dt = float(np.median(np.diff(ts)))
    # Heuristic units: ns/us/ms/s
    if dt > 1e6:
        return rel / 1e9
    if dt > 1e3:
        return rel / 1e6
    if dt > 1.0:
        return rel / 1e3
    return rel


def split_stereo_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if frame.ndim != 3 or frame.shape[1] < 2:
        raise ValueError(f"Unexpected frame shape for stereo split: {frame.shape}")
    w = frame.shape[1]
    mid = w // 2
    left = frame[:, :mid, :]
    right = frame[:, mid:, :]
    return left, right


def calculate_dataset_statistics(parquet_paths: List[Path]) -> Dict[str, Any]:
    all_low_dim_data = []
    for parquet_path in sorted(parquet_paths):
        all_low_dim_data.append(pd.read_parquet(parquet_path))

    if not all_low_dim_data:
        return {}

    all_low_dim_data = pd.concat(all_low_dim_data, axis=0)
    dataset_statistics: Dict[str, Any] = {}
    for le_modality in all_low_dim_data.columns:
        first_value = all_low_dim_data[le_modality].iloc[0]
        if isinstance(first_value, (str, dict)):
            continue
        np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]])
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }

    return dataset_statistics


class OwnToLeRobotConverter:
    def __init__(self, fps: int):
        self.fps = fps
        self.features = Features(
            {
                "states": Sequence(Value("float32")),
                "action": Sequence(Value("float32")),
                "timestamp": Value("float32"),
                "frame_index": Value("int64"),
                "episode_index": Value("int64"),
                "index": Value("int64"),
                "task_index": Value("int64"),
                "next.done": Value("bool"),
            }
        )
        self.lengths_by_episode: Dict[int, int] = {}
        self.num_episodes = 0
        self.total_frames = 0
        self.chunks_size = 1000
        self.instruction = "custom task"

    def make_one_episode(
        self,
        episode_index: int,
        episode_dir: Path,
        out_base: Path,
        action_patterns: List[str],
    ) -> Tuple[int, int, Dict[str, Any]]:
        chunk_id = episode_index // self.chunks_size
        chunk_path = out_base / f"chunk-{chunk_id:03d}"
        chunk_path.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_path / f"episode_{episode_index:06d}.parquet"

        ego_chunk_dir = out_base.parent / "videos" / f"chunk-{chunk_id:03d}" / "egocentric"
        ego_chunk_dir.mkdir(parents=True, exist_ok=True)
        ego_vid_path = ego_chunk_dir / f"episode_{episode_index:06d}.mp4"
        tmp_dir = out_base / f"_tmp_ep_{episode_index:06d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        parquet_tmp = tmp_dir / "episode.parquet"
        video_tmp = tmp_dir / "episode.mp4"

        action_map = load_h5_arrays(episode_dir / "action.h5")
        state_path = episode_dir / "state.h5"
        state_map = load_h5_arrays(state_path)

        action_keys = pick_keys(list(action_map.keys()), action_patterns)
        required_action_keys = [
            RAW_BODY_QUAT_KEY,
            RAW_SMPL_POSE_KEY,
            RAW_LEFT_TRIGGER_KEY,
            RAW_RIGHT_TRIGGER_KEY,
        ]
        missing_action_keys = [key for key in required_action_keys if key not in action_map]
        if missing_action_keys:
            raise ValueError(
                f"Missing required action keys {missing_action_keys} in {episode_dir / 'action.h5'}. "
                f"Available: {sorted(action_map.keys())}"
            )

        action_ts = require_key(action_map, "timestamp_realtime", episode_dir / "action.h5")
        state_vecs, state_ts = build_state_map(state_map, state_path)

        image_paths = sorted((episode_dir / "images").glob("frame_*.jpg"))
        if not image_paths:
            raise ValueError(f"No frames found in {episode_dir / 'images'}")

        seq_lengths = [len(image_paths)]
        seq_lengths.extend(action_map[k].shape[0] for k in required_action_keys)
        seq_lengths.append(action_ts.shape[0])
        n = int(min(seq_lengths))
        if n <= 1:
            raise ValueError(f"Episode too short after alignment: {episode_dir} (n={n})")

        image_paths = image_paths[:n]
        action_ts = action_ts[:n, 0] if action_ts.ndim == 2 else action_ts[:n]
        aligned_state_indices = align_state_indices(action_ts.astype(np.float64, copy=False), state_ts)
        rows: List[Dict[str, Any]] = []
        left_frames: List[np.ndarray] = []
        action_stats = None
        state_stats = None

        for i in range(n):
            act = build_action_vector(action_map, i)
            state_obs = state_vecs[int(aligned_state_indices[i])].tolist()
            frame = iio.imread(image_paths[i])
            left, _ = split_stereo_frame(frame)
            left_frames.append(left)
            row = {
                "states": state_obs,
                "action": act,
                "timestamp": float(i * (1.0 / self.fps)),
                "frame_index": i,
                "episode_index": episode_index,
                "index": i,
                "task_index": 0,
                "next.done": (i == n - 1),
            }
            rows.append(row)

            a = np.asarray(act, dtype=np.float32)
            s = np.asarray(state_obs, dtype=np.float32)
            if action_stats is None:
                action_stats = {"min": a.copy(), "max": a.copy(), "sum": a.copy(), "sumsq": a ** 2, "count": 1}
            else:
                action_stats["min"] = np.minimum(action_stats["min"], a)
                action_stats["max"] = np.maximum(action_stats["max"], a)
                action_stats["sum"] += a
                action_stats["sumsq"] += a ** 2
                action_stats["count"] += 1

            if state_stats is None:
                state_stats = {"min": s.copy(), "max": s.copy(), "sum": s.copy(), "sumsq": s ** 2, "count": 1}
            else:
                state_stats["min"] = np.minimum(state_stats["min"], s)
                state_stats["max"] = np.maximum(state_stats["max"], s)
                state_stats["sum"] += s
                state_stats["sumsq"] += s ** 2
                state_stats["count"] += 1

        ds = Dataset.from_list(rows, features=self.features)
        ds.to_parquet(str(parquet_tmp))
        iio.imwrite(video_tmp, left_frames, fps=self.fps, codec="libx264")
        os.replace(parquet_tmp, parquet_path)
        os.replace(video_tmp, ego_vid_path)
        shutil.rmtree(tmp_dir)

        if action_stats is None or state_stats is None:
            raise RuntimeError(f"Failed to collect stats: {episode_dir}")

        action_mean = (action_stats["sum"] / action_stats["count"]).tolist()
        action_std = np.sqrt(
            np.maximum(
                action_stats["sumsq"] / action_stats["count"] - np.square(action_stats["sum"] / action_stats["count"]),
                0,
            )
        ).tolist()
        state_mean = (state_stats["sum"] / state_stats["count"]).tolist()
        state_std = np.sqrt(
            np.maximum(
                state_stats["sumsq"] / state_stats["count"] - np.square(state_stats["sum"] / state_stats["count"]),
                0,
            )
        ).tolist()
        episode_stats = {
            "episode_index": episode_index,
            "stats": {
                "states": {
                    "min": state_stats["min"].tolist(),
                    "max": state_stats["max"].tolist(),
                    "mean": state_mean,
                    "std": state_std,
                    "count": [len(rows)],
                },
                "action": {
                    "min": action_stats["min"].tolist(),
                    "max": action_stats["max"].tolist(),
                    "mean": action_mean,
                    "std": action_std,
                    "count": [len(rows)],
                },
                "timestamp": {
                    "min": [float(rows[0]["timestamp"])],
                    "max": [float(rows[-1]["timestamp"])],
                    "mean": [float(0.5 * (rows[0]["timestamp"] + rows[-1]["timestamp"]))],
                    "std": [float(len(rows) / (2 * self.fps * math.sqrt(3)))],
                    "count": [len(rows)],
                },
            },
        }
        append_jsonl_line_atomic(out_base.parent / "meta" / "episodes_stats.jsonl", episode_stats)

        return episode_index, len(rows), {
            "action_keys": required_action_keys,
            "state_dim": len(rows[0]["states"]),
            "action_dim": len(rows[0]["action"]),
        }

    def run(
        self,
        data_root: Path,
        work_dir: Path,
        chunks_size: int,
        action_patterns: List[str],
        instruction: str,
    ) -> None:
        self.chunks_size = chunks_size
        self.instruction = instruction

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        ep_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and re.match(r"episode_\d+", p.name)])
        if not ep_dirs:
            raise ValueError(f"No episode_* folders found under {data_root}")

        print(f"Found {len(ep_dirs)} episodes under {data_root}")
        dim_info = None

        for ep_index, ep_dir in enumerate(ep_dirs):
            print(f"[{ep_index + 1}/{len(ep_dirs)}] processing {ep_dir.name}")
            epi, n_frames, info = self.make_one_episode(
                episode_index=ep_index,
                episode_dir=ep_dir,
                out_base=data_dir,
                action_patterns=action_patterns,
            )
            self.lengths_by_episode[epi] = n_frames
            if dim_info is None:
                dim_info = info

        self.num_episodes = len(self.lengths_by_episode)
        self.total_frames = sum(self.lengths_by_episode.values())
        print(f"Done. episodes={self.num_episodes}, frames={self.total_frames}")
        if dim_info:
            print(
                "Picked keys:",
                f"action={dim_info['action_keys']}, state_dim={dim_info['state_dim']}, action_dim={dim_info['action_dim']}",
            )

    def write_meta(self, out_dir: Path) -> None:
        meta_dir = out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dataset_cursor = 0
        episode_rows = []
        for ep_idx in sorted(self.lengths_by_episode.keys()):
            n = self.lengths_by_episode[ep_idx]
            episode_rows.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [0],
                    "length": n,
                    "dataset_from_index": dataset_cursor,
                    "dataset_to_index": dataset_cursor + (n - 1),
                    "robot_type": "custom",
                    "instruction": self.instruction,
                }
            )
            dataset_cursor += n

        episodes_df = pd.DataFrame(episode_rows)
        tasks_df = pd.DataFrame(
            [
                {
                    "task_index": 0,
                    "task": "default/custom_task",
                    "category": "default",
                    "description": self.instruction,
                }
            ]
        )

        features_meta = {
            "observation.images.egocentric": {
                "dtype": "video",
                "shape": [480, 320, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(self.fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "states": {"dtype": "float32", "shape": [-1]},
            "action": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        info = InfoDict(
            codebase_version=CODE_VERSION,
            robot_type="custom",
            total_episodes=self.num_episodes,
            total_frames=self.total_frames,
            total_tasks=1,
            total_videos=self.num_episodes,
            total_chunks=math.ceil(self.num_episodes / self.chunks_size),
            chunks_size=self.chunks_size,
            fps=self.fps,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
            features=features_meta,
        )

        (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))
        with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            for row in tasks_df.to_dict(orient="records"):
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
            for row in episodes_df.to_dict(orient="records"):
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

        parquet_files = list((out_dir / "data").glob("chunk-*/*.parquet"))
        stats = calculate_dataset_statistics(parquet_files)
        with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        print(f"Wrote meta files into {meta_dir}")


def parse_patterns(raw: str, defaults: List[str]) -> List[str]:
    if not raw.strip():
        return defaults
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="Post-traning-3-30")
    parser.add_argument("--work-dir", type=str, default="_lerobot_build_own")
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--instruction", type=str, default="custom task")
    parser.add_argument(
        "--action-patterns",
        type=str,
        default=",".join(DEFAULT_ACTION_PATTERNS),
        help="Comma-separated regex patterns to pick action datasets in action.h5",
    )
    parser.add_argument("--clean", action="store_true", help="Delete work-dir before converting")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)
    for d in [work_dir / "data", work_dir / "videos", work_dir / "meta"]:
        d.mkdir(parents=True, exist_ok=True)

    converter = OwnToLeRobotConverter(fps=args.fps)
    converter.run(
        data_root=data_root,
        work_dir=work_dir,
        chunks_size=args.chunks_size,
        action_patterns=parse_patterns(args.action_patterns, DEFAULT_ACTION_PATTERNS),
        instruction=args.instruction,
    )
    converter.write_meta(work_dir)


if __name__ == "__main__":
    main()
