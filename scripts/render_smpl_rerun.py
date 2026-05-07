import argparse
import time
from pathlib import Path

import numpy as np


EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


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
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _load_action_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"Expected action array with shape [T, D], got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    r6 = np.asarray(rot6d, dtype=np.float32).reshape(6)
    # Matches the exporter order: [r00, r01, r10, r11, r20, r21].
    a1 = np.asarray([r6[0], r6[2], r6[4]], dtype=np.float32)
    a2 = np.asarray([r6[1], r6[3], r6[5]], dtype=np.float32)

    n1 = float(np.linalg.norm(a1))
    if n1 < 1e-8:
        return np.eye(3, dtype=np.float32)
    b1 = a1 / n1

    a2 = a2 - np.dot(b1, a2) * b1
    n2 = float(np.linalg.norm(a2))
    if n2 < 1e-8:
        return np.eye(3, dtype=np.float32)
    b2 = a2 / n2

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32, copy=False)


def _apply_root_rotation(joints_xyz: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    root = joints_xyz[0:1]
    centered = joints_xyz - root
    return centered @ rotmat.T + root


def _actions_to_joints(
    actions: np.ndarray,
    smpl_layout: str,
    use_body_rot: bool,
    body_rot6d_start: int,
) -> np.ndarray:
    if smpl_layout != "ego86":
        raise ValueError(f"Unsupported --smpl-layout {smpl_layout!r}. This script currently supports ego86.")
    if actions.shape[1] != 86:
        raise ValueError(f"Expected ego86 action array with shape [T, 86], got {actions.shape}.")

    joints = actions[:, :72].reshape(actions.shape[0], 24, 3).copy()
    if use_body_rot:
        rot6d_end = body_rot6d_start + 6
        if rot6d_end > actions.shape[1]:
            raise ValueError(
                f"body_rot6d slice [{body_rot6d_start}:{rot6d_end}] is outside action dim {actions.shape[1]}."
            )
        for t in range(joints.shape[0]):
            joints[t] = _apply_root_rotation(joints[t], _rot6d_to_rotmat(actions[t, body_rot6d_start:rot6d_end]))
    return joints


def _get_triggers(actions: np.ndarray, smpl_layout: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if smpl_layout != "ego86":
        raise ValueError(f"Unsupported --smpl-layout {smpl_layout!r}. This script currently supports ego86.")
    if actions.shape[1] != 86:
        raise ValueError(f"Expected ego86 action array with shape [T, 86], got {actions.shape}.")
    return actions[:, 78], actions[:, 79]


def _trigger_to_rgb(trigger_value: float) -> np.ndarray:
    t = float(np.clip(trigger_value, 0.0, 1.0))
    return np.asarray([int(255 * t), 40, int(255 * (1.0 - t))], dtype=np.uint8)


def _apply_coord_fix(joints_xyz: np.ndarray, coord_fix: str) -> np.ndarray:
    rot = _rotation_matrix_from_name(coord_fix)
    return joints_xyz @ rot.T


def _log_skeleton(
    rr,
    prefix: str,
    joints_xyz: np.ndarray,
    color_rgb: list[int],
    left_trigger: float | None = None,
    right_trigger: float | None = None,
) -> None:
    line_segments = [joints_xyz[[a, b]] for a, b in EDGES]
    rr.log(
        f"{prefix}/joints",
        rr.Points3D(
            positions=joints_xyz,
            colors=[color_rgb],
            radii=0.012,
        ),
    )
    rr.log(
        f"{prefix}/bones",
        rr.LineStrips3D(
            line_segments,
            colors=[color_rgb],
            radii=0.006,
        ),
    )
    if left_trigger is not None and right_trigger is not None:
        rr.log(
            f"{prefix}/end_effectors",
            rr.Points3D(
                positions=np.asarray([joints_xyz[22], joints_xyz[23]], dtype=np.float32),
                colors=np.asarray([_trigger_to_rgb(left_trigger), _trigger_to_rgb(right_trigger)], dtype=np.uint8),
                radii=[0.03, 0.03],
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="Rerun visualizer for ego86 SMPL rollout actions.")
    parser.add_argument("--pred-actions", required=True, help="Path to predicted action .npy")
    parser.add_argument("--gt-actions", default=None, help="Optional path to ground-truth action .npy")
    parser.add_argument("--smpl-layout", default="ego86", choices=["ego86"])
    parser.add_argument("--body-rot6d-start", type=int, default=80)
    parser.add_argument("--use-body-rot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--coord-fix",
        default="none",
        choices=["none", "x90", "xm90", "y90", "ym90", "z90", "zm90", "x180", "y180", "z180"],
    )
    parser.add_argument("--sequence-name", default="ego_smpl_rollout_86d")
    parser.add_argument("--recording-id", default=None)
    parser.add_argument("--save-rrd", default=None)
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--show-pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linger-seconds", type=float, default=1.5)
    args = parser.parse_args()

    if not args.show_pred and not args.show_gt:
        raise ValueError("At least one of --show-pred / --show-gt must be enabled.")
    if args.show_gt and args.gt_actions is None:
        raise ValueError("--show-gt requires --gt-actions.")

    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "This script requires `rerun-sdk`. Install it locally first, for example: pip install rerun-sdk"
        ) from exc

    pred_actions = _load_action_array(Path(args.pred_actions).expanduser().resolve())
    gt_actions = _load_action_array(Path(args.gt_actions).expanduser().resolve()) if args.gt_actions is not None else None

    pred_joints = _actions_to_joints(
        pred_actions,
        smpl_layout=args.smpl_layout,
        use_body_rot=args.use_body_rot,
        body_rot6d_start=args.body_rot6d_start,
    )
    pred_left_trigger, pred_right_trigger = _get_triggers(pred_actions, args.smpl_layout)
    gt_joints = None
    gt_left_trigger = None
    gt_right_trigger = None
    if gt_actions is not None:
        gt_joints = _actions_to_joints(
            gt_actions,
            smpl_layout=args.smpl_layout,
            use_body_rot=args.use_body_rot,
            body_rot6d_start=args.body_rot6d_start,
        )
        gt_left_trigger, gt_right_trigger = _get_triggers(gt_actions, args.smpl_layout)
        if gt_joints.shape[0] != pred_joints.shape[0]:
            raise ValueError(f"Pred/GT lengths do not match: pred={pred_joints.shape[0]}, gt={gt_joints.shape[0]}")

    pred_joints = _apply_coord_fix(pred_joints, args.coord_fix)
    if gt_joints is not None:
        gt_joints = _apply_coord_fix(gt_joints, args.coord_fix)

    rr.init(
        application_id=args.sequence_name,
        recording_id=args.recording_id,
        spawn=args.spawn,
    )
    if args.save_rrd is not None:
        rr.save(str(Path(args.save_rrd).expanduser().resolve()))

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "info",
        rr.TextDocument(
            "\n".join(
                [
                    f"layout={args.smpl_layout}",
                    f"pred_actions={Path(args.pred_actions).expanduser().resolve()}",
                    f"gt_actions={Path(args.gt_actions).expanduser().resolve() if args.gt_actions else None}",
                    f"frames={pred_joints.shape[0]}",
                    "ego86_layout=[0:72]=smpl_joints_xyz, [72:78]=joint_pos_last6, [78]=left_trigger, [79]=right_trigger, [80:86]=body_rot6d",
                    f"use_body_rot={args.use_body_rot}",
                    f"body_rot6d=[{args.body_rot6d_start}:{args.body_rot6d_start + 6}]",
                    f"coord_fix={args.coord_fix}",
                ]
            )
        ),
    )

    for t in range(pred_joints.shape[0]):
        rr.set_time("frame", sequence=t)
        if args.show_pred:
            _log_skeleton(
                rr,
                "pred",
                pred_joints[t],
                [255, 80, 80],
                float(pred_left_trigger[t]),
                float(pred_right_trigger[t]),
            )
        if args.show_gt and gt_joints is not None:
            _log_skeleton(
                rr,
                "gt",
                gt_joints[t],
                [80, 140, 255],
                float(gt_left_trigger[t]) if gt_left_trigger is not None else None,
                float(gt_right_trigger[t]) if gt_right_trigger is not None else None,
            )

    if args.spawn and args.linger_seconds > 0:
        time.sleep(float(args.linger_seconds))


if __name__ == "__main__":
    main()
