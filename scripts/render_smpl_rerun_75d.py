import argparse
from pathlib import Path

import numpy as np
import torch

from infer_ego import _axis_angle_to_matrix, _matrix_to_axis_angle, _rotation_matrix_from_name


EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11), (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23), (12, 15),
]


def _load_action_array(path: Path) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 3:
        raise ValueError(
            f"Expected rollout action array with shape [T, 75], but got {arr.shape}. "
            "Please pass rollout arrays, not chunk-wise episode arrays."
        )
    if arr.ndim != 2 or arr.shape[1] != 75:
        raise ValueError(f"Expected [T, 75] action array, got {arr.shape}.")
    return torch.from_numpy(arr).to(dtype=torch.float32)


def _build_joints_from_action(
    action_txd: torch.Tensor,
    smpl_model_path: Path,
    gender: str,
    global_basis_fix: str = "none",
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
            "This script requires `smplx`. Install it in the current environment first."
        ) from exc

    action = action_txd.detach().to(device="cpu", dtype=torch.float32)
    global_orient = action[:, :3]
    body_pose = action[:, 3:72]
    transl = action[:, 72:75]

    if global_basis_fix != "none":
        basis = torch.from_numpy(_rotation_matrix_from_name(global_basis_fix)).to(
            device=global_orient.device,
            dtype=global_orient.dtype,
        )
        global_rot = _axis_angle_to_matrix(global_orient)
        global_rot = basis.unsqueeze(0) @ global_rot @ basis.transpose(0, 1).unsqueeze(0)
        global_orient = _matrix_to_axis_angle(global_rot)

    model = smplx.create(
        str(smpl_model_path),
        model_type="smpl",
        gender=gender,
        use_pca=False,
        batch_size=action.shape[0],
    ).to(device="cpu")
    out = model(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        return_verts=False,
    )
    return out.joints[:, :24].detach().cpu().numpy()


def _log_skeleton(rr, prefix: str, joints_xyz: np.ndarray, color_rgb):
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


def _apply_coord_fix(joints_xyz: np.ndarray, coord_fix: str) -> np.ndarray:
    rot = _rotation_matrix_from_name(coord_fix)
    return joints_xyz @ rot.T


def main():
    parser = argparse.ArgumentParser(description="Temporary Rerun visualizer for 75D standard SMPL actions.")
    parser.add_argument("--pred-actions", required=True, help="Path to pred action .npy")
    parser.add_argument("--gt-actions", default=None, help="Optional path to gt action .npy")
    parser.add_argument("--smpl-model-path", default="/home/zhipy/Documents/smpl/SMPL_python_v.1.1.0/")
    parser.add_argument("--smpl-gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--global-basis-fix", default="none", choices=["none", "x90", "xm90", "y90", "ym90", "z90", "zm90", "x180", "y180", "z180"])
    parser.add_argument("--coord-fix", default="none", choices=["none", "x90", "xm90", "y90", "ym90", "z90", "zm90", "x180", "y180", "z180"])
    parser.add_argument("--sequence-name", default="ego_smpl_rollout_75d")
    parser.add_argument("--recording-id", default=None)
    parser.add_argument("--save-rrd", default=None)
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--show-pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-gt", action=argparse.BooleanOptionalAction, default=True)
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

    pred_actions = _load_action_array(Path(args.pred_actions).resolve())
    gt_actions = _load_action_array(Path(args.gt_actions).resolve()) if args.gt_actions is not None else None

    pred_joints = _build_joints_from_action(
        action_txd=pred_actions,
        smpl_model_path=Path(args.smpl_model_path).resolve(),
        gender=args.smpl_gender,
        global_basis_fix=args.global_basis_fix,
    )
    pred_joints = pred_joints[:-5]
    gt_joints = None
    if gt_actions is not None:
        gt_joints = _build_joints_from_action(
            action_txd=gt_actions,
            smpl_model_path=Path(args.smpl_model_path).resolve(),
            gender=args.smpl_gender,
            global_basis_fix=args.global_basis_fix,
        )
        if gt_joints.shape[0] != pred_joints.shape[0]:
            raise ValueError(
                f"Pred/GT lengths do not match: pred={pred_joints.shape[0]}, gt={gt_joints.shape[0]}"
            )

    pred_joints = _apply_coord_fix(pred_joints, args.coord_fix)
    if gt_joints is not None:
        gt_joints = _apply_coord_fix(gt_joints, args.coord_fix)

    rr.init(
        application_id=args.sequence_name,
        recording_id=args.recording_id,
        spawn=args.spawn,
    )

    if args.save_rrd is not None:
        rr.save(str(Path(args.save_rrd).resolve()))

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for t in range(pred_joints.shape[0]):
        rr.set_time("frame", sequence=t)
        if args.show_pred:
            _log_skeleton(rr, "pred", pred_joints[t], [255, 80, 80])
        if args.show_gt and gt_joints is not None:
            _log_skeleton(rr, "gt", gt_joints[t], [80, 140, 255])


if __name__ == "__main__":
    main()
