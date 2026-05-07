import math

import numpy as np


RAW_BODY_QUAT_KEY = "body_quat_w"


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {quat.shape}")
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


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


def quaternion_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(quat).tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def recover_raw_smpl_root_axis_angle(body_quat_w: np.ndarray) -> np.ndarray:
    # Source-aligned inverse of the collection pipeline in pico_manager_thread_server.py:
    # 1) global_rots = global_rots * R_y(180)
    # 2) global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    # 3) global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)
    #
    # With wxyz convention, that forward chain is:
    #   q_buf = q_x90 * q_raw * q_y180 * q_base_inv
    #
    # Therefore the inverse is:
    #   q_raw = inv(q_x90) * q_buf * q_base * inv(q_y180)
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


def forward_project_body_quat_w(root_axis_angle: np.ndarray) -> np.ndarray:
    # This matches the source collection path exactly under wxyz convention:
    #   q_raw --(R_y 180)--> q_raw * q_y180
    #   --(smpl_root_ytoz_up)--> q_x90 * q_raw * q_y180
    #   --(remove_smpl_base_rot)--> q_x90 * q_raw * q_y180 * q_base_inv
    q_raw = angle_axis_to_quaternion_wxyz(root_axis_angle)
    q_x90 = angle_axis_to_quaternion_wxyz(np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64))
    q_y180 = angle_axis_to_quaternion_wxyz(np.array([0.0, np.pi, 0.0], dtype=np.float64))
    q_base = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    q_base_inv = quat_conj_wxyz(q_base)
    return quat_normalize_wxyz(
        quat_mul_wxyz(
            quat_mul_wxyz(
                quat_mul_wxyz(q_x90, q_raw),
                q_y180,
            ),
            q_base_inv,
        )
    ).astype(np.float32, copy=False)


def quaternion_distance_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = quat_normalize_wxyz(q1)
    q2 = quat_normalize_wxyz(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))
