"""Invert v3 window features into robot and box states in the window's own frame.

The frame has its origin at the last root position's XY (world height kept) and
its X axis along the last root heading, as in encode_window. Generated windows
have no world pose; decoding them here is enough for kinematic checks and replay.

Root and box orientations come from their 6D features by Gram-Schmidt. A joint
angle is the rotation vector of its 6D rotation projected on the joint axis; the
residual off-axis rotation measures how far a generated joint left its hinge.
The box frame is the canonical one (X most forward, Z most up at the last frame),
with full extents from the size condition in the same X/Y/Z order.
"""

import torch

from .features import FEATURE_SLICES_V3


def rotation_from_6d(six):
    """[..., 6] rotated X then Z axes -> [..., 3, 3] proper rotation matrices (columns X, Y, Z)."""
    x = torch.nn.functional.normalize(six[..., :3], dim=-1)
    z = six[..., 3:] - (six[..., 3:] * x).sum(-1, keepdim=True) * x
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def matrix_to_wxyz(m):
    """Rotation matrices -> unit WXYZ quaternions with non-negative w."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    q = torch.stack(
        (
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ),
        -1,
    )
    # Pick the numerically largest component and derive the others from it.
    k = q.argmax(-1, keepdim=True)
    w = torch.stack(
        (
            q[..., 0],
            m[..., 2, 1] - m[..., 1, 2],
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 1, 0] - m[..., 0, 1],
        ),
        -1,
    )
    x = torch.stack(
        (
            m[..., 2, 1] - m[..., 1, 2],
            q[..., 1],
            m[..., 0, 1] + m[..., 1, 0],
            m[..., 0, 2] + m[..., 2, 0],
        ),
        -1,
    )
    y = torch.stack(
        (
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 0, 1] + m[..., 1, 0],
            q[..., 2],
            m[..., 1, 2] + m[..., 2, 1],
        ),
        -1,
    )
    z = torch.stack(
        (
            m[..., 1, 0] - m[..., 0, 1],
            m[..., 0, 2] + m[..., 2, 0],
            m[..., 1, 2] + m[..., 2, 1],
            q[..., 3],
        ),
        -1,
    )
    candidates = torch.stack((w, x, y, z), -2)
    out = candidates.gather(-2, k.unsqueeze(-1).expand(*k.shape[:-1], 1, 4)).squeeze(-2)
    out = torch.nn.functional.normalize(out, dim=-1)
    return torch.where(out[..., :1] < 0, -out, out)


def rotvec_from_matrix(m):
    """Rotation matrices -> rotation vectors (angle < pi)."""
    q = matrix_to_wxyz(m)
    norm = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, q[..., :1])
    return q[..., 1:] * torch.where(norm > 1e-9, angle / norm.clamp_min(1e-12), torch.full_like(norm, 2.0))


def decode_window(motion, condition, schema):
    """[..., H, 216] motion and [..., 3] box size -> dict of states in the window frame.

    Returns root_pos [..., H, 3], root_rot [..., H, 3, 3], joint_pos [..., H, 29],
    joint_off_axis_rad [..., H, 29], key_body_pos [..., H, 4, 3] (world positions in
    the window frame), root/box linear and angular velocities, box_pos, box_rot and
    box_extents [..., 3] (canonical X/Y/Z, constant over the window).
    """
    if motion.shape[-1] != 216 or not schema.box_size_condition:
        raise ValueError("decode_window supports v3 (216 motion channels plus a box-size condition)")
    f = {name: motion[..., a:b] for name, (a, b) in FEATURE_SLICES_V3.items()}
    root_rot = rotation_from_6d(f["root_rotation_6d"])
    joints = rotation_from_6d(f["joint_rotations_6d"].reshape(*motion.shape[:-1], 29, 6))
    rotvec = rotvec_from_matrix(joints)
    axes = torch.as_tensor(schema.joint_axes, dtype=motion.dtype, device=motion.device)
    angle = (rotvec * axes).sum(-1)
    root = f["root_position"]
    key = f["key_body_positions"].reshape(*motion.shape[:-1], 4, 3) + root.unsqueeze(-2)
    return {
        "root_pos": root,
        "root_rot": root_rot,
        "joint_pos": angle,
        "joint_off_axis_rad": (rotvec - angle.unsqueeze(-1) * axes).norm(dim=-1),
        "key_body_pos": key,
        "root_linear_velocity": f["root_linear_velocity"],
        "root_angular_velocity": f["root_angular_velocity"],
        "box_pos": f["box_position"],
        "box_rot": rotation_from_6d(f["box_rotation_6d"]),
        "box_linear_velocity": f["box_linear_velocity"],
        "box_angular_velocity": f["box_angular_velocity"],
        "box_extents": condition,
    }
