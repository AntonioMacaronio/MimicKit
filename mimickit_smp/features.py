"""Versioned Z-up, WXYZ robot-object representation shared with live rewards."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import torch

# Frame layout used by the portable cache and live-simulation adapter.
STATE_SLICES = {
    "root_pos": (0, 3),
    "root_quat_wxyz": (3, 7),
    "joint_pos": (7, 36),
    "key_body_pos": (36, 48),
    "root_linear_velocity": (48, 51),
    "root_angular_velocity": (51, 54),
    "box_center_pos": (54, 57),
    "box_quat_wxyz": (57, 61),
    "box_linear_velocity": (61, 64),
    "box_angular_velocity": (64, 67),
    "box_full_extents": (67, 70),
}
FEATURE_SLICES = {
    "root_position": (0, 3),
    "root_rotation_6d": (3, 9),
    "joint_rotations_6d": (9, 183),
    "key_body_positions": (183, 195),
    "root_linear_velocity": (195, 198),
    "root_angular_velocity": (198, 201),
    "box_position": (201, 204),
    "box_rotation_6d": (204, 210),
    "box_linear_velocity": (210, 213),
    "box_angular_velocity": (213, 216),
    "box_full_extents": (216, 219),
}
# v3 removes box size from the denoised motion and passes it as a clean condition.
FEATURE_SLICES_V3 = {k: v for k, v in FEATURE_SLICES.items() if k != "box_full_extents"}
CONDITION_SLICES_V3 = {"box_size": (0, 3)}
KEY_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_rubber_hand_link",
    "right_rubber_hand_link",
)
FEATURE_VERSIONS = ("g1_box_smp_v1", "g1_box_smp_v2", "g1_box_smp_v3")
CURRENT_FEATURE_VERSION = "g1_box_smp_v3"
BOX_AXIS_TIE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FeatureSchema:
    joint_names: tuple[str, ...]
    joint_axes: tuple[tuple[float, float, float], ...]
    robot_kinematics_sha256: str
    key_body_names: tuple[str, ...] = KEY_BODIES
    sample_rate_hz: float = 50.0
    window_frames: int = 16
    version: str = CURRENT_FEATURE_VERSION

    def __post_init__(self):
        if self.version not in FEATURE_VERSIONS:
            raise ValueError("Unsupported feature schema version")
        if len(self.joint_names) != 29 or len(set(self.joint_names)) != 29:
            raise ValueError("Expected 29 unique, ordered G1 joints")
        axes = torch.as_tensor(self.joint_axes, dtype=torch.float64)
        if axes.shape != (29, 3) or not torch.allclose(axes.norm(dim=-1), torch.ones(29, dtype=axes.dtype)):
            raise ValueError("Expected 29 unit local joint axes")
        if len(self.key_body_names) != 4 or len(set(self.key_body_names)) != 4:
            raise ValueError("Expected four unique hand/foot body origins")
        if not self.sample_rate_hz > 0 or self.window_frames < 2:
            raise ValueError("Invalid time window")

    @property
    def span_seconds(self):
        return (self.window_frames - 1) / self.sample_rate_hz

    @property
    def box_size_condition(self):
        """v3 conditions on canonical box size instead of denoising it."""
        return self.version == "g1_box_smp_v3"

    @property
    def feature_dim(self):
        return 216 if self.box_size_condition else 219

    @property
    def condition_dim(self):
        return 3 if self.box_size_condition else 0

    def current_features(self):
        """Upgrade the encoding contract without modifying cached mesh-frame states."""
        return replace(self, version=CURRENT_FEATURE_VERSION)

    def to_dict(self):
        data = asdict(self)
        data.update(
            state_slices=STATE_SLICES,
            feature_slices=FEATURE_SLICES_V3 if self.box_size_condition else FEATURE_SLICES,
            frame_order="oldest_to_newest",
            units="meters_radians_seconds",
            up_axis="Z",
            quaternion_order="WXYZ",
            rotation_6d="rotated_x_then_z_axes",
            heading_reference="last_frame_root_yaw",
            position_reference="last_frame_root_xy_with_world_height",
            key_body_reference="each_frame_root_position_last_frame_heading",
            box_pose_reference="center_of_object_local_visual_mesh_AABB",
            velocity_convention="world_spatial_linear_and_angular_velocity_rotated_by_last_heading",
            source_velocity_estimator="backward_difference_at_target_rate",
        )
        if self.version != "g1_box_smp_v1":
            data.update(
                box_input_axes="source_mesh_axes_with_matching_local_full_extents",
                box_feature_axes="right_handed_Z_most_up_X_most_root_forward",
                box_axis_selection_frame="last_frame_of_window_in_final_root_heading",
                box_axis_mapping="one_signed_permutation_for_entire_window",
                box_axis_tie_break="Z_then_X_then_Y_for_up_axis_X_then_Y_then_Z_for_forward_axis",
                box_axis_tie_tolerance=BOX_AXIS_TIE_TOLERANCE,
            )
        if self.version == "g1_box_smp_v2":
            data.update(box_feature_extents="input_extents_permuted_to_feature_X_Y_Z")
        if self.box_size_condition:
            data.update(
                condition_slices=CONDITION_SLICES_V3,
                box_size_condition="canonical_X_Y_Z_full_extents_at_last_frame_not_noised",
            )
        return data

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            joint_names=tuple(data["joint_names"]),
            joint_axes=tuple(tuple(x) for x in data["joint_axes"]),
            robot_kinematics_sha256=data["robot_kinematics_sha256"],
            key_body_names=tuple(data["key_body_names"]),
            sample_rate_hz=data["sample_rate_hz"],
            window_frames=data["window_frames"],
            version=data["version"],
        )
        # Reject metadata with altered conventions even if dimensions still match.
        if json.dumps(obj.to_dict(), sort_keys=True) != json.dumps(data, sort_keys=True):
            raise ValueError("Feature convention mismatch")
        return obj


def quat_mul(a, b):
    av, bv = torch.broadcast_tensors(a[..., 1:], b[..., 1:])
    return torch.cat(
        (
            a[..., :1] * b[..., :1] - (av * bv).sum(-1, keepdim=True),
            a[..., :1] * bv + b[..., :1] * av + torch.cross(av, bv, dim=-1),
        ),
        dim=-1,
    )


def quat_rotate(q, v):
    xyz, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(xyz, uv, dim=-1))


def rotation_6d(q):
    x = torch.zeros_like(q[..., :3])
    z = torch.zeros_like(x)
    x[..., 0], z[..., 2] = 1, 1
    return torch.cat((quat_rotate(q, x), quat_rotate(q, z)), dim=-1)


def _lexicographic_axis(candidates, eligible, components):
    """Choose by physical directions, never by the source mesh's axis numbering.

    candidates is [..., xyz_component, candidate_axis]. Tolerance makes exact
    geometric ties robust to quaternion roundoff under a mesh-axis relabeling.
    """
    for component in components:
        score = candidates[..., component, :]
        best = score.masked_fill(~eligible, -torch.inf).amax(-1, keepdim=True)
        eligible = eligible & (score >= best - BOX_AXIS_TIE_TOLERANCE)
    return eligible.to(torch.int64).argmax(-1)


def canonical_box_axes(quats, extents):
    """Return heading-local 6D box rotation and matching full extents.

    Input shapes: [..., H, 4] WXYZ orientations already in the final root heading,
    [..., H, 3] source-mesh extents. Select a proper signed axis permutation at
    the window's last frame and apply it to *every* frame. This preserves the
    oriented cuboid, its tilt, temporal rotation, and world angular velocity.
    Equivalent signed/permuted mesh frames produce the same features. The map
    is piecewise smooth; symmetry-sector boundaries between windows remain.
    """
    basis = torch.eye(3, dtype=quats.dtype, device=quats.device)
    # Columns are the three local mesh axes expressed in the heading frame.
    axes = quat_rotate(quats.unsqueeze(-2), basis).transpose(-1, -2)
    last = axes[..., -1, :, :]
    up_sign = torch.where(last[..., 2, :] >= 0, 1.0, -1.0)
    up_candidates = last * up_sign.unsqueeze(-2)
    iz = _lexicographic_axis(up_candidates, torch.ones_like(up_sign, dtype=torch.bool), (2, 0, 1))
    forward_sign = torch.where(last[..., 0, :] >= 0, 1.0, -1.0)
    forward_candidates = last * forward_sign.unsqueeze(-2)
    remaining = torch.arange(3, device=quats.device) != iz.unsqueeze(-1)
    ix = _lexicographic_axis(forward_candidates, remaining, (0, 1, 2))
    iy = 3 - ix - iz  # The remaining unsigned axis; Y's sign is set by Z cross X.

    def select_axis(index, sign):
        gather = index[..., None, None, None].expand(*axes.shape[:-1], 1)
        direction = axes.gather(-1, gather).squeeze(-1)
        chosen_sign = sign.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        return direction * chosen_sign[..., None, None]

    x = select_axis(ix, forward_sign)
    z = select_axis(iz, up_sign)
    # X and Z fully specify a proper frame: Y = Z cross X. No source handedness leaks in.
    order = torch.stack((ix, iy, iz), -1).unsqueeze(-2).expand_as(extents)
    return torch.cat((x, z), -1), extents.gather(-1, order)


def encode_window(states: torch.Tensor, schema: FeatureSchema):
    """Encode [..., H, 70] live or cached states into (motion, condition).

    v3 returns [..., H, 216] motion features and a [..., 3] box-size condition:
    canonical X (most forward), Y, Z (most up) full extents at the window's last
    frame, which the prior receives without noise. v1/v2 return [..., H, 219]
    features with the extents as denoised channels, and no condition.

    Caller maps named joints/body origins into schema order and supplies unit
    WXYZ quaternions, world-frame velocities, and full (not half) box extents.
    This tensor-only function is piecewise differentiable and supports CPU/CUDA batches.
    It performs no temporal differencing or simulator access.
    """
    if states.shape[-2:] != (schema.window_frames, 70):
        raise ValueError(f"Expected (..., {schema.window_frames}, 70), got {states.shape}")
    root, q = states[..., :3], states[..., 3:7]
    last = q[..., -1, :]
    w, x, y, z = last.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    inv = torch.stack((torch.cos(yaw / 2), zero, zero, -torch.sin(yaw / 2)), -1).unsqueeze(-2)
    origin = root[..., -1:, :] * root.new_tensor([1, 1, 0])
    local_root = quat_rotate(inv, root - origin)
    joint_angles = states[..., 7:36] / 2
    axes = torch.as_tensor(schema.joint_axes, dtype=states.dtype, device=states.device)
    joint_q = torch.cat((torch.cos(joint_angles)[..., None], torch.sin(joint_angles)[..., None] * axes), -1)
    key = states[..., 36:48].reshape(*states.shape[:-1], 4, 3)
    key = quat_rotate(inv.unsqueeze(-2), key - root.unsqueeze(-2)).flatten(-2)
    box_quat = quat_mul(inv, states[..., 57:61])
    if schema.version == "g1_box_smp_v1":
        # Explicit legacy encoding only; datasets upgrade to the current schema.
        box_rotation, box_extents = rotation_6d(box_quat), states[..., 67:70]
    else:
        box_rotation, box_extents = canonical_box_axes(box_quat, states[..., 67:70])
    motion = [
        local_root,
        rotation_6d(quat_mul(inv, q)),
        rotation_6d(joint_q).flatten(-2),
        key,
        quat_rotate(inv, states[..., 48:51]),
        quat_rotate(inv, states[..., 51:54]),
        quat_rotate(inv, states[..., 54:57] - origin),
        box_rotation,
        quat_rotate(inv, states[..., 61:64]),
        quat_rotate(inv, states[..., 64:67]),
    ]
    if schema.box_size_condition:
        return torch.cat(motion, dim=-1), box_extents[..., -1, :]
    return torch.cat([*motion, box_extents], dim=-1), None


def encode_motion(states: torch.Tensor, schema: FeatureSchema) -> torch.Tensor:
    """Motion features only; see encode_window for the box-size condition."""
    return encode_window(states, schema)[0]
