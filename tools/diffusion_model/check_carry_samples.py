"""Kinematic and contact plausibility of generated carry windows, compared with real ones.

Reads sample_carry_prior.py output, decodes both sets with mimickit_smp.decode and
runs MuJoCo forward kinematics on the cache's robot skeleton. Real windows give
the reference for each number; a generated window is plausible when its numbers
look like the real ones. Needs MuJoCo; no diffusion model is loaded.

- body_fk_error: generated hand/foot positions vs FK of the generated root and joints;
- velocity_error: root/box velocity channels vs finite differences of their positions;
- joint_off_axis: rotation a generated joint has off its hinge axis;
- joint_limit_violation: decoded angles outside the robot's joint ranges;
- hand_box: distance of the nearer hand origin to the box surface while the box is lifted;
- box_below_floor: box bottom below the floor;
- foot_slide: horizontal speed of a foot whose origin is within 8 cm of the floor.

    python tools/diffusion_model/check_carry_samples.py --samples <run>/samples_test.npz --out <run>/sample_checks.json
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from mimickit_smp.decode import decode_window, matrix_to_wxyz
from mimickit_smp.features import FeatureSchema

RATE = 50.0


def box_signed_distance(points, center, rotation, extents):
    """points [..., 3] -> signed distance to an oriented cuboid (negative inside)."""
    local = np.einsum("...ji,...j->...i", rotation, points - center)
    q = np.abs(local) - extents / 2
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(-1), 0.0)
    return outside + inside


def metrics(decoded, model, data, body_ids, limits):
    root = decoded["root_pos"].numpy()
    joints, key = decoded["joint_pos"].numpy(), decoded["key_body_pos"].numpy()
    quat = matrix_to_wxyz(decoded["root_rot"]).numpy()
    n, width = joints.shape[:2]
    fk = np.zeros_like(key)
    for i in range(n):
        for t in range(width):
            data.qpos[:3], data.qpos[3:7], data.qpos[7:36] = root[i, t], quat[i, t], joints[i, t]
            mujoco.mj_kinematics(model, data)
            fk[i, t] = data.xpos[body_ids]
    box_pos, box_rot = decoded["box_pos"].numpy(), decoded["box_rot"].numpy()
    extents = decoded["box_extents"].numpy()[:, None, :]
    half_height = 0.5 * np.einsum("ntij,nj->nti", np.abs(box_rot), extents[:, 0])[..., 2]
    bottom = box_pos[..., 2] - half_height
    lifted = bottom > 0.10
    hands = np.stack(
        [box_signed_distance(fk[:, :, k], box_pos, box_rot, extents) for k in (2, 3)], axis=-1
    ).min(-1)
    foot_box = np.stack(
        [box_signed_distance(fk[:, :, k], box_pos, box_rot, extents) for k in (0, 1)], axis=-1
    ).min(-1)
    # Floor skills: the box moves along the floor (push with hands, slide with feet).
    box_speed = np.linalg.norm(np.diff(box_pos[..., :2], axis=1), axis=-1) * RATE
    moving = np.r_["1", box_speed[:, :1], box_speed] > 0.10
    touching = np.minimum(hands, foot_box) < 0.05
    moving_windows = moving.any(1)
    feet = fk[:, :, :2]
    speed = np.linalg.norm(np.diff(feet[..., :2], axis=1), axis=-1) * RATE
    planted = (feet[:, 1:, :, 2] < 0.08) & (feet[:, :-1, :, 2] < 0.08)
    root_fd = np.diff(root, axis=1) * RATE
    box_fd = np.diff(box_pos, axis=1) * RATE
    lo, hi = limits
    violation = np.maximum(lo - joints, joints - hi).clip(min=0)

    def q(x, p=(0.5, 0.95)):
        x = np.asarray(x).ravel()
        return [float(v) for v in np.quantile(x, p)] if len(x) else [float("nan")] * len(p)

    return {
        "body_fk_error_cm_p50_p95": q(100 * np.linalg.norm(fk - key, axis=-1)),
        "root_velocity_error_m_s_p50_p95": q(
            np.linalg.norm(decoded["root_linear_velocity"].numpy()[:, 1:] - root_fd, axis=-1)
        ),
        "box_velocity_error_m_s_p50_p95": q(
            np.linalg.norm(decoded["box_linear_velocity"].numpy()[:, 1:] - box_fd, axis=-1)
        ),
        "joint_off_axis_deg_p50_p95": q(np.degrees(decoded["joint_off_axis_rad"].numpy())),
        "joint_limit_violation_frames": float((violation.max(-1) > 0.02).mean()),
        "lifted_frames": float(lifted.mean()),
        "hand_box_cm_p50_p95_when_lifted": q(100 * hands[lifted]),
        "hands_off_box_frames_when_lifted": float((hands[lifted] > 0.08).mean())
        if lifted.any()
        else float("nan"),
        "box_below_floor_frames": float((bottom < -0.02).mean()),
        "planted_foot_speed_m_s_p50_p95": q(speed[planted]),
        "box_moving_frames": float(moving.mean()),
        "hand_box_cm_p50_p95_when_moving": q(100 * hands[moving]),
        "foot_box_cm_p50_p95_when_moving": q(100 * foot_box[moving]),
        "moving_windows_without_contact": float((~(touching & moving).any(1))[moving_windows].mean())
        if moving_windows.any()
        else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    s = np.load(args.samples, allow_pickle=False)
    index = Path(str(s["index"]))
    schema = FeatureSchema.from_dict(json.loads(index.read_text())["feature_schema"]).current_features()
    if schema.sha256 != str(s["feature_schema_sha256"]):
        raise ValueError("Samples were made with a different feature schema")
    model = mujoco.MjModel.from_xml_path(str(index.parent / "robot_kinematics.xml"))
    data = mujoco.MjData(model)
    body_ids = [model.body(name).id for name in schema.key_body_names]
    hinge = [model.joint(name).id for name in schema.joint_names]
    limits = (model.jnt_range[hinge, 0], model.jnt_range[hinge, 1])
    size = torch.from_numpy(s["box_size"]).double()
    report = {
        "samples": str(args.samples),
        "windows": int(len(size)),
        "generated_per_size": int(s["generated_motion"].shape[1]) if s["generated_motion"].ndim == 4 else 1,
        "window_frames": schema.window_frames,
    }
    classes = s["class_ids"] if "class_ids" in s.files else np.zeros(len(size), np.int64)
    names = s["class_names"].tolist() if "class_names" in s.files else ["carry"]
    for name in ("real", "generated"):
        motion = torch.from_numpy(s[f"{name}_motion"]).double()
        cond, labels = size, classes
        if motion.dim() == 4:  # [windows, samples per size, frames, 216]
            cond = size.repeat_interleave(motion.shape[1], 0)
            labels = np.repeat(classes, motion.shape[1])
            motion = motion.flatten(0, 1)
        decoded = decode_window(motion, cond, schema)
        report[name] = metrics(decoded, model, data, body_ids, limits)
        # Per skill (carry / push / slide) when the samples carry class IDs.
        if len(np.unique(labels)) > 1:
            for c in np.unique(labels):
                keep = torch.from_numpy(labels == c)
                subset = {k: v[keep] for k, v in decoded.items()}
                report[f"{name}_{names[c]}"] = metrics(subset, model, data, body_ids, limits)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    width = max(len(k) for k in report["real"])
    columns = ["real", "generated"] + sorted(
        k for k, v in report.items() if isinstance(v, dict) and k.startswith(("real_", "generated_"))
    )
    fmt = lambda v: " / ".join(f"{x:.3g}" for x in v) if isinstance(v, list) else f"{v:.3g}"  # noqa: E731
    print(f"{'metric':{width}s}  " + "  ".join(f"{c:>20s}" for c in columns))
    for k in report["real"]:
        print(f"{k:{width}s}  " + "  ".join(f"{fmt(report[c][k]):>20s}" for c in columns))


if __name__ == "__main__":
    main()
