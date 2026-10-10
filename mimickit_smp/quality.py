"""Per-frame quality checks on native-rate retargeted G1-box motion.

Each check marks bad source frames; the dataset then drops every window that
touches a bad frame of a selected check, so a recording keeps its clean parts.

- physical: robot-box or robot-floor penetration, box below the floor, joint
  speed spikes, and box-tracking glitches (implausible box speed);
- chatter: frame-to-frame joint wobble, the 0.3 s RMS of a joint's residual from
  a 78 ms quadratic fit, on any joint. Normal fast motion fits such a short
  window; frame-alternating solver chatter does not;
- foot_slide: a robot foot contact whose least-moving sole point travels farther
  than the person's least-moving toe/ankle joint. Least-moving points make heel
  and toe roll count as stationary on both sides;
- support: the box is off the floor but no robot geom is near it.

Grip type is deliberately not checked: the prior should not exclude grips such
as carrying a long box from below, and support may come from any body part.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

CHECKS = ("physical", "chatter", "foot_slide", "support")
BITS = {name: 1 << i for i, name in enumerate(CHECKS)}
THRESHOLDS = {
    "penetration_m": 0.03,
    "box_below_floor_m": 0.02,  # Resting boxes read -0.2 +- 1.3 cm (p50, p5-p95) in the source mocap.
    "joint_speed_rad_s": 25.0,
    "box_speed_m_s": 3.0,
    "chatter_deg": 1.0,
    "chatter_fit_s": 0.078,
    "chatter_window_s": 0.3,
    "foot_contact_m": 0.003,
    "foot_min_contact_s": 0.11,
    "foot_excess_m": 0.05,
    "support_m": 0.03,
    "box_off_floor_m": 0.02,
}


def check_mask(checks):
    """Bit mask for check names; an empty selection keeps every window."""
    unknown = set(checks) - set(CHECKS)
    if unknown:
        raise ValueError(f"Unknown quality checks {sorted(unknown)}; use {CHECKS}")
    return sum(BITS[name] for name in set(checks))


def measure(qpos, scene_xml, t=THRESHOLDS):
    """Per-frame geometry from the replay scene: the box is geom "largebox", the floor
    geom "ground", and each foot has five sole spheres. Respects its collision masks."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    box, floor = model.geom("largebox").id, model.geom("ground").id
    # Report robot geoms within support distance of the box, not only penetrating ones.
    model.geom_margin[box] = t["support_m"]
    data = mujoco.MjData(model)
    spheres = np.asarray(
        [
            [model.geom(f"{side}_ankle_roll_sphere_{k}_link").id for k in range(1, 6)]
            for side in ("left", "right")
        ]
    )
    n = len(qpos)
    out = {
        "robot_box_penetration_m": np.zeros(n),
        "robot_floor_penetration_m": np.zeros(n),
        "box_floor_gap_m": np.zeros(n),
        "robot_box_nearest_m": np.full(n, np.inf),
        "sole_points": np.zeros((n, 2, 5, 3)),
    }
    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        out["box_floor_gap_m"][i] = mujoco.mj_geomDistance(model, data, box, floor, 10, None)
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            dist = float(contact.dist)
            if box in (a, b):
                if floor not in (a, b):
                    out["robot_box_nearest_m"][i] = min(out["robot_box_nearest_m"][i], dist)
                    out["robot_box_penetration_m"][i] = max(out["robot_box_penetration_m"][i], -dist)
            elif floor in (a, b):
                out["robot_floor_penetration_m"][i] = max(out["robot_floor_penetration_m"][i], -dist)
        out["sole_points"][i] = data.geom_xpos[spheres]
    out["sole_radius_m"] = float(model.geom_size[spheres[0, 0], 0])
    return out


def physical_flags(qpos, fps, m, t=THRESHOLDS):
    joint_speed = np.r_[0.0, np.abs(np.diff(qpos[:, 7:36], axis=0)).max(1) * fps]
    box_speed = np.r_[0.0, np.linalg.norm(np.diff(qpos[:, 36:39], axis=0), axis=1) * fps]
    return (
        (m["robot_box_penetration_m"] > t["penetration_m"])
        | (m["robot_floor_penetration_m"] > t["penetration_m"])
        | (m["box_floor_gap_m"] < -t["box_below_floor_m"])
        | (joint_speed > t["joint_speed_rad_s"])
        | (box_speed > t["box_speed_m_s"])
    )


def chatter_flags(joints, fps, t=THRESHOLDS):
    fit = max(5, int(round(t["chatter_fit_s"] * fps)) | 1)
    residual = joints - savgol_filter(joints, fit, 2, axis=0)
    window = max(1, int(round(t["chatter_window_s"] * fps)))
    rms = np.sqrt(np.maximum(uniform_filter1d(residual**2, window, axis=0), 0.0))
    return np.degrees(rms).max(1) > t["chatter_deg"]


def _intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return zip(edges[::2], edges[1::2])


def foot_slide_flags(sole_points, sole_radius, toes, ankles, fps, t=THRESHOLDS):
    """sole_points [T, feet, points, 3]; toes/ankles [T, feet, 3] of the person, same frame and scale."""
    flags = np.zeros(len(sole_points), bool)
    min_frames = int(round(t["foot_min_contact_s"] * fps))
    for foot in range(sole_points.shape[1]):
        touch = ((sole_points[:, foot, :, 2] - sole_radius) < t["foot_contact_m"]).any(-1)
        for a, b in _intervals(touch):
            if b - a < min_frames:
                continue
            robot = np.linalg.norm(
                sole_points[b - 1, foot, :, :2] - sole_points[a, foot, :, :2], axis=-1
            ).min()
            person = min(
                np.linalg.norm(toes[b - 1, foot, :2] - toes[a, foot, :2]),
                np.linalg.norm(ankles[b - 1, foot, :2] - ankles[a, foot, :2]),
            )
            if robot - person > t["foot_excess_m"]:
                flags[a:b] = True
    return flags


def support_flags(m, t=THRESHOLDS):
    return (m["box_floor_gap_m"] > t["box_off_floor_m"]) & (m["robot_box_nearest_m"] > t["support_m"])


def frame_flags(qpos, fps, scene_xml, human_joints, human_joint_names, t=THRESHOLDS):
    """uint8 bit flags (see BITS) per native frame, plus the measurements behind them."""
    m = measure(qpos, scene_xml, t)
    names = list(human_joint_names)
    toes = human_joints[:, [names.index("LeftToeBase"), names.index("RightToeBase")]]
    ankles = human_joints[:, [names.index("LeftFoot"), names.index("RightFoot")]]
    flags = np.zeros(len(qpos), np.uint8)
    flags[physical_flags(qpos, fps, m, t)] |= BITS["physical"]
    flags[chatter_flags(qpos[:, 7:36], fps, t)] |= BITS["chatter"]
    flags[foot_slide_flags(m["sole_points"], m["sole_radius_m"], toes, ankles, fps, t)] |= BITS["foot_slide"]
    flags[support_flags(m, t)] |= BITS["support"]
    return flags, m


def resample_flags(native_flags, native_times, query_times):
    """A resampled state inherits the flags of both native frames it interpolates."""
    k = np.clip(np.searchsorted(native_times, query_times, side="right") - 1, 0, len(native_flags) - 1)
    return native_flags[k] | native_flags[np.minimum(k + 1, len(native_flags) - 1)]
