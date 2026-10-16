"""Left-right mirror augmentation of a prepared cache (reflection across world y = 0).

A mirrored window is the same scene reflected through a vertical plane: positions
and linear velocities negate y, rotations R become M R M (WXYZ quaternion
(w, -x, y, -z)), angular velocities become (-wx, wy, -wz), and left/right joints
and body origins swap. The G1 is left-right symmetric, so the reflected robot is
again a valid robot pose. Each joint's mirror sign comes from the cache's robot
skeleton: set the joint alone, reflect every body origin with left and right
bodies swapped, and keep the partner-joint sign that reproduces it. A reflected
cuboid is the same cuboid (its local y flip is a symmetry), so box extents and
the size condition are unchanged; canonical box axes are recomputed by the encoder.

Quality flags, lift bounds and splits carry over: reflection preserves every
penetration, contact and slide measurement. Mirrored segments keep their actor's
split and get the trial suffix "__mirror".

    python -m mimickit_smp.mirror --index <cache>/index.json --output <new cache>
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .prepare import file_hash, save_json

REFLECT = np.array([1.0, -1.0, 1.0])
QUAT = np.array([1.0, -1.0, 1.0, -1.0])  # WXYZ of M R M for M = diag(1, -1, 1)
PSEUDO = np.array([-1.0, 1.0, -1.0])  # Angular velocity is a pseudovector.
KEY_BODY_SWAP = [1, 0, 3, 2]  # left ankle, right ankle, left hand, right hand


def swap_side(name):
    return name.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")


def mirror_map(model, joint_names, probe=0.4):
    """Partner joint index and sign per joint, derived from forward kinematics."""
    import mujoco

    data = mujoco.MjData(model)
    bodies = [model.body(b).name for b in range(model.nbody)]
    body_partner = [bodies.index(swap_side(n)) for n in bodies]
    adr = [model.jnt_qposadr[model.joint(n).id] for n in joint_names]

    def fk(q):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        return data.xpos.copy()

    zero = fk(np.zeros(len(adr)))
    # Bodies whose name-swapped partner is not their mirror image (e.g. the G1 sole spheres,
    # numbered in the same lateral order on both feet) are relabelings, not asymmetry of
    # the robot; they are left out of the comparison and reported.
    gap = np.abs(zero[body_partner] * REFLECT - zero).max(-1)
    used = gap < 1e-4
    relabeled = sorted(bodies[b] for b in np.flatnonzero(~used))
    partner, signs, errors, margins = [], [], [], []
    for j, name in enumerate(joint_names):
        p = joint_names.index(swap_side(name))
        q = np.zeros(len(adr))
        q[j] = probe
        target = (fk(q)[body_partner] * REFLECT)[used]
        err = {}
        for sign in (1.0, -1.0):
            q2 = np.zeros(len(adr))
            q2[p] = sign * probe
            err[sign] = float(np.abs(fk(q2)[used] - target).max())
        sign = min(err, key=err.get)
        partner.append(p)
        signs.append(sign)
        errors.append(err[sign])
        margins.append(err[-sign])
    info = {
        "zero_pose_asymmetry_m": float(gap[used].max()),
        "max_probe_error_m": max(errors),
        "min_wrong_sign_error_m": min(margins),
        "relabeled_bodies_excluded": relabeled,
    }
    return np.asarray(partner), np.asarray(signs), info


def mirror_states(states, partner, signs):
    """[..., T, 70] cached states -> their left-right mirror image."""
    s = np.asarray(states)
    out = s.copy()
    out[..., 0:3] = s[..., 0:3] * REFLECT
    out[..., 3:7] = s[..., 3:7] * QUAT
    out[..., 7 + partner] = s[..., 7:36] * signs
    key = s[..., 36:48].reshape(*s.shape[:-1], 4, 3)[..., KEY_BODY_SWAP, :] * REFLECT
    out[..., 36:48] = key.reshape(*s.shape[:-1], 12)
    out[..., 48:51] = s[..., 48:51] * REFLECT
    out[..., 51:54] = s[..., 51:54] * PSEUDO
    out[..., 54:57] = s[..., 54:57] * REFLECT
    out[..., 57:61] = s[..., 57:61] * QUAT
    out[..., 61:64] = s[..., 61:64] * REFLECT
    out[..., 64:67] = s[..., 64:67] * PSEUDO
    return out.astype(s.dtype)


def fk_key_bodies(model, key_body_names, states):
    """Hand/foot origins by forward kinematics of cached root and joint states."""
    import mujoco

    data = mujoco.MjData(model)
    ids = [model.body(n).id for n in key_body_names]
    out = np.zeros((len(states), 12))
    for i, row in enumerate(states):
        data.qpos[:] = row[:36]
        mujoco.mj_kinematics(model, data)
        out[i] = data.xpos[ids].reshape(-1)
    return out


def mirror_cache(index_path, output, check_segments=40):
    import mujoco

    from .features import FeatureSchema

    index_path, output = Path(index_path).resolve(), Path(output).resolve()
    index = json.loads(index_path.read_text())
    if index.get("status") != "complete":
        raise ValueError("Refusing an incomplete prepared dataset")
    if any(s.get("mirrored") for s in index["segments"]):
        raise ValueError("Cache is already mirrored")
    root = index_path.parent
    schema = FeatureSchema.from_dict(index["feature_schema"])
    model = mujoco.MjModel.from_xml_path(str(root / "robot_kinematics.xml"))
    partner, signs, symmetry = mirror_map(model, list(schema.joint_names))
    if (
        symmetry["max_probe_error_m"] > 1e-4
        or symmetry["zero_pose_asymmetry_m"] > 1e-4
        or symmetry["min_wrong_sign_error_m"] < 1e-2
    ):
        raise ValueError(f"Robot skeleton is not left-right symmetric enough to mirror: {symmetry}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "states").mkdir()
    (output / "quality").mkdir()
    shutil.copy2(root / "robot_kinematics.xml", output / "robot_kinematics.xml")
    segments, fk_error = [], 0.0
    rng = np.random.default_rng(0)
    checked = set(
        rng.choice(
            len(index["segments"]), min(check_segments, len(index["segments"])), replace=False
        ).tolist()
    )
    for k, s in enumerate(index["segments"]):
        states = np.load(root / s["states"], allow_pickle=False)
        name = Path(s["states"]).stem
        copies = {}
        for suffix, array in (("", states), ("__mirror", mirror_states(states, partner, signs))):
            path = output / "states" / f"{name}{suffix}.npy"
            np.save(path, array, allow_pickle=False)
            copies[suffix] = path
        if k in checked:
            mirrored = np.load(copies["__mirror"])
            fk_error = max(
                fk_error,
                float(
                    np.abs(fk_key_bodies(model, schema.key_body_names, mirrored) - mirrored[:, 36:48]).max()
                ),
            )
        flags = {}
        if "quality_flags" in s:
            for suffix in copies:
                path = output / "quality" / f"{name}{suffix}.npy"
                shutil.copy2(root / s["quality_flags"], path)
                flags[suffix] = path
        for suffix, path in copies.items():
            record = dict(s, states=str(path.relative_to(output)), states_sha256=file_hash(path))
            if suffix:
                record.update(trial_id=s["trial_id"] + suffix, mirrored=True, mirror_of=s["trial_id"])
            if flags:
                record.update(
                    quality_flags=str(flags[suffix].relative_to(output)),
                    quality_flags_sha256=file_hash(flags[suffix]),
                )
            segments.append(record)
        if (k + 1) % 500 == 0:
            print(f"Mirrored {k + 1}/{len(index['segments'])} segments", flush=True)
    if fk_error > 1e-3:
        raise ValueError(f"Mirrored body origins disagree with forward kinematics by {fk_error:.4f} m")
    mirrored_index = dict(
        index,
        segments=segments,
        mirror_augmentation={
            "source_index": str(index_path),
            "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "plane": "world y = 0 (reflect y); rotations M R M; left/right joints and bodies swapped",
            "joint_partner": [schema.joint_names[p] for p in partner],
            "joint_sign": signs.tolist(),
            **symmetry,
            "max_fk_error_m_checked_segments": fk_error,
            "checked_segments": len(checked),
            "code_sha256": file_hash(__file__),
        },
    )
    summary = dict(index["summary"])
    for key in ("segments", "states", "windows", "lifted_windows"):
        summary[key] = 2 * summary[key]
    summary["split_segments"] = {k: 2 * v for k, v in summary["split_segments"].items()}
    if "skill_segments" in summary:
        summary["skill_segments"] = {k: 2 * v for k, v in summary["skill_segments"].items()}
    if "quality_flagged_states" in summary:
        summary["quality_flagged_states"] = {k: 2 * v for k, v in summary["quality_flagged_states"].items()}
    mirrored_index["summary"] = summary
    save_json(output / "index.json", mirrored_index)
    save_json(output / "status.json", {"status": "complete", **summary})
    print(json.dumps({**symmetry, "max_fk_error_m": fk_error, "segments": len(segments)}, indent=2))
    return mirrored_index


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mirror_cache(args.index, args.output)


if __name__ == "__main__":
    main()
