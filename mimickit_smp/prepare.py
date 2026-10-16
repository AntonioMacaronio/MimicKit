"""Prepare portable 50 Hz G1/box state arrays from a selected UMR manifest.

MuJoCo is used only for offline FK, never dynamics. Source files remain read-only.
"""

import argparse
import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import quality
from .features import KEY_BODIES, FeatureSchema

# Source phases kept per policy: 0 pickup context, 1 lifted, 2 set-down context.
# For push and slide, phase 1 is the interval in which the box is moved along the floor.
WINDOW_POLICIES = {"lifted": (1,), "segment": (0, 1, 2)}
# Skill classes; a segment's "skill" (default carry) sets its class_id.
CLASS_MAP = {"carry": 0, "push": 1, "slide": 2}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def robot_xml(scene):
    """Extract exactly the source G1 articulated skeleton, without mesh assets."""
    source = ET.parse(scene).getroot()
    if source.find("include") is not None:
        raise ValueError("Expected the canonical expanded scene XML")
    pelvis = source.find("./worldbody/body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("Missing named pelvis")
    root = ET.Element("mujoco", model="smp_g1_kinematics")
    ET.SubElement(root, "compiler", angle="radian")
    body = copy.deepcopy(pelvis)
    for parent in body.iter():
        for child in list(parent):
            if child.tag not in ("body", "joint", "freejoint", "inertial"):
                parent.remove(child)
    ET.SubElement(root, "worldbody").append(body)
    return ET.tostring(root, encoding="unicode")


def slerp_wxyz(times, quats, query):
    quats = np.asarray(quats, np.float64)
    if not np.allclose(np.linalg.norm(quats, axis=-1), 1, atol=1e-5):
        raise ValueError("Non-unit source quaternion")
    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    return Slerp(times, rotations)(query)


def resample_positive(qpos, times, phase, object_center, sample_rate=50.0):
    """Resample only the contiguous lifted phase (the "lifted" window policy)."""
    return resample_interval(qpos, times, phase, object_center, sample_rate, WINDOW_POLICIES["lifted"])


def resample_interval(qpos, times, phase, object_center, sample_rate=50.0, phases=(1,), flags=None):
    """Interpolation and backward velocities use only the contiguous selected phases.

    Drop the first resampled state: its velocity has no preceding selected sample.
    No padding, wrapping, extrapolation, or future differences are introduced.
    Each resampled state keeps the phase of the latest native frame at or before it.
    Optional native quality flags pass to every state that interpolates, or takes
    its velocity from, a flagged frame.
    """
    qpos, times, phase = np.asarray(qpos), np.asarray(times, np.float64), np.asarray(phase)
    if qpos.shape != (len(times), 43) or not np.isfinite(qpos).all():
        raise ValueError("Expected finite root+29-joint+box states")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or len(phase) != len(times):
        raise ValueError("Invalid source clock or phases")
    ids = np.flatnonzero(np.isin(phase, phases))
    if len(ids) < 3 or not np.array_equal(ids, np.arange(ids[0], ids[-1] + 1)):
        raise ValueError("Expected one contiguous interval of the selected phases")
    lifted = np.flatnonzero(phase == 1)
    if not len(lifted):
        raise ValueError("Expected a lifted interval")
    q, t = qpos[ids], times[ids]
    count = int(np.floor((t[-1] - t[0]) * sample_rate + 1e-9)) + 1
    query = t[0] + np.arange(count) / sample_rate
    query = query[query <= t[-1]]
    if len(query) < 3:
        raise ValueError("Lift too short to derive velocity")
    root_rot = slerp_wxyz(t, q[:, 3:7], query)
    box_rot = slerp_wxyz(t, q[:, 39:43], query)
    root_pos = np.column_stack([np.interp(query, t, q[:, i]) for i in range(3)])
    joints = np.column_stack([np.interp(query, t, q[:, i]) for i in range(7, 36)])
    box_pos = np.column_stack([np.interp(query, t, q[:, i]) for i in range(36, 39)])
    box_pos += box_rot.apply(object_center)

    def angular(rot):
        return (rot[1:] * rot[:-1].inv()).as_rotvec() * sample_rate

    query_phase = phase[ids][np.searchsorted(t, query, side="right") - 1]
    extra = {}
    if flags is not None:
        bracket = quality.resample_flags(np.asarray(flags, np.uint8)[ids], t, query)
        extra["quality_flags"] = bracket[1:] | bracket[:-1]
    return {
        "time": query[1:],
        "root_pos": root_pos[1:],
        "root_quat": root_rot.as_quat()[1:, [3, 0, 1, 2]],
        "joint_pos": joints[1:],
        "root_vel": np.diff(root_pos, axis=0) * sample_rate,
        "root_ang_vel": angular(root_rot),
        "box_pos": box_pos[1:],
        "box_quat": box_rot.as_quat()[1:, [3, 0, 1, 2]],
        "box_vel": np.diff(box_pos, axis=0) * sample_rate,
        "box_ang_vel": angular(box_rot),
        "phase": query_phase[1:],
        "positive_time_bounds": [float(times[lifted[0]]), float(times[lifted[-1]])],
        "interval_time_bounds": [float(t[0]), float(t[-1])],
        **extra,
    }


def process_segment(segment, output, schema_dict, window_policy="lifted"):
    import mujoco

    schema = FeatureSchema.from_dict(schema_dict)
    output = Path(output)
    path = Path(segment["trajectory"])
    if file_hash(path) != segment["trajectory_sha256"]:
        raise ValueError(f"Changed source: {path}")
    with np.load(path, allow_pickle=False) as d:
        if tuple(d["joint_names"].tolist()) != schema.joint_names:
            raise ValueError("Named joint order differs from schema")
        skeleton = robot_xml(str(d["scene_xml"]))
        if hashlib.sha256(skeleton.encode()).hexdigest() != schema.robot_kinematics_sha256:
            raise ValueError(f"Different robot kinematics in {path}")
        frames = d["source_frames"]
        positive = frames[d["phase"] == 1]
        if not np.array_equal(positive, np.arange(segment["lift_start_frame"], segment["lift_stop_frame"])):
            raise ValueError("Source/manifest lifted-frame mismatch")
        vertices = d["object_vertices"]
        extents = np.ptp(vertices, axis=0)
        center = (vertices.min(0) + vertices.max(0)) / 2
        if not np.isfinite(vertices).all() or np.any(extents <= 0):
            raise ValueError("Invalid object-local dimensions")
        if "object_local_aabb_extents_m" in d and not np.allclose(d["object_local_aabb_extents_m"], extents):
            raise ValueError("Object dimension metadata mismatch")
        if not np.allclose(np.diff(d["source_timestamps"]), 1 / float(d["fps"]), atol=1e-7, rtol=0):
            raise ValueError("Invalid native sampling rate")
        native_flags, _ = quality.frame_flags(
            d["qpos"], float(d["fps"]), str(d["scene_xml"]), d["human_joints"], d["human_joint_names"]
        )
        values = resample_interval(
            d["qpos"],
            d["source_timestamps"],
            d["phase"],
            center,
            schema.sample_rate_hz,
            WINDOW_POLICIES[window_policy],
            native_flags,
        )
    model = mujoco.MjModel.from_xml_string(skeleton)
    names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    if model.nq != 36 or names != schema.joint_names:
        raise ValueError("Robot state layout mismatch")
    data = mujoco.MjData(model)
    body_ids = [model.body(name).id for name in schema.key_body_names]
    count = len(values["time"])
    if count < schema.window_frames:
        raise ValueError("Selected segment shorter than one valid window")
    states = np.empty((count, 70), np.float32)
    states[:, :36] = np.concatenate([values[k] for k in ("root_pos", "root_quat", "joint_pos")], axis=-1)
    for i in range(count):
        # Double precision FK before writing the float32 cache.
        data.qpos[:] = np.r_[values["root_pos"][i], values["root_quat"][i], values["joint_pos"][i]]
        mujoco.mj_kinematics(model, data)
        states[i, 36:48] = data.xpos[body_ids].reshape(-1)
    states[:, 48:67] = np.concatenate(
        [values[k] for k in ("root_vel", "root_ang_vel", "box_pos", "box_quat", "box_vel", "box_ang_vel")],
        axis=-1,
    )
    states[:, 67:70] = extents
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite prepared state")
    name = segment["trial_id"]
    if Path(name).name != name:
        raise ValueError("Unsafe segment ID")
    destination = output / "states" / (name + ".npy")
    np.save(destination, states, allow_pickle=False)
    flags_path = output / "quality" / (name + ".npy")
    np.save(flags_path, values["quality_flags"], allow_pickle=False)
    lifted = np.flatnonzero(values["phase"] == 1)
    if not len(lifted) or not np.array_equal(lifted, np.arange(lifted[0], lifted[-1] + 1)):
        raise ValueError("Resampled lift is not one contiguous interval")
    lift_bounds = [int(lifted[0]), int(lifted[-1]) + 1]
    return {
        **{
            k: segment[k]
            for k in (
                "trial_id",
                "motion_id",
                "actor_id",
                "object_id",
                "split",
                "review_flags",
                "quality_status",
                "training_approval",
            )
        },
        "class_id": CLASS_MAP[segment.get("skill", "carry")],
        "class_name": segment.get("skill", "carry"),
        "states": str(destination.relative_to(output)),
        "states_sha256": file_hash(destination),
        "quality_flags": str(flags_path.relative_to(output)),
        "quality_flags_sha256": file_hash(flags_path),
        "quality_flagged_states": {
            check: int(np.count_nonzero(values["quality_flags"] & bit)) for check, bit in quality.BITS.items()
        },
        "quality_flagged_native_frames": {
            check: int(np.count_nonzero(native_flags & bit)) for check, bit in quality.BITS.items()
        },
        "frames": count,
        "valid_windows": count - schema.window_frames + 1,
        "window_policy": window_policy,
        "lift_index_bounds": lift_bounds,
        "lifted_windows": max(0, lift_bounds[1] - lift_bounds[0] - schema.window_frames + 1),
        "phase_frames": {str(k): int((values["phase"] == k).sum()) for k in (0, 1, 2)},
        "source_time_start": float(values["time"][0]),
        "source_time_end": float(values["time"][-1]),
        "source_positive_time_bounds": values["positive_time_bounds"],
        "source_interval_time_bounds": values["interval_time_bounds"],
        "source_trajectory": str(path),
        "source_trajectory_sha256": segment["trajectory_sha256"],
        "box_local_center_offset_m": center.tolist(),
        "box_full_extents_m": extents.tolist(),
    }


def prepare(
    manifest_path, output, workers=8, sample_rate=50.0, window_frames=16, window_policy="lifted", actor_splits=None
):
    """manifest_path: one manifest, or several (e.g. carry and push/slide) merged into one cache.

    actor_splits: optional JSON file with an "actor_splits" map (e.g. a skill manifest
    whose actors were re-split); segments of the listed actors take its split in every
    input manifest. The copied source manifests keep their own splits.
    """
    import mujoco

    if window_policy not in WINDOW_POLICIES:
        raise ValueError(f"window_policy must be one of {sorted(WINDOW_POLICIES)}")
    paths = [
        Path(p).resolve()
        for p in ([manifest_path] if isinstance(manifest_path, (str, Path)) else manifest_path)
    ]
    output = Path(output).resolve()
    manifests = [json.loads(p.read_text()) for p in paths]
    if any(m["status"] != "complete" for m in manifests):
        raise ValueError("Source export is incomplete")
    segments = [s for m in manifests for s in m["segments"]]
    overrides, moved = {}, Counter()
    if actor_splits is not None:
        overrides = json.loads(Path(actor_splits).read_text())["actor_splits"]
        moved = Counter(
            f"{s['actor_id']}: {s['split']} -> {overrides[s['actor_id']]}"
            for s in segments
            if overrides.get(s["actor_id"], s["split"]) != s["split"]
        )
        segments = [dict(s, split=overrides.get(s["actor_id"], s["split"])) for s in segments]
    unknown = {s.get("skill", "carry") for s in segments} - set(CLASS_MAP)
    if unknown:
        raise ValueError(f"Unknown skills {sorted(unknown)}; expected {sorted(CLASS_MAP)}")
    manifest = (
        manifests[0]
        if len(manifests) == 1
        else {"merged_manifests": [str(p) for p in paths], "manifests": manifests}
    )
    if not segments or len({s["trial_id"] for s in segments}) != len(segments):
        raise ValueError("Empty selection or duplicate IDs")
    actors = {}
    for s in segments:
        if s["split"] not in ("train", "validation", "test"):
            raise ValueError("Unrecognized source split")
        if actors.setdefault(s["actor_id"], s["split"]) != s["split"]:
            raise ValueError("Actor leakage across splits in the source manifests")
    with np.load(segments[0]["trajectory"], allow_pickle=False) as d:
        skeleton = robot_xml(str(d["scene_xml"]))
        names = tuple(d["joint_names"].tolist())
    model = mujoco.MjModel.from_xml_string(skeleton)
    axes = tuple(tuple(float(x) for x in model.jnt_axis[model.joint(n).id]) for n in names)
    schema = FeatureSchema(
        names, axes, hashlib.sha256(skeleton.encode()).hexdigest(), KEY_BODIES, sample_rate, window_frames
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "states").mkdir()
    (output / "quality").mkdir()
    (output / "robot_kinematics.xml").write_text(skeleton)
    save_json(output / "source_manifest.json", manifest)
    save_json(output / "status.json", {"status": "preparing", "pid": os.getpid(), "requested": len(segments)})
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_segment, s, output, schema.to_dict(), window_policy): s["trial_id"]
            for s in segments
        }
        for future in as_completed(futures):
            trial = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"trial_id": trial, "error": repr(exc)})
            done = len(results) + len(failures)
            if done % 25 == 0 or done == len(segments):
                print(f"Prepared {done}/{len(segments)}; failures={len(failures)}", flush=True)
                save_json(
                    output / "status.json",
                    {
                        "status": "preparing",
                        "pid": os.getpid(),
                        "processed": done,
                        "requested": len(segments),
                        "failures": failures,
                    },
                )
    results.sort(key=lambda s: s["trial_id"])
    index = {
        "format_version": 1,
        "status": "complete" if not failures else "incomplete",
        "feature_schema": schema.to_dict(),
        "feature_schema_sha256": schema.sha256,
        "source_manifest_sha256": file_hash(paths[0]) if len(paths) == 1 else [file_hash(p) for p in paths],
        "actor_split_overrides": None
        if actor_splits is None
        else {
            "path": str(Path(actor_splits).resolve()),
            "sha256": file_hash(actor_splits),
            "moved_segments": dict(sorted(moved.items())),
        },
        # Multi-skill caches always carry the full map so the class embedding has a fixed size.
        "class_map": dict(CLASS_MAP)
        if len(paths) > 1 or any(s.get("skill", "carry") != "carry" for s in segments)
        else {"carry": 0},
        "window_policy": window_policy,
        "sampling": "uniform_valid_window_starts",
        "quality_policy": "all_selected_flags_preserved",
        "quality_checks": {
            "names": list(quality.CHECKS),
            "bits": quality.BITS,
            "thresholds": quality.THRESHOLDS,
            "scope": "Per-state bit flags; the dataset drops windows touching a selected check",
            "code_sha256": file_hash(quality.__file__),
        },
        "training_approval": "pending_retarget_review",
        "segments": results,
        "failures": failures,
        "summary": {
            "skill_segments": dict(Counter(s["class_name"] for s in results)),
            "segments": len(results),
            "states": sum(s["frames"] for s in results),
            "windows": sum(s["valid_windows"] for s in results),
            "lifted_windows": sum(s["lifted_windows"] for s in results),
            "quality_flagged_states": {
                check: sum(s["quality_flagged_states"][check] for s in results) for check in quality.CHECKS
            },
            "split_segments": dict(Counter(s["split"] for s in results)),
            "quality_counts": dict(Counter(s["quality_status"] for s in results)),
        },
        "producer": {
            "prepare_sha256": file_hash(__file__),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "source_resampling": "linear_positions_joints_shortest_arc_slerp",
            "window_policy_phases": list(WINDOW_POLICIES[window_policy]),
        },
    }
    save_json(output / "index.json", index)
    save_json(output / "status.json", {"status": index["status"], **index["summary"], "failures": failures})
    print(json.dumps(index["summary"], indent=2), flush=True)
    if failures:
        raise RuntimeError("Incomplete cache; failures preserved in index.json")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, nargs="+", required=True, help="One or more smp_manifest.json files"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--actor-splits",
        type=Path,
        help="JSON with an actor_splits map (e.g. a skill manifest); listed actors take its split",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument(
        "--window-policy",
        choices=sorted(WINDOW_POLICIES),
        default="lifted",
        help="lifted: lifted interval only; segment: also pickup and set-down context",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("Use 1 to 32 workers")
    prepare(
        args.manifest,
        args.output,
        args.workers,
        args.sample_rate,
        args.window_frames,
        args.window_policy,
        args.actor_splits,
    )


if __name__ == "__main__":
    main()
