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

from .features import KEY_BODIES, FeatureSchema


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
    """Interpolation and backward velocities use only the contiguous lifted phase.

    Drop the first resampled state: its velocity has no preceding positive sample.
    No padding, wrapping, extrapolation, or future differences are introduced.
    """
    qpos, times = np.asarray(qpos), np.asarray(times, np.float64)
    if qpos.shape != (len(times), 43) or not np.isfinite(qpos).all():
        raise ValueError("Expected finite root+29-joint+box states")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or len(phase) != len(times):
        raise ValueError("Invalid source clock or phases")
    ids = np.flatnonzero(np.asarray(phase) == 1)
    if len(ids) < 3 or not np.array_equal(ids, np.arange(ids[0], ids[-1] + 1)):
        raise ValueError("Expected one contiguous lifted interval")
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
        "positive_time_bounds": [float(t[0]), float(t[-1])],
    }


def process_segment(segment, output, schema_dict):
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
        values = resample_positive(
            d["qpos"], d["source_timestamps"], d["phase"], center, schema.sample_rate_hz
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
        "class_id": 0,
        "class_name": "carry",
        "states": str(destination.relative_to(output)),
        "states_sha256": file_hash(destination),
        "frames": count,
        "valid_windows": count - schema.window_frames + 1,
        "source_time_start": float(values["time"][0]),
        "source_time_end": float(values["time"][-1]),
        "source_positive_time_bounds": values["positive_time_bounds"],
        "source_trajectory": str(path),
        "source_trajectory_sha256": segment["trajectory_sha256"],
        "box_local_center_offset_m": center.tolist(),
        "box_full_extents_m": extents.tolist(),
    }


def prepare(manifest_path, output, workers=8, sample_rate=50.0, window_frames=16):
    import mujoco

    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("Source export is incomplete")
    segments = manifest["segments"]
    if not segments or len({s["trial_id"] for s in segments}) != len(segments):
        raise ValueError("Empty selection or duplicate IDs")
    actors = {}
    for s in segments:
        if s["split"] not in ("train", "validation", "test"):
            raise ValueError("Unrecognized source split")
        if actors.setdefault(s["actor_id"], s["split"]) != s["split"]:
            raise ValueError("Actor leakage in source manifest")
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
    (output / "robot_kinematics.xml").write_text(skeleton)
    save_json(output / "source_manifest.json", manifest)
    save_json(output / "status.json", {"status": "preparing", "pid": os.getpid(), "requested": len(segments)})
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_segment, s, output, schema.to_dict()): s["trial_id"] for s in segments}
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
        "source_manifest_sha256": file_hash(manifest_path),
        "class_map": {"carry": 0},
        "sampling": "uniform_valid_window_starts",
        "quality_policy": "all_selected_flags_preserved",
        "training_approval": "pending_retarget_review",
        "segments": results,
        "failures": failures,
        "summary": {
            "segments": len(results),
            "states": sum(s["frames"] for s in results),
            "windows": sum(s["valid_windows"] for s in results),
            "split_segments": dict(Counter(s["split"] for s in results)),
            "quality_counts": dict(Counter(s["quality_status"] for s in results)),
        },
        "producer": {
            "prepare_sha256": file_hash(__file__),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "source_resampling": "linear_positions_joints_shortest_arc_slerp",
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--window-frames", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("Use 1 to 32 workers")
    prepare(args.manifest, args.output, args.workers, args.sample_rate, args.window_frames)


if __name__ == "__main__":
    main()
