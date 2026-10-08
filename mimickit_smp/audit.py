"""Read back a prepared cache, check windows, and fit training-only statistics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .dataset import CarryWindowDataset, make_dataloader
from .features import encode_motion
from .normalization import fit_normalizer
from .prepare import file_hash, save_json


def audit(index_path, output, normalizer_windows=20_000):
    index_path, output = Path(index_path).resolve(), Path(output).resolve()
    index = json.loads(index_path.read_text())
    datasets = {
        split: CarryWindowDataset(index_path, split=split) for split in ("train", "validation", "test")
    }
    schema = datasets["train"].schema
    output.mkdir(parents=True, exist_ok=False)
    checked = []
    for segment in index["segments"]:
        path = index_path.parent / segment["states"]
        if file_hash(path) != segment["states_sha256"]:
            raise ValueError("Cache hash mismatch")
        states = np.load(path, mmap_mode="r", allow_pickle=False)
        if states.shape != (segment["frames"], 70) or not np.isfinite(states).all():
            raise ValueError("Invalid cached states")
        for a, b in ((3, 7), (57, 61)):
            if not np.allclose(np.linalg.norm(states[:, a:b], axis=-1), 1, atol=1e-5, rtol=0):
                raise ValueError("Invalid cached quaternion")
        if np.any(states[:, 67:] <= 0) or not np.allclose(states[:, 67:], segment["box_full_extents_m"]):
            raise ValueError("Box dimensions changed during the segment")
        first, last = segment["source_time_start"], segment["source_time_end"]
        a, b = segment["source_positive_time_bounds"]
        if (
            first < a
            or last > b
            or not np.isclose(last - first, (len(states) - 1) / schema.sample_rate_hz, atol=1e-8)
        ):
            raise ValueError("Prepared frames escape lifted interval or sample clock")
        count = len(states) - schema.window_frames + 1
        if count != segment["valid_windows"]:
            raise ValueError("Incorrect window index")
        starts = sorted({0, count // 2, count - 1})
        batch = torch.from_numpy(np.stack([states[s : s + schema.window_frames] for s in starts]))
        motion = encode_motion(batch, schema)
        if (
            motion.shape != (len(starts), schema.window_frames, schema.feature_dim)
            or not torch.isfinite(motion).all()
        ):
            raise ValueError("Invalid encoded sample")
        checked.append(
            {
                "trial_id": segment["trial_id"],
                "frames": len(states),
                "windows": count,
                "tested_window_starts": starts,
                "states_sha256": segment["states_sha256"],
            }
        )
    stats = fit_normalizer(datasets["train"], num_windows=normalizer_windows)
    save_json(output / "normalization.json", stats)
    # Exercise worker-process collation and random train sampling, not just direct indexing.
    loader = make_dataloader(datasets["train"], batch_size=32, num_workers=2)
    batch = next(iter(loader))
    if batch["motion"].shape != (32, schema.window_frames, schema.feature_dim):
        raise ValueError("DataLoader shape mismatch")
    del loader
    report = {
        "status": "passed",
        "index_sha256": file_hash(index_path),
        "feature_schema_sha256": schema.sha256,
        "segments": len(checked),
        "sample_rate_hz": schema.sample_rate_hz,
        "window_frames": schema.window_frames,
        "window_span_seconds": schema.span_seconds,
        "feature_dim": schema.feature_dim,
        "split_windows": {k: len(v) for k, v in datasets.items()},
        "split_segments": {k: len(v.segments) for k, v in datasets.items()},
        "split_actors": {k: len({s["actor_id"] for s in v.segments}) for k, v in datasets.items()},
        "quality_counts": dict(Counter(s["quality_status"] for s in index["segments"])),
        "normalizer_training_windows": normalizer_windows,
        "checks": checked,
        "training_approval": "pending_retarget_review",
        "physical_execution_validated": False,
    }
    save_json(output / "validation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalizer-windows", type=int, default=20_000)
    args = parser.parse_args()
    audit(args.index, args.output, args.normalizer_windows)


if __name__ == "__main__":
    main()
