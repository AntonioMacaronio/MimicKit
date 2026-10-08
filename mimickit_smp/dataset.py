"""Memory-mapped windows that cannot cross segments, phases, or actor splits."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .features import FeatureSchema, encode_motion


class CarryWindowDataset(Dataset):
    def __init__(self, index, split="train", quality="all", max_open_files=32):
        self.index_path = Path(index).resolve()
        self.index = json.loads(self.index_path.read_text())
        if self.index.get("status") != "complete":
            raise ValueError("Refusing an incomplete prepared dataset")
        self.cache_schema = FeatureSchema.from_dict(self.index["feature_schema"])
        if self.index["feature_schema_sha256"] != self.cache_schema.sha256:
            raise ValueError("Schema hash mismatch")
        self.schema = self.cache_schema.current_features()
        if split not in ("train", "validation", "test") or quality not in ("all", "unflagged", "approved"):
            raise ValueError("Unknown split or quality policy")
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.split, self.quality = split, quality
        self.max_open_files = max_open_files
        self._maps = OrderedDict()
        actors = {}
        trial_ids = set()
        for s in self.index["segments"]:
            if s["trial_id"] in trial_ids:
                raise ValueError("Duplicate segment")
            trial_ids.add(s["trial_id"])
            if actors.setdefault(s["actor_id"], s["split"]) != s["split"]:
                raise ValueError("Actor leakage across splits")
        self.segments = [
            s
            for s in self.index["segments"]
            if s["split"] == split
            and (quality != "unflagged" or not s["review_flags"])
            and (quality != "approved" or s["training_approval"] == "approved")
        ]
        self.counts = np.asarray(
            [s["frames"] - self.schema.window_frames + 1 for s in self.segments], np.int64
        )
        if not len(self.counts) or np.any(self.counts <= 0):
            raise ValueError(f"No usable windows or short segments for split={split}, quality={quality}")
        self.ends = self.counts.cumsum()
        self.index_sha256 = hashlib.sha256(self.index_path.read_bytes()).hexdigest()

    def __len__(self):
        return int(self.ends[-1])

    def _states(self, segment):
        relative = segment["states"]
        path = (self.index_path.parent / relative).resolve()
        if not path.is_relative_to(self.index_path.parent):
            raise ValueError("Cache path escapes dataset root")
        if relative in self._maps:
            self._maps.move_to_end(relative)
            return self._maps[relative]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (segment["frames"], 70) or array.dtype != np.float32:
            raise ValueError("Cached state layout mismatch")
        self._maps[relative] = array
        if len(self._maps) > self.max_open_files:
            self._maps.popitem(last=False)
        return array

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        clip = int(np.searchsorted(self.ends, index, side="right"))
        start = index - (int(self.ends[clip - 1]) if clip else 0)
        segment = self.segments[clip]
        state = np.array(self._states(segment)[start : start + self.schema.window_frames], copy=True)
        return {
            "states": torch.from_numpy(state),
            "class_id": 0,
            "segment_index": clip,
            "trial_id": segment["trial_id"],
            "actor_id": segment["actor_id"],
            "quality_status": segment["quality_status"],
            "training_approval": segment["training_approval"],
            "start_time": segment["source_time_start"] + start / self.schema.sample_rate_hz,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = OrderedDict()
        return state


class MotionCollator:
    def __init__(self, schema, include_states=False):
        self.schema, self.include_states = schema, include_states

    def __call__(self, samples):
        batch = default_collate(samples)
        states = batch["states"] if self.include_states else batch.pop("states")
        batch["motion"] = encode_motion(states, self.schema)
        return batch


class PackedWindowSampler:
    """Load a selected split once, then gather whole batches without Python sample indexing.

    State storage is about 169 MiB for this training split; window-start offsets
    add about 5 MiB. The offset table includes only valid starts in each segment.
    No duplicated overlapping windows are stored. Sampling indices retain the
    exact ordering/distribution of CarryWindowDataset.
    """

    def __init__(self, dataset, device="cpu"):
        self.schema = dataset.schema
        self.num_windows = len(dataset)
        packed = torch.empty((sum(s["frames"] for s in dataset.segments), 70), dtype=torch.float32)
        starts = []
        offset = 0
        for segment, count in zip(dataset.segments, dataset.counts):
            n = segment["frames"]
            packed[offset : offset + n].copy_(torch.from_numpy(np.array(dataset._states(segment), copy=True)))
            starts.append(torch.arange(int(count), dtype=torch.int64) + offset)
            offset += n
        self.states = packed.to(device)
        self.starts = torch.cat(starts).to(device)
        self.time_offsets = torch.arange(self.schema.window_frames, device=device)

    def gather(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.starts.device)
        if indices.ndim != 1:
            raise ValueError("Expected a vector of window indices")
        # index_select rejects negative/out-of-range indices instead of wrapping.
        starts = self.starts.index_select(0, indices)
        return self.states[starts[:, None] + self.time_offsets]


def make_dataloader(dataset, batch_size=256, shuffle=True, num_workers=0, seed=0, include_states=False):
    """Uniform sampling over valid window starts; long segments contribute more."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=MotionCollator(dataset.schema, include_states),
        persistent_workers=num_workers > 0,
    )
