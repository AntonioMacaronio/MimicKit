"""Memory-mapped windows that cannot cross segments or actor splits.

A "lifted" cache holds only each segment's lifted interval. A "segment" cache
also holds the pickup and set-down context; its windows may span those phases
and stay in the same carry class. stages="lifted" keeps only fully lifted windows.
quality_checks drops every window touching a state flagged by a selected check
(mimickit_smp.quality); stage_weights draws pickup / lifted / set-down windows in
fixed proportions instead of in proportion to their counts, and class_weights does
the same for skill classes (carry / push / slide). For push and slide, "lifted"
is the interval in which the box is moved along the floor.
"""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .features import FeatureSchema, encode_window
from .quality import check_mask

STAGES = ("all", "lifted")
WINDOW_STAGES = ("pickup", "lifted", "setdown")


class CarryWindowDataset(Dataset):
    def __init__(
        self,
        index,
        split="train",
        quality="all",
        stages="all",
        quality_checks=(),
        stage_weights=None,
        class_weights=None,
        max_open_files=32,
    ):
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
        if stages not in STAGES:
            raise ValueError(f"stages must be one of {STAGES}")
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.split, self.quality, self.stages = split, quality, stages
        self.quality_checks = tuple(sorted(set(quality_checks)))
        mask = check_mask(self.quality_checks)
        if mask and "quality_checks" not in self.index:
            raise ValueError("This cache has no quality flags; prepare it again to use quality_checks")
        self.window_policy = self.index.get("window_policy", "lifted")
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
        # A lifted cache is entirely lifted; a segment cache records where the lift is.
        lifts = [s.get("lift_index_bounds", [0, s["frames"]]) for s in self.segments]
        self.lifts = np.asarray(lifts, np.int64).reshape(-1, 2)
        width = self.schema.window_frames
        segment_ids, starts = [], []
        for k, s in enumerate(self.segments):
            if s["frames"] < width:
                raise ValueError(f"Segment shorter than one window: {s['trial_id']}")
            first, last = (
                (self.lifts[k, 0], self.lifts[k, 1] - width)
                if stages == "lifted"
                else (0, s["frames"] - width)
            )
            valid = np.arange(first, last + 1, dtype=np.int64)
            if mask and len(valid):
                bad = (self._flags(s) & mask) != 0
                touched = np.concatenate([[0], np.cumsum(bad)])
                valid = valid[touched[valid + width] == touched[valid]]
            segment_ids.append(np.full(len(valid), k, np.int64))
            starts.append(valid)
        self.window_segments = np.concatenate(segment_ids) if segment_ids else np.zeros(0, np.int64)
        self.window_starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)
        if not len(self.window_starts):
            raise ValueError(
                f"No usable windows for split={split}, quality={quality}, checks={self.quality_checks}"
            )
        # 0 pickup, 1 lifted, 2 set-down; see window_stage.
        lift_start, lift_stop = self.lifts[self.window_segments, 0], self.lifts[self.window_segments, 1]
        self.window_stage_ids = np.where(
            self.window_starts < lift_start, 0, np.where(self.window_starts + width > lift_stop, 2, 1)
        ).astype(np.int64)
        self.stage_counts = {
            name: int((self.window_stage_ids == i).sum()) for i, name in enumerate(WINDOW_STAGES)
        }
        self.stage_weights = self._check_weights(stage_weights)
        # Skill classes (carry/push/slide); single-class caches keep class 0 = carry.
        self.class_map = dict(self.index.get("class_map", {"carry": 0}))
        self.class_names = {i: name for name, i in self.class_map.items()}
        segment_classes = np.asarray([s.get("class_id", 0) for s in self.segments], np.int64)
        self.window_class_ids = segment_classes[self.window_segments]
        self.class_counts = {
            name: int((self.window_class_ids == i).sum()) for name, i in self.class_map.items()
        }
        self.class_weights = self._check_class_weights(class_weights)
        self._groups = {
            int(c): [
                torch.from_numpy(np.flatnonzero((self.window_class_ids == c) & (self.window_stage_ids == k)))
                for k in range(3)
            ]
            for c in np.unique(self.window_class_ids)
        }
        self.index_sha256 = hashlib.sha256(self.index_path.read_bytes()).hexdigest()

    def _flags(self, segment):
        path = (self.index_path.parent / segment["quality_flags"]).resolve()
        if not path.is_relative_to(self.index_path.parent):
            raise ValueError("Cache path escapes dataset root")
        flags = np.load(path, allow_pickle=False)
        if flags.shape != (segment["frames"],) or flags.dtype != np.uint8:
            raise ValueError("Cached quality-flag layout mismatch")
        return flags

    def _check_weights(self, weights):
        if weights is None:
            return None
        weights = {name: float(w) for name, w in dict(weights).items()}
        if (
            set(weights) - set(WINDOW_STAGES)
            or any(w < 0 for w in weights.values())
            or not sum(weights.values())
        ):
            raise ValueError(f"stage_weights needs non-negative weights over {WINDOW_STAGES}")
        empty = [name for name, w in weights.items() if w > 0 and not self.stage_counts[name]]
        if empty:
            raise ValueError(f"stage_weights select stages without windows: {empty}")
        total = sum(weights.values())
        return {name: weights.get(name, 0.0) / total for name in WINDOW_STAGES}

    def _check_class_weights(self, weights):
        if weights is None:
            return None
        weights = {name: float(w) for name, w in dict(weights).items()}
        if (
            set(weights) - set(self.class_map)
            or any(w < 0 for w in weights.values())
            or not sum(weights.values())
        ):
            raise ValueError(f"class_weights needs non-negative weights over {sorted(self.class_map)}")
        empty = [name for name, w in weights.items() if w > 0 and not self.class_counts[name]]
        if empty:
            raise ValueError(f"class_weights select classes without windows: {empty}")
        total = sum(weights.values())
        return {name: weights.get(name, 0.0) / total for name in self.class_map}

    def sample_indices(self, num_samples, generator=None):
        """Random window indices.

        Uniform over windows without weights. Otherwise draw a class (class_weights,
        or in proportion to its windows), then a stage within it (stage_weights, or
        in proportion; stages a class lacks are skipped), then a uniform window.
        """
        if self.stage_weights is None and self.class_weights is None:
            return torch.randint(len(self), (num_samples,), generator=generator)
        classes = sorted(self._groups)

        def class_weight(c):
            if self.class_weights is None:
                return float(sum(len(g) for g in self._groups[c]))
            return self.class_weights[self.class_names[c]]

        weights = torch.tensor([class_weight(c) for c in classes], dtype=torch.float64)
        picks = torch.multinomial(weights, num_samples, replacement=True, generator=generator)
        indices = torch.empty(num_samples, dtype=torch.int64)
        for slot, c in enumerate(classes):
            chosen = torch.nonzero(picks == slot, as_tuple=True)[0]
            if not len(chosen):
                continue
            groups = self._groups[c]
            stage = torch.tensor(
                [
                    0.0 if not len(g) else self.stage_weights[name] if self.stage_weights else float(len(g))
                    for name, g in zip(WINDOW_STAGES, groups)
                ],
                dtype=torch.float64,
            )
            stages = torch.multinomial(stage, len(chosen), replacement=True, generator=generator)
            for k, members in enumerate(groups):
                rows = chosen[stages == k]
                if len(rows):
                    indices[rows] = members[torch.randint(len(members), (len(rows),), generator=generator)]
        return indices

    def __len__(self):
        return len(self.window_starts)

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
        clip, start = int(self.window_segments[index]), int(self.window_starts[index])
        segment = self.segments[clip]
        state = np.array(self._states(segment)[start : start + self.schema.window_frames], copy=True)
        return {
            "states": torch.from_numpy(state),
            "class_id": int(segment.get("class_id", 0)),
            "stage": self.window_stage(clip, start),
            "segment_index": clip,
            "trial_id": segment["trial_id"],
            "actor_id": segment["actor_id"],
            "quality_status": segment["quality_status"],
            "training_approval": segment["training_approval"],
            "start_time": segment["source_time_start"] + start / self.schema.sample_rate_hz,
        }

    def window_stage(self, clip, start):
        """pickup / set-down if the window reaches either context phase, else lifted."""
        lift_start, lift_stop = (int(x) for x in self.lifts[clip])
        if start < lift_start:
            return "pickup"
        if start + self.schema.window_frames > lift_stop:
            return "setdown"
        return "lifted"

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
        batch["motion"], condition = encode_window(states, self.schema)
        if condition is not None:
            batch["box_size"] = condition
        return batch


class PackedWindowSampler:
    """Load a selected split once, then gather whole batches without Python sample indexing.

    States are stored once (70 float32 per 50 Hz state); the offset table holds one
    int64 per valid window start. No duplicated overlapping windows are stored.
    Window indices mean the same windows as in CarryWindowDataset.
    """

    def __init__(self, dataset, device="cpu"):
        self.schema = dataset.schema
        self.num_windows = len(dataset)
        frames = np.asarray([s["frames"] for s in dataset.segments], np.int64)
        offsets = np.concatenate([[0], np.cumsum(frames)[:-1]])
        packed = torch.empty((int(frames.sum()), 70), dtype=torch.float32)
        for segment, offset, n in zip(dataset.segments, offsets, frames):
            packed[offset : offset + n].copy_(torch.from_numpy(np.array(dataset._states(segment), copy=True)))
        self.states = packed.to(device)
        self.starts = torch.from_numpy(offsets[dataset.window_segments] + dataset.window_starts).to(device)
        self.time_offsets = torch.arange(self.schema.window_frames, device=device)

    def gather(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.starts.device)
        if indices.ndim != 1:
            raise ValueError("Expected a vector of window indices")
        # index_select rejects negative/out-of-range indices instead of wrapping.
        starts = self.starts.index_select(0, indices)
        return self.states[starts[:, None] + self.time_offsets]


def make_dataloader(dataset, batch_size=256, shuffle=True, num_workers=0, seed=0, include_states=False):
    """Uniform sampling over valid window starts; long segments contribute more.

    Ignores stage_weights; training draws through CarryWindowDataset.sample_indices.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=MotionCollator(dataset.schema, include_states),
        persistent_workers=num_workers > 0,
    )
