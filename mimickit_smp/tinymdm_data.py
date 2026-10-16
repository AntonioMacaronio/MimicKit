"""Thin adapter to MimicKit's existing TinyMDM trainer; no simulator imports."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import WINDOW_STAGES, CarryWindowDataset, PackedWindowSampler
from .features import encode_window


@dataclass
class ObservationSpec:
    shape: tuple[int, ...]
    dtype: object = np.float32


class HOIMotionPriorData:
    def __init__(self, config, device, split="train"):
        path = os.path.expandvars(config["dataset_index"])
        if "$" in path:
            raise ValueError("Set the environment variable used in dataset_index")
        self.dataset = CarryWindowDataset(
            path,
            split=split,
            quality=config.get("quality_policy", "all"),
            stages=config.get("window_stages", "all"),
            quality_checks=config.get("quality_checks") or (),
            stage_weights=config.get("stage_weights"),
            class_weights=config.get("class_weights"),
        )
        self.schema = self.dataset.schema
        # Skill classes (carry/push/slide); class-conditioned priors use their IDs.
        self.class_map = self.dataset.class_map
        self.num_class = len(self.class_map)
        self.condition_dim = self.schema.condition_dim
        if config.get("feature_schema_sha256", self.schema.sha256) != self.schema.sha256:
            raise ValueError("Checkpoint feature schema differs from dataset")
        self.control_freq = self.schema.sample_rate_hz
        if config["control_freq"] != self.control_freq:
            raise ValueError("Prior control frequency differs from prepared data")
        import yaml

        with open(config["env_config"]) as stream:
            env = yaml.safe_load(stream)
        if env["num_disc_obs_steps"] != self.schema.window_frames:
            raise ValueError("Prior window length differs from prepared data")
        if env.get("feature_schema", self.schema.version) != self.schema.version:
            raise ValueError("Prior feature schema differs from encoder")
        self._device = device
        self._generator = torch.Generator().manual_seed(config.get("seed", 0))
        # Conditions for generated samples come from their own stream, so sampling
        # them never shifts the training batches.
        self._condition_generator = torch.Generator().manual_seed(config.get("seed", 0) + 1)
        self._sampler = PackedWindowSampler(self.dataset, device=config.get("data_device", "cpu"))

    def _encode(self, indices):
        motions, conditions = [], []
        # Bound temporary state/features allocation for large normalization draws.
        for chunk in indices.split(512):
            motion, condition = encode_window(self._sampler.gather(chunk).to(self._device), self.schema)
            motions.append(motion.flatten(1))
            conditions.append(condition)
        return torch.cat(motions), (torch.cat(conditions) if self.condition_dim else None)

    def _labels(self, indices):
        return torch.from_numpy(self.dataset.window_class_ids)[indices].to(self._device)

    def fetch_obs_demo(self, num_samples, return_class=False, return_condition=False):
        """Flattened windows; optionally class IDs and the [N, 3] box-size condition, in that order."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if return_condition and not self.condition_dim:
            raise ValueError(f"Feature schema {self.schema.version} has no condition")
        indices = self.dataset.sample_indices(num_samples, self._generator)
        motion, condition = self._encode(indices)
        outputs = [motion]
        if return_class:
            outputs.append(self._labels(indices))
        if return_condition:
            outputs.append(condition)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def fixed_windows(self, num_samples, seed, stage=None, class_name=None):
        """A reproducible evaluation draw (motion, box size, class IDs): uniform over windows,
        or over the windows of one stage and/or one skill class."""
        generator = torch.Generator().manual_seed(seed)
        keep = np.ones(len(self.dataset), bool)
        if stage is not None:
            keep &= self.dataset.window_stage_ids == WINDOW_STAGES.index(stage)
        if class_name is not None:
            keep &= self.dataset.window_class_ids == self.class_map[class_name]
        members = torch.from_numpy(np.flatnonzero(keep))
        if not len(members):
            raise ValueError(f"No {class_name or ''} {stage or ''} windows in this split")
        indices = members[torch.randint(len(members), (num_samples,), generator=generator)]
        motion, condition = self._encode(indices)
        return motion, condition, self._labels(indices)

    def sample_conditions(self, num_samples, return_class=False):
        """Box sizes (and class IDs) of random training windows, for conditional generation."""
        if not self.condition_dim:
            raise ValueError(f"Feature schema {self.schema.version} has no condition")
        indices = self.dataset.sample_indices(num_samples, self._condition_generator)
        condition = self._encode(indices)[1]
        return (condition, self._labels(indices)) if return_class else condition

    def get_obs_space(self):
        return ObservationSpec((self.schema.window_frames * self.schema.feature_dim,))

    def save_metadata(self, output):
        path = Path(output)
        (path / "feature_schema.json").write_text(json.dumps(self.schema.to_dict(), indent=2) + "\n")
        (path / "dataset_provenance.json").write_text(
            json.dumps(
                {
                    "dataset_index_sha256": self.dataset.index_sha256,
                    "feature_schema_sha256": self.schema.sha256,
                    "cache_schema_sha256": self.dataset.cache_schema.sha256,
                    "training_split": self.dataset.split,
                    "quality_policy": self.dataset.quality,
                    "window_policy": self.dataset.window_policy,
                    "window_stages": self.dataset.stages,
                    "quality_checks": list(self.dataset.quality_checks),
                    "quality_thresholds": self.dataset.index.get("quality_checks", {}).get("thresholds"),
                    "stage_weights": self.dataset.stage_weights,
                    "stage_window_counts": self.dataset.stage_counts,
                    "class_weights": self.dataset.class_weights,
                    "class_window_counts": self.dataset.class_counts,
                    "condition": "box_size" if self.condition_dim else None,
                    "class_map": self.class_map,
                    "training_approval": "pending_retarget_review",
                },
                indent=2,
            )
            + "\n"
        )

    def save_generated_samples(self, samples, output, cond=None, class_labels=None):
        # Features are not qpos. Do not feed joint/object channels into the
        # upstream character-only inverse or claim a physical grasp was generated.
        extra = {} if cond is None else {"box_size": cond.detach().cpu().numpy()}
        if class_labels is not None:
            extra["class_ids"] = class_labels.detach().cpu().numpy()
        np.savez_compressed(
            Path(output) / "hoi_feature_samples.npz",
            motion=samples.detach().cpu().numpy(),
            feature_schema_sha256=self.schema.sha256,
            sample_rate_hz=self.schema.sample_rate_hz,
            **extra,
        )
