"""Thin adapter to MimicKit's existing TinyMDM trainer; no simulator imports."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import CarryWindowDataset, PackedWindowSampler
from .features import encode_motion


@dataclass
class ObservationSpec:
    shape: tuple[int, ...]
    dtype: object = np.float32


class HOIMotionPriorData:
    def __init__(self, config, device):
        path = os.path.expandvars(config["dataset_index"])
        if "$" in path:
            raise ValueError("Set the environment variable used in dataset_index")
        self.dataset = CarryWindowDataset(path, split="train", quality=config.get("quality_policy", "all"))
        self.schema = self.dataset.schema
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
        self._sampler = PackedWindowSampler(self.dataset, device=config.get("data_device", "cpu"))

    def fetch_obs_demo(self, num_samples, return_class=False):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        indices = torch.randint(len(self.dataset), (num_samples,), generator=self._generator)
        # Bound temporary state/features allocation for large normalization draws.
        batches = []
        for chunk in indices.split(512):
            states = self._sampler.gather(chunk).to(self._device)
            batches.append(encode_motion(states, self.schema).flatten(1))
        motion = torch.cat(batches)
        if return_class:
            return motion, torch.zeros(num_samples, dtype=torch.long, device=self._device)
        return motion

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
                    "training_split": "train",
                    "quality_policy": self.dataset.quality,
                    "class_map": {"carry": 0},
                    "training_approval": "pending_retarget_review",
                },
                indent=2,
            )
            + "\n"
        )

    def save_generated_samples(self, samples, output):
        # Features are not qpos. Do not feed joint/object channels into the
        # upstream character-only inverse or claim a physical grasp was generated.
        np.savez_compressed(
            Path(output) / "hoi_feature_samples.npz",
            motion=samples.detach().cpu().numpy(),
            feature_schema_sha256=self.schema.sha256,
            sample_rate_hz=self.schema.sample_rate_hz,
        )
