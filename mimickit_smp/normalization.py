"""Train-only feature statistics, reusable unchanged at reward evaluation."""

import json
from pathlib import Path

import torch

from .dataset import PackedWindowSampler
from .features import encode_window


def fit_normalizer(dataset, num_windows=20_000, batch_size=256, seed=0, min_std=0.03, min_condition_std=1e-3):
    if dataset.split != "train":
        raise ValueError("Normalization must be fitted on the training split only")
    if num_windows < 1 or batch_size < 1 or min_std <= 0 or min_condition_std <= 0:
        raise ValueError("Invalid normalization settings")
    generator = torch.Generator().manual_seed(seed)
    sampler = PackedWindowSampler(dataset)
    condition_dim = dataset.schema.condition_dim
    total = torch.zeros(dataset.schema.feature_dim, dtype=torch.float64)
    squared = torch.zeros_like(total)
    condition_total = torch.zeros(condition_dim, dtype=torch.float64)
    condition_squared = torch.zeros_like(condition_total)
    count = 0
    for start in range(0, num_windows, batch_size):
        indices = torch.randint(len(dataset), (min(batch_size, num_windows - start),), generator=generator)
        motion, condition = encode_window(sampler.gather(indices), dataset.schema)
        motion = motion.double().flatten(0, 1)
        total += motion.sum(0)
        squared += motion.square().sum(0)
        count += len(motion)
        if condition_dim:
            condition_total += condition.double().sum(0)
            condition_squared += condition.double().square().sum(0)
    mean = total / count
    variance = (squared / count - mean.square()).clamp_min(0)
    std = variance.sqrt().clamp_min(min_std)
    stats = {
        "feature_schema_sha256": dataset.schema.sha256,
        "dataset_index_sha256": dataset.index_sha256,
        "split": "train",
        "quality_policy": dataset.quality,
        "window_stages": dataset.stages,
        "num_windows": num_windows,
        "num_frames": count,
        "seed": seed,
        "minimum_std": min_std,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    if condition_dim:
        # One condition per window; box sizes are not floored like motion channels.
        condition_mean = condition_total / num_windows
        condition_std = (condition_squared / num_windows - condition_mean.square()).clamp_min(0).sqrt()
        stats.update(
            minimum_condition_std=min_condition_std,
            condition_mean=condition_mean.tolist(),
            condition_std=condition_std.clamp_min(min_condition_std).tolist(),
        )
    return stats


class FeatureNormalizer(torch.nn.Module):
    def __init__(self, statistics, schema):
        super().__init__()
        if isinstance(statistics, (str, Path)):
            statistics = json.loads(Path(statistics).read_text())
        if statistics["feature_schema_sha256"] != schema.sha256 or statistics["split"] != "train":
            raise ValueError("Normalization schema or training split mismatch")
        self.register_buffer("mean", self._vector(statistics["mean"], schema.feature_dim))
        self.register_buffer("std", self._vector(statistics["std"], schema.feature_dim, positive=True))
        if schema.condition_dim:
            dim = schema.condition_dim
            self.register_buffer("condition_mean", self._vector(statistics["condition_mean"], dim))
            self.register_buffer(
                "condition_std", self._vector(statistics["condition_std"], dim, positive=True)
            )

    @staticmethod
    def _vector(values, dim, positive=False):
        vector = torch.tensor(values, dtype=torch.float32)
        if vector.shape != (dim,):
            raise ValueError("Normalization shape mismatch")
        if not torch.isfinite(vector).all() or (positive and (vector <= 0).any()):
            raise ValueError("Invalid normalization statistics")
        return vector

    def forward(self, motion):
        return (motion - self.mean) / self.std

    def inverse(self, motion):
        return motion * self.std + self.mean

    def normalize_condition(self, condition):
        return (condition - self.condition_mean) / self.condition_std
