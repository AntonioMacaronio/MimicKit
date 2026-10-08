"""Train-only feature statistics, reusable unchanged at reward evaluation."""

import json
from pathlib import Path

import torch

from .dataset import PackedWindowSampler
from .features import encode_motion


def fit_normalizer(dataset, num_windows=20_000, batch_size=256, seed=0, min_std=0.03):
    if dataset.split != "train":
        raise ValueError("Normalization must be fitted on the training split only")
    if num_windows < 1 or batch_size < 1 or min_std <= 0:
        raise ValueError("Invalid normalization settings")
    generator = torch.Generator().manual_seed(seed)
    sampler = PackedWindowSampler(dataset)
    total = torch.zeros(dataset.schema.feature_dim, dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for start in range(0, num_windows, batch_size):
        indices = torch.randint(len(dataset), (min(batch_size, num_windows - start),), generator=generator)
        motion = encode_motion(sampler.gather(indices), dataset.schema).double().flatten(0, 1)
        total += motion.sum(0)
        squared += motion.square().sum(0)
        count += len(motion)
    mean = total / count
    variance = (squared / count - mean.square()).clamp_min(0)
    std = variance.sqrt().clamp_min(min_std)
    return {
        "feature_schema_sha256": dataset.schema.sha256,
        "dataset_index_sha256": dataset.index_sha256,
        "split": "train",
        "quality_policy": dataset.quality,
        "num_windows": num_windows,
        "num_frames": count,
        "seed": seed,
        "minimum_std": min_std,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


class FeatureNormalizer(torch.nn.Module):
    def __init__(self, statistics, schema):
        super().__init__()
        if isinstance(statistics, (str, Path)):
            statistics = json.loads(Path(statistics).read_text())
        if statistics["feature_schema_sha256"] != schema.sha256 or statistics["split"] != "train":
            raise ValueError("Normalization schema or training split mismatch")
        mean, std = (torch.tensor(statistics[k], dtype=torch.float32) for k in ("mean", "std"))
        if mean.shape != (schema.feature_dim,) or std.shape != mean.shape:
            raise ValueError("Normalization shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid normalization statistics")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, motion):
        return (motion - self.mean) / self.std

    def inverse(self, motion):
        return motion * self.std + self.mean
