"""Geometric invariance, schema compatibility and batch-gather regressions."""

import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_data import create_index, schema, states

from mimickit_smp.dataset import CarryWindowDataset, MotionCollator, PackedWindowSampler
from mimickit_smp.features import canonical_box_axes, encode_motion
from mimickit_smp.normalization import FeatureNormalizer
from mimickit_smp.prepare import resample_positive
from mimickit_smp.tinymdm_data import HOIMotionPriorData


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_box_center_offset_rotates_during_resampling(dtype):
    # Nonzero offset exercises SciPy's vector broadcasting in both runtime dtypes.
    times = np.arange(91) / 90
    q = np.zeros((91, 43))
    q[:, 3] = 1
    q[:, 39] = np.cos(times / 2)
    q[:, 42] = np.sin(times / 2)
    offset = np.array([0.1, 0.2, 0.3], dtype=dtype)
    result = resample_positive(q, times, np.ones(91), offset)
    t = result["time"]
    expected = np.c_[
        offset[0] * np.cos(t) - offset[1] * np.sin(t),
        offset[0] * np.sin(t) + offset[1] * np.cos(t),
        np.full(len(t), offset[2]),
    ]
    np.testing.assert_allclose(result["box_pos"], expected, atol=1e-8)


def proper_axis_permutations():
    matrices = []
    for indices in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            p = np.eye(3)[:, indices] * np.array(signs)[None, :]
            if np.linalg.det(p) > 0:
                matrices.append(p)
    return np.stack(matrices)


@pytest.mark.parametrize("angles", [[0, 0, 0], [0, 0, 45], [45, 0, 0], [25, 61, 113]])
def test_mesh_axis_relabeling_preserves_canonical_rotation_and_extents(angles):
    # A single geometric cuboid represented by all 24 proper signed mesh frames.
    permutations = proper_axis_permutations()
    base = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    history = Rotation.from_euler("z", np.linspace(-0.08, 0, 16)[:, None]).as_matrix() @ base
    matrices = history[None] @ permutations[:, None]
    q = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    quats = torch.tensor(q.reshape(24, 16, 4), dtype=torch.float32)
    dims = np.array([0.3, 0.4, 0.5]) @ np.abs(permutations)
    dims = torch.tensor(np.broadcast_to(dims[:, None], (24, 16, 3)).copy(), dtype=quats.dtype)
    rotation, extents = canonical_box_axes(quats, dims)
    torch.testing.assert_close(rotation, rotation[:1].expand_as(rotation), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(extents, extents[:1].expand_as(extents))
    assert (rotation[:, -1, 5] > 0.57).all()  # Canonical Z points most upward.


def test_canonical_box_preserves_geometry_and_temporal_rotation():
    rotations = Rotation.from_euler("zyx", np.c_[np.linspace(44, 46, 16), np.zeros((16, 2))], degrees=True)
    quats = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float64)
    dimensions = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64).expand(16, 3)
    six, extents = canonical_box_axes(quats, dimensions)
    x, z = six[:, :3], six[:, 3:]
    canonical = torch.stack((x, torch.cross(z, x, dim=-1), z), dim=-1)
    original = torch.from_numpy(rotations.as_matrix())
    signs = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=3)), dtype=torch.float64)
    old_corners = (signs * dimensions[:, None] / 2) @ original.transpose(-1, -2)
    new_corners = (signs * extents[:, None] / 2) @ canonical.transpose(-1, -2)
    assert torch.cdist(old_corners, new_corners).amin(-1).max() < 1e-8
    # World increments are unchanged even across the 45-degree canonical-sector cut.
    torch.testing.assert_close(
        canonical[1:] @ canonical[:-1].transpose(-1, -2), original[1:] @ original[:-1].transpose(-1, -2)
    )
    torch.testing.assert_close(extents, extents[:1].expand_as(extents))


def test_face_relabeling_of_cube_has_identical_complete_features():
    x = states()
    x[:, 67:70] = 0.3
    variants = x.repeat(24, 1, 1)
    q = Rotation.from_matrix(proper_axis_permutations()).as_quat()[:, [3, 0, 1, 2]]
    variants[:, :, 57:61] = torch.tensor(q, dtype=x.dtype)[:, None]
    features = encode_motion(variants, schema())
    torch.testing.assert_close(features, features[:1].expand_as(features), atol=1e-6, rtol=1e-6)


def test_packed_gather_matches_scalar_loading_at_all_segment_boundaries(tmp_path):
    dataset = CarryWindowDataset(create_index(tmp_path))
    sampler = PackedWindowSampler(dataset)
    indices = torch.tensor([0, 2, 3, 7, 3, 0])
    scalar = MotionCollator(dataset.schema, include_states=True)([dataset[int(i)] for i in indices])
    packed = sampler.gather(indices)
    torch.testing.assert_close(packed, scalar["states"], rtol=0, atol=0)
    torch.testing.assert_close(encode_motion(packed, dataset.schema), scalar["motion"], rtol=0, atol=0)
    with pytest.raises((IndexError, RuntimeError)):
        sampler.gather(torch.tensor([-1]))
    with pytest.raises((IndexError, RuntimeError)):
        sampler.gather(torch.tensor([len(dataset)]))


def test_adapter_keeps_sampling_seed_with_vectorized_gather(tmp_path):
    import yaml

    index = create_index(tmp_path)
    env = tmp_path / "env.yaml"
    env.write_text(yaml.safe_dump({"num_disc_obs_steps": 16}))
    config = {"dataset_index": str(index), "env_config": str(env), "control_freq": 50, "seed": 12}
    adapter = HOIMotionPriorData(config, "cpu")
    rng = torch.Generator().manual_seed(12)
    for count in (5, 520):  # Includes the adapter's chunk boundary.
        ids = torch.randint(len(adapter.dataset), (count,), generator=rng)
        expected = MotionCollator(adapter.schema)([adapter.dataset[int(i)] for i in ids])["motion"].flatten(1)
        torch.testing.assert_close(adapter.fetch_obs_demo(count), expected, atol=0, rtol=0)


def test_v1_cache_loads_unchanged_but_old_normalization_and_checkpoint_are_rejected(tmp_path):
    import yaml

    index = create_index(tmp_path)
    data = json.loads(index.read_text())
    old = replace(schema(), version="g1_box_smp_v1")
    data["feature_schema"] = old.to_dict()
    data["feature_schema_sha256"] = old.sha256
    index.write_text(json.dumps(data))
    before = index.read_bytes()
    dataset = CarryWindowDataset(index)
    assert dataset.cache_schema.sha256 == old.sha256
    assert dataset.schema.version == "g1_box_smp_v2"
    assert dataset.schema.sha256 != old.sha256
    encode_motion(dataset[0]["states"], dataset.schema)
    assert index.read_bytes() == before
    with pytest.raises(ValueError, match="Normalization schema"):
        FeatureNormalizer({"feature_schema_sha256": old.sha256, "split": "train"}, dataset.schema)
    env = tmp_path / "env.yaml"
    env.write_text(yaml.safe_dump({"num_disc_obs_steps": 16}))
    config = {
        "dataset_index": str(index),
        "env_config": str(env),
        "control_freq": 50,
        "feature_schema_sha256": old.sha256,
    }
    with pytest.raises(ValueError, match="Checkpoint feature schema"):
        HOIMotionPriorData(config, "cpu")
