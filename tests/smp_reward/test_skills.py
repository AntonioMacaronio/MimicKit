"""Skill classes (carry / push / slide): labels in the cache, class-weighted sampling, and a
class- and box-size-conditioned prior."""

import json

import numpy as np
import pytest
import torch
from test_condition_and_stages import tinymdm  # noqa: F401 - pytest fixture
from test_data import schema, states

from mimickit_smp.dataset import WINDOW_STAGES, CarryWindowDataset
from mimickit_smp.prepare import CLASS_MAP


def skill_index(folder, skills=("carry", "carry", "push", "slide", "slide", "slide")):
    """Train segments of 40 states with lifted / moving interval [8, 34) and one class each."""
    s = schema()
    entries = []
    for i, skill in enumerate(skills):
        np.save(folder / f"{i}.npy", states(40).numpy())
        entries.append(
            {
                "trial_id": f"clip_{i}",
                "actor_id": f"actor_{i}",
                "states": f"{i}.npy",
                "split": "train",
                "frames": 40,
                "lift_index_bounds": [8, 34],
                "class_id": CLASS_MAP[skill],
                "class_name": skill,
                "review_flags": [],
                "quality_status": "awaiting_visual_review",
                "training_approval": "pending_retarget_review",
                "source_time_start": 100.0 * i,
            }
        )
    data = {
        "status": "complete",
        "window_policy": "segment",
        "class_map": dict(CLASS_MAP),
        "feature_schema": s.to_dict(),
        "feature_schema_sha256": s.sha256,
        "segments": entries,
    }
    path = folder / "index.json"
    path.write_text(json.dumps(data))
    return path


def test_windows_carry_their_segment_class(tmp_path):
    dataset = CarryWindowDataset(skill_index(tmp_path))
    assert dataset.class_counts == {"carry": 50, "push": 25, "slide": 75}
    assert {dataset[i]["class_id"] for i in range(len(dataset))} == {0, 1, 2}
    np.testing.assert_array_equal(
        dataset.window_class_ids, [dataset[i]["class_id"] for i in range(len(dataset))]
    )


def test_class_weights_then_stage_weights_set_the_mix(tmp_path):
    index = skill_index(tmp_path)
    dataset = CarryWindowDataset(
        index,
        class_weights={"carry": 1, "push": 1, "slide": 1},
        stage_weights={"pickup": 0.2, "lifted": 0.6, "setdown": 0.2},
    )
    draws = dataset.sample_indices(60000, torch.Generator().manual_seed(0)).numpy()
    classes = np.bincount(dataset.window_class_ids[draws], minlength=3) / len(draws)
    np.testing.assert_allclose(classes, [1 / 3] * 3, atol=0.01)
    for c in range(3):
        mine = dataset.window_stage_ids[draws][dataset.window_class_ids[draws] == c]
        np.testing.assert_allclose(np.bincount(mine, minlength=3) / len(mine), [0.2, 0.6, 0.2], atol=0.02)
    # Without class weights, classes are drawn in proportion to their windows.
    proportional = CarryWindowDataset(index, stage_weights={"lifted": 1})
    shares = np.bincount(
        proportional.window_class_ids[proportional.sample_indices(30000).numpy()], minlength=3
    )
    np.testing.assert_allclose(shares / shares.sum(), [50 / 150, 25 / 150, 75 / 150], atol=0.01)
    assert set(proportional.window_stage_ids[proportional.sample_indices(500).numpy()]) == {
        WINDOW_STAGES.index("lifted")
    }
    with pytest.raises(ValueError, match="class_weights"):
        CarryWindowDataset(index, class_weights={"walk": 1})
    with pytest.raises(ValueError, match="without windows"):
        CarryWindowDataset(skill_index(tmp_path, ("carry", "push")), class_weights={"slide": 1})


def test_adapter_returns_skill_labels(tmp_path):
    import yaml

    from mimickit_smp.tinymdm_data import HOIMotionPriorData

    env = tmp_path / "env.yaml"
    env.write_text(yaml.safe_dump({"num_disc_obs_steps": 16}))
    config = {
        "dataset_index": str(skill_index(tmp_path)),
        "env_config": str(env),
        "control_freq": 50,
        "class_weights": {"carry": 0, "push": 1, "slide": 0},
    }
    adapter = HOIMotionPriorData(config, "cpu")
    assert adapter.num_class == 3 and adapter.class_map == CLASS_MAP
    _, labels, size = adapter.fetch_obs_demo(64, return_class=True, return_condition=True)
    assert set(labels.tolist()) == {1} and size.shape == (64, 3)
    motion, size, labels = adapter.fixed_windows(32, seed=0, class_name="slide")
    assert set(labels.tolist()) == {2} and motion.shape[0] == 32
    cond, labels = adapter.sample_conditions(8, return_class=True)
    assert cond.shape == (8, 3) and set(labels.tolist()) == {1}


def test_skill_conditioned_prior_trains_samples_and_scores(tinymdm):  # noqa: F811
    model = tinymdm(3, arch="CondDiT")
    x = torch.randn(48, 24)
    cond = torch.rand(48, 3) * 0.5 + 0.2
    labels = torch.arange(48) % 3
    model.update_normalizer(x)
    model.update_cond_normalizer(cond)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        loss = model(x, cond=cond, class_labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)
    t = torch.full((2,), 2, dtype=torch.long)
    size = model.cond_normalizer.normalize(torch.full((2, 3), 0.4))
    with torch.no_grad():
        carry = model.dmodel(x[:2], t, cond=size, class_labels=torch.zeros(2, dtype=torch.long))
        slide = model.dmodel(x[:2], t, cond=size, class_labels=torch.full((2,), 2, dtype=torch.long))
    assert not torch.allclose(carry, slide)
    raw = torch.full((2, 3), 0.4)
    push = torch.ones(2, dtype=torch.long)
    assert model.sample(shape=(24,), batch_size=2, device="cpu", cond=raw, class_labels=push).shape == (2, 24)
    sds = model.ESM_SDS_loss(x[:2], t_lst=[1, 2], cond=raw, class_labels=push)
    assert sds.shape == (2, 2) and torch.isfinite(sds).all()
