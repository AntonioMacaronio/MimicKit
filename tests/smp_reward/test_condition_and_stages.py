"""Schema v3 box-size conditioning, segment windows with pickup/set-down, and the conditional prior."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_data import schema, states
from test_review_fixes import proper_axis_permutations

from mimickit_smp.dataset import CarryWindowDataset, MotionCollator, PackedWindowSampler
from mimickit_smp.features import encode_window
from mimickit_smp.normalization import FeatureNormalizer, fit_normalizer
from mimickit_smp.prepare import resample_interval, resample_positive


def rotated_box_states():
    x = states()
    x[:, 67:70] = torch.tensor([0.6, 0.4, 0.3])
    q = Rotation.from_euler("zyx", np.c_[np.linspace(20, 30, 16), np.full(16, 8), np.zeros(16)], degrees=True)
    x[:, 57:61] = torch.tensor(q.as_quat()[:, [3, 0, 1, 2]], dtype=x.dtype)
    return x


def test_v3_motion_is_v2_without_extents_and_condition_is_canonical_extents():
    x = rotated_box_states()
    v2, none = encode_window(x, replace(schema(), version="g1_box_smp_v2"))
    v3, box_size = encode_window(x, schema())
    assert none is None and v2.shape == (16, 219) and v3.shape == (16, 216)
    torch.testing.assert_close(v3, v2[:, :216])
    torch.testing.assert_close(box_size, v2[-1, 216:219])
    # The rotated box's longest side points most forward, so canonical X is the 0.6 m extent.
    torch.testing.assert_close(box_size, torch.tensor([0.6, 0.4, 0.3]))


def test_schema_dicts_separate_condition_from_denoised_features():
    v2, v3 = replace(schema(), version="g1_box_smp_v2").to_dict(), schema().to_dict()
    assert "box_full_extents" in v2["feature_slices"] and "condition_slices" not in v2
    assert "box_full_extents" not in v3["feature_slices"] and v3["condition_slices"] == {"box_size": (0, 3)}
    assert v2["box_feature_extents"] and "box_feature_extents" not in v3
    assert replace(schema(), version="g1_box_smp_v2").sha256 != schema().sha256


def test_condition_is_identical_for_all_relabelings_of_one_box():
    x = rotated_box_states()
    base = Rotation.from_quat(x[:, 57:61].numpy()[:, [1, 2, 3, 0]]).as_matrix()
    permutations = proper_axis_permutations()
    variants = x.repeat(24, 1, 1)
    q = Rotation.from_matrix((base[None] @ permutations[:, None]).reshape(-1, 3, 3)).as_quat()[
        :, [3, 0, 1, 2]
    ]
    variants[:, :, 57:61] = torch.tensor(q.reshape(24, 16, 4), dtype=x.dtype)
    variants[:, :, 67:70] = torch.tensor(np.array([0.6, 0.4, 0.3]) @ np.abs(permutations), dtype=x.dtype)[
        :, None
    ]
    motion, box_size = encode_window(variants, schema())
    torch.testing.assert_close(motion, motion[:1].expand_as(motion), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(box_size, box_size[:1].expand_as(box_size))


def segment_source(n0=20, n1=140, n2=20):
    times = 10 + np.arange(n0 + n1 + n2) / 90
    q = np.zeros((len(times), 43))
    q[:, 0] = 2 * times
    q[:, 3] = q[:, 39] = 1
    q[:, 38] = np.where(np.arange(len(times)) < n0, 0.15, 0.6)  # Box rests, then is lifted.
    phase = np.r_[np.zeros(n0), np.ones(n1), np.full(n2, 2)].astype(np.uint8)
    return q, times, phase


def test_segment_resampling_keeps_context_phases_inside_the_segment():
    q, times, phase = segment_source()
    full = resample_interval(q, times, phase, np.zeros(3), phases=(0, 1, 2))
    lifted = resample_positive(q, times, phase, np.zeros(3))
    np.testing.assert_allclose(np.diff(full["time"]), 0.02, atol=1e-12)
    assert full["time"][0] >= times[0] + 0.02 - 1e-12 and full["time"][-1] <= times[-1]
    assert set(np.unique(full["phase"])) == {0, 1, 2} and np.all(np.diff(full["phase"].astype(int)) >= 0)
    assert full["positive_time_bounds"] == lifted["positive_time_bounds"] == [times[20], times[159]]
    assert full["interval_time_bounds"] == [times[0], times[-1]]
    np.testing.assert_allclose(full["root_vel"], np.tile([2, 0, 0], (len(full["time"]), 1)), atol=1e-10)
    # Phase follows the latest native frame at or before each resampled time.
    first_lift = np.flatnonzero(full["phase"] == 1)[0]
    assert full["time"][first_lift] >= times[20] and full["time"][first_lift - 1] < times[20]
    assert np.all(lifted["phase"] == 1)


def segment_index(folder, frames=40, lift=(8, 34)):
    s = schema()
    entries = []
    for i, split in enumerate(["train", "train", "validation"]):
        x = states(frames)
        x[:, 67:70] = torch.tensor([0.4 + 0.1 * i, 0.3, 0.2])
        np.save(folder / f"{i}.npy", x.numpy())
        entries.append(
            {
                "trial_id": f"clip_{i}",
                "actor_id": f"actor_{i}",
                "states": f"{i}.npy",
                "split": split,
                "frames": frames,
                "lift_index_bounds": list(lift),
                "review_flags": [],
                "quality_status": "awaiting_visual_review",
                "training_approval": "pending_retarget_review",
                "source_time_start": 100.0 * i,
            }
        )
    data = {
        "status": "complete",
        "window_policy": "segment",
        "feature_schema": s.to_dict(),
        "feature_schema_sha256": s.sha256,
        "segments": entries,
    }
    path = folder / "index.json"
    path.write_text(json.dumps(data))
    return path


def test_stages_select_windows_and_label_pickup_lifted_setdown(tmp_path):
    index = segment_index(tmp_path)
    everything = CarryWindowDataset(index)
    lifted = CarryWindowDataset(index, stages="lifted")
    assert everything.window_policy == "segment"
    assert len(everything) == 2 * (40 - 15) and len(lifted) == 2 * (34 - 8 - 15)
    stages = [everything[i]["stage"] for i in range(25)]
    assert stages[:8] == ["pickup"] * 8 and stages[8:19] == ["lifted"] * 11 and stages[19:] == ["setdown"] * 6
    assert {lifted[i]["stage"] for i in range(len(lifted))} == {"lifted"}
    # The first lifted-only window starts at the lift, 8 frames into the segment.
    assert lifted[0]["start_time"] == everything[8]["start_time"] == 8 / 50
    for dataset in (everything, lifted):
        ids = torch.arange(len(dataset))
        scalar = MotionCollator(dataset.schema, include_states=True)([dataset[int(i)] for i in ids])
        torch.testing.assert_close(PackedWindowSampler(dataset).gather(ids), scalar["states"], rtol=0, atol=0)
        assert scalar["box_size"].shape == (len(dataset), 3)
    with pytest.raises(ValueError, match="stages"):
        CarryWindowDataset(index, stages="pickup")


def test_normalizer_records_condition_statistics(tmp_path):
    train = CarryWindowDataset(segment_index(tmp_path))
    stats = fit_normalizer(train, num_windows=64, batch_size=16)
    assert len(stats["mean"]) == 216 and len(stats["condition_mean"]) == 3
    norm = FeatureNormalizer(stats, train.schema)
    box_size = MotionCollator(train.schema)([train[0], train[30]])["box_size"]
    # Two training boxes differ only in X (0.4 and 0.5 m): normalized X straddles zero.
    normalized = norm.normalize_condition(box_size)
    assert normalized[0, 0] < 0 < normalized[1, 0]
    torch.testing.assert_close(normalized[:, 1:], torch.zeros(2, 2), atol=1e-6, rtol=0)


@pytest.fixture
def tinymdm(tmp_path):
    pytest.importorskip("diffusers")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mimickit"))
    from learning.tinymdm.tinymdm_model import TinyMDMModel

    env = tmp_path / "env.yaml"
    env.write_text("num_disc_obs_steps: 4\n")

    def build(cond_dim, arch="DiT"):
        torch.manual_seed(0)
        config = {
            "T": 4,
            "input_dim": 4 * 6,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "estimate_mode": "epsilon",
            "loss_type": "l1",
            "noise_schedule_mode": "squaredcos_cap_v2",
            "arch_name": arch,
            "num_class": 2,
            "cfg_dropout": 0.1,
            "model_ema": False,
            "normalizer_std_clip": 0.03,
            "env_config": str(env),
            "cond_dim": cond_dim,
        }
        return TinyMDMModel(config, "cpu")

    return build


def test_conditional_prior_trains_samples_and_scores_with_box_size(tinymdm):
    model = tinymdm(3)
    x = torch.randn(32, 24)
    cond = torch.rand(32, 3) * 0.5 + 0.2
    model.update_normalizer(x)
    model.update_cond_normalizer(cond)
    with pytest.raises(ValueError, match="condition"):
        model(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        loss = model(x, cond=cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)
    # The zero-initialized condition embedding has learned to change the denoiser output.
    t = torch.full((2,), 2, dtype=torch.long)
    small, large = torch.full((2, 3), 0.2), torch.full((2, 3), 0.7)
    with torch.no_grad():
        a = model.dmodel(x[:2], t, cond=model.cond_normalizer.normalize(small))
        b = model.dmodel(x[:2], t, cond=model.cond_normalizer.normalize(large))
    assert not torch.allclose(a, b)
    assert model.sample(shape=(24,), batch_size=2, device="cpu", cond=small).shape == (2, 24)
    sds = model.ESM_SDS_loss(x[:2], t_lst=[1, 2], cond=small)
    assert sds.shape == (2, 2) and torch.isfinite(sds).all()
    restored = tinymdm(3)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.cond_normalizer.get_mean(), model.cond_normalizer.get_mean())


def test_unconditional_prior_rejects_condition_and_keeps_its_initialization(tinymdm):
    plain, conditional = tinymdm(0), tinymdm(3)
    with pytest.raises(ValueError, match="without a continuous condition"):
        plain(torch.randn(4, 24), cond=torch.rand(4, 3))
    # New condition layers are created last, so shared parameters start identical.
    shared = plain.dmodel.state_dict()
    for key, value in conditional.dmodel.state_dict().items():
        if not key.startswith("cond_embedder"):
            torch.testing.assert_close(value, shared[key])


def test_classifier_free_wrapper_duplicates_the_condition(tinymdm):
    from learning.tinymdm.cfg_model import ClassifierFreeSampleModel

    model = tinymdm(3, arch="CondDiT")
    wrapper = ClassifierFreeSampleModel(model.dmodel)
    out = wrapper(
        torch.randn(3, 24),
        torch.full((3,), 1, dtype=torch.long),
        class_labels=torch.zeros(3, dtype=torch.long),
        cfg_scale=2.0,
        cond=torch.zeros(3, 3),
    )
    assert out.shape == (3, 24)
