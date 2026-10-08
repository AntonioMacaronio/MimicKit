import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mimickit_smp.dataset import CarryWindowDataset, make_dataloader
from mimickit_smp.features import FeatureSchema, encode_motion, quat_mul, quat_rotate
from mimickit_smp.normalization import FeatureNormalizer, fit_normalizer
from mimickit_smp.prepare import resample_positive


def schema():
    return FeatureSchema(tuple(f"joint_{i}" for i in range(29)), ((0.0, 0.0, 1.0),) * 29, "fixture")


def states(count=16):
    x = torch.zeros(count, 70)
    x[:, 0] = torch.arange(count) * 0.02
    x[:, 2] = 0.8
    x[:, 3] = x[:, 57] = 1
    x[:, 36:48] = torch.tensor([0, 0.1, 0.05, 0, -0.1, 0.05, 0.2, 0.1, 0.9, 0.2, -0.1, 0.9])
    x[:, 48] = 1
    x[:, 54] = x[:, 0] + 0.4
    x[:, 56] = 1
    x[:, 61] = 1
    x[:, 67:70] = torch.tensor([0.4, 0.3, 0.2])
    return x


def test_window_timing_and_feature_conventions():
    s, x = schema(), states()
    assert s.span_seconds == 0.3
    assert (10 - 1) / 30 == s.span_seconds
    out = encode_motion(x, s)
    assert out.shape == (16, 219)
    torch.testing.assert_close(out[:, 0], x[:, 0] - 0.3)
    torch.testing.assert_close(out[:, 2], x[:, 2])
    torch.testing.assert_close(out[:, 195], torch.ones(16))
    torch.testing.assert_close(out[:, 201] - out[:, 0], torch.full((16,), 0.4))
    torch.testing.assert_close(out[:, 216:219], x[:, 67:70])


def test_shared_encoder_is_yaw_translation_invariant_but_object_sensitive():
    s, x = schema(), states()
    q = torch.tensor([np.cos(0.7 / 2), 0, 0, np.sin(0.7 / 2)], dtype=x.dtype)
    offset = x.new_tensor([12, -4, 0])
    moved = x.clone()
    for a, b in [(0, 3), (54, 57)]:
        moved[:, a:b] = quat_rotate(q, x[:, a:b]) + offset
    moved[:, 36:48] = (quat_rotate(q, x[:, 36:48].reshape(16, 4, 3)) + offset).reshape(16, 12)
    for a, b in [(3, 7), (57, 61)]:
        moved[:, a:b] = quat_mul(q, x[:, a:b])
    for a, b in [(48, 51), (51, 54), (61, 64), (64, 67)]:
        moved[:, a:b] = quat_rotate(q, x[:, a:b])
    torch.testing.assert_close(encode_motion(moved, s), encode_motion(x, s), atol=2e-6, rtol=2e-6)
    moved = x.clone()
    moved[:, 56] -= 0.2
    original, changed = encode_motion(x, s), encode_motion(moved, s)
    torch.testing.assert_close(original[:, :201], changed[:, :201])
    assert not torch.equal(original[:, 201:], changed[:, 201:])


def test_native_to_50hz_preserves_lift_and_constant_world_velocities():
    times = 10 + np.arange(180) / 90
    q = np.zeros((len(times), 43))
    q[:, 0] = 2 * times
    q[:, 3] = 1
    q[:, 36] = -times
    q[:, 39:43] = Rotation.from_rotvec(np.c_[np.zeros((180, 2)), 0.5 * times]).as_quat()[:, [3, 0, 1, 2]]
    # Quaternion sign changes must not become artificial spin or invalid interpolation.
    q[::2, 39:43] *= -1
    phase = np.zeros(180, dtype=np.uint8)
    phase[20:160] = 1
    result = resample_positive(q, times, phase, np.zeros(3))
    np.testing.assert_allclose(np.diff(result["time"]), 0.02, atol=1e-12)
    np.testing.assert_allclose(result["root_vel"], np.tile([2, 0, 0], (len(result["time"]), 1)), atol=1e-10)
    np.testing.assert_allclose(
        result["box_ang_vel"], np.tile([0, 0, 0.5], (len(result["time"]), 1)), atol=1e-10
    )
    assert result["time"][0] >= times[20] + 0.02 - 1e-12
    assert result["time"][-1] <= times[159]


def test_disjoint_positive_phases_are_rejected():
    q = np.zeros((50, 43))
    q[:, 3] = q[:, 39] = 1
    phase = np.ones(50)
    phase[20] = 0
    with pytest.raises(ValueError, match="contiguous"):
        resample_positive(q, np.arange(50) / 90, phase, np.zeros(3))


def create_index(folder):
    s = schema()
    entries = []
    for i, (split, count) in enumerate([("train", 18), ("train", 20), ("validation", 18), ("test", 18)]):
        np.save(folder / f"{i}.npy", states(count).numpy())
        entries.append(
            {
                "trial_id": f"clip_{i}",
                "actor_id": f"actor_{i}",
                "states": f"{i}.npy",
                "split": split,
                "frames": count,
                "review_flags": ["flag"] if i == 1 else [],
                "quality_status": "needs_review" if i == 1 else "awaiting_visual_review",
                "training_approval": "pending_retarget_review",
                "source_time_start": 100.0 * i,
            }
        )
    data = {
        "status": "complete",
        "feature_schema": s.to_dict(),
        "feature_schema_sha256": s.sha256,
        "segments": entries,
    }
    path = folder / "index.json"
    path.write_text(json.dumps(data))
    return path


def test_windows_split_boundaries_metadata_and_portability(tmp_path):
    index = create_index(tmp_path)
    dataset = CarryWindowDataset(index)
    assert len(dataset) == 3 + 5
    assert dataset[2]["trial_id"] == "clip_0"
    assert dataset[3]["trial_id"] == "clip_1"
    assert dataset[3]["start_time"] == 100
    assert dataset[7]["start_time"] == 100.08
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    batch = next(iter(make_dataloader(dataset, batch_size=8, shuffle=False)))
    assert batch["motion"].shape == (8, 16, 219)
    assert batch["class_id"].tolist() == [0] * 8
    assert len(CarryWindowDataset(index, quality="unflagged")) == 3
    with pytest.raises(ValueError, match="No usable"):
        CarryWindowDataset(index, quality="approved")
    # Runtime cache requires no absolute source path or MuJoCo model.
    relocated = tmp_path.parent / (tmp_path.name + "_relocated")
    tmp_path.rename(relocated)
    assert len(CarryWindowDataset(relocated / "index.json")) == 8


def test_normalization_train_only_and_frozen_roundtrip(tmp_path):
    index = create_index(tmp_path)
    train = CarryWindowDataset(index)
    stats = fit_normalizer(train, num_windows=8)
    norm = FeatureNormalizer(stats, train.schema)
    batch = next(iter(make_dataloader(train, batch_size=4)))["motion"]
    torch.testing.assert_close(norm.inverse(norm(batch)), batch)
    with pytest.raises(ValueError, match="training split"):
        fit_normalizer(CarryWindowDataset(index, split="validation"), num_windows=8)


def test_actor_leakage_and_changed_schema_are_rejected(tmp_path):
    index = create_index(tmp_path)
    data = json.loads(index.read_text())
    data["segments"][-1]["actor_id"] = data["segments"][0]["actor_id"]
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Actor leakage"):
        CarryWindowDataset(index)
    data = schema().to_dict()
    data["quaternion_order"] = "XYZW"
    with pytest.raises(ValueError, match="convention"):
        FeatureSchema.from_dict(data)


def test_tinymdm_adapter_uses_train_split_and_checks_window_contract(tmp_path):
    import yaml

    from mimickit_smp.tinymdm_data import HOIMotionPriorData

    index = create_index(tmp_path)
    env = tmp_path / "env.yaml"
    env.write_text(yaml.safe_dump({"num_disc_obs_steps": 16}))
    config = {"dataset_index": str(index), "env_config": str(env), "control_freq": 50}
    adapter = HOIMotionPriorData(config, "cpu")
    obs, labels = adapter.fetch_obs_demo(5, return_class=True)
    assert obs.shape == (5, 3504)
    assert labels.tolist() == [0] * 5
    assert {s["split"] for s in adapter.dataset.segments} == {"train"}
    with pytest.raises(ValueError, match="frequency"):
        HOIMotionPriorData(dict(config, control_freq=30), "cpu")
    env.write_text(yaml.safe_dump({"num_disc_obs_steps": 10}))
    with pytest.raises(ValueError, match="window length"):
        HOIMotionPriorData(config, "cpu")


def test_quaternion_sign_and_batched_live_encoding_agree():
    x = states().repeat(3, 1, 1)
    signed = x.clone()
    signed[1, :, 3:7] *= -1
    signed[2, :, 57:61] *= -1
    torch.testing.assert_close(encode_motion(x, schema()), encode_motion(signed, schema()))
    for batch in range(3):
        torch.testing.assert_close(encode_motion(x, schema())[batch], encode_motion(x[batch], schema()))
