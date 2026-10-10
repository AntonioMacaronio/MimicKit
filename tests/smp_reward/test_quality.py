"""Per-frame quality checks, their resampling into the cache, and filtered / stage-weighted windows."""

import json

import numpy as np
import pytest
import torch
from test_data import schema, states

from mimickit_smp import quality
from mimickit_smp.dataset import CarryWindowDataset, MotionCollator, PackedWindowSampler
from mimickit_smp.prepare import resample_interval

FPS = 90.0


def test_chatter_flags_frame_alternation_but_not_fast_smooth_motion():
    t = np.arange(450) / FPS
    joints = np.tile(np.sin(2 * np.pi * 1.5 * t)[:, None], (1, 29))  # 1 rad at 1.5 Hz on every joint
    assert not quality.chatter_flags(joints, FPS).any()
    noisy = joints.copy()
    noisy[200:245, 5] += np.radians(2.0) * (-1.0) ** np.arange(45)
    flags = quality.chatter_flags(noisy, FPS)
    assert flags[200:245].all() and not flags[:150].any() and not flags[300:].any()


def soles(count, foot_xy):
    """Two feet of five sole points on the floor; foot_xy [T, 2] moves the left foot."""
    points = np.zeros((count, 2, 5, 3))
    points[..., 0] = np.linspace(-0.1, 0.1, 5)
    points[:, 1, :, 1] = -0.2
    points[:, 0, :, :2] += foot_xy[:, None]
    points[..., 2] = 0.02  # Sphere radius 0.02 m: touching the floor.
    return points


def test_foot_slide_compares_robot_and_person_pivots():
    n = 60
    still = np.zeros((n, 2, 3))
    planted = soles(n, np.zeros((n, 2)))
    assert not quality.foot_slide_flags(planted, 0.02, still, still, FPS).any()
    slide = soles(n, np.c_[np.linspace(0, 0.12, n), np.zeros(n)])
    flags = quality.foot_slide_flags(slide, 0.02, still, still, FPS)
    assert flags.all()
    # The person slid as far: no excess, so no flag.
    person = np.zeros((n, 2, 3))
    person[:, 0, 0] = np.linspace(0, 0.12, n)
    assert not quality.foot_slide_flags(slide, 0.02, person, person, FPS).any()
    # Rolling over the toe: the toe point stays put while the heel moves; not a slide.
    roll = planted.copy()
    roll[:, 0, :4, 0] += np.linspace(0, 0.08, n)[:, None]
    assert not quality.foot_slide_flags(roll, 0.02, still, still, FPS).any()
    # Contacts shorter than foot_min_contact_s are ignored.
    short = slide.copy()
    short[8:, 0, :, 2] = 0.2
    assert not quality.foot_slide_flags(short, 0.02, still, still, FPS).any()


def test_support_flags_need_a_lifted_box_with_nothing_near():
    m = {
        "box_floor_gap_m": np.array([0.0, 0.5, 0.5, 0.01]),
        "robot_box_nearest_m": np.array([np.inf, np.inf, 0.01, np.inf]),
    }
    assert quality.support_flags(m).tolist() == [False, True, False, False]


def test_check_mask_and_resampled_flags():
    assert quality.check_mask(()) == 0
    assert (
        quality.check_mask(["chatter", "physical", "chatter"])
        == quality.BITS["physical"] | quality.BITS["chatter"]
    )
    with pytest.raises(ValueError, match="Unknown"):
        quality.check_mask(["grip"])
    native = np.zeros(10, np.uint8)
    native[4] = 2
    times = np.arange(10) / FPS
    # Queries between frames 3-4 and 4-5 inherit frame 4's flag; others do not.
    query = np.array([3.5, 4.5, 6.0, 8.2]) / FPS
    assert quality.resample_flags(native, times, query).tolist() == [2, 2, 0, 0]


def test_flagged_native_frames_reach_the_states_that_use_them():
    n = 91
    times = np.arange(n) / FPS
    qpos = np.zeros((n, 43))
    qpos[:, 3] = qpos[:, 39] = 1
    phase = np.ones(n, np.uint8)
    flags = np.zeros(n, np.uint8)
    flags[45] = quality.BITS["chatter"]
    out = resample_interval(qpos, times, phase, np.zeros(3), 50.0, (1,), flags)
    assert out["quality_flags"].shape == out["time"].shape
    bad = out["time"][out["quality_flags"] > 0]
    # States interpolating frame 45 (44-46 brackets), plus the next state whose velocity uses one.
    assert bad.min() > times[44] and bad.min() < times[46]
    assert bad.max() < times[46] + 1 / 50 + 1e-9
    assert 2 <= len(bad) <= 3
    assert "quality_flags" not in resample_interval(qpos, times, phase, np.zeros(3))


def flagged_index(folder, flags_by_segment=None):
    """Two train segments with pickup / lifted / set-down phases and per-state quality flags."""
    s = schema()
    entries = []
    for i, count in enumerate((40, 40)):
        np.save(folder / f"{i}.npy", states(count).numpy())
        flags = np.zeros(count, np.uint8) if flags_by_segment is None else flags_by_segment[i]
        np.save(folder / f"{i}_quality.npy", flags)
        entries.append(
            {
                "trial_id": f"clip_{i}",
                "actor_id": f"actor_{i}",
                "states": f"{i}.npy",
                "quality_flags": f"{i}_quality.npy",
                "split": "train",
                "frames": count,
                "lift_index_bounds": [8, 34],
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
        "quality_checks": {"names": list(quality.CHECKS), "bits": quality.BITS},
        "segments": entries,
    }
    path = folder / "index.json"
    path.write_text(json.dumps(data))
    return path


def test_quality_checks_drop_exactly_the_windows_touching_selected_flags(tmp_path):
    first = np.zeros(40, np.uint8)
    first[20] = quality.BITS["chatter"]
    first[30] = quality.BITS["foot_slide"]
    index = flagged_index(tmp_path, [first, np.zeros(40, np.uint8)])
    everything = CarryWindowDataset(index)
    assert len(everything) == 2 * 25
    chatter = CarryWindowDataset(index, quality_checks=["chatter"])
    starts = [
        chatter[i]["start_time"] * 50 for i in range(len(chatter)) if chatter[i]["trial_id"] == "clip_0"
    ]
    # Windows [s, s + 16) touching state 20 start at 5..20; all others survive.
    assert np.allclose(starts, [s for s in range(25) if not 5 <= s <= 20])
    both = CarryWindowDataset(index, quality_checks=["chatter", "foot_slide"])
    starts = {round(both[i]["start_time"] * 50) for i in range(len(both)) if both[i]["trial_id"] == "clip_0"}
    assert starts == {s for s in range(25) if not 5 <= s <= 24}
    assert len(both) == len(starts) + 25
    ids = torch.arange(len(both))
    scalar = MotionCollator(both.schema, include_states=True)([both[int(i)] for i in ids])
    torch.testing.assert_close(PackedWindowSampler(both).gather(ids), scalar["states"], rtol=0, atol=0)
    unflagged = json.loads(index.read_text())
    del unflagged["quality_checks"]
    index.write_text(json.dumps(unflagged))
    assert len(CarryWindowDataset(index)) == 50
    with pytest.raises(ValueError, match="no quality flags"):
        CarryWindowDataset(index, quality_checks=["chatter"])


def test_a_fully_flagged_segment_contributes_no_windows(tmp_path):
    index = flagged_index(tmp_path, [np.full(40, quality.BITS["support"], np.uint8), np.zeros(40, np.uint8)])
    dataset = CarryWindowDataset(index, quality_checks=["support"])
    assert len(dataset) == 25 and {dataset[i]["trial_id"] for i in range(25)} == {"clip_1"}
    with pytest.raises(ValueError, match="No usable"):
        CarryWindowDataset(
            flagged_index(tmp_path, [np.full(40, 1, np.uint8)] * 2), quality_checks=["physical"]
        )


def test_stage_weights_set_the_stage_mix(tmp_path):
    index = flagged_index(tmp_path)
    uniform = CarryWindowDataset(index)
    assert uniform.stage_counts == {"pickup": 16, "lifted": 22, "setdown": 12}
    assert uniform.stage_weights is None
    weighted = CarryWindowDataset(index, stage_weights={"pickup": 1, "lifted": 2, "setdown": 1})
    assert weighted.stage_weights == {"pickup": 0.25, "lifted": 0.5, "setdown": 0.25}
    draws = weighted.sample_indices(40000, torch.Generator().manual_seed(0))
    shares = np.bincount(weighted.window_stage_ids[draws.numpy()], minlength=3) / len(draws)
    np.testing.assert_allclose(shares, [0.25, 0.5, 0.25], atol=0.01)
    assert len(set(draws.tolist())) == len(weighted)  # Every window is still drawn.
    assert [weighted[int(i)]["stage"] for i in np.flatnonzero(weighted.window_stage_ids == 2)[:1]] == [
        "setdown"
    ]
    only = CarryWindowDataset(index, stage_weights={"lifted": 1})
    assert set(only.window_stage_ids[only.sample_indices(500).numpy()]) == {1}
    for bad in ({"carry": 1}, {"lifted": -1}, {"lifted": 0}):
        with pytest.raises(ValueError, match="stage_weights"):
            CarryWindowDataset(index, stage_weights=bad)
    with pytest.raises(ValueError, match="without windows"):
        CarryWindowDataset(index, stages="lifted", stage_weights={"pickup": 1, "lifted": 1})
    # Without weights the draw matches the previous uniform sampler exactly.
    a = uniform.sample_indices(64, torch.Generator().manual_seed(3))
    torch.testing.assert_close(
        a, torch.randint(len(uniform), (64,), generator=torch.Generator().manual_seed(3))
    )


def test_adapter_reads_quality_checks_and_stage_weights(tmp_path):
    import yaml

    from mimickit_smp.tinymdm_data import HOIMotionPriorData

    first = np.zeros(40, np.uint8)
    first[:8] = quality.BITS["physical"]  # Every pickup window of clip_0 touches a flagged state.
    index = flagged_index(tmp_path, [first, np.zeros(40, np.uint8)])
    env = tmp_path / "env.yaml"
    env.write_text(yaml.safe_dump({"num_disc_obs_steps": 16}))
    config = {
        "dataset_index": str(index),
        "env_config": str(env),
        "control_freq": 50,
        "quality_checks": ["physical"],
        "stage_weights": {"pickup": 0.5, "lifted": 0.5},
    }
    adapter = HOIMotionPriorData(config, "cpu")
    assert adapter.dataset.stage_counts["pickup"] == 8 and adapter.dataset.stage_weights["setdown"] == 0
    assert adapter.fetch_obs_demo(32).shape == (32, 16 * adapter.schema.feature_dim)
    adapter.save_metadata(tmp_path)
    provenance = json.loads((tmp_path / "dataset_provenance.json").read_text())
    assert provenance["quality_checks"] == ["physical"]
    assert provenance["stage_window_counts"] == {"pickup": 8, "lifted": 22, "setdown": 12}


def scene(folder):
    """A box, a floor and five sole spheres per foot, as in the replay scenes."""
    feet = "".join(
        f'<geom name="{side}_ankle_roll_sphere_{k}_link" type="sphere" size="0.02" pos="{0.04 * k - 0.12} {y} 0"/>'
        for side, y in (("left", 0.1), ("right", -0.1))
        for k in range(1, 6)
    )
    xml = f"""<mujoco><worldbody>
      <geom name="ground" type="plane" size="5 5 0.1"/>
      <body name="robot" pos="0 0 0.02"><freejoint/>{feet}
        <geom name="hand" type="sphere" size="0.03" pos="0.4 0 0.93"/></body>
      <body name="largebox_link"><freejoint/><geom name="largebox" type="box" size="0.15 0.15 0.15"/></body>
    </worldbody></mujoco>"""
    path = folder / "scene.xml"
    path.write_text(xml)
    return path


def test_measure_reads_penetration_support_and_soles_from_the_scene(tmp_path):
    pytest.importorskip("mujoco")
    n = 3
    qpos = np.zeros((n, 14))
    qpos[:, 2] = 0.02  # Sole spheres touch the floor; the hand sphere bottom is at 0.92 m.
    qpos[:, 3] = qpos[:, 10] = 1
    # Frame 0: box on the floor. Frame 1: lifted, hand 1 cm away. Frame 2: lifted, nothing near.
    qpos[:, 7:10] = [[0.4, 0, 0.15], [0.4, 0, 0.76], [2.0, 0, 0.76]]
    m = quality.measure(qpos, scene(tmp_path))
    np.testing.assert_allclose(m["box_floor_gap_m"], [0.0, 0.61, 0.61], atol=1e-6)
    assert m["robot_box_nearest_m"][1] == pytest.approx(0.01, abs=1e-6) and np.isinf(
        m["robot_box_nearest_m"][2]
    )
    assert quality.support_flags(m).tolist() == [False, False, True]
    assert m["sole_points"].shape == (n, 2, 5, 3) and m["sole_radius_m"] == pytest.approx(0.02)
    np.testing.assert_allclose(m["sole_points"][..., 2], 0.02)
