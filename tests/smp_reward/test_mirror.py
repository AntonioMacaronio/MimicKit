"""Left-right mirror augmentation: reflection identities, FK-derived joint signs, encoder consistency."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_data import schema
from test_decode import moving_states

from mimickit_smp.features import encode_window
from mimickit_smp.mirror import PSEUDO, mirror_map, mirror_states, swap_side

PARTNER = np.r_[6:12, 0:6, 12:15, 22:29, 15:22]  # G1 order: left leg, right leg, waist, left arm, right arm
SIGNS = np.ones(29)
SIGNS[
    [1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28]
] = -1.0  # G1: every roll and yaw joint flips


def test_mirroring_twice_is_the_identity_and_swaps_sides():
    x = moving_states()[0].numpy()
    once = mirror_states(x, PARTNER, SIGNS)
    np.testing.assert_allclose(mirror_states(once, PARTNER, SIGNS), x, atol=1e-12)
    np.testing.assert_allclose(once[:, 1], -x[:, 1])
    np.testing.assert_allclose(
        once[:, 36:39] * [1, -1, 1], x[:, 39:42]
    )  # left ankle <- reflected right ankle
    np.testing.assert_allclose(once[:, 67:70], x[:, 67:70])
    assert (
        swap_side("left_hip_roll_joint") == "right_hip_roll_joint"
        and swap_side("waist_yaw_joint") == "waist_yaw_joint"
    )


def test_mirrored_rotations_negate_yaw_and_keep_angular_velocity_consistent():
    euler = np.c_[np.linspace(10, 60, 20), np.linspace(-5, 8, 20), np.linspace(3, -4, 20)]
    rot = Rotation.from_euler("ZYX", euler, degrees=True)
    x = np.zeros((20, 70))
    x[:, 3:7] = rot.as_quat()[:, [3, 0, 1, 2]]
    x[:, 57:61] = x[:, 3:7]
    x[1:, 51:54] = (rot[1:] * rot[:-1].inv()).as_rotvec() * 50
    m = mirror_states(x, PARTNER, SIGNS)
    mrot = Rotation.from_quat(m[:, [4, 5, 6, 3]])
    np.testing.assert_allclose(mrot.as_euler("ZYX", degrees=True), euler * [-1, 1, -1], atol=1e-9)
    # Backward-difference angular velocity of the mirrored rotations equals the mirrored velocity.
    np.testing.assert_allclose((mrot[1:] * mrot[:-1].inv()).as_rotvec() * 50, m[1:, 51:54], atol=1e-9)
    np.testing.assert_allclose(m[1:, 51:54], x[1:, 51:54] * PSEUDO, atol=1e-12)


def test_encoded_mirror_negates_lateral_features_and_keeps_the_box_size():
    s = schema()
    x = moving_states()[0]
    m = torch.from_numpy(mirror_states(x.numpy(), PARTNER, SIGNS))
    motion, cond = encode_window(x, s)
    mirrored, mirrored_cond = encode_window(m, s)
    torch.testing.assert_close(mirrored_cond, cond)
    torch.testing.assert_close(mirrored[..., 1], -motion[..., 1])  # root lateral position
    torch.testing.assert_close(mirrored[..., 0], motion[..., 0])


def test_joint_signs_come_from_forward_kinematics(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    leg = """<body name="{s}_hip_link" pos="0 {y} 0">
        <joint name="{s}_hip_pitch_joint" axis="0 1 0"/><joint name="{s}_hip_roll_joint" axis="1 0 0"/>
        <joint name="{s}_hip_yaw_joint" axis="0 0 1"/><geom size="0.01"/>
        <body name="{s}_foot_link" pos="0.05 0 -0.4"><geom size="0.01"/></body></body>"""
    xml = f"""<mujoco><worldbody><body name="pelvis"><freejoint/><geom size="0.01"/>
        {leg.format(s="left", y=0.1)}{leg.format(s="right", y=-0.1)}
        <body name="torso_link" pos="0.02 0 0.3"><joint name="waist_yaw_joint" axis="0 0 1"/>
        <joint name="waist_roll_joint" axis="1 0 0"/><geom size="0.01"/>
        <body name="head_link" pos="0.1 0.05 0.2"><geom size="0.01"/></body></body>
        </body></worldbody></mujoco>"""
    # head_link is off-center, so it is no mirror image of itself: reported and left out.
    model = mujoco.MjModel.from_xml_string(xml)
    names = [model.joint(i).name for i in range(1, model.njnt)]
    partner, signs, info = mirror_map(model, names)
    assert info["relabeled_bodies_excluded"] == ["head_link"]
    symmetric = mujoco.MjModel.from_xml_string(xml.replace('pos="0.1 0.05 0.2"', 'pos="0.1 0 0.2"'))
    partner, signs, info = mirror_map(symmetric, names)
    assert info["relabeled_bodies_excluded"] == [] and info["max_probe_error_m"] < 1e-9
    assert info["min_wrong_sign_error_m"] > 0.01
    expected = {"hip_pitch": 1, "hip_roll": -1, "hip_yaw": -1, "waist_yaw": -1, "waist_roll": -1}
    for name, p, sign in zip(names, partner, signs):
        assert names[p] == swap_side(name)
        assert sign == next(v for k, v in expected.items() if k in name)
