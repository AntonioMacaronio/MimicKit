"""decode_window inverts encode_window in the window's own frame."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from test_data import schema, states

from mimickit_smp.decode import decode_window, matrix_to_wxyz, rotation_from_6d
from mimickit_smp.features import encode_window


def moving_states():
    x = states(16).double()
    rng = np.random.default_rng(0)
    x[:, 7:36] = torch.from_numpy(rng.uniform(-2.5, 2.5, (16, 29)))
    root = Rotation.from_euler(
        "zyx", np.c_[np.linspace(30, 50, 16), np.full(16, 5), np.full(16, -3)], degrees=True
    )
    x[:, 3:7] = torch.from_numpy(root.as_quat()[:, [3, 0, 1, 2]])
    box = Rotation.from_euler(
        "zyx", np.c_[np.linspace(10, 20, 16), np.full(16, 4), np.zeros(16)], degrees=True
    )
    x[:, 57:61] = torch.from_numpy(box.as_quat()[:, [3, 0, 1, 2]])
    x[:, 67:70] = torch.tensor([0.6, 0.4, 0.3])
    return x, root


def test_decode_recovers_joints_root_and_box_in_the_window_frame():
    x, root = moving_states()
    s = schema()
    motion, cond = encode_window(x, s)
    out = decode_window(motion, cond, s)
    torch.testing.assert_close(out["joint_pos"], x[:, 7:36], atol=1e-9, rtol=0)
    assert out["joint_off_axis_rad"].max() < 1e-9
    # Window frame: rotate world by minus the last root yaw about the last root XY.
    yaw = root[-1].as_euler("ZYX")[0]  # Intrinsic Z-Y-X yaw, as encode_window extracts it.
    inv = Rotation.from_euler("z", -yaw)
    origin = x[-1, :3].numpy() * [1, 1, 0]
    np.testing.assert_allclose(out["root_pos"].numpy(), inv.apply(x[:, :3].numpy() - origin), atol=1e-9)
    np.testing.assert_allclose(out["root_rot"].numpy(), (inv * root).as_matrix(), atol=1e-9)
    key = x[:, 36:48].reshape(16, 4, 3).numpy()
    np.testing.assert_allclose(
        out["key_body_pos"].numpy(), inv.apply((key - origin).reshape(-1, 3)).reshape(16, 4, 3), atol=1e-9
    )
    np.testing.assert_allclose(out["box_pos"].numpy(), inv.apply(x[:, 54:57].numpy() - origin), atol=1e-9)
    # The canonical box axes are a signed permutation of the mesh axes with matching extents.
    box = (inv * Rotation.from_quat(x[:, [58, 59, 60, 57]].numpy())).as_matrix()
    relation = np.einsum("tij,tik->tjk", box, out["box_rot"].numpy())
    assert np.allclose(np.abs(relation).max(1), 1, atol=1e-9)
    order = np.abs(relation[-1]).argmax(0)
    np.testing.assert_allclose(out["box_extents"].numpy(), x[-1, 67:70].numpy()[order], atol=1e-12)


def test_quaternion_conversion_handles_all_largest_components():
    rotations = Rotation.from_rotvec(np.random.default_rng(1).normal(size=(500, 3)) * 1.5)
    q = matrix_to_wxyz(torch.from_numpy(rotations.as_matrix()))
    expected = rotations.as_quat()[:, [3, 0, 1, 2]]
    expected *= np.sign(expected[:, :1])
    np.testing.assert_allclose(q.numpy(), expected, atol=1e-9)
    six = torch.from_numpy(
        np.concatenate([rotations.as_matrix()[:, :, 0], rotations.as_matrix()[:, :, 2]], -1)
    )
    np.testing.assert_allclose(rotation_from_6d(six).numpy(), rotations.as_matrix(), atol=1e-9)
