import numpy as np
import pytest
import scipy.spatial.transform

from sim2real.rl_tasks.spatial import (
    Transformation,
    mujoco_to_vicon_frame,
    vicon_to_mujoco_frame,
)


def _sample_quat():
    quat = np.random.uniform(low=-0.2, high=0.2, size=(3,))
    return np.concatenate((quat, [1.0 - np.linalg.norm(quat)]))


def test_vicon_to_mujoco_pos():
    pos = np.random.normal(size=(3,))
    quat = _sample_quat()
    robot_pose = Transformation(
        np.random.normal(size=(3,)),
        scipy.spatial.transform.Rotation.from_quat(_sample_quat()),
    )
    pos_mujoco, quat_mujoco = vicon_to_mujoco_frame(pos, quat, robot_pose)
    pos_vicon, quat_vicon = mujoco_to_vicon_frame(pos_mujoco, quat_mujoco, robot_pose)
    assert pos == pytest.approx(pos_vicon)
    assert scipy.spatial.transform.Rotation.from_quat(quat).approx_equal(
        scipy.spatial.transform.Rotation(quat_vicon)
    )
