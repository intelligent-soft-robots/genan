import collections
from typing import Optional, Tuple

import jax
import numpy as np
import scipy.spatial.transform

from sim2real.rl_tasks.physics_backend.physics_backend import ObjData

Transformation = collections.namedtuple("Transformation", ["translation", "rotation"])

# Transformation from the Mujoco robot frame to the Vicon robot frame
EDGE_POS_VICON_ROBOT_FRAME = np.array([-0.0801, 0.119, 0.01909])
EDGE_ROT_VICON_ROBOT_FRAME = scipy.spatial.transform.Rotation.from_euler(
    "z", -np.pi / 2
)

# Transformation from the Mujoco robot frame to the Mujoco world frame
ROBOT_POS_MUJOCO_FRAME = np.array([0.0, 0.0, 1.216])
ROBOT_ROT_MUJOCO_FRAME = scipy.spatial.transform.Rotation.from_euler("z", -np.pi / 2)

BALL_IN_CUP_Z_OFFSET_STRING_ATTACH = -0.079290
BALL_IN_CUP_STRING_LENGTH = 0.2


def vicon_to_robot_frame(
    pos: np.ndarray,
    quat: np.ndarray,
    vicon_robot_pose: Transformation,
) -> Tuple[np.ndarray, np.ndarray]:
    rot = scipy.spatial.transform.Rotation.from_quat(quat)
    pos_vicon_base = vicon_robot_pose.rotation.apply(
        pos - vicon_robot_pose.translation, inverse=True
    )
    rot_vicon_base = vicon_robot_pose.rotation.inv() * rot
    pos_mujoco_base = EDGE_ROT_VICON_ROBOT_FRAME.apply(
        pos_vicon_base - EDGE_POS_VICON_ROBOT_FRAME, inverse=True
    )
    rot_mujoco_base = EDGE_ROT_VICON_ROBOT_FRAME.inv() * rot_vicon_base
    return pos_mujoco_base, rot_mujoco_base.as_quat()


def robot_to_vicon_frame(
    pos: np.ndarray,
    quat: np.ndarray,
    vicon_robot_pose: Transformation,
) -> Tuple[np.ndarray, np.ndarray]:
    rot = scipy.spatial.transform.Rotation.from_quat(quat)
    pos_vicon_base = EDGE_ROT_VICON_ROBOT_FRAME.apply(pos) + EDGE_POS_VICON_ROBOT_FRAME
    rot_vicon_base = EDGE_ROT_VICON_ROBOT_FRAME * rot
    pos_vicon_world = (
        vicon_robot_pose.rotation.apply(pos_vicon_base) + vicon_robot_pose.translation
    )
    rot_vicon_world = vicon_robot_pose.rotation * rot_vicon_base
    return pos_vicon_world, rot_vicon_world.as_quat()


def mujoco_to_robot_frame(
    pos: np.ndarray, quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    rot = scipy.spatial.transform.Rotation.from_quat(quat)
    pos_robot_frame = ROBOT_ROT_MUJOCO_FRAME.apply(
        pos - ROBOT_POS_MUJOCO_FRAME, inverse=True
    )
    rot_robot_frame = ROBOT_ROT_MUJOCO_FRAME.inv() * rot
    return pos_robot_frame, rot_robot_frame.as_quat()


def robot_to_mujoco_frame(
    pos: np.ndarray, quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    rot = scipy.spatial.transform.Rotation.from_quat(quat)
    pos_robot_frame = ROBOT_ROT_MUJOCO_FRAME.apply(pos) + ROBOT_POS_MUJOCO_FRAME
    rot_robot_frame = ROBOT_ROT_MUJOCO_FRAME * rot
    return pos_robot_frame, rot_robot_frame.as_quat()


def vicon_to_mujoco_frame(
    pos: np.ndarray, quat: np.ndarray, robot_pose: Transformation
) -> Tuple[np.ndarray, np.ndarray]:
    pos_robot, quat_robot = vicon_to_robot_frame(pos, quat, robot_pose)
    pos_mujoco, quat_mujoco = robot_to_mujoco_frame(pos_robot, quat_robot)
    return pos_mujoco, quat_mujoco


def mujoco_to_vicon_frame(
    pos: np.ndarray, quat: np.ndarray, robot_pose: Transformation
) -> Tuple[np.ndarray, np.ndarray]:
    pos_robot, quat_robot = mujoco_to_robot_frame(pos, quat)
    pos_mujoco, quat_mujoco = robot_to_vicon_frame(pos_robot, quat_robot, robot_pose)
    return pos_mujoco, quat_mujoco


def vel_vicon_to_robot_frame(
    vel: np.ndarray,
    vicon_robot_pose: Transformation,
) -> np.ndarray:
    vel_vicon_base = vicon_robot_pose.rotation.apply(vel, inverse=True)
    vel_mujoco_base = EDGE_ROT_VICON_ROBOT_FRAME.apply(
        vel_vicon_base - EDGE_POS_VICON_ROBOT_FRAME, inverse=True
    )
    return vel_mujoco_base


def vel_robot_to_vicon_frame(
    vel: np.ndarray,
    vicon_robot_pose: Transformation,
) -> np.ndarray:
    vel_vicon_base = EDGE_ROT_VICON_ROBOT_FRAME.apply(vel)
    vel_vicon_world = vicon_robot_pose.rotation.apply(vel_vicon_base)
    return vel_vicon_world


def vel_mujoco_to_robot_frame(vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return ROBOT_ROT_MUJOCO_FRAME.apply(vel, inverse=True)


def vel_robot_to_mujoco_frame(vel: np.ndarray) -> np.ndarray:
    return ROBOT_ROT_MUJOCO_FRAME.apply(vel)


def vel_vicon_to_mujoco_frame(
    vel: np.ndarray, robot_pose: Transformation
) -> np.ndarray:
    vel_robot = vel_vicon_to_robot_frame(vel, robot_pose)
    vel_mujoco = vel_robot_to_mujoco_frame(vel_robot)
    return vel_mujoco


def vel_mujoco_to_vicon_frame(
    vel: np.ndarray, robot_pose: Transformation
) -> np.ndarray:
    vel_robot, quat_robot = vel_mujoco_to_robot_frame(vel)
    vel_mujoco, quat_mujoco = vel_robot_to_vicon_frame(vel_robot, robot_pose)
    return vel_mujoco


def ball_pos_vel_cup_frame(
    cup_data: ObjData, ball_pos: jax.Array, ball_vel: Optional[jax.Array]
) -> Tuple[jax.Array, jax.Array]:
    rot_to_cup = jax.scipy.spatial.transform.Rotation.from_quat(cup_data.quat).inv()
    ball_pos_rel = rot_to_cup.apply(ball_pos - cup_data.pos)
    ball_vel_rel = (
        rot_to_cup.apply(ball_vel - cup_data.linvel) if ball_vel is not None else None
    )
    return ball_pos_rel, ball_vel_rel
