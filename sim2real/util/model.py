from pathlib import Path
from typing import Literal, Sequence

import jax.numpy as jnp
import mujoco
import numpy as np


def _attach_racket(model_spec: mujoco.MjSpec) -> None:
    racket_xml_path = (
        Path(__file__).parents[2]
        / "assets"
        / "Pamy_Arm_URDF_230227_2"
        / "xml"
        / "racket.xml"
    )
    racket_model = mujoco.MjSpec.from_file(str(racket_xml_path))
    j4 = model_spec.body("J4")
    ee_mount = j4.add_frame()
    ee_mount.attach_body(racket_model.body("racket"), "", "")
    model_spec.add_exclude(bodyname1="robot_arm", bodyname2="racket")
    for dof in range(1, 5):
        model_spec.add_exclude(bodyname1=f"J{dof}", bodyname2="racket")


def _attach_cup(model_spec: mujoco.MjSpec) -> None:
    cup_xml_path = (
        Path(__file__).parents[2]
        / "assets"
        / "Pamy_Arm_URDF_230227_2"
        / "xml"
        / "cup.xml"
    )
    cup_model = mujoco.MjSpec.from_file(str(cup_xml_path))

    j4 = model_spec.body("J4")
    ee_mount = j4.add_frame()
    ee_mount.attach_body(cup_model.body("cup_connector_arm_side"), "", "")
    for cup_body in ["cup", "cup_connector_arm_side", "cup_connector_cup_side"]:
        model_spec.add_exclude(bodyname1="robot_arm", bodyname2=cup_body)
        for dof in range(1, 5):
            model_spec.add_exclude(bodyname1=f"J{dof}", bodyname2=cup_body)


def _attach_cup_ball(model_spec: mujoco.MjSpec) -> None:
    ball_xml_path = (
        Path(__file__).parents[2]
        / "assets"
        / "Pamy_Arm_URDF_230227_2"
        / "xml"
        / "cup_ball.xml"
    )
    ball_model = mujoco.MjSpec.from_file(str(ball_xml_path))
    ball_frame = model_spec.worldbody.add_frame(pos=[0.2, 0.0, 1.8])
    ball_frame.attach_body(ball_model.body("ball"))
    string = model_spec.add_tendon(
        name="string",
        limited=True,
        range=[0, 0.2],
        width=0.002,
    )
    string.wrap_site("string_attach")
    string.wrap_site("ball_attach")
    # The ball can only collide with the J4 mesh --> Disable all other contacts
    model_spec.add_exclude(bodyname1="robot_arm", bodyname2="ball")
    for dof in range(1, 4):
        model_spec.add_exclude(bodyname1=f"J{dof}", bodyname2="ball")


def _add_table(model_spec: mujoco.MjSpec, table_position: Sequence[float]) -> None:
    xml_path = (
        Path(__file__).parents[2]
        / "assets"
        / "Pamy_Arm_URDF_230227_2"
        / "xml"
        / "table.xml"
    )
    table_model = mujoco.MjSpec.from_file(str(xml_path))
    # Table position is the position of the plane on top of the table --> Subtract the table thickness
    table_frame = model_spec.worldbody.add_frame(
        pos=np.asarray(table_position) - np.array([0.0, 0.0, 0.02])
    )
    table_frame.attach_body(table_model.body("table"))
    model_spec.add_exclude(bodyname1="robot_arm", bodyname2="table")
    for dof in range(1, 5):
        model_spec.add_exclude(bodyname1=f"J{dof}", bodyname2="table")
        model_spec.add_exclude(bodyname1=f"J{dof}", bodyname2="net")
    model_spec.add_exclude(bodyname1="racket", bodyname2="table")
    model_spec.add_exclude(bodyname1="racket", bodyname2="net")
    model_spec.add_exclude(bodyname1="ball", bodyname2="table")


def _add_table_tennis_ball(model_spec: mujoco.MjSpec) -> None:
    xml_path = (
        Path(__file__).parents[2]
        / "assets"
        / "Pamy_Arm_URDF_230227_2"
        / "xml"
        / "table_tennis_ball.xml"
    )
    ball_model = mujoco.MjSpec.from_file(str(xml_path))
    ball_frame = model_spec.worldbody.add_frame()
    ball_frame.attach_body(ball_model.body("ball"))
    model_spec.add_exclude(bodyname1="robot_arm", bodyname2="ball")
    for dof in range(1, 5):
        model_spec.add_exclude(bodyname1=f"J{dof}", bodyname2="ball")
    # Racket-ball interactions are handled by custom contact model.
    model_spec.add_exclude(bodyname1="racket", bodyname2="ball")


def create_model_spec(
    context: Literal["ball_in_cup", "cup", "racket", "table_tennis"],
    dt: float = 0.002,
    table_position: Sequence[float] = (0.6, 1.5, 0.76),
) -> mujoco.MjSpec:
    assert context in [
        "ball_in_cup",
        "cup",
        "racket",
        "table_tennis",
    ], f"Invalid context: {context}."
    model_spec = mujoco.MjSpec.from_file(
        str(
            Path(__file__).parents[2]
            / "assets"
            / "Pamy_Arm_URDF_230227_2"
            / "xml"
            / "pamy_no_muscles.xml"
        )
    )
    if context in ["racket", "table_tennis"]:
        _attach_racket(model_spec)
    if context in ["ball_in_cup", "cup"]:
        _attach_cup(model_spec)
    if context == "ball_in_cup":
        _attach_cup_ball(model_spec)
    if context == "table_tennis":
        _add_table(model_spec, table_position)
        _add_table_tennis_ball(model_spec)
    model_spec.option.timestep = dt
    return model_spec


def quat_from_mujoco(quat: jnp.ndarray) -> jnp.ndarray:
    # MuJoCo represents quaternions as [w, x, y, z], scipy as [x, y, z, w]
    return jnp.concatenate((quat[..., 1:], quat[..., :1]), axis=-1)
