import argparse
import logging
import time
from pathlib import Path

import hydra.utils
import imageio
import mujoco.egl
import mujoco.viewer
import numpy as np
import yaml
from tqdm import tqdm

from sim2real.rl_tasks.pamy_ball_in_cup.pamy_ball_in_cup import PamyBallInCupEnv
from sim2real.rl_tasks.pamy_reacher.pamy_reacher import PamyReacherEnv
from sim2real.rl_tasks.physics_backend.physics_backend import ObjData
from sim2real.util.model import create_model_spec
from sim2real.util.table_tennis import ball_pos_vel_rel_racket, ball_racket_contact

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def render_frame(model, mj_data, opt, cam, scene, context, fb_width, fb_height):
    mujoco.mjv_updateScene(
        model,
        mj_data,
        opt,
        None,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    rect = mujoco.MjrRect(0, 0, fb_width, fb_height)
    mujoco.mjr_render(
        rect,
        scene,
        context,
    )
    pixels = np.zeros((fb_height, fb_width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(
        pixels,
        None,
        rect,
        context,
    )
    return np.flipud(pixels)


def duplicate_visual_geoms(model_spec: mujoco.MjSpec, suffix="_goal_marker"):
    robot_arm = model_spec.worldbody.find_child("robot_arm")
    _duplicate_visual_geoms(model_spec, robot_arm, robot_arm, suffix)


def _duplicate_visual_geoms(
    model_spec: mujoco.MjSpec,
    orig_parent_body: mujoco.MjsBody,
    parent_body: mujoco.MjsBody,
    suffix: str,
):
    for orig_child_body in orig_parent_body.bodies:
        quat = np.array(
            model_spec.resolve_orientation(
                model_spec.compiler.degree,
                model_spec.compiler.eulerseq,
                orig_child_body.alt,
            )
        )
        # No euler tag results in an all zero quaternion, which produces some weird orientation.
        if np.all(quat == 0.0):
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        new_child_body = parent_body.add_body(
            name=orig_child_body.name + suffix,
            pos=orig_child_body.pos,
            quat=quat,
        )
        _duplicate_visual_geoms(model_spec, orig_child_body, new_child_body, suffix)

    for joint in orig_parent_body.joints:
        parent_body.add_joint(name=joint.name + suffix, axis=joint.axis)

    for geom in orig_parent_body.geoms:
        if getattr(geom, "contype", None) == 0:
            parent_body.add_geom(
                name=geom.name + suffix,
                meshname=geom.meshname,
                type=geom.type,
                size=geom.size,
                pos=geom.pos,
                quat=geom.quat,
                rgba=[1.0, 0.0, 0.0, 0.2],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--slowdown", type=int, default=0)
    parser.add_argument("--keep-reset", action="store_true")
    parser.add_argument("--outname", type=str)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--camera-look-at", type=float, nargs=3, default=[0.25, 0.0, 1.42]
    )
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", type=float, default=1.3)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    args = parser.parse_args()

    model_path = Path(__file__).parents[2] / "assets" / "Pamy_Arm_URDF_230227_2"

    with (args.path / "evaluate_transfer_config" / "config.yaml").open("r") as f:
        eval_transfer_cfg = yaml.safe_load(f)

    train_logs_path = (
        Path(__file__).parents[2] / "logs" / "train" / eval_transfer_cfg["train_logs"]
    )
    with (train_logs_path / "train_rl_agent_config" / "config.yaml").open("r") as f:
        train_rl_agent_cfg = yaml.safe_load(f)
    env = hydra.utils.instantiate(
        train_rl_agent_cfg["env"],
        backend=train_rl_agent_cfg["backend"],
    )

    data = np.load(next(args.path.glob("*.npz")), allow_pickle=True)
    task = data["task"]
    data_from_o80_backend = "o80_iteration" in data

    model_spec = create_model_spec("racket" if task == "reacher" else task)

    if task == "reacher":
        duplicate_visual_geoms(model_spec, suffix="_goal_marker")
    model = model_spec.compile()

    mj_data = mujoco.MjData(model)

    reset_len = 3000

    if args.outname is not None:
        fb_width = 1920
        fb_height = 1088
        # TODO: Dont hardcode the action repeat
        fps = round(1 / (model.opt.timestep * (1 if data_from_o80_backend else 5)))
        writer = imageio.get_writer(
            (Path(__file__).parents[2] / "out" / f"{args.outname}").with_suffix(".mp4"),
            fps=fps,
        )
        gl_context = mujoco.egl.GLContext(fb_width, fb_height)
        gl_context.make_current()
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        mujoco.mjr_resizeOffscreen(fb_width, fb_height, context)
        opt = mujoco.MjvOption()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array(args.camera_look_at)
        cam.elevation = args.camera_elevation
        cam.distance = args.camera_distance
        cam.azimuth = args.camera_azimuth
    else:
        writer = None

    last_o80_iteration = None
    if writer is None:
        viewer = mujoco.viewer.launch_passive(model, mj_data)
        viewer.cam.lookat[:] = np.array(args.camera_look_at)
        viewer.cam.elevation = args.camera_elevation
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.camera_azimuth
    else:
        viewer = None

    scene = mujoco.MjvScene(model, maxgeom=10000)
    success_rates = []
    data_files = sorted(
        [p for p in args.path.iterdir() if p.suffix == ".npz"],
        key=lambda p: int(p.stem[len("ep_") :]),
    )
    num_episodes = 0
    successes = 0
    for p in tqdm(data_files):
        if p.suffix == ".npz":
            data = np.load(p, allow_pickle=True)
            observations = data["observations"]
            num_obs_per_dof = observations.shape[-1] // 4
            if "o80_positions" in data:
                positions = data["o80_positions"][
                    None, (0 if args.keep_reset else reset_len) :
                ]
            else:
                positions = data["positions"]
            if "o80_iterations" in data:
                iterations = data["o80_iterations"][
                    (0 if args.keep_reset else reset_len) :
                ]
            else:
                iterations = np.arange(positions.shape[1])
            if "o80_timestamps" in data:
                timestamps = data["o80_timestamps"][
                    (0 if args.keep_reset else reset_len) :
                ]
            else:
                timestamps = np.arange(positions.shape[1]) * 0.002

            goals = observations[..., 0, -4:]

            obj_datas = data["object_datas"].item()
            if args.max_episodes is not None:
                positions = positions[: args.max_episodes - num_episodes]
            for i, pos_traj in enumerate(tqdm(positions, disable=len(positions) <= 1)):
                if args.max_episodes is not None and num_episodes >= args.max_episodes:
                    break
                if task == "reacher":
                    mj_data.qpos[4:8] = goals[i]
                start_time = time.time()
                for j, (qpos, it, t) in enumerate(
                    zip(pos_traj, iterations, timestamps)
                ):
                    backend_infos = data["backend_infos"].item()
                    obj_data_idx = (
                        np.nanargmin(np.abs(backend_infos["timestamp"][i] - t))
                        if "timestamp" in backend_infos
                        else j
                    )
                    obj_data = {
                        k: ObjData(
                            v.pos[i, obj_data_idx],
                            v.quat[i, obj_data_idx],
                            v.linvel[i, obj_data_idx],
                            v.rotvel[i, obj_data_idx],
                            (
                                v.timestamp[i, obj_data_idx]
                                if v.timestamp is not None
                                else None
                            ),
                        )
                        for k, v in obj_datas.items()
                    }
                    mj_data.qpos[:4] = qpos
                    if task == "ball_in_cup" or task == "table_tennis":
                        ball_pos = obj_data["ball"].pos
                        ball_quat = obj_data["ball"].quat
                        ball_id = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_BODY, "ball"
                        )
                        ball_qpos_start_idx = model.jnt_qposadr[
                            model.body_jntadr[ball_id]
                        ]
                        mj_data.qpos[ball_qpos_start_idx : ball_qpos_start_idx + 3] = (
                            ball_pos
                        )
                        mj_data.qpos[
                            ball_qpos_start_idx + 3 : ball_qpos_start_idx + 7
                        ] = ball_quat
                    mujoco.mj_forward(model, mj_data)
                    if isinstance(env, PamyReacherEnv):
                        curr_success = np.linalg.norm(qpos - goals[i]) <= 0.1
                    elif isinstance(env, PamyBallInCupEnv):
                        curr_success = env.is_success(obj_data).item()
                    else:
                        curr_success = ball_racket_contact(
                            ball_pos_vel_rel_racket(obj_data, None)[0],
                            tolerance=0.005,
                        )
                    if j == len(pos_traj) - 1:
                        successes += curr_success
                    for k in range(model.ngeom):
                        if model.geom(k).name.endswith("_goal_marker"):
                            if (
                                "o80_positions" in data
                                and args.keep_reset
                                and i < reset_len
                            ):
                                model.geom_rgba[k] = np.array([0.0, 0.0, 0.0, 0.0])
                            elif curr_success:
                                model.geom_rgba[k] = np.array([0.0, 1.0, 0.0, 0.2])
                            else:
                                model.geom_rgba[k] = np.array([1.0, 0.0, 0.0, 0.2])
                        elif model.geom(k).name == "ball_geom":
                            if curr_success:
                                model.geom_rgba[k] = np.array([0.0, 1.0, 0.0, 1.0])
                            else:
                                model.geom_rgba[k] = np.array([1.0, 0.0, 0.0, 1.0])
                    if viewer is not None:
                        viewer.sync()
                        while (
                            time.time() - start_time
                            < (j + 1) * model.opt.timestep * args.slowdown
                        ):
                            pass
                    if writer is not None:
                        frame = render_frame(
                            model,
                            mj_data,
                            opt,
                            cam,
                            scene,
                            context,
                            fb_width,
                            fb_height,
                        )
                        if last_o80_iteration is not None:
                            # If frames are missing, repeat the current frame to keep the synchronization
                            for _ in range(it - last_o80_iteration):
                                writer.append_data(frame)
                        else:
                            writer.append_data(frame)
                    last_o80_iteration = it
                num_episodes += 1
                if task == "table_tennis":
                    landing_point_table_frame = (
                        data["final_states"].item().task.landing_point_table_frame[i]
                    )
                    landing_point = (
                        landing_point_table_frame + env.backend.table_position
                    )
                    mj_data.qpos[ball_qpos_start_idx : ball_qpos_start_idx + 3] = (
                        landing_point
                    )
                    mujoco.mj_forward(model, mj_data)
                    if writer is not None:
                        frame = render_frame(
                            model,
                            mj_data,
                            opt,
                            cam,
                            scene,
                            context,
                            fb_width,
                            fb_height,
                        )
                        for _ in range(fps):
                            writer.append_data(frame)
                    else:
                        viewer.sync()
                        time.sleep(1.0)

                print(f"Success rate: {successes / num_episodes * 100.0:.2f}%")
                if isinstance(env, PamyReacherEnv):
                    print(f"Distances to goal: {qpos - goals[i]}")
            last_iteration_last_episode = (
                data["o80_iterations"][-1] if "o80_iterations" in data else None
            )
            if task == "reacher":
                success_metric_name = "target_dist/success_rate_norm_1steps_dist0.1"
            elif task == "ball_in_cup":
                success_metric_name = "ball_in_cup/success_rate"
            elif task == "table_tennis":
                success_metric_name = "table_tennis/success_rate_dist_0.685"
            success_rates.append(data["episode_info"].item()[success_metric_name])
            if args.max_episodes is not None and num_episodes >= args.max_episodes:
                break
    print(f"success rate: {np.mean(success_rates) * 100.0:.2f}%")
    if viewer is not None:
        viewer.close()
