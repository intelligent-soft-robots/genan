import argparse
import time
from pathlib import Path

import hydra
import imageio
import mujoco.egl
import mujoco.viewer
import numpy as np
import yaml
from tqdm import tqdm

from sim2real.rl_tasks.pamy_reacher.pamy_reacher import PamyReacherEnv
from sim2real.rl_tasks.physics_backend.physics_backend import ObjData
from sim2real.util.model import create_model_spec


def sample_ball_pos():
    cos_ball_polar_angle_min = np.cos(np.pi)
    cos_ball_polar_angle = np.random.uniform(
        low=cos_ball_polar_angle_min,
        high=1.0,
    )
    if cos_ball_polar_angle > 0.8:
        r = np.random.uniform(0.02, 0.2 - 1e-6)
    else:
        r = np.random.uniform(0.1, 0.2 - 1e-6)
    ball_z = r * cos_ball_polar_angle
    r_xy = np.sqrt(r**2 - ball_z**2)
    # Sample from [-0.8 pi, 0.8 pi] to avoid positions in which the ball collides with the link.
    ball_azimuth = np.random.uniform(
        low=-0.8 * np.pi,
        high=0.8 * np.pi,
    )
    ball_pos_rel = np.stack(
        [r_xy * np.cos(ball_azimuth), r_xy * np.sin(ball_azimuth), ball_z],
        axis=-1,
    )
    return ball_pos_rel


def duplicate_visual_geoms(
    model_spec: mujoco.MjSpec, suffix: str, pos_offset: np.ndarray
):
    orig_robot_arm = model_spec.worldbody.find_child("robot_arm")
    quat = np.array(
        model_spec.resolve_orientation(
            model_spec.compiler.degree,
            model_spec.compiler.eulerseq,
            orig_robot_arm.alt,
        )
    )
    # No euler tag results in an all zero quaternion, which produces some weird orientation.
    if np.all(quat == 0.0):
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    robot_arm = model_spec.worldbody.add_body(
        name=orig_robot_arm.name + suffix,
        pos=orig_robot_arm.pos + pos_offset,
        quat=quat,
    )
    _duplicate_visual_geoms(model_spec, orig_robot_arm, robot_arm, suffix)

    orig_ball = model_spec.worldbody.find_child("ball")
    if orig_ball is not None:
        tmp = mujoco.MjSpec()
        ball = tmp.worldbody.add_body(
            name=orig_ball.name + suffix,
            pos=orig_ball.pos,
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        ball_frame = model_spec.worldbody.add_frame(
            pos=model_spec.worldbody.frames[0].pos + pos_offset
        )
        ball_frame.attach_body(ball)
        ball.add_joint(name="ball_freejoint" + suffix, type=mujoco.mjtJoint.mjJNT_FREE)
        for geom in orig_ball.geoms:
            ball.add_geom(
                name=geom.name + suffix,
                meshname=geom.meshname,
                type=geom.type,
                size=geom.size,
                pos=geom.pos,
                quat=geom.quat,
                rgba=geom.rgba,
            )
        for site in orig_ball.sites:
            ball.add_site(name=site.name + suffix, pos=site.pos)
        string = model_spec.add_tendon(
            name="string" + suffix,
            limited=True,
            range=[0, 0.2],
            width=0.002,
        )
        string.wrap_site("string_attach" + suffix)
        string.wrap_site("ball_attach" + suffix)


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
                rgba=geom.rgba,
            )
    for site in orig_parent_body.sites:
        parent_body.add_site(name=site.name + suffix, pos=site.pos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--slowdown", type=int, default=0)
    parser.add_argument("--keep-reset", action="store_true")
    parser.add_argument("--outname", type=str)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--camera-look-at", type=float, nargs=3, default=[0.2, 0.0, 1.8]
    )
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", type=float, default=1.8)
    parser.add_argument("--camera-azimuth", type=float, default=89.5)
    parser.add_argument("--robots-x", type=int, default=17)
    parser.add_argument("--robots-y", type=int, default=35)
    parser.add_argument("--resolution-x", type=int, default=4096)
    parser.add_argument("--resolution-y", type=int, default=2160)
    parser.add_argument("--reduce-framerate", type=int, default=2)
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

    model_spec = create_model_spec("racket" if task == "reacher" else "ball_in_cup")

    robot_dist = 1.2
    for x in range(-(args.robots_x // 2), args.robots_x // 2 + 1):
        for y in range(0, args.robots_y):
            if not (x == 0 and y == 0):
                duplicate_visual_geoms(
                    model_spec,
                    f"{x}_{y}",
                    np.array([x * robot_dist, y * robot_dist, 0]),
                )

    model = model_spec.compile()
    mj_data = mujoco.MjData(model)

    if args.outname is not None:
        # TODO: Dont hardcode the action repeat
        out_path = (Path(__file__).parents[2] / "out" / f"{args.outname}").with_suffix(
            ".mp4"
        )
        writer = imageio.get_writer(
            out_path,
            fps=round(
                1
                / (
                    model.opt.timestep
                    * (1 if data_from_o80_backend else 5)
                    * args.reduce_framerate
                )
            ),
        )
        print(f"Rendering to: {out_path}.")
        gl_context = mujoco.egl.GLContext(args.resolution_x, args.resolution_y)
        gl_context.make_current()
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        mujoco.mjr_resizeOffscreen(args.resolution_x, args.resolution_y, context)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
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

    scene = mujoco.MjvScene(model, maxgeom=100000)
    success_rates = []
    data_files = sorted(
        [p for p in args.path.iterdir() if p.suffix == ".npz"],
        key=lambda p: int(p.stem[len("ep_") :]),
    )

    reset_len = 3000
    num_episodes = 0
    successes = 0
    print("Started rendering!")
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

            num_robots = args.robots_x * args.robots_y
            if args.max_episodes is not None:
                positions = positions[: num_robots * (args.max_episodes - num_episodes)]

            num_splits = positions.shape[0] // num_robots
            positions_chunked = np.split(
                positions[: num_splits * num_robots, :: args.reduce_framerate],
                num_splits,
            )
            obj_datas_chunked = [{} for _ in range(num_splits)]
            for k, obj_data in data["object_datas"].item().items():
                pos_chunked = np.split(
                    np.array(obj_data.pos[: num_splits * num_robots]), num_splits
                )
                quat_chunked = np.split(
                    np.array(obj_data.quat[: num_splits * num_robots]), num_splits
                )
                linvel_chunked = np.split(
                    np.array(obj_data.linvel[: num_splits * num_robots]), num_splits
                )
                rotvel_chunked = np.split(
                    np.array(obj_data.rotvel[: num_splits * num_robots]), num_splits
                )
                timestamp_chunked = np.split(
                    np.array(obj_data.timestamp[: num_splits * num_robots]), num_splits
                )
                for s in range(num_splits):
                    obj_datas_chunked[s][k] = ObjData(
                        pos_chunked[s][:, :: args.reduce_framerate],
                        quat_chunked[s][:, :: args.reduce_framerate],
                        linvel_chunked[s][:, :: args.reduce_framerate],
                        rotvel_chunked[s][:, :: args.reduce_framerate],
                        timestamp_chunked[s][:, :: args.reduce_framerate],
                    )
            for i, (pos_traj, obj_data_traj) in enumerate(
                tqdm(
                    zip(positions_chunked, obj_datas_chunked),
                    disable=len(positions) <= 1,
                )
            ):
                if args.max_episodes is not None and num_episodes >= args.max_episodes:
                    break
                if task == "reacher":
                    mj_data.qpos[4:8] = goals[i]
                start_time = time.time()
                for j, (qpos, it, t) in enumerate(
                    tqdm(
                        zip(pos_traj.swapaxes(0, 1), iterations, timestamps),
                        total=pos_traj.shape[1],
                    )
                ):
                    for x in range(args.robots_x):
                        for y in range(args.robots_y):
                            r = x * args.robots_y + y
                            x_id = x - args.robots_x // 2
                            y_id = y

                            j1_body_name = (
                                f"J1{x_id}_{y_id}" if x_id != 0 or y_id != 0 else "J1"
                            )
                            robot_arm_id = mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_BODY,
                                j1_body_name,
                            )
                            robot_arm_qadr = model.jnt_qposadr[
                                model.body_jntadr[robot_arm_id]
                            ]

                            backend_infos = data["backend_infos"].item()
                            obj_data_idx = (
                                np.nanargmin(
                                    np.abs(backend_infos["timestamp"][:, i] - t)
                                )
                                if "timestamp" in backend_infos
                                else j
                            )
                            curr_obj_data = {
                                k: ObjData(
                                    v.pos[r, j],
                                    v.quat[r, j],
                                    v.linvel[r, j],
                                    v.rotvel[r, j],
                                    (
                                        v.timestamp[r, j]
                                        if v.timestamp is not None
                                        else None
                                    ),
                                )
                                for k, v in obj_data_traj.items()
                            }
                            mj_data.qpos[robot_arm_qadr : robot_arm_qadr + 4] = qpos[r]
                            if task == "ball_in_cup":
                                ball_pos = curr_obj_data["ball"].pos + np.array(
                                    [
                                        robot_dist * (x - args.robots_x // 2),
                                        robot_dist * y,
                                        0,
                                    ]
                                )
                                ball_quat = curr_obj_data["ball"].quat
                                ball_name = (
                                    f"ball{x_id}_{y_id}"
                                    if x_id != 0 or y_id != 0
                                    else "ball"
                                )
                                ball_id = mujoco.mj_name2id(
                                    model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    ball_name,
                                )
                                ball_qadr = model.jnt_dofadr[model.body_jntadr[ball_id]]
                                ball_joint_name = (
                                    f"ball_freejoint{x_id}_{y_id}"
                                    if x_id != 0 or y_id != 0
                                    else "ball_freejoint"
                                )
                                ball_qadr = model.jnt_qposadr[
                                    mujoco.mj_name2id(
                                        model,
                                        mujoco.mjtObj.mjOBJ_JOINT,
                                        ball_joint_name,
                                    )
                                ]
                                if not np.any(np.isnan(ball_pos)):
                                    mj_data.qpos[ball_qadr : ball_qadr + 3] = ball_pos
                                    mj_data.qpos[ball_qadr + 3 : ball_qadr + 7] = (
                                        ball_quat
                                    )
                            mujoco.mj_forward(model, mj_data)
                            if isinstance(env, PamyReacherEnv):
                                curr_success = np.linalg.norm(qpos - goals[i]) <= 0.1
                            else:
                                curr_success = env.is_success(curr_obj_data).item()
                            if j == len(pos_traj) - 1:
                                successes += curr_success
                            ball_geom_name = (
                                f"ball_geom{x_id}_{y_id}"
                                if x_id != 0 or y_id != 0
                                else "ball_geom"
                            )
                            for k in range(model.ngeom):
                                if model.geom(k).name.endswith("_goal_marker"):
                                    if (
                                        "o80_positions" in data
                                        and args.keep_reset
                                        and i < reset_len
                                    ):
                                        model.geom_rgba[k] = np.array(
                                            [0.0, 0.0, 0.0, 0.0]
                                        )
                                    elif curr_success:
                                        model.geom_rgba[k] = np.array(
                                            [0.0, 1.0, 0.0, 0.2]
                                        )
                                    else:
                                        model.geom_rgba[k] = np.array(
                                            [1.0, 0.0, 0.0, 0.2]
                                        )
                                elif model.geom(k).name == ball_geom_name:
                                    if curr_success:
                                        model.geom_rgba[k] = np.array(
                                            [0.0, 1.0, 0.0, 1.0]
                                        )
                                    else:
                                        model.geom_rgba[k] = np.array(
                                            [1.0, 0.0, 0.0, 1.0]
                                        )
                    if viewer is not None:
                        viewer.sync()
                        while (
                            time.time() - start_time
                            < (j + 1) * model.opt.timestep * args.slowdown
                        ):
                            pass
                    if writer is not None:
                        mujoco.mjv_updateScene(
                            model,
                            mj_data,
                            opt,
                            None,
                            cam,
                            mujoco.mjtCatBit.mjCAT_ALL,
                            scene,
                        )
                        rect = mujoco.MjrRect(
                            0, 0, args.resolution_x, args.resolution_y
                        )
                        mujoco.mjr_render(
                            rect,
                            scene,
                            context,
                        )
                        pixels = np.zeros(
                            (args.resolution_y, args.resolution_x, 3), dtype=np.uint8
                        )
                        mujoco.mjr_readPixels(
                            pixels,
                            None,
                            rect,
                            context,
                        )
                        frame = np.flipud(pixels)
                        if last_o80_iteration is not None:
                            # If frames are missing, repeat the current frame to keep the synchronization
                            for _ in range(it - last_o80_iteration):
                                writer.append_data(frame)
                        else:
                            writer.append_data(frame)
                        imageio.imwrite("out/debug_frame.png", frame)
                    last_o80_iteration = it
                num_episodes += 1
                print(f"Success rate: {successes / num_episodes * 100.0:.2f}%")
            last_iteration_last_episode = (
                data["o80_iterations"][-1] if "o80_iterations" in data else None
            )
            success_metric_name = (
                "target_dist/success_rate_norm_1steps_dist0.1"
                if task == "reacher"
                else "ball_in_cup/success_rate"
            )
            success_rates.append(data["episode_info"].item()[success_metric_name])
            if args.max_episodes is not None and num_episodes >= args.max_episodes:
                break
    print(f"success rate: {np.mean(success_rates) * 100.0:.2f}%")
    if viewer is not None:
        viewer.close()
