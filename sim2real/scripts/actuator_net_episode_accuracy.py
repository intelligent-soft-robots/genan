import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List

import jax.numpy as jnp
import numpy as np
import omegaconf
import scipy.stats
from flax import nnx
from tqdm import tqdm

from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.models import (
    restore_actuator_net_checkpoint,
    restore_actuator_net_rl_checkpoint,
)
from sim2real.actuator_network.simulation_accuracy_evaluator import (
    SimulationAccuracyEvaluator,
)
from sim2real.data_collection.cocontraction import Cocontraction
from sim2real.data_collection.trajectory_dataset import TrajectoryDataset
from sim2real.data_processing.process_data import process_data
from sim2real.rl_tasks.physics_backend.physics_backend import get_an_timestep
from sim2real.scripts.train_actuator_network import split_datasets
from sim2real.util.config_utils import get_an_training_config

logger = logging.getLogger(__name__)


def evaluate_model(
    actuator_net_exp_name: str,
    actuator_net_name: str,
    rl_training: bool,
    extra_data_names: List[str],
    eval_log_names: List[str],
    warmup_trajs: int,
):
    max_hist_len = 20  # TODO: Dont hardcode
    actuator_net_train_cfg = get_an_training_config(
        actuator_net_exp_name, actuator_net_name, rl_training
    )
    if rl_training:
        actuator_ensemble = restore_actuator_net_rl_checkpoint(
            actuator_net_exp_name, actuator_net_name
        )
    else:
        actuator_ensemble = restore_actuator_net_checkpoint(
            actuator_net_exp_name, actuator_net_name
        )
    dt = 1 / actuator_net_train_cfg.target_frequency

    datasets = []
    for data_exp_name in [actuator_net_exp_name] + extra_data_names:
        data_exp_path = Path(__file__).parents[2] / "exps" / data_exp_name
        raw_data_path = data_exp_path / "raw_data.pkl"
        with raw_data_path.open("rb") as f:
            raw_data = pickle.load(f)

        if data_exp_name == actuator_net_exp_name:
            curr_warmup_trajs = actuator_net_train_cfg.get("num_warmup_trajectories", 0)
        else:
            curr_warmup_trajs = warmup_trajs
        raw_data = raw_data[curr_warmup_trajs:]
        # Throw away the first couple of datapoints of each episode to make sure that actuator nets are always evaluated
        # on the same samples, independent of the history length.
        steps_to_remove = 2 * (
            max_hist_len - actuator_net_train_cfg["actuator_net"]["num_history_states"]
        )
        raw_data = raw_data[
            :,
            steps_to_remove:,
        ]
        datasets.append((data_exp_name, raw_data))

    traj_len = datasets[0][1].positions.shape[1]
    an_exp_path = Path(__file__).parents[2] / "exps" / actuator_net_exp_name
    data_collection_config = omegaconf.OmegaConf.load(
        an_exp_path / "collect_data_config" / "config.yaml"
    )
    with (an_exp_path / "robot_config.json").open("r") as f:
        robot_config = json.load(f)

    cocontraction = Cocontraction(
        robot_config["min_pressures_ago"],
        robot_config["max_pressures_ago"],
        robot_config["min_pressures_antago"],
        robot_config["max_pressures_antago"],
        data_collection_config.ndp,
    )
    for eval_log_name in eval_log_names:
        eval_log_path = Path(__file__).parents[2] / "out" / eval_log_name
        positions = []
        controls = []
        desired_pressures = []
        observed_pressures = []
        timestamps = []
        for p in eval_log_path.iterdir():
            if p.suffix == ".npz":
                data = np.load(p, allow_pickle=True)
                t = (
                    data["o80_timestamps"] * 1e-6
                    if data["o80_timestamps"].dtype == int
                    else data["o80_timestamps"]
                )
                if (t[1:] - t[:-1]).max() > 0.0015:
                    logger.warning(
                        f"Found invalid timestep: {(t[1:] - t[:-1]).max():.4f}. Skipping trajectory..."
                    )
                    continue
                first_step = np.argwhere(
                    np.isclose(t, data["backend_infos"].item()["timestamp"][0])
                ).item()
                pos = data["o80_positions"][first_step : first_step + traj_len]
                p_des = data["o80_desired_pressures"][
                    first_step : first_step + traj_len - 1
                ]
                u = cocontraction.pressures_to_control(p_des[:, :, 0], p_des[:, :, 1])
                positions.append(pos)
                controls.append(u)
                desired_pressures.append(p_des)
                observed_pressures.append(
                    data["o80_observed_pressures"][first_step : first_step + traj_len]
                )
                timestamps.append(t[first_step : first_step + traj_len])
        positions = np.stack(positions, axis=0)
        controls = np.stack(controls, axis=0)
        desired_pressures = np.stack(desired_pressures, axis=0)
        observed_pressures = np.stack(observed_pressures, axis=0)
        timestamps = np.stack(timestamps, axis=0)
        datasets.append(
            (
                eval_log_name,
                TrajectoryDataset.from_data(
                    positions,
                    desired_pressures,
                    observed_pressures,
                    controls,
                    timestamps,
                ),
            )
        )

    abs_errors = {}
    disag = {}
    for data_name, raw_data in datasets:
        positions, positions_raw, observed_pressures, controls, torques = process_data(
            raw_data,
            actuator_net_train_cfg["end_effector"],
            (
                actuator_net_train_cfg.filter
                if "filter" in actuator_net_train_cfg
                else None
            ),
            actuator_net_train_cfg.target_frequency,
            actuator_net_train_cfg.get("actuation_delay", 0),
            plot_visualizations=False,
        )

        dataset = Dataset(
            positions, positions_raw, observed_pressures, controls, torques
        )

        train_data, val_data, train_data_stats = split_datasets(
            actuator_net_train_cfg,
            dataset,
            dt,
            train_split=0.8 if data_name == actuator_net_exp_name else 1.0,
        )

        an_timestep = get_an_timestep(
            actuator_net_exp_name, actuator_net_name, rl_training
        )
        rngs = nnx.Rngs(backend=42)
        num_train_episodes = (
            2000 if data_name == actuator_net_exp_name else 1000 - warmup_trajs
        )
        num_val_episodes = 500 if data_name == actuator_net_exp_name else 0
        sim_accuracy_evaluator = SimulationAccuracyEvaluator(
            rngs,
            train_data,
            val_data,
            None,
            an_timestep,
            actuator_net_train_cfg.actuator_net,
            actuator_net_train_cfg.end_effector,
            num_train_episodes=num_train_episodes,
            num_val_episodes=num_val_episodes,
            num_steps_per_episode=1000,
        )
        (
            (train_err_pos_an, _, _, train_disag),
            (val_err_pos_an, _, _, val_disag),
            _,
        ) = sim_accuracy_evaluator.evaluate(actuator_ensemble)

        # assert torch.all(val_gt_pos_data == val_gt_pos_an)
        # assert torch.all(train_gt_pos_data == train_gt_pos_an)

        if data_name == actuator_net_exp_name:
            abs_errors["train"] = np.array(jnp.abs(train_err_pos_an))
            disag["train"] = np.array(train_disag)
            abs_errors["val"] = np.array(jnp.abs(val_err_pos_an))
            disag["val"] = np.array(val_disag)
        else:
            abs_errors["test"] = np.array(jnp.abs(train_err_pos_an))
            disag["test"] = np.array(train_disag)

    return (
        abs_errors,
        disag,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--an", type=str, nargs="*", default=[])
    parser.add_argument("--an-rl", type=str, nargs="*", default=[])
    parser.add_argument("--extra-data-names", type=str, nargs="*", default=[])
    parser.add_argument("--eval-log-names", type=str, nargs="*", default=[])
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--outname", type=str, required=True)
    parser.add_argument("--warmup-trajs", type=str, default=200)
    args = parser.parse_args()

    # Not supported right now
    assert len(args.extra_data_names) <= 1
    assert len(args.eval_log_names) == 0

    acuator_net_configs = [(name, False) for name in args.an] + [
        (name, True) for name in args.an_rl
    ]

    def vector_mean(*inp, axis):
        return [np.mean(i, axis=axis) for i in inp]

    out_path = (Path(__file__).parents[2] / "out" / args.outname).with_suffix(".pkl")
    if args.append:
        with open(out_path, "rb") as f:
            results = pickle.load(f)
        assert results["step"] == args.step
    else:
        results = {"step": args.step}
    for an_name, rl_training in tqdm(acuator_net_configs):
        if (
            args.append
            and an_name in results
            and "abs_error" in results[an_name]
            and "disag" in results[an_name]
        ):
            continue
        else:
            try:
                abs_errors, disags = evaluate_model(
                    args.exp_name,
                    an_name,
                    rl_training,
                    args.extra_data_names,
                    args.eval_log_names,
                    args.warmup_trajs,
                )
            except Exception as e:
                logger.warning(f"{type(e).__name__}: {e}")
                continue

            print(f"Evaluation of {an_name} done")
            results[an_name] = {}
            for metric_name, eval_results in [
                ("abs_error", abs_errors),
                ("disag", disags),
            ]:
                results[an_name][metric_name] = {}
                for d, res in eval_results.items():
                    res = np.mean(res, axis=-1)[..., :: args.step]
                    cis = scipy.stats.bootstrap(res.T, vector_mean).confidence_interval
                    results[an_name][metric_name][d] = (
                        res.mean(axis=0),
                        cis.low,
                        cis.high,
                    )

                if args.outname is not None:
                    outpath = (
                        Path(__file__).parents[2] / "out" / args.outname
                    ).with_suffix(".pkl")
                    with open(outpath, "wb") as f:
                        pickle.dump(results, f)
