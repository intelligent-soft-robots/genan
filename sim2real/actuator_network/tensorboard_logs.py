from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter

from sim2real.actuator_network.models import ActuatorEnsemble


def mean_squared_error(
    pred: jax.typing.ArrayLike, gt: jax.typing.ArrayLike
) -> jax.typing.ArrayLike:
    return ((gt[None] - pred) ** 2).mean(axis=1)


def mean_abs_error(
    pred: jax.typing.ArrayLike, gt: jax.typing.ArrayLike
) -> jax.typing.ArrayLike:
    return jnp.abs(gt[None] - pred).mean(axis=1)


def normalized_mean_abs_error(
    pred: jax.typing.ArrayLike, gt: jax.typing.ArrayLike
) -> jax.typing.ArrayLike:
    return jnp.abs(gt[None] - pred).mean(axis=1) / jnp.abs(gt[None]).mean(axis=1)


def mean_abs_deviation(pred: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    return jnp.abs(pred.mean(axis=0, keepdims=True) - pred).mean(axis=(0, 1))


SIM_EVAL_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def log_sim_errors(
    summary_writer: SummaryWriter,
    sim_eval_steps: List[int],
    errors_per_split: List[Tuple[str, Optional[jax.typing.ArrayLike], str]],
    sim_timestep: float,
    epoch: int,
) -> None:
    """Write the sim_error_deg_* scalars for a list of (tag, errors, suffix) triples."""
    subsampling_factor = round(sim_timestep / 0.001)
    for name, errors, suffix in errors_per_split:
        if errors is None:
            continue
        for step in sim_eval_steps:
            if errors.shape[1] >= step:
                summary_writer.add_scalar(
                    f"sim_error_deg_avg{suffix}/{name}/step{step * subsampling_factor:04d}",
                    jnp.abs(errors[:, step - 1]).mean().item() * 180.0 / np.pi,
                    epoch,
                )
        for dof in range(4):
            for step in sim_eval_steps:
                if errors.shape[1] >= step:
                    summary_writer.add_scalar(
                        f"sim_error_deg_joints{suffix}/{name}/step{step * subsampling_factor:04d}/dof{dof}",
                        jnp.abs(errors[:, step - 1, dof]).mean().item() * 180.0 / np.pi,
                        epoch,
                    )


class TensorboardLogs:
    def __init__(
        self,
        outpath: Path,
        train_data_in: jnp.ndarray,
        train_data_torques: jnp.ndarray,
        val_data_in: jnp.ndarray,
        val_data_torques: jnp.ndarray,
        extra_eval_data_in: Optional[jnp.ndarray],
        extra_eval_data_torques: Optional[jnp.ndarray],
        num_samples: int = 10000,
    ):
        self._accuracy_metrics = {
            "mean_squared_error": mean_squared_error,
            "mean_abs_error": mean_abs_error,
            "normalized_mean_abs_error": normalized_mean_abs_error,
        }
        self._disagreement_metrics = {
            "std": lambda x: x.std(axis=0).mean(axis=0),
            "mean_abs_deviation": mean_abs_deviation,
        }
        self.sim_eval_steps = SIM_EVAL_STEPS
        self._summary_writer = SummaryWriter(str(outpath))
        train_indices = np.random.choice(
            np.arange(train_data_in.shape[0], dtype=int),
            num_samples,
            replace=False,
        )
        val_indices = np.random.choice(
            np.arange(val_data_in.shape[0], dtype=int),
            min(num_samples, val_data_in.shape[0]),
            replace=False,
        )
        self._train_data_in = train_data_in[train_indices]
        self._train_data_torques = train_data_torques[train_indices]
        self._val_data_in = val_data_in[val_indices]
        self._val_data_torques = val_data_torques[val_indices]
        if extra_eval_data_in is not None:
            extra_eval_indices = np.random.choice(
                np.arange(extra_eval_data_in.shape[0], dtype=int),
                num_samples,
                replace=False,
            )
            self._extra_eval_data_in = extra_eval_data_in[extra_eval_indices]
            self._extra_eval_data_torques = extra_eval_data_torques[extra_eval_indices]
        else:
            self._extra_eval_data_in = None
            self._extra_eval_data_torques = None
        self._add_custom_scalars()

    def update_metrics(
        self,
        model: ActuatorEnsemble,
        epoch: int,
    ):
        for data_name, data_in, data_torques in [
            ("train", self._train_data_in, self._train_data_torques),
            ("val", self._val_data_in, self._val_data_torques),
            ("extra", self._extra_eval_data_in, self._extra_eval_data_torques),
        ]:
            if data_in is not None:
                accuracy_metrics, disagreement_metrics = self._compute_metrics(
                    model, data_in, data_torques
                )
                for metric_name, value in accuracy_metrics.items():
                    for dof in range(4):
                        self._summary_writer.add_scalar(
                            f"accuracy_joints/{data_name}/{metric_name}/dof{dof}",
                            value[:, dof].mean().item(),
                            epoch,
                        )
                    for net in range(model.ensemble_size):
                        self._summary_writer.add_scalar(
                            f"accuracy_networks/{data_name}/{metric_name}/network{net}",
                            value[net].mean().item(),
                            epoch,
                        )
                    self._summary_writer.add_scalar(
                        f"accuracy_avg/{data_name}/{metric_name}",
                        value.mean().item(),
                        epoch,
                    )
                for metric_name, value in disagreement_metrics.items():
                    for dof in range(4):
                        self._summary_writer.add_scalar(
                            f"disagreement_joints/{data_name}/{metric_name}/dof{dof}",
                            value[dof].mean().item(),
                            epoch,
                        )
                    self._summary_writer.add_scalar(
                        f"disagreement_avg/{data_name}/{metric_name}",
                        value.mean().item(),
                        epoch,
                    )

    def update_loss(
        self,
        train_losses: Optional[np.ndarray],
        val_losses: Optional[np.ndarray],
        extra_eval_losses: Optional[np.ndarray],
        epoch: int,
    ) -> None:
        for name, losses in [
            ("train", train_losses),
            ("val", val_losses),
            ("extra", extra_eval_losses),
        ]:
            if losses is not None:
                self._summary_writer.add_scalar(
                    f"loss/{name}", losses.mean().item(), epoch
                )
                for i, loss in enumerate(losses):
                    self._summary_writer.add_scalar(
                        f"loss_networks/{name}/network{i}", loss.item(), epoch
                    )

    def update_sim_error(
        self,
        train_pos_errors: jax.typing.ArrayLike,
        val_pos_errors: jax.typing.ArrayLike,
        extra_eval_pos_errors: Optional[jax.typing.ArrayLike],
        train_pos_errors_quant: jax.typing.ArrayLike,
        val_pos_errors_quant: jax.typing.ArrayLike,
        extra_eval_pos_errors_quant: Optional[jax.typing.ArrayLike],
        train_pos_errors_filt: jax.typing.ArrayLike,
        val_pos_errors_filt: jax.typing.ArrayLike,
        extra_eval_pos_errors_filt: Optional[jax.typing.ArrayLike],
        sim_timestep: float,
        epoch: int,
    ) -> None:
        log_sim_errors(
            self._summary_writer,
            self.sim_eval_steps,
            [
                ("train/position", train_pos_errors, ""),
                ("val/position", val_pos_errors, ""),
                ("extra/position", extra_eval_pos_errors, ""),
                ("train/position", train_pos_errors_quant, "_quant"),
                ("val/position", val_pos_errors_quant, "_quant"),
                ("extra/position", extra_eval_pos_errors_quant, "_quant"),
                ("train/position", train_pos_errors_filt, "_filt"),
                ("val/position", val_pos_errors_filt, "_filt"),
                ("extra/position", extra_eval_pos_errors_filt, "_filt"),
            ],
            sim_timestep,
            epoch,
        )

    def _add_custom_scalars(self) -> None:
        common_sub = [1, 2, 5, 10]
        all_sim_eval_steps = sorted(
            list(set([s * f for s in self.sim_eval_steps for f in common_sub]))
        )
        dataset_names = ["train", "val", "extra"]
        if self._extra_eval_data_in is not None:
            dataset_names.append("extra")
        custom_metrics = {
            "accuracy_avg": {
                metric_name: [
                    "Multiline",
                    [
                        f"accuracy_avg/{data_name}/{metric_name}"
                        for data_name in dataset_names
                    ],
                ]
                for metric_name in self._accuracy_metrics.keys()
            },
            "accuracy_joints": {
                f"{metric_name}/{data_name}": [
                    "Multiline",
                    [
                        f"accuracy_joints/{data_name}/{metric_name}/dof{dof}"
                        for dof in range(4)
                    ],
                ]
                for metric_name in self._accuracy_metrics.keys()
                for data_name in dataset_names
            },
            "disagreement_avg": {
                f"{metric_name}_{metric_name_suffix}": [
                    "Multiline",
                    [
                        f"disagreement_avg/{data_name}/{metric_name}_{metric_name_suffix}"
                        for data_name in dataset_names
                    ],
                ]
                for metric_name in self._disagreement_metrics.keys()
                for metric_name_suffix in ["not_destandardized", "destandardized"]
            },
            "disagreement_joints": {
                f"{metric_name}_{metric_name_suffix}/{data_name}": [
                    "Multiline",
                    [
                        f"disagreement_joints/{data_name}/{metric_name}_{metric_name_suffix}/dof{dof}"
                        for dof in range(4)
                    ],
                ]
                for metric_name in self._disagreement_metrics.keys()
                for metric_name_suffix in ["not_destandardized", "destandardized"]
                for data_name in dataset_names
            },
            "sim_error_deg_avg": {
                f"position/step{step:04d}": [
                    "Multiline",
                    [
                        f"sim_error_deg_avg/{data_name}/position/step{step:04d}"
                        for data_name in dataset_names
                    ],
                ]
                for step in all_sim_eval_steps
            },
            "sim_error_deg_joints": {
                f"position/{data_name}/step{step:04d}": [
                    "Multiline",
                    [
                        f"sim_error_deg_joints/{data_name}/position/step{step:04d}/dof{dof}"
                        for dof in range(4)
                    ],
                ]
                for step in all_sim_eval_steps
                for data_name in dataset_names
            },
        }
        self._summary_writer.add_custom_scalars(custom_metrics)

    def _compute_metrics(
        self, model: ActuatorEnsemble, data_in: jnp.ndarray, data_torques: jnp.ndarray
    ) -> Tuple[Dict[str, jax.typing.ArrayLike], Dict[str, jax.typing.ArrayLike]]:
        model.eval()
        # TODO: Set the rng key properly
        torques_pred, torques_pred_raw, _ = model.forward_all(
            data_in,
            jax.random.PRNGKey(42),
        )
        torques_gt = data_torques

        accuracy_metrics = {}
        for name, metric in self._accuracy_metrics.items():
            accuracy_metrics[name] = metric(
                torques_pred,
                torques_gt,
            )

        if model.ensemble_size > 1:
            disagreement_metrics = {}
            for name, metric in self._disagreement_metrics.items():
                disagreement_metrics[f"{name}_not_destandardized"] = metric(
                    torques_pred_raw
                )
                disagreement_metrics[f"{name}_destandardized"] = metric(torques_pred)
        else:
            disagreement_metrics = {}

        return accuracy_metrics, disagreement_metrics
