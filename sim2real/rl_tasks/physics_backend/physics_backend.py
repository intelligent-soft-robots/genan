import abc
import collections
import functools
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import jax.random
from flax import nnx
from jax.tree_util import register_pytree_node_class

from sim2real.actuator_network.models import ActuatorEnsemble
from sim2real.util.config_utils import get_an_training_config
from sim2real.util.encoder_simulation import encoder_quantization

logger = logging.getLogger(__name__)

ObjData = collections.namedtuple(
    "ObjData", ["pos", "quat", "linvel", "rotvel", "timestamp"]
)


def get_an_timestep(
    exp_name: str, actuator_net_name: str, rl_training: bool = False
) -> float:
    an_train_cfg = get_an_training_config(exp_name, actuator_net_name, rl_training)
    return 1.0 / an_train_cfg.target_frequency


@register_pytree_node_class
class Buffers:
    def __init__(
        self,
        positions: jnp.ndarray,
        controls: jnp.ndarray,
    ):
        self.positions = positions
        self.controls = controls

    @classmethod
    def init(
        cls,
        position: jnp.ndarray,
        prev_position: jnp.ndarray,
        control: jnp.ndarray,
        history_length: int,
    ) -> "Buffers":
        position_hist = jnp.concatenate(
            (prev_position[:, None].repeat(history_length, axis=1), position[:, None]),
            axis=1,
        )
        control_hist = control[:, None].repeat(history_length, axis=1)
        return cls(position_hist, control_hist)

    def advance_pos(
        self,
        position: jnp.ndarray,
    ) -> "Buffers":
        positions = self.positions.at[..., :-1, :].set(self.positions[..., 1:, :])
        positions = positions.at[..., -1, :].set(position.astype(jnp.float32))
        return Buffers(positions, self.controls)

    def advance_ctrl(self, control: jnp.ndarray) -> "Buffers":
        controls = self.controls.at[..., :-1, :].set(self.controls[..., 1:, :])
        controls = controls.at[..., -1, :].set(control)
        return Buffers(self.positions, controls)

    def tree_flatten(self):
        children = (self.positions, self.controls)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class PhysicsBackend(abc.ABC):
    def __init__(
        self,
        dt: float,
        history_length: int,
        simulate_encoder_quantization: bool,
        table_position: Sequence[float],
    ):
        self.dt = dt
        self.history_length = history_length
        self.simulate_encoder_quantization = simulate_encoder_quantization
        self.table_position = jnp.array(table_position, dtype=jnp.float32)
        self.num_dofs = 4

    @functools.partial(nnx.jit, static_argnames=("self", "obj_frames", "ensemble_size"))
    def reset(
        self,
        rng: jax.random.PRNGKey,
        control: jnp.ndarray,
        positions: Optional[jnp.ndarray],
        prev_positions: Optional[jnp.ndarray],
        obj_frames: Sequence[
            Tuple[str, Optional[str]]
        ] = (),  # List of (object name, frame)
        obj_poses: Sequence[
            Tuple[jnp.ndarray, jnp.ndarray]
        ] = (),  # List of (pos, quat)
        ensemble_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> Tuple["Buffers", Dict[str, ObjData], Any, Dict[str, Any]]:
        return self._reset_no_jit(
            rng,
            control,
            positions,
            prev_positions,
            obj_frames,
            obj_poses,
            ensemble_size,
            dry_run,
        )

    def _reset_no_jit(
        self,
        rng: jax.random.PRNGKey,
        control: jnp.ndarray,
        positions: Optional[jnp.ndarray],
        prev_positions: Optional[jnp.ndarray],
        obj_frames: Sequence[
            Tuple[str, Optional[str]]
        ] = (),  # List of (object name, frame)
        obj_poses: Sequence[
            Tuple[jnp.ndarray, jnp.ndarray]
        ] = (),  # List of (pos, quat)
        ensemble_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> Tuple["Buffers", Dict[str, ObjData], Any, Dict[str, Any]]:
        position, prev_position, obj_data, state, info = self._reset(
            rng,
            control,
            positions,
            prev_positions,
            obj_frames,
            obj_poses,
            ensemble_size=ensemble_size,
            dry_run=dry_run,
        )

        buffers = Buffers.init(
            position,
            prev_position,
            control,
            self.history_length,
        )
        return buffers, obj_data, state, info

    @functools.partial(nnx.jit, static_argnames=("self", "obj_frames", "ensemble_size"))
    def reset_with_history(
        self,
        rng: jax.random.PRNGKey,
        position_hist: jnp.ndarray,
        control_hist: jnp.ndarray,
        obj_frames: Sequence[Tuple[str, Optional[str]]] = (),
        obj_poses: Sequence[Tuple[jnp.ndarray, jnp.ndarray]] = (),
        ensemble_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> Tuple["Buffers", Dict[str, ObjData], Any, Dict[str, Any]]:
        return self._reset_with_history_no_jit(
            rng,
            position_hist,
            control_hist,
            obj_frames,
            obj_poses,
            ensemble_size,
            dry_run,
        )

    def _reset_with_history_no_jit(
        self,
        rng: jax.random.PRNGKey,
        position_hist: jnp.ndarray,
        control_hist: jnp.ndarray,
        obj_frames: Sequence[Tuple[str, Optional[str]]] = (),
        obj_poses: Sequence[Tuple[jnp.ndarray, jnp.ndarray]] = (),
        ensemble_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> Tuple["Buffers", Dict[str, ObjData], Any, Dict[str, Any]]:
        instances_shape = position_hist.shape[:-2]
        _, _, obj_data, state, info = self._reset(
            rng,
            control_hist[..., -1, :],
            position_hist[..., -1, :],
            position_hist[..., -2, :],
            obj_frames,
            obj_poses,
            ensemble_size=ensemble_size,
            dry_run=dry_run,
        )

        control_hist = jnp.concatenate(
            (jnp.zeros((*instances_shape, 1, self.num_dofs)), control_hist), axis=-2
        )
        buffers = Buffers(
            position_hist.astype(jnp.float32),
            control_hist.astype(jnp.float32),
        )
        return buffers, obj_data, state, info

    @functools.partial(
        nnx.jit,
        static_argnames=(
            "self",
            "enable_obj_tracking_randomization",
        ),
    )
    def step(
        self,
        rng: jax.random.PRNGKey,
        control: jnp.ndarray,
        buffers: "Buffers",
        state: Any,
        actuator_ensemble: Optional[ActuatorEnsemble] = None,
        dry_run: bool = False,
        enable_obj_tracking_randomization: bool = True,
    ) -> Tuple["Buffers", Any, Dict[str, ObjData], Dict[str, Any]]:
        return self._step_no_jit(
            rng,
            control,
            buffers,
            state,
            actuator_ensemble,
            dry_run,
            enable_obj_tracking_randomization,
        )

    def _step_no_jit(
        self,
        rng: jax.random.PRNGKey,
        control: jnp.ndarray,
        buffers: "Buffers",
        state: Any,
        actuator_ensemble: Optional[ActuatorEnsemble] = None,
        dry_run: bool = False,
        enable_obj_tracking_randomization: bool = True,
    ) -> Tuple["Buffers", Any, Dict[str, ObjData], Dict[str, Any]]:
        buffers = buffers.advance_ctrl(control)

        position, obj_data, state, info = self._step(
            rng,
            buffers,
            state,
            actuator_ensemble=actuator_ensemble,
            dry_run=dry_run,
            enable_obj_tracking_randomization=enable_obj_tracking_randomization,
        )

        buffers = buffers.advance_pos(position)
        return buffers, obj_data, state, info

    @functools.partial(
        nnx.jit,
        static_argnames=(
            "self",
            "history_reset",
        ),
    )
    def episode(
        self,
        rng: jax.random.PRNGKey,
        positions_reset: jnp.ndarray,
        controls_reset: jnp.ndarray,
        controls_episode: jnp.ndarray,
        history_reset: bool = True,
        actuator_ensemble: Optional[ActuatorEnsemble] = None,
        dry_run: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, ObjData], Dict[str, Any]]:
        if history_reset:
            buffers, obj_data, state, info = self._reset_with_history_no_jit(
                rng,
                positions_reset,
                controls_reset,
                ensemble_size=actuator_ensemble.ensemble_size,
                dry_run=dry_run,
            )
        else:
            buffers, obj_data, state, info = self._reset_no_jit(
                rng,
                controls_reset[..., -1, :],
                positions_reset[..., -1, :],
                positions_reset[..., -2, :],
                ensemble_size=actuator_ensemble.ensemble_size,
                dry_run=dry_run,
            )
        return self._steps(
            rng,
            controls_episode,
            buffers,
            obj_data,
            state,
            actuator_ensemble,
            dry_run,
        )

    def _steps(
        self,
        rng: jax.random.PRNGKey,
        control: jnp.ndarray,
        buffers: "Buffers",
        obj_data: Dict[str, ObjData],
        state: Any,
        actuator_ensemble: Optional[ActuatorEnsemble],
        dry_run: bool,
    ) -> Tuple[jnp.ndarray, Dict[str, ObjData], Dict[str, Any]]:
        @nnx.scan(in_axes=(-2, nnx.Carry), out_axes=(-2, -2, 0, nnx.Carry))
        def _step_fun(control, carry):
            (_rng, _buffers, _obj_data, _state) = carry

            rng_step, _rng = jax.random.split(_rng)
            _buffers, _obj_data, _state, _info = self._step_no_jit(
                rng_step,
                control,
                _buffers,
                _state,
                actuator_ensemble,
                dry_run,
            )
            return (
                _buffers.positions[..., -1, :],
                _obj_data,
                _info,
                (_rng, _buffers, _obj_data, _state),
            )

        positions, obj_data, info, _ = _step_fun(
            control,
            (rng, buffers, obj_data, state),
        )
        if "actuator_network" in info:
            an_info = info["actuator_network"]
            an_info.update(
                {
                    "input": (
                        jnp.moveaxis(an_info["input"], 0, -2)
                        if "input" in an_info
                        else None
                    ),
                    "output": (
                        jnp.moveaxis(an_info["output"], 0, -2)
                        if "output" in an_info
                        else None
                    ),
                    "model_indices": (
                        jnp.moveaxis(an_info["model_indices"], 0, -1)
                        if "model_indices" in an_info
                        else None
                    ),
                }
            )
        return positions, obj_data, info

    def close(self) -> None:
        return

    @abc.abstractmethod
    def _reset(
        self,
        rng: jax.random.PRNGKey,
        control: jnp.ndarray,
        position: Optional[jnp.ndarray],
        prev_position: Optional[jnp.ndarray],
        obj_frames: Sequence[Tuple[str, Optional[str]]],
        obj_poses: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
        ensemble_size: Optional[int],
        dry_run: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, ObjData], Any, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def _step(
        self,
        rng: jax.random.PRNGKey,
        buffers: Buffers,
        state: Any,
        actuator_ensemble: Optional[ActuatorEnsemble],
        dry_run: bool,
        enable_obj_tracking_randomization: bool,
    ) -> Tuple[jnp.ndarray, Dict[str, ObjData], Any, Dict[str, Any]]:
        pass

    def process_buffers_for_obs(self, buffers: Buffers) -> Buffers:
        if self.simulate_encoder_quantization:
            return Buffers(
                encoder_quantization(buffers.positions),
                buffers.controls,
            )
        else:
            return buffers
