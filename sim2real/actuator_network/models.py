import functools
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import flax.nnx as nnx
import gymnasium
import hydra
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import omegaconf
import orbax.checkpoint as ocp
import skrl.agents.jax
from jax.sharding import SingleDeviceSharding
from mujoco import mjx

from sim2real.actuator_network.pack_an_input import unpack_an_input
from sim2real.actuator_network.standardize import de_standardize, standardize
from sim2real.rl_tasks.agent import load_agent
from sim2real.util.config_utils import (
    get_an_input_size,
    get_an_input_size_from_cfg,
    get_an_training_config,
)
from sim2real.util.encoder_simulation import (
    ENCODER_RESOLUTION,
    ENCODER_RESOLUTION_NP,
    encoder_quantization,
)
from sim2real.util.model import create_model_spec

NUM_DOFS = 4


class FrozenParam(nnx.Variable):
    """Non-trainable parameter for randomized prior functions (Osband et al., 2018).

    Using a distinct Variable type ensures nnx.value_and_grad (which targets
    nnx.Param by default) never computes gradients for prior weights.
    """

    pass


def _make_layer_stack(
    rngs: nnx.Rngs,
    input_dim: int,
    output_dim: int,
    num_neurons: int,
    num_hidden_layers: int,
    use_batch_norm: bool,
) -> Tuple[nnx.Module, ...]:
    layers = []
    layers.append(nnx.Linear(input_dim, num_neurons, rngs=rngs))
    for i in range(num_hidden_layers):
        if use_batch_norm:
            layers.append(nnx.BatchNorm(num_neurons, rngs=rngs))
        layers.append(
            nnx.Linear(
                num_neurons,
                num_neurons,
                rngs=rngs,
            )
        )
    if use_batch_norm:
        layers.append(
            nnx.BatchNorm(
                num_neurons,
                rngs=rngs,
            )
        )
    layers.append(nnx.Linear(num_neurons, output_dim, rngs=rngs))
    return tuple(layers)


def _freeze_layer_params(layer: nnx.Module) -> None:
    """Replace nnx.Param with FrozenParam in a single layer in-place."""
    if isinstance(layer, nnx.Linear):
        layer.kernel = FrozenParam(layer.kernel.value)
        layer.bias = FrozenParam(layer.bias.value)
    elif isinstance(layer, nnx.BatchNorm):
        layer.scale = FrozenParam(layer.scale.value)
        layer.bias = FrozenParam(layer.bias.value)


# TODO: Clean up the code (fix method names, unify ActuatorEnsemble with subclasses, remove duplicate code)
class ActuatorEnsemble(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        ensemble_size: int,
        num_history_states: int,
        step_size: int,
        pressure_input: bool,
        velocity_input: bool,
        position_history: bool,
        velocity_history: bool,
        control_history: bool,
        delta_history: bool,
        num_neurons: int = 512,
        num_hidden_layers: int = 2,
        output_dim: int = 4,
        activation_fn: str = "tanh",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        use_randomized_prior: bool = False,
        prior_scale: float = 1.0,
        input_mean: Optional[jax.Array] = None,
        input_std: Optional[jax.Array] = None,
        output_mean: Optional[jax.Array] = None,
        output_std: Optional[jax.Array] = None,
        **kwargs,
    ):
        assert dropout_rate == 0.0, "Dropout not implemented yet"

        input_dim = get_an_input_size(
            num_history_states,
            velocity_input,
            pressure_input,
            position_history,
            velocity_history,
            control_history,
        )

        self.layers = self._build_layers(
            rngs,
            ensemble_size,
            input_dim,
            output_dim,
            num_neurons,
            num_hidden_layers,
            use_batch_norm,
        )

        self.use_randomized_prior = use_randomized_prior
        if use_randomized_prior:
            # Second call to _build_layers advances the RNG state, giving different
            # random initializations for the prior vs. the main network.
            self.prior_layers = self._build_layers(
                rngs,
                ensemble_size,
                input_dim,
                output_dim,
                num_neurons,
                num_hidden_layers,
                use_batch_norm,
            )
            for layer in self.prior_layers:
                _freeze_layer_params(layer)
            self.prior_scale = prior_scale

        self.activation_fn = activation_fn
        if activation_fn.lower() == "tanh":
            self.activation_fn = nnx.tanh
        elif activation_fn.lower() == "relu":
            self.activation_fn = nnx.relu
        elif activation_fn.lower() == "leaky_relu":
            self.activation_fn = nnx.leaky_relu
        elif activation_fn.lower() == "elu":
            self.activation_fn = nnx.elu
        else:
            raise ValueError(f"Invalid activation function {activation_fn}.")

        self.input_mean = nnx.Variable(
            (
                input_mean.astype(jnp.float32)
                if input_mean is not None
                else jnp.zeros(input_dim, dtype=jnp.float32)
            ),
            collection="constants",
        )
        self.input_std = nnx.Variable(
            (
                input_std.astype(jnp.float32)
                if input_std is not None
                else jnp.ones(input_dim, dtype=jnp.float32)
            ),
            collection="constants",
        )
        self.output_mean = nnx.Variable(
            (
                output_mean.astype(jnp.float32)
                if output_mean is not None
                else jnp.zeros(output_dim, dtype=jnp.float32)
            ),
            collection="constants",
        )
        self.output_std = nnx.Variable(
            (
                output_std.astype(jnp.float32)
                if output_std is not None
                else jnp.ones(output_dim, dtype=jnp.float32)
            ),
            collection="constants",
        )

        # Verifications
        assert self.input_mean.shape == (
            input_dim,
        ), f"Invalid input_mean.shape, expected ({input_dim},), got {input_mean.shape}."
        assert self.input_std.shape == (input_dim,)
        assert self.output_mean.shape == (output_dim,)
        assert self.output_std.shape == (output_dim,)

        self.pressure_input = pressure_input
        self.velocity_input = velocity_input
        self.position_history = position_history
        self.velocity_history = velocity_history
        self.control_history = control_history
        self.delta_history = delta_history

        self.num_history_states = num_history_states
        self.step_size = step_size
        self.ensemble_size = ensemble_size

    def _build_layers(
        self,
        rngs: nnx.Rngs,
        ensemble_size: int,
        input_dim: int,
        output_dim: int,
        num_neurons: int,
        num_hidden_layers: int,
        use_batch_norm: bool,
    ) -> Tuple[nnx.Module, ...]:
        @nnx.split_rngs(splits=ensemble_size)
        @nnx.vmap
        def make_model(_rngs: nnx.Rngs) -> Tuple[nnx.Module, ...]:
            return _make_layer_stack(
                _rngs,
                input_dim,
                output_dim,
                num_neurons,
                num_hidden_layers,
                use_batch_norm,
            )

        return make_model(rngs)

    def _apply_layers(
        self, layers: Tuple[nnx.Module, ...], x_std: jax.Array
    ) -> jax.Array:
        layers_graphdef, layers_state = nnx.split(layers)

        @jax.vmap
        def net_forward(_layers_state, _x):
            merged_layers = nnx.merge(layers_graphdef, _layers_state)
            _x = merged_layers[0](_x)
            for layer in merged_layers[1:]:
                _x = layer(self.activation_fn(_x))
            return _x

        return net_forward(layers_state, x_std)

    def __call__(
        self,
        x: jax.Array,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, jax.Array, Any]:
        x = self.forward(x)
        x_destd = de_standardize(x, self.output_mean.value, self.output_std.value)
        return x_destd, x, None

    def forward(self, x: jax.Array) -> jax.Array:
        x_std = standardize(x, self.input_mean.value, self.input_std.value)
        result = self._apply_layers(self.layers, x_std)

        if self.use_randomized_prior:
            result = result + self.prior_scale * self._apply_layers(
                self.prior_layers, x_std
            )

        return result

    def forward_all(
        self,
        x: jax.Array,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, jax.Array, List[jax.Array]]:
        return self.forward_all_no_jit(x, rng)

    def forward_all_no_jit(
        self, x: jax.Array, rng: jax.random.PRNGKey
    ) -> Tuple[jax.Array, jax.Array, Any]:
        x = x[None].repeat(self.ensemble_size, axis=0)
        return self(x, rng)


class PerDoFActuatorEnsemble(ActuatorEnsemble):
    """Ensemble with a separate network per DoF instead of one network for all DoFs.

    Each network predicts the torque of a single DoF. By default it only receives the
    inputs belonging to that DoF, with all_dofs_input=True it receives the full input.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        ensemble_size: int,
        num_history_states: int,
        step_size: int,
        pressure_input: bool,
        velocity_input: bool,
        position_history: bool,
        velocity_history: bool,
        control_history: bool,
        delta_history: bool,
        all_dofs_input: bool = False,
        output_dim: int = NUM_DOFS,
        **kwargs,
    ):
        assert (
            output_dim % NUM_DOFS == 0
        ), f"output_dim must be divisible by {NUM_DOFS}, got {output_dim}."
        self.all_dofs_input = all_dofs_input

        super().__init__(
            rngs,
            ensemble_size,
            num_history_states,
            step_size,
            pressure_input,
            velocity_input,
            position_history,
            velocity_history,
            control_history,
            delta_history,
            output_dim=output_dim,
            **kwargs,
        )

    def _build_layers(
        self,
        rngs: nnx.Rngs,
        ensemble_size: int,
        input_dim: int,
        output_dim: int,
        num_neurons: int,
        num_hidden_layers: int,
        use_batch_norm: bool,
    ) -> Tuple[nnx.Module, ...]:
        assert (
            input_dim % NUM_DOFS == 0
        ), f"input_dim must be divisible by {NUM_DOFS}, got {input_dim}."
        input_dim_per_dof = input_dim if self.all_dofs_input else input_dim // NUM_DOFS
        output_dim_per_dof = output_dim // NUM_DOFS

        # The parameters of the resulting layers have shape [ensemble_size, num_dofs, ...]
        @nnx.split_rngs(splits=ensemble_size)
        @nnx.vmap
        @nnx.split_rngs(splits=NUM_DOFS)
        @nnx.vmap
        def make_model(_rngs: nnx.Rngs) -> Tuple[nnx.Module, ...]:
            return _make_layer_stack(
                _rngs,
                input_dim_per_dof,
                output_dim_per_dof,
                num_neurons,
                num_hidden_layers,
                use_batch_norm,
            )

        return make_model(rngs)

    def _apply_layers(
        self, layers: Tuple[nnx.Module, ...], x_std: jax.Array
    ) -> jax.Array:
        layers_graphdef, layers_state = nnx.split(layers)

        def dof_forward(_layers_state, _x):
            merged_layers = nnx.merge(layers_graphdef, _layers_state)
            _x = merged_layers[0](_x)
            for layer in merged_layers[1:]:
                _x = layer(self.activation_fn(_x))
            return _x

        if self.all_dofs_input:
            # Every DoF network sees the whole input
            x_per_dof = x_std
            dof_in_axes = (0, None)
        else:
            # The input is packed as [..., num_dofs, entries_per_dof]
            x_per_dof = jnp.moveaxis(
                x_std.reshape(*x_std.shape[:-1], NUM_DOFS, -1), -2, 1
            )
            dof_in_axes = (0, 0)

        net_forward = jax.vmap(jax.vmap(dof_forward, in_axes=dof_in_axes, out_axes=-2))
        result = net_forward(layers_state, x_per_dof)
        return result.reshape(*result.shape[:-2], -1)


def save_actuator_net_checkpoint(
    actuator_ensemble: ActuatorEnsemble, checkpoint_path: Path
) -> None:
    _, state = nnx.split(actuator_ensemble)
    save_actuator_net_state_checkpoint(state, checkpoint_path)


def save_actuator_net_state_checkpoint(
    state: nnx.statelib.State, checkpoint_path: Path
) -> None:
    checkpointer = ocp.StandardCheckpointer()
    state_filtered = {
        k: v
        for k, v in state.items()
        if k not in ["mjx_model", "mjx_data", "output_bins"]
    }
    checkpointer.save(checkpoint_path, state_filtered)
    checkpointer.wait_until_finished()


def restore_actuator_net_checkpoint_from_path(
    checkpoint_path: Path, actuator_net_cfg: omegaconf.DictConfig | Dict[str, Any]
) -> "ActuatorEnsemble":
    checkpointer = ocp.StandardCheckpointer()
    empty_an = hydra.utils.instantiate(
        actuator_net_cfg,
        rngs=nnx.Rngs(params=42),
    )
    abstract_an = nnx.eval_shape(lambda: empty_an)
    graphdef, abstract_state = nnx.split(abstract_an)
    abstract_state = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(
            a.shape, a.dtype, sharding=SingleDeviceSharding(jax.devices()[0])
        ),
        abstract_state,
    )

    an_state_restored = checkpointer.restore(checkpoint_path, target=abstract_state)
    checkpointer.wait_until_finished()
    actuator_net = nnx.merge(graphdef, an_state_restored)
    # TODO: Load this stuff automatically, then remove empty_an
    if hasattr(actuator_net, "mjx_model"):
        actuator_net.mjx_model = empty_an.mjx_model
        actuator_net.mjx_data = empty_an.mjx_data
    return actuator_net


def restore_actuator_net_checkpoint(
    exp_name: str, actuator_net_name: str, checkpoint_name: str = "model"
) -> "ActuatorEnsemble":
    actuator_net_train_cfg = get_an_training_config(exp_name, actuator_net_name)
    checkpoint_path = (
        Path(__file__).parents[2]
        / "exps"
        / exp_name
        / "actuator_network"
        / actuator_net_name
        / checkpoint_name
    )
    return restore_actuator_net_checkpoint_from_path(
        checkpoint_path, actuator_net_train_cfg.actuator_net
    )


def convert_actuator_net_rl(
    agent: skrl.agents.jax.Agent, train_cfg: omegaconf.DictConfig
) -> "ActuatorEnsemble":
    model = hydra.utils.instantiate(
        train_cfg.actuator_net,
        rngs=nnx.Rngs(params=42),  # Parameters are overwritten anyway
        ensemble_size=1,
        input_mean=agent._observation_preprocessor.running_mean,
        input_std=jnp.sqrt(agent._observation_preprocessor.running_variance)
        + agent._observation_preprocessor.epsilon,
        output_mean=jnp.zeros(4, dtype=jnp.float32),
        output_std=jnp.ones(4, dtype=jnp.float32),
    )

    layers_rl_sorted = [
        l[1]
        for l in sorted(
            agent.policy.state_dict.params["params"]["net_container"].items(),
            key=lambda l: int(l[0][7:]),
        )
    ]
    for l_an, l_rl in zip(
        model.layers,
        layers_rl_sorted,
    ):
        l_an.bias.value = jnp.array(l_rl["bias"][None])
        l_an.kernel.value = jnp.array(l_rl["kernel"][None])

    return model


def restore_actuator_net_rl_checkpoint(
    exp_name: str, actuator_net_name: str, checkpoint_name: str = "best_agent.pickle"
) -> "ActuatorEnsemble":
    actuator_net_train_cfg = get_an_training_config(
        exp_name, actuator_net_name, rl_training=True
    )

    # This seems to be the only way to resolve the interpolations in the config
    train_cfg = omegaconf.OmegaConf.create(
        omegaconf.OmegaConf.to_container(actuator_net_train_cfg, resolve=True)
    )

    inp_size = get_an_input_size_from_cfg(train_cfg.actuator_net)
    dummy_observation_space = gymnasium.spaces.Box(
        -np.inf * np.ones(inp_size, dtype=np.float32),
        np.inf * np.ones(inp_size, dtype=np.float32),
    )
    dummy_action_space = gymnasium.spaces.Box(
        -np.inf * np.ones(4, dtype=np.float32),
        np.inf * np.ones(4, dtype=np.float32),
    )

    exps_path = Path(__file__).parents[2] / "exps"
    checkpoint_path = (
        exps_path
        / exp_name
        / "actuator_network_rl"
        / actuator_net_name
        / "checkpoints"
        / checkpoint_name
    )
    agent = load_agent(
        checkpoint_path,
        train_cfg.agent,
        dummy_observation_space,
        dummy_action_space,
        1,
    )

    return convert_actuator_net_rl(agent, train_cfg)
