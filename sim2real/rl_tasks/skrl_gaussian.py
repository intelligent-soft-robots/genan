import math
import textwrap
from functools import partial
from typing import Any, Literal

import flax
import flax.linen as nn  # noqa
import gymnasium
import jax
import jax.numpy as jnp
from skrl import config
from skrl.models.jax import Model
from skrl.utils.model_instantiators.jax.common import (
    generate_containers,
    one_hot_encoding,
)  # noqa
from skrl.utils.spaces.jax import (
    compute_space_limits,
    unflatten_tensorized_space,
)  # noqa

LOG_SQRT_2_PI = math.log(math.sqrt(2 * math.pi))
HALF_LOG_2_PI_PLUS = 0.5 + 0.5 * math.log(2 * math.pi)


def gaussian_model(
    *,
    observation_space: gymnasium.Space | None = None,
    state_space: gymnasium.Space | None = None,
    action_space: gymnasium.Space | None = None,
    device: str | jax.Device | None = None,
    clip_actions: bool = False,
    clip_mean_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    reduction: Literal["mean", "sum", "prod", "none"] = "sum",
    initial_log_std: float = 0,
    fixed_log_std: bool = False,
    network: list[dict[str, Any]] = [],
    output: str | list[str] = "",
    return_source: bool = False,
) -> Model | str:
    """Instantiate a :class:`~skrl.models.jax.gaussian.GaussianMixin`-based model.

    :param observation_space: Observation space. The ``num_observations`` property will contain the size of the space.
    :param state_space: State space. The ``num_states`` property will contain the size of the space.
    :param action_space: Action space. The ``num_actions`` property will contain the size of the space.
    :param device: Data allocation and computation device. If not specified, the default device will be used.
    :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
    :param clip_mean_actions: Flag to indicate whether the mean actions should be clipped to the action space.
        If ``True``, the mean actions will be clipped before sampling the actions.
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
    :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
    :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
    :param reduction: Reduction method for returning the log probability density function.
        If ``"none"``, the log probability density function is returned as a tensor of shape
        ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
    :param initial_log_std: Initial value for the log standard deviation.
    :param fixed_log_std: Whether the log standard deviation parameter should be fixed.
        Fixed parameters have the gradient computation deactivated.
    :param network: Network definition.
    :param output: Output expression.
    :param return_source: Whether to return the source string containing the model class used to
        instantiate the model rather than the model instance.

    :return: Gaussian model instance or definition source (if ``return_source`` is True).
    """
    # parse model definition
    containers, output = generate_containers(
        network, output, embed_output=True, indent=1
    )

    # network definitions
    networks = []
    forward: list[str] = []
    for container in containers:
        networks.append(
            f'self.{container["name"]}_container = {container["sequential"]}'
        )
        forward.append(
            f'{container["name"]} = self.{container["name"]}_container({container["input"]})'
        )
    # process output
    if output["modules"]:
        networks.append(f'self.output_layer = {output["modules"][0]}')
        forward.append(f'output = self.output_layer({container["name"]})')
    if output["output"]:
        forward.append(f'output = {output["output"]}')
    else:
        forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

    # build substitutions and indent content
    networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]
    if fixed_log_std:
        log_std_parameter = (
            f'jnp.full(shape={output["size"]}, fill_value={float(initial_log_std)})'
        )
    else:
        log_std_parameter = f'self.param("log_std_parameter", lambda _: jnp.full(shape={output["size"]}, fill_value={float(initial_log_std)}))'

    template = f"""class GaussianModel(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        role="",
        **kwargs,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )

    def setup(self):
        {networks}
        self.log_std_parameter = {log_std_parameter}

    def __call__(self, inputs, role=""):
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        {forward}
        return output, {{"log_std": self.log_std_parameter}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["GaussianModel"](
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_mean_actions=clip_mean_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        reduction=reduction,
    )


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("reduction"))
def _gaussian(
    loc,
    log_std,
    log_std_min,
    log_std_max,
    clip_actions_min,
    clip_actions_max,
    clip_mean_actions_min,
    clip_mean_actions_max,
    taken_actions,
    key,
    reduction,
):
    # clip mean actions
    loc = jnp.clip(loc, a_min=clip_mean_actions_min, a_max=clip_mean_actions_max)

    # clamp log standard deviations
    log_std = jnp.clip(log_std, a_min=log_std_min, a_max=log_std_max)

    # distribution
    scale = jnp.exp(log_std)

    # sample actions
    actions = jax.random.normal(key, loc.shape, dtype=jnp.float32) * scale + loc

    # clip actions
    actions = jnp.clip(actions, a_min=clip_actions_min, a_max=clip_actions_max)

    # log of the probability density function
    taken_actions = actions if taken_actions is None else taken_actions
    log_prob = (
        -jnp.square(taken_actions - loc) / (2 * jnp.square(scale))
        - jnp.log(scale)
        - LOG_SQRT_2_PI
    )

    if reduction is not None:
        log_prob = reduction(log_prob, axis=-1)
    if log_prob.ndim != actions.ndim:
        log_prob = jnp.expand_dims(log_prob, -1)

    return loc, actions, log_prob, log_std, scale


@jax.jit
def _entropy(scale):
    return HALF_LOG_2_PI_PLUS + jnp.log(scale)


class GaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_mean_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param clip_mean_actions: Flag to indicate whether the mean actions should be clipped to the action space.
            If ``True``, the mean actions will be clipped before sampling the actions.
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
        :param reduction: Reduction method for returning the log probability density function.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid.
        """
        self._g_clip_actions = clip_actions
        self._g_clip_mean_actions = clip_mean_actions

        self._g_clip_actions_min, self._g_clip_actions_max = compute_space_limits(
            self.action_space, device=self.device
        )
        self._g_clip_actions_min = self._g_clip_actions_min.astype(jnp.float32)
        self._g_clip_actions_max = self._g_clip_actions_max.astype(jnp.float32)
        self._g_clip_mean_actions_min, self._g_clip_mean_actions_max = (
            self._g_clip_actions_min,
            self._g_clip_actions_max,
        )
        if not self._g_clip_actions:
            self._g_clip_actions_min = jnp.full_like(self._g_clip_actions_min, -jnp.inf)
            self._g_clip_actions_max = jnp.full_like(self._g_clip_actions_max, jnp.inf)
        if not self._g_clip_mean_actions:
            self._g_clip_mean_actions_min = jnp.full_like(
                self._g_clip_mean_actions_min, -jnp.inf
            )
            self._g_clip_mean_actions_max = jnp.full_like(
                self._g_clip_mean_actions_max, jnp.inf
            )

        self._g_clip_log_std = clip_log_std
        self._g_log_std_min = min_log_std if self._g_clip_log_std else -jnp.inf
        self._g_log_std_max = max_log_std if self._g_clip_log_std else jnp.inf

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            jnp.mean
            if reduction == "mean"
            else (
                jnp.sum
                if reduction == "sum"
                else jnp.prod if reduction == "prod" else None
            )
        )

        self._g_i = 0
        self._g_key = config.jax.key

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.IncorrectPostInitOverrideError
        flax.linen.Module.__post_init__(self)

    def act(
        self, inputs: dict[str, Any], *, role: str = "", params: jax.Array | None = None
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Act stochastically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.
        :param params: Parameters used to compute the output. If not provided, internal parameters will be used.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing the following extra output values:

            - ``"log_std"``: log of the standard deviation.
            - ``"log_prob"``: log of the probability density function.
            - ``"mean_actions"``: mean actions (network output after optional clipping).
        """
        with jax.default_device(self.device):
            self._g_i += 1
            subkey = jax.random.fold_in(self._g_key, self._g_i)
            inputs["key"] = subkey

        # map from observations/states to mean actions and log standard deviations
        mean_actions, outputs = self.apply(
            self.state_dict.params if params is None else params, inputs, role
        )

        mean_actions, actions, log_prob, log_std, stddev = _gaussian(
            mean_actions,
            outputs["log_std"],
            self._g_log_std_min,
            self._g_log_std_max,
            self._g_clip_actions_min,
            self._g_clip_actions_max,
            self._g_clip_mean_actions_min,
            self._g_clip_mean_actions_max,
            inputs.get("taken_actions", None),
            subkey,
            self._g_reduction,
        )

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        # avoid jax.errors.UnexpectedTracerError
        outputs["log_std"] = log_std
        outputs["stddev"] = stddev

        return actions, outputs

    def get_entropy(self, stddev: jax.Array, *, role: str = "") -> jax.Array:
        """Compute and return the entropy of the model.

        :param stddev: Model standard deviation.
        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        return _entropy(stddev)
