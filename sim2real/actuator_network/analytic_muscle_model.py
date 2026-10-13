"""Analytic pneumatic-artificial-muscle model as a drop-in replacement for ActuatorEnsemble.

The structure follows Biyajima et al., "Development of a Variable-Stiffness Musculoskeletal
Percussion Robot and Realization of Single-Stroke Motion through Sim-to-Real Transfer"
(IECON 2025): a static Festo force characteristic F(P, h), a contraction ratio h(theta) derived
from the joint geometry, and a wire-slack condition. It is adapted to PAMY2, which has four
joints driven by antagonistic pairs of FESTO DMSP-20-1200N-AM-CM muscles.

The model is stateless: the desired pressures follow from the control history that is already
part of the actuator-network input, so it satisfies the same interface as ActuatorEnsemble and
works with the existing MjxBackend / SimulationAccuracyEvaluator pipeline.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from sim2real.actuator_network.pack_an_input import unpack_an_input
from sim2real.data_collection.cocontraction import Cocontraction
from sim2real.util.config_utils import get_an_input_size

NUM_DOFS = 4
NUM_MUSCLES_PER_DOF = 2

# DMSP-20 data sheet (Festo, "Fluidic muscle DMSP", 2022/04): 20 mm inner diameter, theoretical
# force 1500 N at the maximum operating pressure of 6 bar, maximum contraction 25 % of the
# nominal length.
DMSP20_INNER_DIAMETER = 0.020
DMSP20_MAX_PRESSURE = 6e5
DMSP20_THEORETICAL_FORCE = 1500.0
DMSP20_MAX_CONTRACTION = 0.25

# Conversion from the integer pressure units used by the pam_interface valves to gauge pascal:
# p_bar = 2e-4 * p_int - 1.7888.
PRESSURE_SCALE_PA = 20.0
PRESSURE_OFFSET_PA = -178880.0


def datasheet_force_curve_parameters(
    inner_diameter: float = DMSP20_INNER_DIAMETER,
    max_pressure: float = DMSP20_MAX_PRESSURE,
    theoretical_force: float = DMSP20_THEORETICAL_FORCE,
    max_contraction: float = DMSP20_MAX_CONTRACTION,
) -> Tuple[float, float, float]:
    """Chou-Hannaford coefficients (a, b) and elastic stiffness that match the data sheet.

    The braid angle is chosen so that the uncontracted muscle reaches the theoretical force at
    the maximum operating pressure, and the elastic term is chosen so that the force vanishes at
    the maximum contraction. Returns (a, b, elastic_stiffness).
    """
    area = math.pi * inner_diameter**2 / 4.0
    # a - b = 3 * cot(alpha)^2 - csc(alpha)^2 = (3 * cos(alpha)^2 - 1) / sin(alpha)^2
    force_factor = theoretical_force / (area * max_pressure)
    sin_sq = 2.0 / (force_factor + 3.0)
    cos_sq = 1.0 - sin_sq
    a = 3.0 * cos_sq / sin_sq
    b = 1.0 / sin_sq
    # Pressure-driven force that is left at the maximum contraction, which the elastic term of the
    # membrane has to cancel.
    residual_force = area * max_pressure * (a * (1.0 - max_contraction) ** 2 - b)
    elastic_stiffness = residual_force / max_contraction
    return a, b, elastic_stiffness


class FrozenLagParam(nnx.Variable):
    """Pressure-lag parameter held fixed at its independently calibrated value.

    A distinct Variable type (rather than nnx.Param) ensures nnx.grad, which targets nnx.Param,
    never produces a gradient for it.
    """

    pass


def _log_per_dof(value: float | Sequence[float]) -> jax.Array:
    """Broadcast a scalar or per-DoF initial value to a log-parametrized array of shape [dofs]."""
    return jnp.log(jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float32), (NUM_DOFS,)))


def muscle_force(
    pressure: jax.Array,
    contraction: jax.Array,
    area: jax.Array | float,
    a: jax.Array,
    b: jax.Array,
    contraction_correction: jax.Array,
    elastic_stiffness: jax.Array,
) -> jax.Array:
    """Static force of a fluidic muscle, clipped at zero where the wire goes slack.

    F(P, h) is strictly decreasing in h and crosses zero exactly at the no-load contraction
    h0(P), so clipping at zero implements the slack condition of Eq. (3) in the paper
    (no tension for h > h0) without an explicit branch.
    """
    # Clipping keeps the force bounded if the optimizer pushes the contraction outside the range
    # in which the data sheet curve is defined.
    braid_factor = jnp.clip(1.0 - contraction_correction * contraction, 0.0, 1.0)
    force = (
        area * pressure * (a * braid_factor**2 - b) - elastic_stiffness * contraction
    )
    return jnp.clip(force, min=0.0)


class AnalyticMuscleEnsemble(nnx.Module):
    """Physically parametrized actuator model with the ActuatorEnsemble interface.

    Trainable physical parameters are nnx.Param so that nnx.grad picks them up; robot constants
    and parameters that are held fixed are stored in the "constants" collection.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        num_history_states: int,
        step_size: int,
        pressure_input: bool,
        velocity_input: bool,
        position_history: bool,
        velocity_history: bool,
        control_history: bool,
        delta_history: bool,
        ndp: Sequence[float],
        min_pressures_ago: Sequence[int],
        max_pressures_ago: Sequence[int],
        min_pressures_antago: Sequence[int],
        max_pressures_antago: Sequence[int],
        moment_arms: Sequence[Sequence[float]],
        antagonist_pull_signs: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        ensemble_size: int = 1,
        fit_force_gains: bool = False,
        fit_pressure_lag: bool = False,
        timestep: float = 0.002,
        nominal_length: float = 1.2,
        inner_diameter: float = DMSP20_INNER_DIAMETER,
        pressure_scale_pa: float = PRESSURE_SCALE_PA,
        pressure_offset_pa: float = PRESSURE_OFFSET_PA,
        init_tendon_slack: float = 0.1,
        elastic_stiffness_scale: float = 1.0,
        init_lag_time_constant: float | Sequence[float] = 0.06,
        init_coulomb_torque: float = 0.5,
        init_friction_coefficient: float = 0.05,
        init_viscous_damping: float = 0.05,
        velocity_smoothing: float = 0.02,
        **kwargs,
    ):
        assert ensemble_size == 1, "The analytic muscle model is deterministic."
        assert velocity_input, "The friction model needs the joint velocity."
        assert (
            control_history
        ), "Modelling the pressure lag requires the control history."

        self.ensemble_size = ensemble_size
        self.pressure_input = pressure_input
        self.velocity_input = velocity_input
        self.position_history = position_history
        self.velocity_history = velocity_history
        self.control_history = control_history
        self.delta_history = delta_history
        self.num_history_states = num_history_states
        self.step_size = step_size

        # unpack_an_input only needs these keys, and keeping a plain dict avoids carrying an
        # omegaconf object through jit.
        self.actuator_net_cfg = {
            "num_history_states": num_history_states,
            "step_size": step_size,
            "pressure_input": pressure_input,
            "velocity_input": velocity_input,
            "position_history": position_history,
            "velocity_history": velocity_history,
            "control_history": control_history,
            "delta_history": delta_history,
        }

        # The pressure limits and the cocontraction level are robot constants, but they have to
        # live in the module state (rather than in a Cocontraction object) so that nnx can trace
        # through them.
        cocontraction = Cocontraction(
            min_pressures_ago,
            max_pressures_ago,
            min_pressures_antago,
            max_pressures_antago,
            ndp,
        )
        self.min_pressures_ago = nnx.Variable(
            cocontraction.min_pressures_ago_jax, collection="constants"
        )
        self.max_pressures_ago = nnx.Variable(
            cocontraction.max_pressures_ago_jax, collection="constants"
        )
        self.min_pressures_antago = nnx.Variable(
            cocontraction.min_pressures_antago_jax, collection="constants"
        )
        self.max_pressures_antago = nnx.Variable(
            cocontraction.max_pressures_antago_jax, collection="constants"
        )
        self.ndp = nnx.Variable(cocontraction.ndp_jax, collection="constants")
        self.symmetric_cocontraction_factor = nnx.Variable(
            cocontraction.symmetric_cocontraction_factor_jax, collection="constants"
        )
        self.context_timestep = step_size * timestep
        self.nominal_length = nominal_length
        self.muscle_area = math.pi * inner_diameter**2 / 4.0
        self.pressure_scale_pa = pressure_scale_pa
        self.pressure_offset_pa = pressure_offset_pa
        self.velocity_smoothing = velocity_smoothing

        a_init, b_init, elastic_init = datasheet_force_curve_parameters(
            inner_diameter=inner_diameter
        )
        self.log_a = nnx.Param(jnp.array(math.log(a_init), dtype=jnp.float32))
        self.log_b = nnx.Param(jnp.array(math.log(b_init), dtype=jnp.float32))
        self.log_contraction_correction = nnx.Param(jnp.zeros((), dtype=jnp.float32))
        self.log_elastic_stiffness = nnx.Param(
            jnp.array(
                math.log(elastic_init * elastic_stiffness_scale), dtype=jnp.float32
            )
        )

        muscle_shape = (NUM_DOFS, NUM_MUSCLES_PER_DOF)
        self.log_tendon_slack = nnx.Param(
            jnp.full(muscle_shape, math.log(init_tendon_slack), dtype=jnp.float32)
        )

        # The muscle that pulls the joint in the positive direction shortens as the joint angle
        # grows, so its contraction ratio grows with theta and its antagonist's shrinks.
        pull_signs = jnp.asarray(antagonist_pull_signs, dtype=jnp.float32)
        self.antagonist_pull_signs = nnx.Variable(pull_signs, collection="constants")
        self.contraction_signs = nnx.Variable(
            jnp.stack((-pull_signs, pull_signs), axis=-1), collection="constants"
        )

        moment_arms_init = jnp.asarray(moment_arms, dtype=jnp.float32)
        assert (
            moment_arms_init.shape == muscle_shape
        ), f"Expected moment_arms of shape {muscle_shape}, got {moment_arms_init.shape}."
        self.log_moment_arms = nnx.Variable(
            jnp.log(moment_arms_init), collection="constants"
        )
        log_force_gains = jnp.zeros(muscle_shape, dtype=jnp.float32)
        if fit_force_gains:
            self.log_force_gains = nnx.Param(log_force_gains)
        else:
            self.log_force_gains = nnx.Variable(log_force_gains, collection="constants")

        self.log_coulomb_torque = nnx.Param(
            jnp.full(NUM_DOFS, math.log(init_coulomb_torque), dtype=jnp.float32)
        )
        self.log_friction_coefficient = nnx.Param(
            jnp.full(NUM_DOFS, math.log(init_friction_coefficient), dtype=jnp.float32)
        )
        self.log_viscous_damping = nnx.Param(
            jnp.full(NUM_DOFS, math.log(init_viscous_damping), dtype=jnp.float32)
        )
        self.torque_bias = nnx.Param(jnp.zeros(NUM_DOFS, dtype=jnp.float32))

        # The pressure lag is identified beforehand from the recorded desired/observed pressures,
        # which measure it far more directly than joint motion does. Fitting it again here would
        # only let it absorb downstream effects, so it is frozen by default.
        lag_variable = nnx.Param if fit_pressure_lag else FrozenLagParam
        self.log_lag_time_constant = lag_variable(_log_per_dof(init_lag_time_constant))
        self.context_lags = nnx.Variable(
            # Index 0 of the control context is the oldest sample.
            jnp.arange(num_history_states, -1, -1, dtype=jnp.float32)
            * self.context_timestep,
            collection="constants",
        )

        # The analytic model works in physical units, so the standardization the interface
        # expects is the identity.
        input_dim = get_an_input_size(
            num_history_states,
            velocity_input,
            pressure_input,
            position_history,
            velocity_history,
            control_history,
        )
        self.input_mean = nnx.Variable(
            jnp.zeros(input_dim, dtype=jnp.float32), collection="constants"
        )
        self.input_std = nnx.Variable(
            jnp.ones(input_dim, dtype=jnp.float32), collection="constants"
        )
        self.output_mean = nnx.Variable(
            jnp.zeros(NUM_DOFS, dtype=jnp.float32), collection="constants"
        )
        self.output_std = nnx.Variable(
            jnp.ones(NUM_DOFS, dtype=jnp.float32), collection="constants"
        )

    def desired_pressures(self, control_context: jax.Array) -> jax.Array:
        """Gauge pressures in pascal for shape [..., context, dofs] controls.

        Returns shape [..., context, dofs, muscles].
        """
        pressures_ago, pressures_antago = Cocontraction._control_to_pressures(
            control_context,
            self.min_pressures_ago.value,
            self.max_pressures_ago.value,
            self.min_pressures_antago.value,
            self.max_pressures_antago.value,
            self.ndp.value,
            self.symmetric_cocontraction_factor.value,
            round_to_int=False,
        )
        pressures_int = jnp.stack((pressures_ago, pressures_antago), axis=-1)
        return self.pressure_scale_pa * pressures_int + self.pressure_offset_pa

    def lag_kernel(self) -> jax.Array:
        """First-order lag as an FIR kernel over the control history, shape [context, dofs].

        Only the total lag is identifiable from the recorded pressures: a dead time and a time
        constant trade off against each other along an almost perfectly flat valley, so the lag
        is modelled with a single time constant.

        The weights are built in log space and normalized with a softmax, which keeps the kernel
        summing to one for any time constant, including ones long enough that most of the
        impulse response falls outside the available history.
        """
        time_constant = jnp.exp(self.log_lag_time_constant.value)
        lags = self.context_lags.value[:, None]
        return jax.nn.softmax(-lags / time_constant[None], axis=0)

    def pressures(self, control_context: jax.Array) -> jax.Array:
        """Actual muscle pressures in pascal, shape [..., dofs, muscles]."""
        pressures_desired = self.desired_pressures(control_context)
        return jnp.einsum("...tdm,td->...dm", pressures_desired, self.lag_kernel())

    def contraction_ratios(self, positions: jax.Array) -> jax.Array:
        """Contraction ratio of every muscle, shape [..., dofs, muscles].

        With an uncontracted muscle the wire is slack by tendon_slack metres, so the muscle has
        to shorten by that much before it can transmit any force. The joint angle adds or removes
        slack depending on which side of the joint the muscle sits on.
        """
        moment_arms = jnp.exp(self.log_moment_arms.value)
        slack = (
            jnp.exp(self.log_tendon_slack.value)
            + self.contraction_signs.value * moment_arms * positions[..., None]
        )
        return slack / self.nominal_length

    def muscle_forces(
        self, positions: jax.Array, control_context: jax.Array
    ) -> jax.Array:
        """Tendon tension of every muscle in newton, shape [..., dofs, muscles]."""
        forces = muscle_force(
            self.pressures(control_context),
            self.contraction_ratios(positions),
            self.muscle_area,
            jnp.exp(self.log_a.value),
            jnp.exp(self.log_b.value),
            jnp.exp(self.log_contraction_correction.value),
            jnp.exp(self.log_elastic_stiffness.value),
        )
        return forces * jnp.exp(self.log_force_gains.value)

    def friction_torques(self, velocities: jax.Array, forces: jax.Array) -> jax.Array:
        """Tendon and joint friction, shape [..., dofs].

        The Coulomb part grows with the total tendon tension because the friction in the pulleys
        is driven by the normal load, which is what makes cocontraction stiffen the joint.
        """
        moment_arms = jnp.exp(self.log_moment_arms.value)
        normal_load = (forces * moment_arms).sum(axis=-1)
        coulomb = (
            jnp.exp(self.log_coulomb_torque.value)
            + jnp.exp(self.log_friction_coefficient.value) * normal_load
        )
        direction = jnp.tanh(velocities / self.velocity_smoothing)
        return (
            -direction * coulomb - jnp.exp(self.log_viscous_damping.value) * velocities
        )

    def joint_torques(self, an_input: jax.Array) -> jax.Array:
        contexts = unpack_an_input(an_input, self.actuator_net_cfg)
        positions = contexts["position_context"][..., -1, :]
        velocities = contexts["velocity_context"][..., -1, :]
        control_context = contexts["control_context"]

        forces = self.muscle_forces(positions, control_context)
        moment_arms = jnp.exp(self.log_moment_arms.value)
        muscle_torques = self.antagonist_pull_signs.value * (
            moment_arms[..., 1] * forces[..., 1] - moment_arms[..., 0] * forces[..., 0]
        )
        friction = self.friction_torques(velocities, forces)
        return muscle_torques + friction + self.torque_bias.value

    def eval(self, **kwargs) -> None:
        return

    def train(self, **kwargs) -> None:
        return

    def __call__(
        self, x: jax.Array, rng: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jax.Array, jax.Array, Any]:
        torques = self.joint_torques(x)
        return torques, torques, None

    def forward(self, x: jax.Array) -> jax.Array:
        return self.joint_torques(x)

    def forward_all(
        self, x: jax.Array, rng: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jax.Array, jax.Array, List[jax.Array]]:
        return self.forward_all_no_jit(x, rng)

    def forward_all_no_jit(
        self, x: jax.Array, rng: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jax.Array, jax.Array, Any]:
        return self(x[None], rng)

    def physical_parameters(self) -> Dict[str, Any]:
        """Fitted parameters converted back to SI units, for logging and inspection."""
        parameters = {
            "braid_coefficient_a": float(jnp.exp(self.log_a.value)),
            "braid_coefficient_b": float(jnp.exp(self.log_b.value)),
            "contraction_correction": float(
                jnp.exp(self.log_contraction_correction.value)
            ),
            "elastic_stiffness_n": float(jnp.exp(self.log_elastic_stiffness.value)),
            "tendon_slack_m": jnp.exp(self.log_tendon_slack.value).tolist(),
            "moment_arms_m": jnp.exp(self.log_moment_arms.value).tolist(),
            "force_gains": jnp.exp(self.log_force_gains.value).tolist(),
            "coulomb_torque_nm": jnp.exp(self.log_coulomb_torque.value).tolist(),
            "friction_coefficient": jnp.exp(
                self.log_friction_coefficient.value
            ).tolist(),
            "viscous_damping_nms": jnp.exp(self.log_viscous_damping.value).tolist(),
            "torque_bias_nm": self.torque_bias.value.tolist(),
        }
        parameters["lag_time_constant_s"] = jnp.exp(
            self.log_lag_time_constant.value
        ).tolist()
        return parameters
