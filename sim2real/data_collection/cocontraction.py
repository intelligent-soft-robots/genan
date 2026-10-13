import logging
from typing import Sequence, Tuple

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class Cocontraction:
    def __init__(
        self,
        min_pressures_agos: Sequence[int],
        max_pressures_agos: Sequence[int],
        min_pressures_antagos: Sequence[int],
        max_pressures_antagos: Sequence[int],
        ndp: Sequence[float],
    ):
        self.min_pressures_ago_np = np.asarray(min_pressures_agos)
        self.max_pressures_ago_np = np.asarray(max_pressures_agos)
        self.min_pressures_antago_np = np.asarray(min_pressures_antagos)
        self.max_pressures_antago_np = np.asarray(max_pressures_antagos)
        self.ndp_np = np.asarray(ndp)
        # Scaling factor that ensures that for the whole control range [-1, 1] that either agonist or the antagonist
        # pressure changes. See sect. 3.2 in Büchler et al., "Learning to Control Highly Accelerated Ballistic Movements
        # on Muscular Robots" (the minus instead of plus is a sign error in the paper).
        self.symmetric_cocontraction_factor_np = 0.5 + np.clip(
            np.abs(self.ndp_np - 0.5), 0.0, 0.5
        )

        self.min_pressures_ago_jax = jnp.array(self.min_pressures_ago_np).astype(
            jnp.float32
        )
        self.max_pressures_ago_jax = jnp.array(self.max_pressures_ago_np).astype(
            jnp.float32
        )
        self.min_pressures_antago_jax = jnp.array(self.min_pressures_antago_np).astype(
            jnp.float32
        )
        self.max_pressures_antago_jax = jnp.array(self.max_pressures_antago_np).astype(
            jnp.float32
        )
        self.ndp_jax = jnp.array(self.ndp_np).astype(jnp.float32)
        self.symmetric_cocontraction_factor_jax = jnp.array(
            self.symmetric_cocontraction_factor_np
        ).astype(jnp.float32)

    @classmethod
    def from_robot_config(cls, robot_config, ndp: Sequence[float]) -> "Cocontraction":
        return cls(
            robot_config.min_pressures_ago,
            robot_config.max_pressures_ago,
            robot_config.min_pressures_antago,
            robot_config.max_pressures_antago,
            ndp,
        )

    def control_to_pressures(
        self, control: jnp.ndarray | np.ndarray, round_to_int: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[np.ndarray, np.ndarray]:
        if isinstance(control, jnp.ndarray):
            return self._control_to_pressures(
                control,
                self.min_pressures_ago_jax,
                self.max_pressures_ago_jax,
                self.min_pressures_antago_jax,
                self.max_pressures_antago_jax,
                self.ndp_jax,
                self.symmetric_cocontraction_factor_jax,
                round_to_int,
            )
        else:
            return self._control_to_pressures(
                np.asarray(control),
                self.min_pressures_ago_np,
                self.max_pressures_ago_np,
                self.min_pressures_antago_np,
                self.max_pressures_antago_np,
                self.ndp_np,
                self.symmetric_cocontraction_factor_np,
                round_to_int,
            )

    @classmethod
    def _control_to_pressures(
        cls,
        control: jnp.ndarray | np.ndarray,
        min_pressures_ago: jnp.ndarray | np.ndarray,
        max_pressures_ago: jnp.ndarray | np.ndarray,
        min_pressures_antago: jnp.ndarray | np.ndarray,
        max_pressures_antago: jnp.ndarray | np.ndarray,
        ndp: jnp.ndarray | np.ndarray,
        symmetric_cocontraction_factor: jnp.ndarray | np.ndarray,
        round_to_int: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[np.ndarray, np.ndarray]:
        range_agos = max_pressures_ago - min_pressures_ago
        p_agos = min_pressures_ago + range_agos * (
            ndp - symmetric_cocontraction_factor * control
        )
        range_antagos = max_pressures_antago - min_pressures_antago
        p_antagos = min_pressures_antago + range_antagos * (
            ndp + symmetric_cocontraction_factor * control
        )
        p_agos = p_agos.clip(min_pressures_ago, max_pressures_ago)
        p_antagos = p_antagos.clip(min_pressures_antago, max_pressures_antago)
        # Rounding to the integer pressure units the valves accept destroys the gradient, so
        # differentiable users (the analytic muscle model) ask for the continuous mapping.
        if not round_to_int:
            return p_agos, p_antagos
        return p_agos.round().astype(np.int32), p_antagos.round().astype(np.int32)

    def pressures_to_control(
        self,
        pressures_ago: jnp.ndarray | np.ndarray,
        pressures_antago: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray:
        if isinstance(pressures_ago, jnp.ndarray):
            return self._pressures_to_control(
                pressures_ago,
                pressures_antago,
                jnp,
                self.min_pressures_ago_jax,
                self.max_pressures_ago_jax,
                self.min_pressures_antago_jax,
                self.max_pressures_antago_jax,
                self.ndp_jax,
                self.symmetric_cocontraction_factor_jax,
            )
        else:
            return self._pressures_to_control(
                np.asarray(pressures_ago),
                np.asarray(pressures_antago),
                np,
                self.min_pressures_ago_np,
                self.max_pressures_ago_np,
                self.min_pressures_antago_np,
                self.max_pressures_antago_np,
                self.ndp_np,
                self.symmetric_cocontraction_factor_np,
            )

    @classmethod
    def _pressures_to_control(
        cls,
        pressures_ago: jnp.ndarray | np.ndarray,
        pressures_antago: jnp.ndarray | np.ndarray,
        xp,
        min_pressures_ago: jnp.ndarray | np.ndarray,
        max_pressures_ago: jnp.ndarray | np.ndarray,
        min_pressures_antago: jnp.ndarray | np.ndarray,
        max_pressures_antago: jnp.ndarray | np.ndarray,
        ndp: jnp.ndarray | np.ndarray,
        symmetric_cocontraction_factor: jnp.ndarray | np.ndarray,
    ):
        pressures_ago = xp.asarray(pressures_ago)
        pressures_antago = xp.asarray(pressures_antago)
        pressures = xp.stack((pressures_ago, pressures_antago), axis=-1)

        min_pressures = xp.stack((min_pressures_ago, min_pressures_antago), axis=1)
        max_pressures = xp.stack((max_pressures_ago, max_pressures_antago), axis=1)
        ranges = max_pressures - min_pressures

        controls = xp.clip(
            (
                xp.stack(
                    (-jnp.ones_like(pressures_ago), jnp.ones_like(pressures_ago)),
                    axis=-1,
                )
                * ((pressures - min_pressures) / ranges - ndp[:, None])
                / symmetric_cocontraction_factor[:, None]
            ),
            -1,
            1,
        )

        return cls._select_closest_controls(
            pressures,
            controls,
            xp,
            min_pressures_ago,
            max_pressures_ago,
            min_pressures_antago,
            max_pressures_antago,
            ndp,
            symmetric_cocontraction_factor,
        )

    @classmethod
    def _select_closest_controls(
        cls,
        pressures: jnp.ndarray,
        controls: jnp.ndarray,
        xp,
        min_pressures_ago: jnp.ndarray | np.ndarray,
        max_pressures_ago: jnp.ndarray | np.ndarray,
        min_pressures_antago: jnp.ndarray | np.ndarray,
        max_pressures_antago: jnp.ndarray | np.ndarray,
        ndp: jnp.ndarray | np.ndarray,
        symmetric_cocontraction_factor: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray:
        # Map the calculated controls back to pressures (needed for selecting the closest control in terms of pressure
        # change).
        pressures_from_controls_ago = xp.stack(
            cls._control_to_pressures(
                controls[..., 0],
                min_pressures_ago,
                max_pressures_ago,
                min_pressures_antago,
                max_pressures_antago,
                ndp,
                symmetric_cocontraction_factor,
            ),
            axis=-1,
        )
        pressures_from_controls_antago = xp.stack(
            cls._control_to_pressures(
                controls[..., 1],
                min_pressures_ago,
                max_pressures_ago,
                min_pressures_antago,
                max_pressures_antago,
                ndp,
                symmetric_cocontraction_factor,
            ),
            axis=-1,
        )

        # For the control signal of the agonist and antagonist, calculate each pressure difference needed to achieve
        # that control signal. Note: If the control signal was calculated from the agonist pressure, there is only a
        # difference in the antagonist and vice-versa: if it was calculated from the antagonist pressures, there is only
        # a difference in the agonist.
        dp_from_ctrl_ago = xp.abs(pressures - pressures_from_controls_ago).mean(axis=-1)
        dp_from_ctrl_antago = xp.abs(pressures - pressures_from_controls_antago).mean(
            axis=-1
        )

        return xp.where(
            dp_from_ctrl_ago <= dp_from_ctrl_antago, controls[..., 0], controls[..., 1]
        )
