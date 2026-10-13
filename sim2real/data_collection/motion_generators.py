import abc
import random
from typing import Literal, Optional, Sequence

import numpy as np
import scipy


class MotionGenerator(abc.ABC):
    def __init__(
        self,
        duration: float,
        generator_timestep: float,
        backend_timestep: float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        noise_scale_min: float,
        noise_scale_max: float,
    ):
        self.duration = duration
        self.backend_timestep = backend_timestep
        self.generator_timestep = generator_timestep
        assert np.isclose(self.generator_timestep % self.backend_timestep, 0.0)
        self.lower_limits = np.asarray(lower_limits)
        self.upper_limits = np.asarray(upper_limits)
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max

    def generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        start = np.clip(
            2 * (start - self.lower_limits) / (self.upper_limits - self.lower_limits)
            - 1,
            -1,
            1,
        )
        signal = self._generate_signal(start, last_pos_traj)
        noise_scale = np.random.uniform(
            self.noise_scale_min, self.noise_scale_max, size=4
        )
        signal = signal + np.random.normal(scale=noise_scale, size=signal.shape)
        signal = np.clip(signal, -1, 1)
        assert np.all(self.lower_limits == -1) and np.all(
            self.upper_limits == 1
        )  # Other limits are not supported at the moment
        num_repeats = round(self.generator_timestep / self.backend_timestep)
        signal = signal.repeat(num_repeats, axis=0)
        return (self.upper_limits - self.lower_limits) / 2 * (
            signal + 1
        ) + self.lower_limits

    @abc.abstractmethod
    def _generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        pass


class SineMotionGenerator(MotionGenerator):
    def __init__(
        self,
        duration: float,
        generator_timestep: float,
        backend_timestep: float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        noise_scale_min: float = 0.05,
        noise_scale_max: float = 0.05,
        frequencies_low: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        frequencies_high: Sequence[float] = (0.5, 0.5, 0.5, 0.5),
    ):
        super().__init__(
            duration,
            generator_timestep,
            backend_timestep,
            lower_limits,
            upper_limits,
            noise_scale_min,
            noise_scale_max,
        )
        self.frequencies_low = np.asarray(frequencies_low)
        self.frequencies_high = np.asarray(frequencies_high)

    def _generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        wave_amplitudes = np.random.uniform(0, 1, size=4)
        wave_frequency = np.random.uniform(self.frequencies_low, self.frequencies_high)
        y_offset = (1 - wave_amplitudes) * np.random.uniform(-1, 1, size=4)
        t = np.arange(0, self.duration, self.generator_timestamp)[:, None]

        # Choose the phase such that the sine starts as close as possible to current position
        closest_position_on_wave = np.clip(
            (start - y_offset) / wave_amplitudes, -1.0, 1.0
        )
        phase1 = np.arcsin(closest_position_on_wave)
        phase2 = np.pi - phase1
        phase = random.choice([phase1, phase2])

        samples = np.clip(
            wave_amplitudes * np.sin(2 * np.pi * wave_frequency * t + phase) + y_offset,
            -1,
            1,
        )

        return samples


class OrnsteinUhlenbeckMotionGenerator(MotionGenerator):
    def __init__(
        self,
        duration: float,
        generator_timestep: float,
        backend_timestep: float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        theta: float = 0.01,
        sigma_min: float = 0.03,
        sigma_max: float = 0.03,
        mu: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        mu_adaptation_per_ep: Optional[float] = None,
        max_mu_adaptation: float = 0.5,
        mean_update_coef: float = 0.01,
        target_pos_mean_deg: Sequence[float] = (0.0, 40.0, 0.0, 0.0),
        ratio_linear_segments: float = 0.05,
        min_len_linear_segments: int = 20,
        max_len_linear_segments: int = 100,
        linear_segment_noise_std: float = 0.025,
        action_repeat: int = 1,
        noise_distribution: Literal["normal", "uniform"] = "normal",
    ):
        super().__init__(
            duration,
            generator_timestep,
            backend_timestep,
            lower_limits,
            upper_limits,
            0.0,
            0.0,
        )
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu = np.asarray(mu)
        self.mu_adaptation_per_ep = mu_adaptation_per_ep
        self.max_mu_adaptation = max_mu_adaptation
        self.target_pos_mean = np.array(target_pos_mean_deg) * np.pi / 180.0
        self.pos_mean = self.target_pos_mean
        self.mean_update_coef = mean_update_coef
        self.num_ep = 0
        self.linear_segment_start_prob = (
            2
            * ratio_linear_segments
            * action_repeat
            / (max_len_linear_segments + min_len_linear_segments)
        )
        self.min_len_linear_segments = min_len_linear_segments
        self.max_len_linear_segments = max_len_linear_segments
        self.linear_segment_noise_std = linear_segment_noise_std
        self.action_repeat = action_repeat
        self.noise_distribution = noise_distribution

    def _generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        if last_pos_traj is not None:
            self.pos_mean = (
                self.mean_update_coef * last_pos_traj.mean(axis=0)
                + (1 - self.mean_update_coef) * self.pos_mean
            )
        if self.mu_adaptation_per_ep is not None:
            self.mu += self.mu_adaptation_per_ep * (
                self.target_pos_mean - self.pos_mean
            )
            self.mu = np.clip(self.mu, -self.max_mu_adaptation, self.max_mu_adaptation)
        sigma = np.random.uniform(self.sigma_min, self.sigma_max, size=4)
        theta = sigma * self.theta
        trajectory = np.empty((int(self.duration / self.generator_timestep), 4))
        state = start
        linear_segments_remain_steps = np.zeros(4, dtype=int)
        linear_segments_slope = np.zeros(4, dtype=float)
        for i in range(0, trajectory.shape[0], self.action_repeat):
            if self.linear_segment_start_prob > 0.0:
                for j in range(len(linear_segments_remain_steps)):
                    if linear_segments_remain_steps[j] <= 0:
                        linear_segments_remain_steps[j] = np.random.choice(
                            [
                                0.0,
                                np.random.uniform(
                                    self.min_len_linear_segments,
                                    self.max_len_linear_segments,
                                ),
                            ],
                            p=[
                                1 - self.linear_segment_start_prob,
                                self.linear_segment_start_prob,
                            ],
                        )
                        linear_segments_slope[j] = np.random.uniform(-0.01, 0.01)
            if self.noise_distribution == "normal":
                noise = np.random.randn(4) * sigma
            else:
                # TODO: Dont hardcode the delta limit here
                curr_lim_lower = np.clip(-1 - state, a_min=-0.01, a_max=None)
                curr_lim_upper = np.clip(1 - state, a_min=None, a_max=0.01)
                noise = np.random.uniform(curr_lim_lower, curr_lim_upper, size=4)
            delta = np.where(
                linear_segments_remain_steps <= 0,
                theta * (self.mu - state) + noise,
                linear_segments_slope
                + np.random.randn(4) * self.linear_segment_noise_std,
            )
            for k in range(self.action_repeat):
                state += delta
                state = np.clip(state, -1, 1)
                trajectory[i + k, :] = state
            linear_segments_remain_steps = np.clip(
                linear_segments_remain_steps - self.action_repeat, 0, a_max=None
            )
        self.num_ep += 1
        return trajectory


class IntegratedOrnsteinUhlenbeckMotionGenerator(MotionGenerator):
    def __init__(
        self,
        duration: float,
        generator_timestep: float,
        backend_timestep: float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        theta: float | Sequence[float] = 5e-4,
        theta_delta: float | Sequence[float] = 5e-3,
        sigma_min: float = 0.003,
        sigma_max: float = 0.003,
        mu: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        reflecting_limits: bool = False,
    ):
        super().__init__(
            duration,
            generator_timestep,
            backend_timestep,
            lower_limits,
            upper_limits,
            0.0,
            0.0,
        )
        if np.isscalar(theta):
            self.theta = theta * np.ones(4)
        else:
            self.theta = np.asarray(theta)
        if np.isscalar(theta_delta):
            self.theta_delta = theta_delta * np.ones(4)
        else:
            self.theta_delta = np.asarray(theta_delta)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu = np.asarray(mu)
        self.reflecting_limits = reflecting_limits

    def _generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        sigma = np.random.uniform(self.sigma_min, self.sigma_max, size=4)
        trajectory = np.empty((int(self.duration / self.generator_timestep), 4))
        state = start
        # TODO: Dont hardcode the delta limits here!
        delta_state = np.random.uniform(-0.01, 0.01, size=(4,))
        for i in range(trajectory.shape[0]):
            delta_state += (
                self.theta * (self.mu - state)
                - self.theta_delta * delta_state
                + sigma * np.random.randn(4)
            )
            delta_state = np.clip(delta_state, -0.01, 0.01)
            state += delta_state
            if self.reflecting_limits:
                state = np.where(
                    state > 1,
                    2 - state,
                    state,
                )
                state = np.where(
                    state < -1,
                    -2 - state,
                    state,
                )
            else:
                state = np.clip(state, -1, 1)
            trajectory[i, :] = state
        return trajectory


class PolynomialMotionGenerator(MotionGenerator):
    def __init__(
        self,
        duration: float,
        generator_timestep: float,
        backend_timestep: float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        noise_scale_min: Optional[float] = None,
        noise_scale_max: Optional[float] = None,
        max_derivative: float = 0.2,
    ):
        super().__init__(
            duration,
            generator_timestep,
            backend_timestep,
            lower_limits,
            upper_limits,
            noise_scale_min,
            noise_scale_max,
        )
        self.max_derivative = max_derivative

    def _generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        # Sample an endpoint that is reachable with the given maximum derivative
        end_limit_lower = np.clip(
            start - self.max_derivative / self.duration + 0.05, -0.9, 0.9
        )
        end_limit_upper = np.clip(
            start + self.max_derivative / self.duration - 0.05, -0.9, 0.9
        )
        end = np.random.uniform(end_limit_lower, end_limit_upper)
        while True:
            a = np.random.uniform(-1.0, 1.0, size=4)
            b = np.random.uniform(-1.0, 1.0, size=4)
            c, d = self.solve(0.0, start, self.duration, end, a, b)
            if np.all(
                self.calc_max_abs_derivative(a, b, c) < self.max_derivative
            ) and np.all(self.calc_max_abs_value(a, b, c, d) < 1.0):
                break
        t = np.arange(0, self.duration, self.generator_timestamp)[:, None]
        return self.eval_fun(t, a, b, c, d)

    @staticmethod
    def eval_fun(t, a, b, c, d):
        return a * t**3 + b * t**2 + c * t + d

    @staticmethod
    def eval_der(t, a, b, c):
        return a * t**2 + b * t + c

    @staticmethod
    def solve(t1, y1, t2, y2, a, b):
        c = (y1 - y2 - a * (t1**3 - t2**3) - b * (t1**2 - t2**2)) / (t1 - t2)
        d = y1 - a * t1**3 - b * t1**2 - c * t1
        return c, d

    def calc_max_abs_derivative(self, a, b, c):
        t_max = b / (3 * a)
        return np.max(
            [
                np.abs(self.eval_der(np.zeros(4), a, b, c)),
                np.where(
                    np.logical_and(0 < t_max, t_max < self.duration),
                    np.abs(self.eval_der(t_max, a, b, c)),
                    np.zeros(4),
                ),
                np.abs(self.eval_der(np.ones(4) * self.duration, a, b, c)),
            ],
            axis=0,
        )

    def calc_max_abs_value(self, a, b, c, d):
        sqrt_component = b**2 / (9 * a**2) - c / (3 * a)
        root = np.sqrt(np.where(sqrt_component > 0, sqrt_component, np.nan))
        t_max1 = -b / (3 * a) + root
        t_max2 = -b / (3 * a) - root
        return np.max(
            [
                np.where(
                    np.logical_and(0 < t_max1, t_max1 < self.duration),
                    np.abs(self.eval_fun(t_max1, a, b, c, d)),
                    np.zeros_like(t_max1),
                ),
                np.where(
                    np.logical_and(0 < t_max2, t_max2 < self.duration),
                    np.abs(self.eval_fun(t_max2, a, b, c, d)),
                    np.zeros_like(t_max2),
                ),
            ],
            axis=0,
        )


class LinearMotionGenerator(MotionGenerator):
    def __init__(
        self,
        duration: float,
        generator_timestep: float,
        backend_timestep: float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        noise_scale_min: float,
        noise_scale_max: float,
        max_ctrl_diff: float,
    ):
        super().__init__(
            duration,
            generator_timestep,
            backend_timestep,
            lower_limits,
            upper_limits,
            noise_scale_min,
            noise_scale_max,
        )
        self.max_ctrl_diff = max_ctrl_diff

    def _generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        trajectory = np.nan * np.ones((int(self.duration / self.generator_timestep), 4))
        trajectory[0] = start
        slope = np.random.uniform(-self.max_ctrl_diff, self.max_ctrl_diff, size=4)
        for i in range(1, int(self.duration / self.generator_timestep)):
            trajectory[i] = trajectory[i - 1] + slope
            neg_slope = np.random.uniform(-self.max_ctrl_diff, -1e-6, size=4)
            pos_slope = np.random.uniform(1e-6, self.max_ctrl_diff, size=4)
            slope = np.where(trajectory[i] < self.lower_limits, pos_slope, slope)
            slope = np.where(trajectory[i] > self.upper_limits, neg_slope, slope)
            trajectory[i] = np.clip(trajectory[i], self.lower_limits, self.upper_limits)
        return trajectory


class SplineMotionGenerator(MotionGenerator):
    def __init__(
        self,
        duration: float,
        generator_timestep: float,
        backend_timestep: float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        noise_scale_min: float,
        noise_scale_max: float,
        points_mean: Sequence[float],
        points_std: Sequence[float],
        distance_between_points_s: float,
        max_avg_ctrl_diff: float,
    ):
        super().__init__(
            duration,
            generator_timestep,
            backend_timestep,
            lower_limits,
            upper_limits,
            noise_scale_min,
            noise_scale_max,
        )
        self.points_mean = np.array(points_mean)
        self.points_std = np.array(points_std)
        self.steps_between_points = round(
            distance_between_points_s / self.generator_timestep
        )
        self.max_avg_ctrl_diff = max_avg_ctrl_diff

    def _generate_signal(
        self, start: np.ndarray, last_pos_traj: Optional[np.ndarray]
    ) -> np.ndarray:
        points = [start]
        num_steps = round(self.duration / self.generator_timestep)
        for i in range(0, num_steps, self.steps_between_points):
            points.append(
                np.random.normal(loc=self.points_mean, scale=self.points_std, size=4)
            )
        f = scipy.interpolate.make_interp_spline(
            np.concatenate(
                (
                    np.arange(0, num_steps, self.steps_between_points) - 1,
                    [num_steps - 1],
                ),
                axis=0,
            ),
            np.stack(points, axis=0),
        )
        return f(np.arange(num_steps))
