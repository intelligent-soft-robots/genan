import atexit
import contextlib
import sys
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from skrl import config, logger
from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.jax import MultiAgent
from skrl.trainers.jax.base import TrainerCfg
from skrl.trainers.jax.sequential import SequentialTrainerCfg
from skrl.utils import ScopedTimer


class CustomTrainer:

    def __init__(
        self,
        *,
        cfg: TrainerCfg,
        env: Wrapper | MultiAgentEnvWrapper,
        agents: Agent | MultiAgent | list[Agent] | list[MultiAgent],
        scopes: list[int] | None = None,
        post_interaction_callback: Optional[
            Callable[[Agent, Wrapper, int], None]
        ] = None,
    ) -> None:
        """Base trainer class for implementing custom trainers.

        :param cfg: Configuration dictionary.
        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param scopes: Number of environments for each simultaneous agent to train/evaluate on.
        """
        self.env = env
        self.agents = agents
        self.scopes = scopes if scopes is not None else []
        self._post_interaction_callback = post_interaction_callback

        self.cfg = cfg
        self.cfg.expand()

        # setup agents
        self.num_simultaneous_agents = 1
        self._setup_agents()

        # register environment closing if configured
        if self.cfg.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.jax.is_distributed:
            if config.jax.rank:
                self.cfg.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer.

        :return: Representation of the trainer as string.
        """
        string = f"Trainer: {type(self).__name__}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += (
            f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        )
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.scopes):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup simultaneous agents.

        :raises ValueError: Invalid setup.
        """
        # validate agents and their scopes
        if isinstance(self.agents, (tuple, list)):
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.scopes = [(0, self.env.num_envs)]
            # simultaneous agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.scopes):
                    logger.warning(
                        "The agents' scopes are empty. They will be generated to be as equal as possible"
                    )
                    self.scopes = [int(self.env.num_envs / len(self.agents))] * len(
                        self.agents
                    )
                    if sum(self.scopes):
                        self.scopes[-1] += self.env.num_envs - sum(self.scopes)
                    else:
                        raise ValueError(
                            f"The number of simultaneous agents ({len(self.agents)}) is greater than "
                            f"the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.scopes) != len(self.agents):
                    raise ValueError(
                        f"The number of simultaneous agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.scopes)})"
                    )
                elif sum(self.scopes) != self.env.num_envs:
                    raise ValueError(
                        f"The scopes ({sum(self.scopes)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # generate agents' scopes
                index = 0
                for i in range(len(self.scopes)):
                    index += self.scopes[i]
                    self.scopes[i] = (index - self.scopes[i], index)
            else:
                raise ValueError("A list of agents is expected")
        # non-simultaneous agent
        else:
            self.num_simultaneous_agents = 1
            self.scopes = [(0, self.env.num_envs)]

    def train(self) -> None:
        """Train a single/multi-agent.

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render environments
        - Record transitions
        - Post-interaction
        - Reset environments

        :raises AssertionError: If the method is called in a simultaneous agents setup.
        """
        assert self.num_simultaneous_agents == 1, (
            "This method is not allowed for simultaneous agents. "
            "Inherit from `Trainer` and reimplement the `.train()` method instead."
        )

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with contextlib.nullcontext():
                # compute actions
                with ScopedTimer() as timer:
                    actions, outputs = self.agents.act(
                        observations,
                        states,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )
                    self.agents.track_data(
                        "Stats / Inference time (ms)", timer.elapsed_time_ms
                    )

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = (
                        self.env.step(actions)
                    )
                    next_states = self.env.state()
                    self.agents.track_data(
                        "Stats / Env stepping time (ms)", timer.elapsed_time_ms
                    )

                # render the environments
                if not self.cfg.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.cfg.timesteps,
                )

                # log environment info
                if "episode" in infos:
                    for k, v in infos["episode"].items():
                        if isinstance(v, (jnp.ndarray, np.ndarray)) and v.size == 1:
                            v = v.item()
                        if isinstance(v, float):
                            self.agents.track_data(f"Info / {k}", v)

            # post-interaction
            self.agents.post_interaction(
                timestep=timestep, timesteps=self.cfg.timesteps
            )
            for p in jax.tree.flatten(self.agents.policy.state_dict)[0]:
                assert not jnp.any(jnp.isnan(p)), "Found NaN parameters after update."
            if self._post_interaction_callback is not None:
                self._post_interaction_callback(self.agents, self.env, timestep)

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - single environment
            else:
                # check condition to reset
                # - multi-agent
                if self.env.num_agents > 1:
                    should_reset = not self.env.agents
                # - single-agent
                else:
                    should_reset = terminated.any() or truncated.any()
                # explicit reset
                if should_reset:
                    with contextlib.nullcontext():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

    def eval(self) -> None:
        """Evaluate a single/multi-agent.

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render environments
        - Record transitions
        - Reset environments

        :raises AssertionError: If the method is called in a simultaneous agents setup.
        """
        assert self.num_simultaneous_agents == 1, (
            "This method is not allowed for simultaneous agents. "
            "Inherit from `Trainer` and reimplement the `.eval()` method instead."
        )

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with contextlib.nullcontext():
                # compute actions
                with ScopedTimer() as timer:
                    actions, outputs = self.agents.act(
                        observations,
                        states,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )
                    self.agents.track_data(
                        "Stats / Inference time (ms)", timer.elapsed_time_ms
                    )
                actions = (
                    actions
                    if self.cfg.stochastic_evaluation
                    else outputs.get("mean_actions", actions)
                )

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = (
                        self.env.step(actions)
                    )
                    next_states = self.env.state()
                    self.agents.track_data(
                        "Stats / Env stepping time (ms)", timer.elapsed_time_ms
                    )

                # render the environments
                if not self.cfg.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.cfg.timesteps,
                )

                # log environment info
                if "episode" in infos:
                    for k, v in infos["episode"].items():
                        if isinstance(v, (jnp.ndarray, np.ndarray)) and v.size == 1:
                            v = v.item()
                        if isinstance(v, float):
                            self.agents.track_data(f"Info / {k}", v)

            # post-interaction
            super(self.agents.__class__, self.agents).post_interaction(
                timestep=timestep, timesteps=self.cfg.timesteps
            )

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - single environment
            else:
                # check condition to reset
                # - multi-agent
                if self.env.num_agents > 1:
                    should_reset = not self.env.agents
                # - single-agent
                else:
                    should_reset = terminated.any() or truncated.any()
                # explicit reset
                if should_reset:
                    with contextlib.nullcontext():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states


# fmt: off
# [start-config-dict-jax]
SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-jax]
# fmt: on


class CustomSequentialTrainer(CustomTrainer):
    def __init__(
        self,
        *,
        env: Wrapper | MultiAgentEnvWrapper,
        agents: Agent | MultiAgent | list[Agent] | list[MultiAgent],
        scopes: list[int] | None = None,
        cfg: SequentialTrainerCfg | dict = {},
        post_interaction_callback: Optional[
            Callable[[Agent, Wrapper, int], None]
        ] = None,
    ) -> None:
        """Sequential trainer.

        Train agents sequentially, i.e., one after the other, in each interaction with the environment.

        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param scopes: Number of environments for each simultaneous agent to train/evaluate on.
        :param cfg: Configuration dictionary.
        """
        self.cfg: SequentialTrainerCfg
        super().__init__(
            env=env,
            agents=agents,
            scopes=scopes if scopes is not None else [],
            cfg=SequentialTrainerCfg(**cfg) if isinstance(cfg, dict) else cfg,
            post_interaction_callback=post_interaction_callback,
        )

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Train agents sequentially.

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render environments
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(True)
        else:
            self.agents.enable_training_mode(True)

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            super().train()
            return

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with contextlib.nullcontext():
                # compute actions
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.cfg.timesteps,
                        )
                        agent.track_data(
                            "Stats / Inference time (ms)", timer.elapsed_time_ms
                        )
                    _actions.append(actions)
                    _outputs.append(outputs)
                actions = jnp.vstack(_actions)

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = (
                        self.env.step(actions)
                    )
                    next_states = self.env.state()
                    elapsed_time_ms = timer.elapsed_time_ms
                    for agent in self.agents:
                        agent.track_data(
                            "Stats / Env stepping time (ms)", elapsed_time_ms
                        )

                # render the environments
                if not self.cfg.headless:
                    self.env.render()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=(
                            states[scope[0] : scope[1]] if states is not None else None
                        ),
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=(
                            next_states[scope[0] : scope[1]]
                            if next_states is not None
                            else None
                        ),
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )

            # log environment info
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, (jnp.ndarray, np.ndarray)) and v.size == 1:
                        v = v.item()
                    if isinstance(v, float):
                        for agent in self.agents:
                            agent.track_data(f"Info / {k}", v)

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)
                for p in jax.tree.flatten(agent.policy.state_dict)[0]:
                    assert not jnp.any(
                        jnp.isnan(p)
                    ), "Found NaN parameters after update."
            if self._post_interaction_callback is not None:
                self._post_interaction_callback(self.agents, self.env, timestep)

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - single environment
            else:
                raise RuntimeError(
                    "Sequential trainer is not supported for single environment"
                )

    def eval(self) -> None:
        """Evaluate agents sequentially.

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions (sequentially)
        - Interact with the environments
        - Render environments
        - Record transitions
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(False)
        else:
            self.agents.enable_training_mode(False)

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            super().eval()
            return

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with contextlib.nullcontext():
                # compute actions
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.cfg.timesteps,
                        )
                        agent.track_data(
                            "Stats / Inference time (ms)", timer.elapsed_time_ms
                        )
                    _actions.append(
                        actions
                        if self.cfg.stochastic_evaluation
                        else outputs.get("mean_actions", actions)
                    )
                    _outputs.append(outputs)
                actions = jnp.vstack(_actions)

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = (
                        self.env.step(actions)
                    )
                    next_states = self.env.state()
                    elapsed_time_ms = timer.elapsed_time_ms
                    for agent in self.agents:
                        agent.track_data(
                            "Stats / Env stepping time (ms)", elapsed_time_ms
                        )

                # render the environments
                if not self.cfg.headless:
                    self.env.render()

                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=(
                            states[scope[0] : scope[1]] if states is not None else None
                        ),
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=(
                            next_states[scope[0] : scope[1]]
                            if next_states is not None
                            else None
                        ),
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )

            # log environment info
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, (jnp.ndarray, np.ndarray)) and v.size == 1:
                        v = v.item()
                    if isinstance(v, float):
                        self.agents.track_data(f"Info / {k}", v)

            # post-interaction
            for agent in self.agents:
                super(agent.__class__, agent).post_interaction(
                    timestep=timestep, timesteps=self.cfg.timesteps
                )

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - single environment
            else:
                raise RuntimeError(
                    "Sequential trainer is not supported for single environment"
                )
