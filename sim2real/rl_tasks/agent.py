from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import skrl.agents.jax.ppo
import skrl.memories.jax
from skrl.models.jax import Model
from skrl.utils.model_instantiators.jax import deterministic_model

from sim2real.rl_tasks.skrl_gaussian import gaussian_model


def create_models(
    num_hidden_layers: int,
    num_neurons: int,
    activation_fn: str,
    observation_space: gymnasium.spaces.Box,
    action_space: gymnasium.spaces.Box,
    initial_log_std: float,
    device: Optional[str],
) -> Dict[str, Model]:
    network = [
        {
            "name": "net",
            "input": "OBSERVATIONS",
            "layers": [num_neurons] * (num_hidden_layers + 1),
            "activations": activation_fn,
        }
    ]

    if np.any(np.isinf(action_space.low)) or np.any(np.isinf(action_space.high)):
        policy_output = "ACTIONS"
    else:
        # Otherwise the soft clipping of tanh(ACTIONS) is problematic
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
        policy_output = "tanh(ACTIONS)"

    models = {
        "policy": gaussian_model(
            observation_space=observation_space,
            action_space=action_space,
            network=network,
            output=policy_output,
            initial_log_std=initial_log_std,
            max_log_std=0.5,
            device=device,
            clip_actions=True,
        ),
        "value": deterministic_model(
            observation_space=observation_space,
            action_space=action_space,
            network=network,
            output="ONE",
            device=device,
        ),
    }
    for role, model in models.items():
        model.init_state_dict(role=role)

        # Convert the model parameters to single precision
        model.state_dict = jax.tree.map(
            lambda x: (
                x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x
            ),
            model.state_dict,
        )
    return models


def create_agent(
    cfg: omegaconf.DictConfig | Dict[str, Any],
    observation_space: gymnasium.spaces.Box,
    action_space: gymnasium.spaces.Box,
    num_envs: int,
    device: Optional[str] = None,
) -> skrl.agents.jax.ppo.PPO:
    if isinstance(cfg, omegaconf.DictConfig):
        cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    cfg["models"]["observation_space"] = observation_space
    cfg["models"]["action_space"] = action_space

    memory = skrl.memories.jax.RandomMemory(
        memory_size=cfg["cfg"]["rollouts"],
        num_envs=num_envs,
        device=device,
    )

    if (
        "observation_preprocessor_kwargs" in cfg["cfg"]
        and "size" in cfg["cfg"]["observation_preprocessor_kwargs"]
    ):
        cfg["cfg"]["observation_preprocessor_kwargs"]["size"] = observation_space
    agent = hydra.utils.instantiate(
        cfg,
        memory=memory,
        observation_space=observation_space.shape[0],
        action_space=action_space.shape[0],
        device=device,
    )
    agent.init()
    return agent


def load_agent(
    path: Path,
    agent_cfg: omegaconf.DictConfig | Dict[str, Any],
    observation_space: gymnasium.spaces.Box,
    action_space: gymnasium.spaces.Box,
    num_envs: int,
    device: Optional[str] = None,
) -> skrl.agents.jax.ppo.PPO:
    agent = create_agent(agent_cfg, observation_space, action_space, num_envs, device)
    agent.load(str(path))
    return agent
