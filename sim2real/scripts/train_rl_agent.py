import logging
import subprocess
from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import skrl.trainers.jax
from flax import nnx

from sim2real.rl_tasks.agent import create_agent
from sim2real.rl_tasks.custom_trainer import CustomSequentialTrainer
from sim2real.rl_tasks.pamy_reacher.pamy_reacher import PamyReacherEnv
from sim2real.rl_tasks.skrl_env_wrapper import SkrlEnvWrapper
from sim2real.util.config_utils import get_an_training_config

skrl.config.jax.backend = "jax"

logger = logging.getLogger(__name__)


def create_env(cfg: omegaconf.DictConfig | Dict[str, Any]) -> PamyReacherEnv:
    if cfg["seed"] is not None:
        seed = cfg
    else:
        seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)

    if (
        "actuator_net_exp_name" in cfg["backend"]
        and cfg["backend"]["actuator_net_exp_name"] is not None
    ):
        an_train_cfg = get_an_training_config(
            cfg["backend"]["actuator_net_exp_name"], cfg["backend"]["actuator_net_name"]
        )

        backend = hydra.utils.instantiate(cfg["backend"])
        env = hydra.utils.instantiate(
            cfg["env"],
            backend=backend,
            num_history_states=an_train_cfg["actuator_net"]["num_history_states"],
            step_size=an_train_cfg["actuator_net"]["step_size"],
            delta_history=an_train_cfg["actuator_net"]["delta_history"],
        )
    else:
        backend = hydra.utils.instantiate(cfg["backend"])
        env = hydra.utils.instantiate(
            cfg["env"],
            backend=backend,
        )
    return env


def train_rl_agent(cfg: omegaconf.DictConfig) -> None:
    if "seed" in cfg and cfg.seed is not None:
        seed = cfg.seed
    else:
        seed = np.random.randint(0, 2**32 - 1)
    rngs = nnx.Rngs(env=seed)
    env = SkrlEnvWrapper(create_env(cfg), rngs)

    agent = create_agent(
        cfg.agent, env.observation_space, env.action_space, env.num_instances
    )

    cfg_trainer = skrl.trainers.jax.sequential.SequentialTrainerCfg(
        timesteps=cfg.num_timesteps,
        headless=True,
        disable_progressbar=False,
        close_environment_at_exit=True,
    )
    trainer = CustomSequentialTrainer(env=env, agents=agent, cfg=cfg_trainer)
    trainer.train()


@hydra.main(version_base=None, config_path="config", config_name="train_rl_agent")
def main(cfg: omegaconf.DictConfig) -> None:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        logger.info(f"Git commit: {sha}")
    except subprocess.CalledProcessError:
        logger.info("Not in a git repository")
    train_rl_agent(cfg)


if __name__ == "__main__":
    main()
