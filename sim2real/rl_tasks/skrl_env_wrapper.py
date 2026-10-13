from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from skrl.envs.wrappers.jax import Wrapper

from sim2real.rl_tasks.pamy_env import PamyEnv


class SkrlEnvWrapper(Wrapper):
    def __init__(self, env: PamyEnv, rngs: nnx.Rngs) -> None:
        super().__init__(env)
        self._state = None
        self.rngs = rngs
        self.reset_next_step = True

    def reset(self) -> Tuple[jax.Array, Dict[str, Any]]:
        self.reset_next_step = False
        (obs, info), self._state = self._unwrapped.reset(self.rngs.env())
        del info["episode"]
        return obs, info

    def step(
        self, actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Dict[str, Any]]:
        if self.reset_next_step:
            obs, info = self.reset()
            reward = jnp.zeros(self.num_envs, dtype=jnp.float32)
            terminated = jnp.zeros(self.num_envs, dtype=bool)
            truncated = jnp.zeros(self.num_envs, dtype=bool)
        else:
            (obs, reward, terminated, truncated, info), self._state = (
                self._unwrapped.step(actions, self._state, self.rngs.env())
            )
            if jnp.all(terminated) or jnp.all(truncated):
                self.reset_next_step = True
            else:
                if "episode" in info:
                    del info["episode"]
        return obs, reward[:, None], terminated[:, None], truncated[:, None], info

    def close(self) -> None:
        self._unwrapped.close()

    @property
    def num_envs(self) -> int:
        return self._unwrapped.num_instances

    def state(self) -> Union[np.ndarray, jax.Array]:
        """Get the environment state

        :raises NotImplementedError: Not implemented

        :return: State
        :rtype: np.ndarray or jax.Array
        """
        return None

    def render(self, *args, **kwargs) -> Any:
        """Render the environment

        :raises NotImplementedError: Not implemented

        :return: Any value from the wrapped environment
        :rtype: any
        """
        raise NotImplementedError
