from pathlib import Path

import hydra
import numpy as np
import omegaconf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sim2real.util.gpu_memory import limit_gpu_memory_fraction


def compute_objective(run_logs: Path) -> float:
    event_accumulator = EventAccumulator(str(run_logs))
    event_accumulator.Reload()
    key = "Reward / Total reward (mean)"
    scalars = []
    for scalar in event_accumulator.Scalars(key):
        scalars.append((scalar.step, scalar.value))
        scalars.sort(key=lambda x: x[0])

    return -np.mean([s[1] for s in scalars[-10:]])


@hydra.main(version_base=None, config_path="config", config_name="tune_rl_agent")
def main(cfg: omegaconf.DictConfig) -> float:
    limit_gpu_memory_fraction()
    # Imported here and not at module level, because the import initializes a JAX GPU
    # client. At module level that would also happen in the sweeper process, which then
    # preallocates a share of the GPU memory that it never uses for the whole sweep.
    from sim2real.scripts.train_rl_agent import train_rl_agent

    cfg.agent.cfg.experiment.checkpoint_interval = 0
    train_rl_agent(cfg)
    return compute_objective(Path.cwd())


if __name__ == "__main__":
    # run the main function
    main()
