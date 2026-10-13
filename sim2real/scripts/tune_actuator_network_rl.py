import logging
import os
import subprocess
from pathlib import Path

import hydra
import numpy as np
import omegaconf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sim2real.util.gpu_memory import limit_gpu_memory_fraction

logger = logging.getLogger(__name__)

TARGET_ACCURACY_KEY = "sim_error_deg_avg/val/position/step1000"
FAILURE_TARGET_ACCURACY = 100.0


@hydra.main(
    version_base=None, config_path="config", config_name="tune_actuator_network_rl"
)
def main(cfg: omegaconf.DictConfig) -> float:
    raw_data_path = Path(__file__).parents[2] / "exps" / cfg.exp_name / "raw_data.pkl"
    config_path = Path.cwd() / "train_actuator_network_rl_config"
    # Caps the memory pool of the training subprocess, which inherits the environment of
    # this process. No JAX GPU client is created here, so this process itself does not
    # occupy any GPU memory.
    limit_gpu_memory_fraction()
    # There seems to be no way to clear all JAX caches between training runs, so running the training script in the same
    # process leaks memory. For some reason the training script also does not work with Python's subprocessing module,
    # resulting in some cryptic errors. So this abomination is the only way that I found to run multiple training runs
    # and reliably cleaning the memory in between runs.
    training_run = subprocess.run(
        f"python -m sim2real.scripts.train_actuator_network_rl --config-path {config_path} "
        f"--config-name config hydra.run.dir={Path.cwd()} hydra.output_subdir=null "
        f"exp_name={cfg.exp_name} raw_data_path={raw_data_path}",
        shell=True,
        env={**os.environ, "PYTHONPATH": hydra.utils.get_original_cwd()},
    )
    if training_run.returncode != 0:
        logger.warning(
            f"Training run failed with exit code {training_run.returncode}, "
            f"reporting an accuracy of {FAILURE_TARGET_ACCURACY}"
        )
        return FAILURE_TARGET_ACCURACY

    event_accumulator = EventAccumulator(str(Path.cwd()))
    event_accumulator.Reload()
    if TARGET_ACCURACY_KEY not in event_accumulator.Tags()["scalars"]:
        logger.warning(f"Did not find {TARGET_ACCURACY_KEY} in the tensorboard logs")
        return FAILURE_TARGET_ACCURACY

    scalars = sorted(
        (scalar.step, scalar.value)
        for scalar in event_accumulator.Scalars(TARGET_ACCURACY_KEY)
    )
    target_accuracy = float(np.mean([value for _, value in scalars[-10:]]))

    if np.isnan(target_accuracy):
        return FAILURE_TARGET_ACCURACY
    return target_accuracy


if __name__ == "__main__":
    # run the main function
    main()
