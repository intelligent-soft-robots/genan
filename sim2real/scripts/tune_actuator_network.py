import os
import subprocess
from pathlib import Path

import hydra
import numpy as np
import omegaconf


@hydra.main(
    version_base=None, config_path="config", config_name="tune_actuator_network"
)
def main(cfg: omegaconf.DictConfig) -> float:
    raw_data_path = Path(__file__).parents[2] / "exps" / cfg.exp_name / "raw_data.pkl"
    config_path = Path().parent / "train_actuator_network_config"
    # There seems to be no way to clear all JAX caches between training runs, so running the training script in the same
    # process leaks memory. For some reason the training script also does not work with Python's subprocessing module,
    # resulting in some cryptic errors. So this abomination is the only way that I found to run multiple training runs
    # and reliably cleaning the memory in between runs.
    subprocess.run(
        f"python -m sim2real.scripts.train_actuator_network --config-path {config_path.absolute()} "
        f"--config-name config hydra.run.dir={Path.cwd()} hydra.output_subdir=null "
        f"exp_name={cfg.exp_name} raw_data_path={raw_data_path} save_models=False",
        shell=True,
        env={
            "PYTHONPATH": hydra.utils.get_original_cwd(),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(0.8 / cfg.n_jobs),
            **os.environ,
        },
    )

    results_path = Path("results.npz")
    if results_path.exists():
        results = np.load(results_path)
        subsampling_factor = round(1000 / cfg.target_frequency)
        target_error = results["val_pos_errs"][1000 // subsampling_factor - 1].mean()
    else:
        return 100.0
    if np.isnan(target_error):
        return 100.0
    else:
        return float(target_error)


if __name__ == "__main__":
    # run the main function
    main()
