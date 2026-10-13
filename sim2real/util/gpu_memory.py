import os

from hydra.core.hydra_config import HydraConfig

TOTAL_MEM_FRACTION = 0.8


def limit_gpu_memory_fraction() -> None:
    """Cap the XLA memory pool of the calling job so that all parallel jobs of a hydra
    sweep fit on the GPU.

    Must be called before the first import that initializes a JAX GPU client.
    """
    n_jobs = HydraConfig.get().launcher.n_jobs
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(TOTAL_MEM_FRACTION / n_jobs)
