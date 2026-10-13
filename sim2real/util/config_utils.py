from pathlib import Path
from typing import Any, Dict

from omegaconf import omegaconf


def get_an_training_config(
    exp_name: str, actuator_net_name: str, rl_training: bool = False
) -> omegaconf.DictConfig:
    exps_path = Path(__file__).parents[2] / "exps"
    if rl_training:
        an_path = (
            exps_path
            / exp_name
            / "actuator_network_rl"
            / actuator_net_name
            / "train_actuator_network_rl_config"
            / "config.yaml"
        )
    else:
        an_path = (
            exps_path
            / exp_name
            / "actuator_network"
            / actuator_net_name
            / "train_actuator_network_config"
            / "config.yaml"
        )
    return omegaconf.OmegaConf.load(an_path)


def get_an_input_size_from_cfg(an_cfg: Dict[str, Any] | omegaconf.DictConfig) -> int:
    return get_an_input_size(
        an_cfg["num_history_states"],
        an_cfg["velocity_input"],
        an_cfg["pressure_input"],
        an_cfg["position_history"],
        an_cfg["velocity_history"],
        an_cfg["control_history"],
    )


def get_an_input_size(
    num_history_states: int,
    velocity_input: bool,
    pressure_input: bool,
    position_history: bool,
    velocity_history: bool,
    control_history: bool,
) -> int:

    pos_context_len = num_history_states + 1 if position_history else 1
    if velocity_input:
        vel_context_len = num_history_states + 1 if velocity_history else 1
    else:
        vel_context_len = 0
    ctrl_context_len = num_history_states + 1 if control_history else 1
    press_context_len = num_history_states + 1 if pressure_input else 0

    return 4 * (
        pos_context_len + vel_context_len + ctrl_context_len + press_context_len
    )
