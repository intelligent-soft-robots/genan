# import gymnasium as gym
#
# from . import agents
# from .pamy_reacher import PamyReacherEnvCfg
#
# gym.register(
#     id="PamyReacher-v1",
#     entry_point="sim2real.rl_tasks.isaaclab.pamy_reacher.pamy_reacher:PamyReacherEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": PamyReacherEnvCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroHandPPORunnerCfg",
#     },
# )
#
# gym.register(
#     id="PamyReacherTorque-v1",
#     entry_point="sim2real.rl_tasks.isaaclab.pamy_reacher.pamy_reacher_torque:PamyReacherTorqueEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": PamyReacherEnvCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroHandPPORunnerCfg",
#     },
# )
