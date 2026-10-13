# Installation
The process requires `python==3.10.*`.
Older versions might work, but are not tested.
Later versions do not work because the robot framework does not support `python>3.11`.

Install the robot framework according to the [PAM documentation](https://intelligent-soft-robots.github.io/pam_documentation/A1_overview_and_installation.html).

Install the package: `pip install .`

# Usage

## Data collection
Start the MuJoCo simulation of PAMY2 with `launch_pam_mujoco data_collection` or connect to the real robot by running `o80_pamy2`.

In a separate process, start the data collection script with `python -m sim2real.scripts.collect_data exp_name=<exp_name>`. 
See `sim2real/scripts/config/collect_data.yaml` for more config options.
Each config value can be overridden from the commandline by adding `<name>=<value>` to the command.

The collected data will be stored in `exps/<current date>_<exp_name>`.

## Data processing

Run `python -m sim2real.scripts.process_data exp_name=<exp_name>`.
Note that exp_name needs to be the full name of the data directory in the `exps` folder (including the date when the data was recorded).
See `sim2real/scripts/config/process_data.yaml` for config options.

## Training the actuator network

Run `python -m sim2real.scripts.train_actuator_network exp_name=<exp_name> run_name=<run_name> loss=<loss> end_effector=<end_effector>`, where `<exp_name>` is the name of the dataset collected beforehand, `<run_name>` the name of the actuator net training run (determines the directory where the trained network is saved), `<loss>` is either `torque_prediction_loss` or `position_accuracy_loss`, and `<end_effector>` is the end effector that is currently attached to the robot (either `racket` or `cup`).
See `sim2real/scripts/config/train_actuator_network.yaml` for more config options.

The trained actuator network is stored in `exp/<exp_name>/actuator_network/<run_name>/model`.

## Running the policy training

Run `python -m sim2real.scripts.train_rl_agent run_name=<run_name> agent=<agent> env=<env> backend.context=<context> backend.simulate_encoder_quantization=<true/false>`, where `run_name` is a name for the run `<agent>` is one of the RL agents in `scripts/config/agent`, `<env>` is `pamy_reacher`, `pamy_ball_in_cup`, or `pamy_table_tennis`. 

`backend.context` needs to be set according to the environment.

| `env` value         | `backend.context` |
|---------------------|-------------------|
| `pamy_reacher`      | `racket`          |
| `pamy_ball_in_cup`  | `ball_in_cup`     |
| `pamy_table_tennis` | `table_tennis`    | 

Use `backend.simulate_encoder_quantization` to enable or disable the quantization of the joint positions (to simulate the finite joint encoder resolution on the real robot).
See `sim2real/scripts/config/train_rl_agent.yaml` for more config options.
The logs and agent checkpoints will be stored in `logs/train/<date>/<time>_<run_name>`.

## Evaluating trained policies

Run `python -  sim2real.scripts.evaluate_transfer train_logs=<path_to_train_logs> backend=<backend> num_episodes=<num_episodes> backend.simulate_encoder_quantization=<true/false>`, where `<path_to_train_logs>` is the path to the training logs generated in the previous step, `<backend>` is either `mjx` (for evaluation in sim) or `real` for the evaluation on the real robot, and `<num_episodes>` is the number of episodes to execute.
`backend.simulate_encoder_quantization` should be set to `false` on the real system and set to the same value that was used during training when evaluating in sim.