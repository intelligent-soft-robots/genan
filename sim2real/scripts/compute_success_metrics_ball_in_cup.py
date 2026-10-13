import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from sim2real.util.success_rate_cis import calculate_success_rate_cis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outname", type=str)
    args = parser.parse_args()

    num_rollouts = 100
    num_successes_from_experiments = {
        "2026-01-21/16-11-15_ball_in_cup_green_action_pen5000_disag_pen01_start": 75,
        "2026-01-21/16-11-13_ball_in_cup_green_action_pen5000_disag_pen0": 56,
        "2026-01-21/16-11-13_ball_in_cup_green_action_pen5000_disag_pen0_single_model": 74,
        "2026-01-22/15-04-24_ball_in_cup_green_action_pen1000_disag_pen01": 55,
        "2026-01-22/15-30-15_ball_in_cup_green_action_pen5000_disag_pen01_torque_an": 76,
    }

    results = {}
    for policy_name, successes in num_successes_from_experiments.items():
        results[policy_name] = {}
        with np.printoptions(precision=4, suppress=True):
            ci_low, ci_high = calculate_success_rate_cis(successes, num_rollouts)
            results[policy_name]["success_rate"] = [
                float(successes / num_rollouts),
                float(ci_low),
                float(ci_high),
            ]

    if args.outname is not None:
        with (
            (Path(__file__).parents[2] / "out" / args.outname)
            .with_suffix(".yaml")
            .open("w") as f
        ):
            yaml.dump(results, f)
