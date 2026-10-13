import argparse
import logging
from pathlib import Path

import numpy as np
import scipy
import yaml

from sim2real.util.success_rate_cis import calculate_success_rate_cis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", type=str, nargs="+")
    parser.add_argument("--outname", type=str)
    args = parser.parse_args()

    results = {}
    for policy_name in args.names:
        successes_l2_rad = {0.1: [], 0.075: [], 0.05: []}
        successes_avg_rad = {0.1: [], 0.075: [], 0.05: [], 1.0: []}
        successes_max_rad = {0.1: [], 0.075: [], 0.05: []}
        successes_l2_deg = {5.0: [], 4.0: [], 3.0: [], 2.0: []}
        successes_avg_deg = {5.0: [], 4.0: [], 3.0: [], 2.0: [], 1.0: []}
        successes_max_deg = {5.0: [], 4.0: [], 3.0: [], 2.0: []}
        errors = []
        path = Path(__file__).parents[2] / "out" / policy_name
        with (path / "evaluate_transfer_config" / "config.yaml").open("r") as f:
            eval_transfer_cfg = yaml.safe_load(f)

        data = np.load(next(path.glob("*.npz")), allow_pickle=True)
        task = data["task"]
        data_from_o80_backend = "o80_iteration" in data

        reset_len = 3000

        last_o80_iteration = None
        success_rates = []
        data_files = sorted(
            [p for p in path.iterdir() if p.suffix == ".npz"],
            key=lambda p: int(p.stem[len("ep_") :]),
        )
        for p in data_files:
            if p.suffix == ".npz":
                data = np.load(p, allow_pickle=True)
                observations = data["observations"]
                num_obs_per_dof = observations.shape[-1] // 4
                goals = observations[..., 0, -4:]
                # last_positions = data["positions"][..., -1, :]
                last_positions = (
                    data["o80_positions"][None, -1, :]
                    if "o80_positions" in data
                    else data["positions"][:, -1]
                )
                err = last_positions - goals
                errors.append(err)
                for d, l in successes_l2_rad.items():
                    l.append(np.linalg.norm(err * 180.0 / np.pi, axis=-1, ord=2) < d)
                for d, l in successes_avg_rad.items():
                    l.append(np.mean(np.abs(err), axis=-1) < d)
                for d, l in successes_max_rad.items():
                    l.append(np.max(np.abs(err), axis=-1) < d)
                for d, l in successes_l2_deg.items():
                    l.append(np.linalg.norm(err * 180.0 / np.pi, axis=-1, ord=2) < d)
                for d, l in successes_avg_deg.items():
                    l.append(np.mean(np.abs(err) * 180.0 / np.pi, axis=-1) < d)
                for d, l in successes_max_deg.items():
                    l.append(np.max(np.abs(err) * 180.0 / np.pi, axis=-1) < d)

        errors = np.abs(np.concatenate(errors))
        num_rollouts = errors.shape[0]
        print(
            f"=== {policy_name} ({num_rollouts} rollouts) =================================================================="
        )
        to_deg = 180.0 / np.pi
        results[policy_name] = {}
        with np.printoptions(precision=4, suppress=True):
            ci_error = scipy.stats.bootstrap(
                (errors.mean(axis=-1),), np.mean
            ).confidence_interval
            print(
                f"Avg. error: {errors.mean() * to_deg:.4f}° ({ci_error.low * to_deg:.4f}° -- {ci_error.high * to_deg:.4f}°) / {errors.mean():.4f} rad ({ci_error.low:.4f} rad -- {ci_error.high:.4f} rad), all errors: {errors.mean(axis=0) * 180 / np.pi}° / {errors.mean(axis=0)} rad"
            )
            results[policy_name]["avg_pos_errs"] = [
                float(errors.mean()),
                float(ci_error.low),
                float(ci_error.high),
            ]
            for name, res in [
                ("successes_l2_rad", successes_l2_rad),
                ("successes_avg_rad", successes_avg_rad),
                ("successes_max_rad", successes_max_rad),
                ("successes_l2_deg", successes_l2_deg),
                ("successes_avg_deg", successes_avg_deg),
                ("successes_max_deg", successes_max_deg),
            ]:
                print(f"{name}: ", end="")
                for d, s in res.items():
                    s = np.concatenate(s, axis=-1)
                    successes = s.sum()
                    ci_low, ci_high = calculate_success_rate_cis(
                        successes, num_rollouts
                    )
                    success_rate = successes / num_rollouts
                    results[policy_name][f"{name}_{d}"] = [
                        float(success_rate),
                        float(ci_low),
                        float(ci_high),
                    ]
                    print(f"{d}: {success_rate:.2f}, ", end="")
            print("")

    if args.outname is not None:
        with (
            (Path(__file__).parents[2] / "out" / args.outname)
            .with_suffix(".yaml")
            .open("w") as f
        ):
            yaml.dump(results, f)
