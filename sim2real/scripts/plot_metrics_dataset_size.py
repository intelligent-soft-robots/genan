import argparse
import re
from pathlib import Path

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FixedFormatter, FixedLocator

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Plot metrics from YAML file with dual model types"
)
parser.add_argument("yaml_file", type=str, help="Path to YAML file")
parser.add_argument(
    "--runs",
    nargs="+",
    required=True,
    help="List of run names to include in the plot, or '*' for all matching runs",
)
parser.add_argument("--metric", type=str, required=True, help="Metric to plot")
parser.add_argument(
    "--y_label", type=str, default="Metric Value", help="Label for y-axis"
)
parser.add_argument(
    "--y_lim",
    nargs=2,
    type=float,
    default=None,
    metavar=("MIN", "MAX"),
    help="Y-axis limits (optional)",
)
parser.add_argument("--scale-factor", type=float, default=1.0)
parser.add_argument("--portrait", action="store_true")
parser.add_argument(
    "--output", type=str, default=None, help="Output filename (optional)"
)
args = parser.parse_args()

# Load YAML data
with open(args.yaml_file, "r") as f:
    data = yaml.safe_load(f)


# Function to extract trajectory count and model type from run name
def extract_run_info(run_name):
    """Extract trajectory count and model type from run name"""
    match = re.search(r"(\d+)trajs(_single_model)?$", run_name)
    if match:
        count = int(match.group(1))
        model_type = "single_model" if match.group(2) else "ensemble"
        return count, model_type
    return None, None


# Handle wildcard for runs
if len(args.runs) == 1 and args.runs[0] == "*":
    runs = [run for run in data.keys() if extract_run_info(run)[0] is not None]
    if not runs:
        raise ValueError("No runs found with the '*_<num>trajs' pattern")
    args.runs = runs
    print(f"Automatically selected runs: {', '.join(args.runs)}")

# Organize runs by trajectory count and model type
run_groups = {}
for run in args.runs:
    count, model_type = extract_run_info(run)
    if count is None:
        print(
            f"Warning: Run '{run}' doesn't contain trajectory count pattern. Skipping."
        )
        continue

    if count not in run_groups:
        run_groups[count] = {"ensemble": None, "single_model": None}
    run_groups[count][model_type] = run

if not run_groups:
    raise ValueError("No valid runs found with trajectory count pattern")

# Get sorted list of trajectory counts
traj_counts = sorted(run_groups.keys())


# Extract metrics data
def get_metric_data(run, metric, scale_factor):
    try:
        metric_data = data[run][metric]
        val = float(metric_data[0]) * scale_factor
        ci_low = float(metric_data[1]) * scale_factor
        ci_high = float(metric_data[2]) * scale_factor
        return val, val - ci_low, ci_high - val
    except KeyError:
        raise ValueError(f"Metric '{metric}' not found in run '{run}'")
    except (TypeError, IndexError):
        raise ValueError(f"Invalid data format for {run}.{metric}")


# Prepare data structures
metric_ens, metric_single = [], []
metric_ens_err, metric_single_err = [], []

for count in traj_counts:
    group = run_groups[count]

    # Process ensemble run
    if group["ensemble"]:
        run = group["ensemble"]
        val, low, high = get_metric_data(run, args.metric, args.scale_factor)
        metric_ens.append(val)
        metric_ens_err.append((low, high))
    else:
        metric_ens.append(np.nan)
        metric_ens_err.append((np.nan, np.nan))

    # Process single model run
    if group["single_model"]:
        run = group["single_model"]
        val, low, high = get_metric_data(run, args.metric, args.scale_factor)
        metric_single.append(val)
        metric_single_err.append((low, high))
    else:
        metric_single.append(np.nan)
        metric_single_err.append((np.nan, np.nan))

# Convert to numpy arrays
traj_counts = np.array(traj_counts)
metric_ens = np.array(metric_ens)
metric_single = np.array(metric_single)

# Create plot
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot ensemble runs
(line_ens,) = ax.plot(
    traj_counts,
    metric_ens,
    "o-",
    color="C0",
    linewidth=2.5,
    markersize=4,
    label="ensemble",
)

# Plot single model runs
(line_single,) = ax.plot(
    traj_counts,
    metric_single,
    "s-",
    color="C3",
    linewidth=2.5,
    markersize=4,
    label="no ensemble",
)

# Add error bars
ax.errorbar(
    traj_counts,
    metric_ens,
    yerr=np.array(metric_ens_err).T,
    fmt="none",
    ecolor="C0",
    elinewidth=2.0,
    capsize=4,
    capthick=2.0,
    alpha=0.8,
)

ax.errorbar(
    traj_counts,
    metric_single,
    yerr=np.array(metric_single_err).T,
    fmt="none",
    ecolor="C3",
    elinewidth=2.0,
    capsize=4,
    capthick=2.0,
    alpha=0.8,
)

# Set logarithmic x-axis with FIXED TICKS
ax.set_xscale("log")
ax.set_xlabel("dataset size ($\#$ trajectories)", fontsize=14, labelpad=10)

# Explicitly set ticks and labels to prevent TikZ issues
ax.xaxis.set_major_locator(FixedLocator(traj_counts))
ax.xaxis.set_major_formatter(FixedFormatter([str(int(x)) for x in traj_counts]))
ax.tick_params(axis="x", labelsize=12, rotation=45, pad=5)

# Configure y-axis
ax.set_ylabel(args.y_label, fontsize=14, labelpad=10)
ax.tick_params(axis="y", labelsize=12)
if args.y_lim:
    ax.set_ylim(args.y_lim)

# Add grid and style elements
ax.grid(axis="both", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_alpha(0.4)
ax.spines["bottom"].set_alpha(0.4)

# Create legend
ax.legend(loc="best", fontsize=12)
golden_ratio = 1.6180

# Adjust layout and save/show
plt.tight_layout()
if args.output:
    output_path = Path(__file__).parents[2] / "plots" / args.output
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    tikz_output_path = output_path.with_suffix(".tex")
    matplot2tikz.save(output_path.with_suffix(".tex"))
    # matplot2tikz removes the custom tick labels, this is a hack to fix that
    with tikz_output_path.open("r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "xtick style={color=black}," in line:
            new_lines.append(
                f"xtick={{{', '.join([str(int(x)) for x in traj_counts])}}},\n"
            )
        if r"\begin{axis}[" in line:
            if args.portrait:
                new_lines.append("golden width portrait=0.73\\linewidth,")
            else:
                new_lines.append("golden width landscape=0.8\\linewidth,")
    with tikz_output_path.open("w") as f:
        f.writelines(new_lines)

    print(f"Saved plots to {output_path.stem}.pdf and {output_path.stem}.tex")
plt.show()
