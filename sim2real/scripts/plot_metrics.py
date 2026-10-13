import argparse
from pathlib import Path

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm


# Helper function to darken a color
def darken_color(color, factor=0.7):
    """Darken a color by reducing its RGB values"""
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)


# Set up argument parsing
parser = argparse.ArgumentParser(description="Plot metrics from YAML file")
parser.add_argument("yaml_file", type=str, help="Path to YAML file", nargs="+")
parser.add_argument(
    "--runs", nargs="+", required=True, help="List of run names to include in the plot"
)
parser.add_argument(
    "--names",
    nargs="+",
    help="Human-readable names for the runs (same order as --runs)",
)
parser.add_argument(
    "--groups",
    nargs="+",
)
parser.add_argument(
    "--metric", type=str, required=True, help="Metric to plot", nargs="+"
)
parser.add_argument(
    "--output", type=str, default=None, help="Output filename (optional)"
)
parser.add_argument("--scaling-factor", type=float, default=100.0)
parser.add_argument("--ylabel", type=str, default="success rate (%)")
parser.add_argument("--ylim", type=float, default=None)
parser.add_argument("--portrait", action="store_true")
args = parser.parse_args()

group_size = len(args.runs) // len(args.groups) if args.groups is not None else 1

# Validate input
if args.groups is None and len(args.runs) != len(args.names):
    raise ValueError("Number of runs must match number of names")

data = {}
for yaml_file in args.yaml_file:
    # Load YAML data
    with open(yaml_file, "r") as f:
        data.update(yaml.safe_load(f))


# Extract values and confidence intervals
values = []
lower_errors = []
upper_errors = []

for run in args.runs:
    metric_iter = iter(args.metric)
    metric = next(metric_iter)
    while metric not in data[run]:
        metric = next(metric_iter)
    try:
        metric_data = data[run][metric]
        val = float(metric_data[0])
        ci_low = float(metric_data[1])
        ci_high = float(metric_data[2])

        values.append(val)
        lower_errors.append(val - ci_low)
        upper_errors.append(ci_high - val)
    except KeyError:
        raise ValueError(f"Run '{run}' or metric '{metric}' not found in YAML")
    except (TypeError, IndexError):
        raise ValueError(f"Invalid data format for {run}.{metric}")

# Create plot
plt.figure(figsize=(10, 6))
ax = plt.gca()

values = np.array(values)
lower_errors = np.array(lower_errors)
upper_errors = np.array(upper_errors)


if args.groups is None:
    x = np.arange(len(values))
    # Use Set2 Qualitative colormap
    colors = cm.Set2(np.arange(8))
else:
    x = 0
    x_pos = []
    for i in range(len(values)):
        if i % group_size == 0:
            x += 1.6
        else:
            x += 0.85
        x_pos.append(x)
    colors = cm.Set2(np.tile(np.arange(group_size), len(args.runs) // group_size))

edge_colors = [darken_color(c) for c in colors]  # Properly darkened edges
# Plot bars with error bars - THICKER EDGES AND ERROR MARKERS
bar_width = 0.7

bars = ax.bar(
    x_pos,
    values * args.scaling_factor,
    width=bar_width,
    color=colors,
    edgecolor=edge_colors,
    linewidth=3.0,  # Thicker bar edges
    yerr=[lower_errors * args.scaling_factor, upper_errors * args.scaling_factor],
    error_kw={
        "elinewidth": 3,  # Thicker error lines
        "capthick": 3,  # Thicker error caps
        "capsize": 5,  # Slightly larger caps
        "ecolor": "dimgray",
    },
)

# Configure axes and labels - LARGER FONTS
ax.set_ylabel(args.ylabel, fontsize=14, labelpad=10)
# ax.set_xlabel("Configurations", fontsize=14, labelpad=10)
ax.set_xticks(x_pos)
if args.groups is not None:
    # TODO: Do this properly for group sizes != 3
    labels_pad_left = [""] * ((group_size - 1) // 2)
    labels_pad_right = [""] * (group_size - 1 - len(labels_pad_left))

    ax.set_xticklabels(
        [
            item
            for g in args.groups
            for item in (*labels_pad_left, g, *labels_pad_right)
        ],
        fontsize=12,
    )
    ax.legend(bars[:group_size], args.names)
else:
    ax.set_xticklabels(args.names, fontsize=12)  # rotation=45, ha="right",
ax.set_ylim(0, args.ylim)
if args.ylim == 100.0:
    ax.set_yticks(np.arange(0, 101, 10))
ax.tick_params(axis="y", labelsize=12)

# Add grid and style elements
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_alpha(0.4)
ax.spines["bottom"].set_alpha(0.4)
ax.tick_params(axis="both", which="both", length=0)

# Add value labels on bars - MOVED TO BOTTOM OF BAR
for bar in bars:
    height = bar.get_height()
    label_position = 0.04 * plt.gca().get_ylim()[1]

    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        label_position,
        str(int(height)) if args.ylim == 100.0 else f"{height:.2f}",
        ha="center",
        va="center",  # Centered vertically at the label position
        fontsize=18,
        color="white",  # White text for better contrast
        fontweight="bold",
    )

# Adjust layout and save/show
plt.tight_layout()
if args.output:
    plt.savefig(
        (Path(__file__).parents[2] / "plots" / args.output).with_suffix(".pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    tikz_output_path = (Path(__file__).parents[2] / "plots" / args.output).with_suffix(
        ".tex"
    )
    matplot2tikz.save(
        tikz_output_path,
    )
    print(f"Saved plot to {args.output}")
    golden_ratio = 1.6180
    with tikz_output_path.open("r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if r"\begin{axis}[" in line:
            if args.portrait:
                new_lines.append("golden width portrait=0.73\\linewidth,")
            else:
                new_lines.append("golden width landscape=0.8\\linewidth,")
    with tikz_output_path.open("w") as f:
        f.writelines(new_lines)

plt.show()
