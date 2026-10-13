import argparse
import pickle
import re
from pathlib import Path

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# Helper function to darken a color
def darken_color(color, factor=0.7):
    """Darken a color by reducing its RGB values"""
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)


# Set up argument parsing
parser = argparse.ArgumentParser(description="Plot metrics from pickle file")
parser.add_argument("pickle_file", type=str, help="Path to pickle file")
parser.add_argument(
    "--ans", nargs="+", required=True, help="List of analysis names to include"
)
parser.add_argument(
    "--names",
    nargs="+",
    required=True,
    help=(
        "Human-readable names for the analyses (same order as --ans)."
    ),
)
parser.add_argument(
    "--step", type=int, required=True, help="Step number to extract data for"
)
parser.add_argument(
    "--metric",
    type=str,
    required=True,
    choices=["abs_error", "disag"],
    help="Metric type to plot (abs_error or disag)",
)
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["train", "val", "test"],
    help="Data split to use (train, val, test)",
)
parser.add_argument(
    "--output", type=str, default=None, help="Output filename (optional)"
)
parser.add_argument("--scaling-factor", type=float, default=1.0)
parser.add_argument("--xlabel", type=str)
parser.add_argument("--ylabel", type=str, default="Value")
parser.add_argument("--ylim", type=float, default=None)
parser.add_argument("--rotate-tick-labels", action="store_true")
parser.add_argument("--portrait", action="store_true")
parser.add_argument("--main", type=int, default=0)
args = parser.parse_args()

# Validate input
if len(args.ans) != len(args.names):
    raise ValueError("Number of analyses must match number of names")

# Load pickle data
with open(args.pickle_file, "rb") as f:
    data = pickle.load(f)

# Get base step and calculate index
base_step = data["step"]
index = (args.step - 1) // base_step  # Calculate index from step

# Validate index
if index < 0:
    raise ValueError(f"Step {args.step} is smaller than base step {base_step}")

# Extract values and confidence intervals
values = []
lower_errors = []
upper_errors = []

for an_name in args.ans:
    try:
        # Access the nested metric data
        metric_data = data[an_name][args.metric][args.split]

        # Extract values at the specified index
        mean_val = float(metric_data[0][index])
        ci_low = float(metric_data[1][index])
        ci_high = float(metric_data[2][index])

        values.append(mean_val)
        lower_errors.append(mean_val - ci_low)
        upper_errors.append(ci_high - mean_val)

    except KeyError:
        raise ValueError(
            f"Analysis '{an_name}', metric '{args.metric}', or split '{args.split}' not found"
        )
    except IndexError:
        raise ValueError(f"Step {args.step} (index {index}) out of range for {an_name}")
    except TypeError:
        raise ValueError(
            f"Invalid data format for {an_name}.{args.metric}.{args.split}"
        )

# Create plot
plt.figure(figsize=(10, 6))
ax = plt.gca()

values = np.array(values)
lower_errors = np.array(lower_errors)
upper_errors = np.array(upper_errors)

# Use Set2 Qualitative colormap
color_indices = np.arange(8)
color_indices = np.concatenate(
    (color_indices[1 : args.main + 1], [0], color_indices[args.main + 1 :])
)
colors = cm.Set2(color_indices)
edge_colors = [darken_color(c) for c in colors]  # Properly darkened edges

# Plot bars with error bars
x_pos = np.arange(len(values))
bar_width = 0.7

bars = ax.bar(
    x_pos,
    values * args.scaling_factor,
    width=bar_width,
    color=colors,
    edgecolor=edge_colors,
    linewidth=3.0,
    yerr=[lower_errors * args.scaling_factor, upper_errors * args.scaling_factor],
    error_kw={
        "elinewidth": 3,
        "capthick": 3,
        "capsize": 3,
        "ecolor": "dimgray",
    },
)

# One or two backslashes, so that the separator survives shell quoting
LINE_BREAK_PATTERN = re.compile(r"\\+")


def set_tick_labels(labels: list[str]) -> None:
    ax.set_xticklabels(
        labels,
        fontsize=12,
        rotation=45 if args.rotate_tick_labels else 0,
        ha="right" if args.rotate_tick_labels else "center",
        rotation_mode="anchor",
    )


def to_tikz_label(name: str) -> str:
    parts = LINE_BREAK_PATTERN.split(name)
    if len(parts) == 1:
        return name
    # Braces group the label so that pgfplots does not misparse the line break
    return "{" + r"\\".join(parts) + "}"


has_line_breaks = any(LINE_BREAK_PATTERN.search(name) for name in args.names)
plot_names = [LINE_BREAK_PATTERN.sub("\n", name) for name in args.names]
tikz_names = [to_tikz_label(name) for name in args.names]

ax.set_xlabel(args.xlabel)
# Configure axes and labels
ax.set_ylabel(args.ylabel, fontsize=14, labelpad=10)
ax.set_xticks(x_pos)
set_tick_labels(plot_names)
ax.set_ylim(0, args.ylim)
ax.tick_params(axis="y", labelsize=12)

# Add grid and style elements
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_alpha(0.4)
ax.spines["bottom"].set_alpha(0.4)
ax.tick_params(axis="both", which="both", length=0)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    label_position = 0.04 * plt.gca().get_ylim()[1]

    # ax.text(
    #     bar.get_x() + bar.get_width() / 2.0,
    #     label_position,
    #     f"{height:.2f}",
    #     ha="center",
    #     va="center",
    #     fontsize=14,
    #     color="white",
    #     fontweight="bold",
    # )

golden_ratio = 1.6180
if args.portrait:
    plt.gca().set_box_aspect(golden_ratio)
# Adjust layout and save/show
plt.tight_layout()
if args.output:
    output_path = Path(__file__).parents[2] / "plots" / args.output
    tikz_output_path = output_path.with_suffix(".tex")
    plt.savefig(
        output_path.with_suffix(".pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    # pgfplots needs "\\" instead of a newline to break a tick label
    set_tick_labels(tikz_names)
    matplot2tikz.save(tikz_output_path)
    set_tick_labels(plot_names)
    print(f"Saved plot to {output_path}")
    with tikz_output_path.open("r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if "xtick style={color=black}" in line:
            line = line.replace("color=black", "color=black,draw=none")
        if args.rotate_tick_labels and "rotate=45.0,anchor=east" in line:
            line = line.replace("anchor=east", "anchor=north east")
        new_lines.append(line)
        if r"\begin{axis}[" in line:
            if args.portrait:
                new_lines.append("golden width portrait=0.73\\linewidth,")
            else:
                new_lines.append("golden width landscape=0.8\\linewidth,")
        if "xticklabels" in line:
            style = "yshift=3pt,align=center" if has_line_breaks else "yshift=3pt"
            new_lines.append(f"xticklabel style={{{style}}},\n")
        with tikz_output_path.open("w") as f:
            f.writelines(new_lines)

plt.show()
