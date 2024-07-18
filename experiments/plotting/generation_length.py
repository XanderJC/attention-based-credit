import wandb
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
from pkg_resources import resource_filename

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--save_res_local", action="store_true")
parser.add_argument("--save_fig", action="store_true")
args = parser.parse_args()

graph_colour = "black"

plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
plt.rcParams["font.style"] = "normal"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titlecolor"] = graph_colour
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 2
plt.rcParams["font.family"] = "Futura"


plt.rcParams["axes.edgecolor"] = graph_colour  # Spines
plt.rcParams["xtick.color"] = graph_colour  # X-axis tick marks
plt.rcParams["ytick.color"] = graph_colour  # Y-axis tick marks
plt.rcParams["axes.labelcolor"] = graph_colour  # Axis labels

BASE_PATH = resource_filename("abcrl", "/..")

# colours = ["#377eb8", "#4daf4a", "#e41a1c"]
colours = ["#46b4a7", "#96ac4d", "#b55b00"]
# colours = ["#392f5a", "#c9d5a1", "#ff8811"]"#392f5a""#96ac4d"
colours = ["#1fbdaa", "#94ba22", "#9f5000", "#984ea3", "#cb9707"]


def plot_results(res, save=False):
    """'Plot results, where res is a dictionary of rewards per timestep
    we group by abc, rlhf, uniform and plot standard deviation over time."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    handles = []
    for num, (length, min_length, ax) in enumerate(
        zip(["30", "60", "110", "160"], ["20", "40", "90", "140"], axes)
    ):
        for method, name, marker, c, _ in zip(
            ["abc", "rlhf", "uniform", "abcde", "abcde2"],
            ["ABC", "RLHF", "Uniform", "ABC-D", "ABC-D2"],
            ["o", "s", "D", "X", "P"],
            colours,
            range(5),
        ):
            runs = []
            for key, value in res.items():
                if method.lower() == key.split("_")[0]:
                    if length == key.split("_")[3]:
                        # if value[-1] > 6:
                        runs.append(value[:101])
            if method == "rlhf":
                rlhf_runs = runs
            runs = np.array(runs)
            mean = np.mean(runs, axis=0)
            std = np.std(runs, axis=0)
            print(method, length, mean[-1], std[-1])

            (line,) = ax.plot(mean, color=c)
            ax.fill_between(
                range(len(mean)),
                mean - std,
                mean + std,
                alpha=0.2,
                color=c,
                edgecolor="none",
            )
            ax.plot(
                range(0, len(mean), 20),
                mean[0 : len(mean) : 20],
                marker,
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=c,
                label=method,
            )
            if num == 0:
                handle = mlines.Line2D(
                    [],
                    [],
                    color=c,
                    marker=marker,
                    markersize=10,
                    markeredgewidth=3,
                    markerfacecolor="white",
                    linewidth=3,
                    linestyle="-",
                    label=name,
                )
                handles.append(handle)

        if num == 3 or num == 2 or num == 1 or num == 0:
            inset_bounds = [0.4, 0.08, 0.45, 0.45]

            # Create the inset plot
            ax_inset = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=inset_bounds,
                bbox_transform=ax.transAxes,
            )
            for run in rlhf_runs:
                ax_inset.plot(run, color=colours[1], linewidth=0.5)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.grid(True)
            for spine in ax_inset.spines.values():
                spine.set_edgecolor("gray")
                # spine.set_linewidth(1.5)

        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.set_title(f"Length: {min_length} - {length}")
        ax.set_xlabel("Timestep")
        if num == 0:
            ax.set_ylabel("Reward")
        ax.set_ylim(0, 12.5)
        # Set the background color of the figure and axes as transparent
        # fig.patch.set_facecolor("none")
        ax.patch.set_facecolor("none")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(
            axis="both", which="minor", length=3, width=1, color=graph_colour
        )
        ax.tick_params(
            axis="both", which="major", length=6, width=2, color=graph_colour
        )

        # Remove the top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

    print(len(handles))
    legend = fig.legend(
        handles=handles,
        fontsize=16,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
    )
    for text, color in zip(legend.get_texts(), colours):
        text.set_color("white")
        text.set_weight("bold")
        text.set_path_effects([path_effects.withStroke(linewidth=6, foreground=color)])

    plt.tight_layout()
    if save:
        plt.savefig(
            f"{BASE_PATH}/results/figures/generation_length.png",
            bbox_inches="tight",
        )
    # plt.show()


api = wandb.Api()

if args.use_wandb:
    runs = api.runs("alex-abc/generation_length_seeded")

    res = {}
    for run in runs:
        rewards = run.history()["env/reward_mean"].to_list()
        if len(rewards) == 151:
            res[run.name] = rewards

    if args.save_res_local:
        with open(f"{BASE_PATH}/results/numerics/generation_length.pkl", "wb") as f:
            pickle.dump(res, f)

else:
    try:
        with open(f"{BASE_PATH}/results/numerics/generation_length.pkl", "rb") as f:
            res = pickle.load(f)
    except Exception as e:
        print(
            "No local results found. Please run with --use_wandb to fetch results from wandb."
        )

plot_results(res, save=args.save_fig)
