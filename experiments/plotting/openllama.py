import wandb
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
from pkg_resources import resource_filename

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--save_res_local", action="store_true")
parser.add_argument("--save_fig", action="store_true")
args = parser.parse_args()

plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
plt.rcParams["font.style"] = "normal"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 2
plt.rcParams["font.family"] = "Futura"

graph_colour = "black"

plt.rcParams["axes.edgecolor"] = graph_colour  # Spinespip
plt.rcParams["xtick.color"] = graph_colour  # X-axis tick marks
plt.rcParams["ytick.color"] = graph_colour  # Y-axis tick marks
plt.rcParams["axes.labelcolor"] = graph_colour  # Axis labels

colours = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ff7f00"]
colours = ["#46b4a7", "#96ac4d", "#b55b00", "#392f5a", "#dea811"]
colours = ["#1fbdaa", "#94ba22", "#9f5000", "#984ea3", "#cb9707"]

BASE_PATH = resource_filename("abcrl", "/..")


def plot_results(res, save=False):
    """'Plot results, where res is a dictionary of rewards per timestep
    we group by abc, rlhf, uniform and plot standard deviation over time."""
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    axes = [ax1, ax2]
    handles = []
    means = {}
    for method, name, marker, c, coords, _ in zip(
        ["abc", "rlhf", "uniform", "abcde", "abcde2"],
        ["ABC", "RLHF", "Uniform", "ABC-D", "ABC-D2"],
        ["o", "s", "D", "X", "P"],
        colours,
        [(45, 10.0), (45, 7.9), (45, 6.7), (45, 9.2), (45, 8.5)],
        range(3),
    ):
        runs = []
        for key, value in res.items():
            if method.lower() == key.split("_")[0]:
                runs.append(value[:201])
        runs = np.array(runs)
        mean = np.mean(runs, axis=0)
        means[method] = mean
        std = np.std(runs, axis=0)
        print(method, mean[-1], std[-1])

        """
        window = 1
        mean = np.convolve(mean, np.ones(window) / window, mode="valid")
        std = np.convolve(std, np.ones(window) / window, mode="valid")
        """

        (line,) = axes[0].plot(mean, color=c, zorder=10 if method == "abc" else 0)

        axes[0].fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            alpha=0.2,
            color=c,
            edgecolor="none",
        )

        axes[0].plot(
            range(0, len(mean), 40),
            mean[0 : len(mean) : 40],
            marker,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=c,
            label=method,
            zorder=11,
        )
        handle = mlines.Line2D(
            [],
            [],
            color=c,
            marker=marker,
            markersize=8,
            markeredgewidth=2,
            markerfacecolor="white",
            linestyle="-",
            label=name,
        )
        handles.append(handle)

        runs = []
        for key, value in values.items():
            if method.lower() == key.split("_")[0]:
                runs.append(value[:201])
        runs = np.array(runs)
        mean = np.mean(runs, axis=0)
        means[method] = mean
        std = np.std(runs, axis=0)
        print(method, mean[-1], std[-1])
        (line,) = axes[1].plot(mean, color=c)
        axes[1].plot(
            range(0, len(mean), 40),
            mean[0 : len(mean) : 40],
            marker,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=c,
            label=method,
        )
        axes[1].fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            alpha=0.2,
            color=c,
            edgecolor="none",
        )

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

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

        ax.set_xlabel("Timestep")

        axes[0].set_ylabel("Reward")
        axes[1].set_ylabel("PPO Value Loss")
    """
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
        fontsize=14,
    )"""
    legend = axes[1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        fontsize=12,
        ncol=3,
    )
    for text, color in zip(legend.get_texts(), colours):
        text.set_color("white")
        text.set_weight("bold")
        text.set_path_effects([path_effects.withStroke(linewidth=6, foreground=color)])

    legend2 = axes[0].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        fontsize=12,
        ncol=3,
    )
    for text, color in zip(legend2.get_texts(), colours):
        text.set_color("white")
        text.set_weight("bold")
        text.set_path_effects([path_effects.withStroke(linewidth=6, foreground=color)])

    # ax.xlim = (0, 250)
    # ax.set_ylim(0, 11)
    # Set the background color of the figure and axes as transparent
    # fig.patch.set_facecolor("none")

    fig1.tight_layout()
    fig2.tight_layout()
    if save:
        fig1.savefig(
            f"{BASE_PATH}/results/figures/openllama_reward.png", bbox_inches="tight"
        )
        fig2.savefig(
            f"{BASE_PATH}/results/figures/openllama_value.png", bbox_inches="tight"
        )
    # plt.show()


api = wandb.Api()

if args.use_wandb:
    runs = api.runs("alex-abc/openllama_seeded")

    res = {}
    values = {}
    for run in runs:
        rewards = run.history()["env/reward_mean"].to_list()
        value_loss = run.history()["ppo/loss/value"].to_list()
        if len(rewards) == 201:
            res[run.name] = rewards
            values[run.name] = value_loss

    if args.save_res_local:
        with open(f"{BASE_PATH}/results/numerics/openllama.pkl", "wb") as f:
            pickle.dump((res, values), f)

else:
    try:
        with open(f"{BASE_PATH}/results/numerics/openllama.pkl", "rb") as f:
            res, values = pickle.load(f)
    except:
        print(
            "No local results found. Please run with --use_wandb to fetch results from wandb."
        )

plot_results(res, save=args.save_fig)
