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


colours = ["#1fbdaa", "#94ba22", "#9f5000"]

BASE_PATH = resource_filename("abcrl", "/..")


def plot_results(rewards_dict, kl_dict, save=False):
    """'Plot results, where res is a dictionary of rewards per timestep
    we group by abc, rlhf, uniform and plot standard deviation over time."""
    # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    handles = []
    kls_list = []
    rewards_list = []
    for method, name, c, marker, _ in zip(
        ["abc", "rlhf", "uniform"],
        ["ABC", "RLHF", "Uniform"],
        colours,
        ["o", "s", "D"],
        range(2),
    ):
        rewards = []
        kls = []
        for key, value in rewards_dict.items():
            if method.lower() == key.split("_")[0]:
                if kl_dict[key][-1] > 0:
                    rewards.append(value)
                    kls.append(kl_dict[key])
        rewards = np.array(rewards)
        mean_rewards = np.mean(rewards, axis=0)
        rewards_list.append(mean_rewards)

        kls = np.array(kls)
        mean_kls = np.mean(kls, axis=0)
        kls_list.append(mean_kls)

        time = np.linspace(1, 0, len(mean_kls))  # Generate a sequence from 0 to 1

        for i in range(len(mean_kls)):
            """
            ax.scatter(
                mean_kls[i],
                mean_rewards[i],
                color=c,
                alpha=time[i],
                label=name if i == 150 else "",
                edgecolors="none",
                marker=marker,
            )"""
            ax.scatter(
                mean_kls[i],
                mean_rewards[i],
                facecolor="white",
                alpha=time[i],
                edgecolors=c,
                lw=2,
                marker=marker,
                zorder=10,
                label=name if i == 0 else "",
            )
        ax.plot(
            mean_kls,
            mean_rewards,
            color=c,
            alpha=0.2,
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

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    legend = fig.legend(
        handles=handles,
        loc="lower right",
        fontsize=16,
        bbox_to_anchor=(0.95, 0.2),
    )
    for text, color in zip(
        legend.get_texts(),
        colours,
    ):
        text.set_color("white")
        text.set_weight("bold")
        text.set_path_effects([path_effects.withStroke(linewidth=6, foreground=color)])

    x_min = -1
    x_max = 21
    y_min = 0
    y_max = 10.5
    x = np.linspace(x_min, x_max, 100)
    y = 0.2 * x

    plt.plot(x, y, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    plt.fill_between(x, y, y_max, color="green", alpha=0.05, edgecolor="none")
    plt.fill_between(x, y_min, y, color="red", alpha=0.05, edgecolor="none")

    ax.set_xlabel(r"KL Divergence - $D_{KL}(\pi_{\theta}||\pi_{ref})$")
    ax.set_ylabel("Reward")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0, 5, 10, 15, 20])
    # Set the background color of the figure and axes as transparent
    # fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="both", which="minor", length=3, width=1, color=graph_colour)
    ax.tick_params(axis="both", which="major", length=6, width=2, color=graph_colour)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    if save:
        plt.savefig(f"{BASE_PATH}/results/figures/kl_tradeoff.png", bbox_inches="tight")
    # plt.show()


api = wandb.Api()

if args.use_wandb:
    runs = api.runs("alex-abc/IMDb_seeded")

    rewards_dict = {}
    kl_dict = {}
    for run in runs:
        rewards = run.history()["env/reward_mean"].to_list()
        kls = run.history()["objective/kl"].to_list()
        if len(rewards) == 151:
            rewards_dict[run.name] = rewards
            kl_dict[run.name] = kls

    if args.save_res_local:
        with open(f"{BASE_PATH}/results/numerics/kl_tradeoff.pkl", "wb") as f:
            pickle.dump((rewards_dict, kl_dict), f)

else:
    try:
        with open(f"{BASE_PATH}/results/numerics/kl_tradeoff.pkl", "rb") as f:
            rewards_dict, kl_dict = pickle.load(f)
    except:
        print(
            "No local results found. Please run with --use_wandb to fetch results from wandb."
        )

plot_results(rewards_dict, kl_dict, save=args.save_fig)
