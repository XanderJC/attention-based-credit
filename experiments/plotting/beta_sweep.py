import wandb
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 2
plt.rcParams["font.family"] = "Futura"

graph_colour = "black"

plt.rcParams["axes.edgecolor"] = graph_colour  # Spinespip
plt.rcParams["xtick.color"] = graph_colour  # X-axis tick marks
plt.rcParams["ytick.color"] = graph_colour  # Y-axis tick marks
plt.rcParams["axes.labelcolor"] = graph_colour  # Axis labels

colours = ["#1fbdaa", "#94ba22", "#9f5000", "#984ea3", "#cb9707"]

BASE_PATH = resource_filename("abcrl", "/..")


def plot_results(res, save=False):
    """'Plot results, where res is a dictionary of rewards per timestep
    we group by abc, rlhf, uniform and plot standard deviation over time."""
    runs = [[] for _ in range(11)]
    for key, value in res.items():
        i = int(int(key.split("_")[1]) / 10)
        # runs[i].append(value[-10:])
        runs[i].append(value)
    print([len(run) for run in runs])

    runs = [np.array(run).mean(axis=1) for run in runs]

    mean_perf = [np.mean(run) for run in runs]
    std_perf = [np.std(run) for run in runs]
    beta = [i / 10 for i in range(11)]

    fig, ax = plt.subplots()

    alphas = [0.1 + 0.07 * i for i in range(len(beta))]

    for i in range(len(beta)):
        ax.bar(
            beta[i],
            mean_perf[i],
            width=0.05,
            yerr=std_perf[i],
            color=colours[0],
            align="center",
            alpha=alphas[i],
            ecolor="grey",
            capsize=8,
            error_kw={"elinewidth": 2},
        )

    # Make x and y tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.annotate(
        "",
        xy=(0, 8.3),  # coordinates of point to which arrow points
        xytext=(0, 8.75),  # coordinates where text starts
        arrowprops=dict(arrowstyle="->", color=colours[1], linestyle="-", linewidth=3),
    )
    ax.annotate(
        "Equivalent to RLHF",
        xy=(-0.03, 8.8),
        weight="bold",
        fontsize=18,
        color=colours[1],
    )

    ax.set_ylim(7, 9.1)

    ax.set_xlabel(r"Beta - $\beta$", fontweight="bold")
    ax.set_ylabel("Mean Reward", fontweight="bold")
    # Set the background color of the figure and axes as transparent
    ax.patch.set_facecolor("none")

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="both", which="minor", length=3, width=1, color=graph_colour)
    ax.tick_params(axis="both", which="major", length=6, width=2, color=graph_colour)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    if save:
        plt.savefig(f"{BASE_PATH}/results/figures/beta_sweep.png", bbox_inches="tight")
    # plt.show()


api = wandb.Api()

if args.use_wandb:
    runs = api.runs("alex-abc/beta_sweep_seeded")

    res = {}
    for run in runs:
        rewards = run.history()["env/reward_mean"].to_list()
        if len(rewards) == 251:
            res[run.name] = rewards

    if args.save_res_local:
        with open(f"{BASE_PATH}/results/numerics/beta_sweep.pkl", "wb") as f:
            pickle.dump(res, f)

else:
    try:
        with open(f"{BASE_PATH}/results/numerics/beta_sweep.pkl", "rb") as f:
            res = pickle.load(f)
    except:
        print(
            "No local results found. Please run with --use_wandb to fetch results from wandb."
        )

plot_results(res, save=args.save_fig)
