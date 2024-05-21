import matplotlib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from adm.utils.toy import Toy

# Sourced from: https://raw.githubusercontent.com/AvivNavon/nash-mtl/main/experiments/toy/utils.py


def get_opt(scale):
    F = Toy(scale=scale)
    yy = -8.3552
    x = np.linspace(-7, 7, 1000)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).float()

    Ys = F.batch_forward(Xs)
    opt_l0, opt_l1 = Ys[Ys.sum(dim=1).argmin()]
    return opt_l0, opt_l1


def plot_2d_pareto(trajectories: dict, scale):
    """Adaptation of code from: https://github.com/Cranial-XIX/CAGrad"""
    fig, ax = plt.subplots(figsize=(6, 5))

    F = Toy(scale=scale)

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(torch.from_numpy(res["traj"])))

    yy = -8.3552
    x = np.linspace(-7, 7, 100)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).float()
    Ys = F.batch_forward(Xs)

    plt.plot(
        Ys.numpy()[:, 0],
        Ys.numpy()[:, 1],
        "-",
        linewidth=8,
        color="#72727A",
        label="Pareto Front",
    )  # Pareto front

    for i, tt in enumerate(losses):
        plt.scatter(
            tt[0, 0],
            tt[0, 1],
            color="k",
            s=150,
            zorder=10,
            label="Initial Point" if i == 0 else None,
        )
        colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, tt.shape[0]))
        # colors = matplotlib.cm.plasma_r(np.linspace(0.1, 0.6, tt.shape[0]))
        plt.scatter(tt[:, 0], tt[:, 1], color=colors, s=5, zorder=9, rasterized=True)

    sns.despine()
    ax.set_xlabel(r"$\mathcal{L}_1$", size=30)
    ax.set_ylabel(r"$\mathcal{L}_2$", size=30)
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    opt_l0, opt_l1 = get_opt(scale)
    plt.scatter(opt_l0, opt_l1, marker='*', color="black", s=500, zorder=100)

    return ax, fig
