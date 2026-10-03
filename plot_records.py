# plot_records.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from train.arguments import parse_args


def main():
    args = parse_args()
    data = torch.load(args.data_path)
    t = data["tlist"].numpy()
    x = data["train"]["x"].numpy()[:, :, :2]   # (N, T, 2): channels of qubit 1
    y = data["train"]["y"].numpy().reshape(len(x), -1)[:, 0]   # labels of qubit 1
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    names = {0: "prepared ground", 1: "prepared excited"}
    colors = {0: "tab:blue", 1: "tab:red"}

    # 1. Raw example records (I quadrature)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for lab, ax in zip((0, 1), axes):
        for rec in x[y == lab][:5]:
            ax.plot(t, rec[:, 0], lw=0.7, alpha=0.8)
        ax.set_title(names[lab]); ax.set_xlabel("t (µs)")
    axes[0].set_ylabel("I (noisy)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "raw_records.png"), dpi=150)

    # 2. Class-mean records with standard-error bands
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, q, label in zip(axes, (0, 1), ("I", "Q")):
        for lab in (0, 1):
            r = x[y == lab][:, :, q]
            m, se = r.mean(0), r.std(0) / np.sqrt(len(r))
            ax.plot(t, m, color=colors[lab], label=names[lab])
            ax.fill_between(t, m - 3 * se, m + 3 * se, color=colors[lab], alpha=0.25)
        ax.set_xlabel("t (µs)"); ax.set_ylabel(f"mean {label}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "mean_records.png"), dpi=150)

    # 3. Matched-filter score histograms
    kernel = x[y == 1].mean(0) - x[y == 0].mean(0)
    score = (x * kernel).sum((1, 2))
    plt.figure(figsize=(6, 3.2))
    for lab in (0, 1):
        plt.hist(score[y == lab], bins=80, alpha=0.6, color=colors[lab], label=names[lab])
    plt.xlabel("matched-filter score"); plt.ylabel("count"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "mf_histogram.png"), dpi=150)

    print(f"Saved figures to {fig_dir}")
    plt.show()


if __name__ == "__main__":
    main()