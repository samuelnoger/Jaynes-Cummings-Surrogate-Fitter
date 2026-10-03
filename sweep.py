# sweep.py
import json
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

T1_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0]
SEEDS = [0, 1, 2]
T_RO = 2.0
SIGMA = 1.0
DATA_PATH = "data/sweep_dataset.pt"
CKPT_DIR = "checkpoints/sweep/"
RESULTS = "sweep_results.json"

PATTERNS = {
    "integrated": r"integrated threshold\s*:\s*([0-9.]+)",
    "matched":    r"matched filter\s*:\s*([0-9.]+)",
    "cnn":        r"CNN\s*:\s*([0-9.]+)",
}


def run(cmd):
    res = subprocess.run([sys.executable, "-m", *cmd], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout


def run_sweep():
    results = {k: [] for k in PATTERNS}            # each: list over T1 of list over seeds

    for t1 in T1_VALUES:
        per_seed = {k: [] for k in PATTERNS}
        for seed in SEEDS:
            print(f"--- T1 = {t1} us, seed = {seed} ---")
            run(["data.generate_data",
                 "--T1", str(t1), "--t-ro", str(T_RO), "--sigma", str(SIGMA),
                 "--n-train", "15000", "--n-test", "4000",
                 "--seed", str(seed), "--data-path", DATA_PATH])
            out = run(["train.train",
                       "--epochs", "20", "--arch", "cnn", "--device", "mps",
                       "--seed", str(seed),
                       "--data-path", DATA_PATH, "--checkpoint-dir", CKPT_DIR])
            for k, pat in PATTERNS.items():
                m = re.search(pat, out)
                if m is None:
                    raise RuntimeError(f"Could not parse '{k}' from output:\n{out}")
                per_seed[k].append(float(m.group(1)))
            print("   " + " | ".join(f"{k}: {per_seed[k][-1]:.4f}" for k in PATTERNS))
        for k in PATTERNS:
            results[k].append(per_seed[k])

    with open(RESULTS, "w") as f:
        json.dump({"t1": T1_VALUES, "t_ro": T_RO, "sigma": SIGMA, "seeds": SEEDS, **results}, f, indent=2)
    return results


def plot(results):
    styles = {"cnn": ("1D CNN", "o"), "matched": ("Matched filter", "s"),
              "integrated": ("Integrated threshold", "^")}
    plt.figure(figsize=(8, 5))
    for k, (label, marker) in styles.items():
        arr = np.array(results[k])                  # (n_T1, n_seeds)
        plt.errorbar(T1_VALUES, arr.mean(1), yerr=arr.std(1), marker=marker,
                     capsize=3, linewidth=1.8, label=label)
    plt.axvline(T_RO, color="gray", linestyle="--", label=f"Readout window ({T_RO} µs)")
    plt.xscale("log")
    plt.xlabel("Qubit T1 (µs)")
    plt.ylabel("Assignment fidelity")
    plt.title(f"Readout fidelity vs. T1 (σ = {SIGMA}, mean ± std over {len(SEEDS)} seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("t1_fidelity_sweep.png", dpi=300)
    print("Saved t1_fidelity_sweep.png")


if __name__ == "__main__":
    plot(run_sweep())