# sweep.py
"""Parameter sweeps for the readout project.

    python sweep.py --sweep t1        # single qubit, vary T1
    python sweep.py --sweep zeta      # two qubits, vary the cross-dispersive shift (leak = 0)
    python sweep.py --sweep leak      # two qubits, vary the linear leakage (zeta = 0)
    python sweep.py --sweep all
    python sweep.py --sweep zeta --replot    # redraw from the saved JSON without rerunning

Each point = generate a dataset, train the network, evaluate the baselines, for several seeds.
Results go to <results-dir>/sweeps/sweep_<name>.json and <results-dir>/figures/sweep_<name>.png.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from baselines import LABELS

T_RO = 2.0
RESULTS_DIR = "results"


def json_path(name):
    return os.path.join(RESULTS_DIR, "sweeps", f"sweep_{name}.json")


def fig_path(name):
    return os.path.join(RESULTS_DIR, "figures", f"sweep_{name}.png")

SWEEPS = {
    "t1": dict(
        flag="--T1", values=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0], n_qubits=1,
        fixed={"--sigma": 1.0},
        sizes={"--n-train": 15000, "--n-val": 2000, "--n-test": 4000},
        xlabel="Qubit T1 (µs)", log=True, vline=T_RO, title="Readout fidelity vs. T1"),
    "zeta": dict(
        flag="--zeta", values=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0], n_qubits=2,
        fixed={"--sigma": 1.0, "--leak": 0.0},
        sizes={"--n-train": 5000, "--n-val": 500, "--n-test": 1500},   # per joint state (x4)
        xlabel="Cross-dispersive shift ζ (rad/µs)", log=False, vline=None,
        title="Two-qubit readout vs. cross-dispersive shift"),
    "leak": dict(
        flag="--leak", values=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5], n_qubits=2,
        fixed={"--sigma": 1.0, "--zeta": 0.0},
        sizes={"--n-train": 5000, "--n-val": 500, "--n-test": 1500},
        xlabel="Linear signal leakage", log=False, vline=None,
        title="Two-qubit readout vs. signal leakage"),
}

STYLES = {                      # kind: (legend label, marker); drawn in this order
    "net":        ("Neural network", "o"),
    "gbm":        ("Gradient boosting", "D"),
    "lda":        ("Linear (LDA, all channels)", "v"),
    "lda_indep":  ("Linear (LDA, own channels)", "P"),
    "matched":    ("Matched filter (own channels)", "s"),
    "integrated": ("Integrated threshold (own channels)", "^"),
}


def run(module_args):
    res = subprocess.run([sys.executable, "-m", *module_args], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Command failed: {module_args}")
    return res.stdout


def parse_fidelities(output, arch):
    """Mean fidelity per method from the printed lines (the number right after the colon)."""
    names = {**LABELS, "net": arch.upper()}
    found = {}
    for kind, label in names.items():
        m = re.search(rf"^\s*{re.escape(label)}\s*:\s*([0-9.]+)", output, re.M)
        if m:
            found[kind] = float(m.group(1))
    return found


def flatten(d):
    return [str(x) for kv in d.items() for x in kv]


def run_sweep(name, args):
    cfg = SWEEPS[name]
    fixed = dict(cfg["fixed"])
    if args.sigma is not None:
        fixed["--sigma"] = args.sigma
    data_path, ckpt_dir = f"data/sweep_{name}.pt", f"checkpoints/sweep_{name}/"

    results = {}                                   # kind -> [per value -> [per seed]]
    for v in cfg["values"]:
        per_seed = {}
        for seed in args.seeds:
            print(f"--- [{name}] {cfg['flag']} = {v}, seed = {seed} ---")
            run(["data.generate_data", "--n-qubits", str(cfg["n_qubits"]),
                 cfg["flag"], str(v), "--t-ro", str(T_RO), "--seed", str(seed),
                 "--data-path", data_path, *flatten(fixed), *flatten(cfg["sizes"])])
            out = run(["train.train", "--epochs", str(args.epochs), "--arch", args.arch,
                       "--device", args.device, "--seed", str(seed),
                       "--data-path", data_path, "--checkpoint-dir", ckpt_dir])
            found = parse_fidelities(out, args.arch)
            if not found:
                raise RuntimeError(f"Could not parse any fidelity from:\n{out}")
            for k, f in found.items():
                per_seed.setdefault(k, []).append(f)
            print("   " + " | ".join(f"{k}: {f:.4f}" for k, f in found.items()))
        for k, vals in per_seed.items():
            results.setdefault(k, []).append(vals)

    payload = {"sweep": name, "flag": cfg["flag"], "values": cfg["values"], "fixed": fixed,
               "seeds": args.seeds, "arch": args.arch, "results": results}
    os.makedirs(os.path.dirname(json_path(name)), exist_ok=True)
    with open(json_path(name), "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def plot(payload):
    name, cfg = payload["sweep"], SWEEPS[payload["sweep"]]
    plt.figure(figsize=(8, 5))
    for kind, (label, marker) in STYLES.items():
        if kind not in payload["results"]:
            continue
        if kind == "net":
            label = {"cnn": "1D CNN", "gru": "GRU"}[payload["arch"]]
        arr = np.array(payload["results"][kind])           # (n_values, n_seeds)
        plt.errorbar(payload["values"], arr.mean(1), yerr=arr.std(1), marker=marker,
                     capsize=3, linewidth=1.8, label=label)
    if cfg["vline"] is not None:
        plt.axvline(cfg["vline"], color="gray", linestyle="--", label=f"Readout window ({cfg['vline']} µs)")
    if cfg["log"]:
        plt.xscale("log")
    fixed = ", ".join(f"{k.lstrip('-')} = {v}" for k, v in payload["fixed"].items())
    plt.title(f"{cfg['title']}\n({fixed}; mean ± std over {len(payload['seeds'])} seeds)", fontsize=10)
    plt.xlabel(cfg["xlabel"])
    plt.ylabel("Assignment fidelity" + (" (mean over qubits)" if cfg["n_qubits"] > 1 else ""))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left" if cfg["n_qubits"] > 1 else "lower right", fontsize=8)
    plt.tight_layout()
    path = fig_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description="Readout sweeps")
    p.add_argument("--sweep", choices=[*SWEEPS, "all"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--arch", choices=["cnn", "gru"], default="cnn")
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--sigma", type=float, default=None, help="override the noise level of the sweep")
    p.add_argument("--results-dir", default="results", help="root folder for figures and result files")
    p.add_argument("--replot", action="store_true", help="redraw from saved JSON, no simulation")
    args = p.parse_args()
    global RESULTS_DIR
    RESULTS_DIR = args.results_dir

    for name in (SWEEPS if args.sweep == "all" else [args.sweep]):
        if args.replot:
            with open(json_path(name)) as f:
                payload = json.load(f)
        else:
            payload = run_sweep(name, args)
        plot(payload)


if __name__ == "__main__":
    main()