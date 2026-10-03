# data/generate_data.py
import os
import torch

from sim.engine import simulate_readout
from sim.engine_2q import simulate_readout_2q
from train.arguments import parse_args


def main():
    args = parse_args()
    phys = dict(T1=args.T1, t_ro=args.t_ro, dt=args.dt, kappa=args.kappa,
                chi=args.chi, eps=args.eps, sigma=args.sigma)
    if args.n_qubits == 2:
        phys.update(zeta=args.zeta, leak=args.leak)
        simulate = simulate_readout_2q
    else:
        simulate = simulate_readout
    print(f"n_qubits = {args.n_qubits}, physics: {phys}")

    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    out = {"phys": phys, "n_qubits": args.n_qubits}
    for i, (name, n) in enumerate(sizes.items()):
        tlist, X, y = simulate(n, seed=args.seed + 1000 + i, **phys)   # independent seed per split
        y = torch.from_numpy(y).reshape(len(y), -1)                    # always (N, n_qubits)
        out[name] = {"x": torch.from_numpy(X), "y": y}
        print(f"{name}: records {tuple(X.shape)}, labels {tuple(y.shape)}")
    out["tlist"] = torch.tensor(tlist, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.data_path) or ".", exist_ok=True)
    torch.save(out, args.data_path)
    print(f"Saved dataset to {args.data_path}")


if __name__ == "__main__":
    main()