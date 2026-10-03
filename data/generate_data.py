# data/generate_data.py
import os
import torch

from sim.engine import simulate_readout
from train.arguments import parse_args


def main():
    args = parse_args()
    phys = dict(T1=args.T1, t_ro=args.t_ro, dt=args.dt, kappa=args.kappa,
                chi=args.chi, eps=args.eps, sigma=args.sigma)
    print(f"Physics: {phys}")

    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    out = {"phys": phys}
    for i, (name, n) in enumerate(sizes.items()):
        tlist, X, y = simulate_readout(n, seed=args.seed + 1000 + i, **phys)  # independent seed per split
        out[name] = {"x": torch.from_numpy(X), "y": torch.from_numpy(y)}
        print(f"{name}: {tuple(X.shape)}")
    out["tlist"] = torch.tensor(tlist, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.data_path) or ".", exist_ok=True)
    torch.save(out, args.data_path)
    print(f"Saved dataset to {args.data_path}")


if __name__ == "__main__":
    main()