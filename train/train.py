# /train/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model.readout_model import ReadoutCNN, ReadoutGRU
from baselines import (assignment_fidelity, fidelity_per_qubit, evaluate_baselines,
                       format_line, LABELS)
from train.arguments import parse_args


@torch.no_grad()
def predict(model, x, device, bs=2048):
    model.eval()
    out = [model(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)]
    return (torch.cat(out) > 0).numpy().astype(int)             # (N, n_qubits)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")

    data = torch.load(args.data_path)
    tr, va, te = data["train"], data["val"], data["test"]
    ytr, yva, yte = (s["y"].reshape(len(s["y"]), -1).float() for s in (tr, va, te))   # (N, n_qubits)
    n_qubits, in_channels, T = ytr.shape[1], tr["x"].shape[2], tr["x"].shape[1]
    print(f"--- Readout classifier training ({args.arch}) on {device} ---")
    print(f"n_qubits = {n_qubits}, input channels = {in_channels}, samples per record = {T}")
    print(f"Dataset physics: {data['phys']}")

    scale = tr["x"].std()                        # normalise inputs by the training std
    xtr, xva, xte = tr["x"] / scale, va["x"] / scale, te["x"] / scale

    if args.arch == "cnn":
        model = ReadoutCNN(T, in_channels=in_channels, n_outputs=n_qubits, hidden=args.hidden_neurons)
    else:
        model = ReadoutGRU(in_channels=in_channels, n_outputs=n_qubits, hidden=args.hidden_neurons)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.BCEWithLogitsLoss()           # one independent binary decision per qubit
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)

    best_fid, best_state = 0.0, None
    pbar = tqdm(range(1, args.epochs + 1), desc="Training")
    for epoch in pbar:
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        scheduler.step()

        fid = assignment_fidelity(predict(model, xva, device), yva.numpy())
        if fid > best_fid:                       # keep best epoch by validation fidelity
            best_fid = fid
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pbar.set_postfix({'Loss': f"{running / len(xtr):.3e}", 'ValFid': f"{fid:.4f}"})

    model.load_state_dict(best_state)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"readout_{args.arch}.pth")
    torch.save({"state_dict": best_state, "scale": scale.item(), "arch": args.arch,
                "hidden": args.hidden_neurons, "seq_len": T, "in_channels": in_channels,
                "n_qubits": n_qubits, "phys": data["phys"]}, ckpt_path)

    nn_fids = fidelity_per_qubit(predict(model, xte, device), yte.numpy())
    base = evaluate_baselines(tr, te)
    print("\nTest assignment fidelity (mean over qubits first, then per qubit)")
    for kind, fids in base.items():
        print(format_line(LABELS[kind], fids))
    print(format_line(args.arch.upper() if args.arch == "gru" else "CNN", nn_fids))
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()