# /train/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import ReadoutCNN, ReadoutGRU
from baselines import assignment_fidelity, evaluate_baselines
from train import parse_args


@torch.no_grad()
def predict(model, x, device, bs=2048):
    model.eval()
    out = [model(x[i:i + bs].to(device)).squeeze(1).cpu() for i in range(0, len(x), bs)]
    return (torch.cat(out) > 0).numpy().astype(int)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    print(f"--- Readout classifier training ({args.arch}) on {device} ---")

    data = torch.load(args.data_path)
    tr, va, te = data["train"], data["val"], data["test"]
    print(f"Dataset physics: {data['phys']}")

    scale = tr["x"].std()                        # normalise inputs by the training std
    xtr, xva, xte = tr["x"] / scale, va["x"] / scale, te["x"] / scale
    T = xtr.shape[1]

    if args.arch == "cnn":
        model = ReadoutCNN(T, hidden=args.hidden_neurons)
    else:
        model = ReadoutGRU(hidden=args.hidden_neurons)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(xtr, tr["y"]), batch_size=args.batch_size, shuffle=True)

    best_fid, best_state = 0.0, None
    pbar = tqdm(range(1, args.epochs + 1), desc="Training")
    for epoch in pbar:
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            
        scheduler.step()

        fid = assignment_fidelity(predict(model, xva, device), va["y"].numpy())
        if fid > best_fid:                       # keep best epoch by validation fidelity
            best_fid = fid
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pbar.set_postfix({'Loss': f"{running / len(xtr):.3e}", 'ValFid': f"{fid:.4f}"})

    model.load_state_dict(best_state)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"readout_{args.arch}.pth")
    torch.save({"state_dict": best_state, "scale": scale.item(), "arch": args.arch,
                "hidden": args.hidden_neurons, "seq_len": T, "phys": data["phys"]}, ckpt_path)

    nn_fid = assignment_fidelity(predict(model, xte, device), te["y"].numpy())
    base = evaluate_baselines(tr, te)
    print("\nTest assignment fidelity")
    print(f"  integrated threshold : {base['integrated']:.4f}")
    print(f"  matched filter       : {base['matched']:.4f}")
    print(f"  {args.arch.upper():<21s}: {nn_fid:.4f}")
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()