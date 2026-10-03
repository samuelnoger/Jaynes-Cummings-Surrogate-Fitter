# baselines.py
import numpy as np
import torch


def assignment_fidelity(pred, y):
    """1 - (P(0|1) + P(1|0)) / 2"""
    pred, y = np.asarray(pred), np.asarray(y)
    p0_given_1 = np.mean(pred[y == 1] == 0)
    p1_given_0 = np.mean(pred[y == 0] == 1)
    return 1.0 - 0.5 * (p0_given_1 + p1_given_0)


def _best_threshold(score, y):
    cands = np.quantile(score, np.linspace(0.01, 0.99, 400))
    fids = [assignment_fidelity(score > c, y) for c in cands]
    return cands[int(np.argmax(fids))]


def _fit_score(x, y, kind):
    """Return a function mapping records (N,T,2) -> scalar score, fitted on train data."""
    if kind == "integrated":
        integ = x.sum(1)                                    # (N, 2): integrate each quadrature
        d = integ[y == 1].mean(0) - integ[y == 0].mean(0)   # axis separating the two states
        return lambda z: z.sum(1) @ d
    if kind == "matched":
        kernel = x[y == 1].mean(0) - x[y == 0].mean(0)      # (T, 2) time-resolved weights
        return lambda z: (z * kernel).sum((1, 2))
    raise ValueError(kind)


def evaluate_baselines(train, test):
    xt, yt = train["x"].numpy(), train["y"].numpy()
    xe, ye = test["x"].numpy(), test["y"].numpy()
    out = {}
    for kind in ("integrated", "matched"):
        f = _fit_score(xt, yt, kind)
        thr = _best_threshold(f(xt), yt)
        out[kind] = assignment_fidelity(f(xe) > thr, ye)
    return out


if __name__ == "__main__":
    data = torch.load("data/readout_dataset.pt")
    res = evaluate_baselines(data["train"], data["test"])
    for k, v in res.items():
        print(f"{k:>10s} filter: assignment fidelity = {v:.4f}")