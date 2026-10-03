# model/readout_model.py
import torch
import torch.nn as nn


class ReadoutCNN(nn.Module):
    """1D CNN over the (I, Q) record. Input (B, T, 2) -> one logit (B, 1).

    No global pooling on purpose: *when* the signal switches (a mid-readout decay)
    is the information that beats a matched filter, so we keep time position by flattening.
    """
    def __init__(self, seq_len, channels=(16, 32, 32), hidden=64):
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(2, c1, 5, padding=2), nn.ReLU(),
            nn.Conv1d(c1, c2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c2, c3, 5, stride=2, padding=2), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 2, seq_len)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x):                       # x: (B, T, 2)
        return self.head(self.conv(x.transpose(1, 2)))


class ReadoutGRU(nn.Module):
    """Small GRU alternative. Input (B, T, 2) -> one logit (B, 1)."""
    def __init__(self, hidden=64, layers=1):
        super().__init__()
        self.gru = nn.GRU(2, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)                      # h: (layers, B, hidden)
        return self.head(h[-1])