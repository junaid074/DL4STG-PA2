# ConvLSTM bits for Q2 (model A/B/C/D).
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (k // 2)
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(
            in_ch + hid_ch, 4 * hid_ch, kernel_size=k,
            padding=pad, dilation=dilation, bias=True,
        )

    def forward(self, u, h, c):
        if h is None:
            b, _, ht, wd = u.shape
            h = u.new_zeros(b, self.hid_ch, ht, wd)
            c = u.new_zeros(b, self.hid_ch, ht, wd)
        gates = self.conv(torch.cat([u, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    # A/B/C: unshuffle 4, two ConvLSTM(64, 3x3), 1x1, shuffle+sigmoid.
    # D can change kernel / dilation / width.

    def __init__(
        self,
        hidden: int = 64,
        kernel: int = 3,
        patch: int = 4,
        dilation: int = 1,
        n_layers: int = 2,
    ):
        super().__init__()
        self.patch = patch
        self.hidden = hidden
        self.n_layers = n_layers
        fold_ch = patch * patch  # 1 channel → patch^2 after unshuffle
        self.cells = nn.ModuleList()
        in_ch = fold_ch
        for _ in range(n_layers):
            self.cells.append(ConvLSTMCell(in_ch, hidden, k=kernel, dilation=dilation))
            in_ch = hidden
        self.head = nn.Conv2d(hidden, fold_ch, kernel_size=1, bias=True)

    def step(self, frame, states):
        # frame: (B, 1, 64, 64)
        u = F.pixel_unshuffle(frame, self.patch)
        new_states = []
        for i, cell in enumerate(self.cells):
            h, c = (None, None) if states is None else states[i]
            h, c = cell(u, h, c)
            new_states.append((h, c))
            u = h
        out = self.head(u)
        out = F.pixel_shuffle(out, self.patch)
        out = torch.sigmoid(out)
        return out, new_states

    def forward(self, x):
        """
        x: (B, 5, 1, 64, 64) → Ŷ: (B, 5, 1, 64, 64)
        9-step rollout, no teacher forcing.
        """
        b = x.shape[0]
        states = None
        preds = []
        # steps 1..5: real inputs; keep only outputs of steps 5..9
        for t in range(5):
            pred, states = self.step(x[:, t], states)
            if t == 4:
                preds.append(pred)
        # steps 6..9: feed own predictions
        for _ in range(4):
            pred, states = self.step(pred, states)
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def model_a() -> ConvLSTMForecaster:
    return ConvLSTMForecaster(hidden=64, kernel=3, patch=4, dilation=1, n_layers=2)


def model_d() -> ConvLSTMForecaster:
    # same size as A, dilation 2 → about 16 px/step reach instead of 8
    return ConvLSTMForecaster(hidden=64, kernel=3, patch=4, dilation=2, n_layers=2)


class BlankPredictor(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 5, 1, 64, 64, device=x.device, dtype=x.dtype)


class CopyLastPredictor(nn.Module):
    def forward(self, x):
        last = x[:, -1:]
        return last.expand(-1, 5, -1, -1, -1).contiguous()


def symmetrised(f: nn.Module):
    """¯f(X) = (1/8) Σ_g g^{-1}(f(g(X)))."""
    from .transforms import G_ELEMENTS, apply_g, inverse_g

    class SymWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            # x: (B, 5, 1, H, W)
            outs = []
            for a, m in G_ELEMENTS:
                gx = apply_g(x, a, m)
                pred = self.base(gx)
                outs.append(inverse_g(pred, a, m))
            return torch.stack(outs, dim=0).mean(dim=0)

        def count_params(self):
            return self.base.count_params()

    return SymWrapper(f)
