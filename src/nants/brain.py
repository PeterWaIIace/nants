"""The ants' brains: one small MLP, reacting only to what is sensed.

Like Langton's ant, there is no memory. What the ant does depends purely on
the field around it.
"""

import torch
import torch.nn as nn

from .ant import CLOCK_DIM, COMPASS_DIM, FACING_DIM


class Brain(nn.Module):
    """A small MLP that maps what an ant senses to what it writes and how it turns."""

    _use_gumbel = False  # class flag: True uses compiler-friendly sampling

    def __init__(
        self, batch, cell_dim, width=16, seed=0, scale=1.0, temp=1.0,
        device="cpu", shared=False, zero_out=False, groups=1,
    ):
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(seed)
        self.gen = gen
        self.batch = batch
        self.cell_dim = cell_dim
        self.temp = temp
        self.device_ = device
        self.groups = groups

        rows = groups if groups > 1 else (1 if shared else batch)
        self.rows = rows

        input_dim = 3 * cell_dim + CLOCK_DIM + COMPASS_DIM + FACING_DIM

        def w(a, b):
            return torch.randn(rows, a, b, generator=gen, device=device) * (scale / a**0.5)

        def b(n):
            return torch.zeros(rows, 1, n, device=device)

        self.w1 = nn.Parameter(w(input_dim, width))
        self.write = nn.Parameter(w(width, cell_dim))
        self.move = nn.Parameter(w(width, 3))
        self.b1 = nn.Parameter(b(width))
        self.b_write = nn.Parameter(b(cell_dim))
        self.b_move = nn.Parameter(b(3))

        if zero_out:
            with torch.no_grad():
                self.write.zero_()

    def forward(self, sense):
        """sense (B,3C) -> d_cell (B,C), move (B,), logp of that move (B,)"""
        batch = sense.shape[0]
        x = sense.view(self.rows, -1, sense.shape[-1])

        h = torch.relu(x @ self.w1 + self.b1)
        d_cell = (h @ self.write + self.b_write).reshape(batch, -1)

        logits = (h @ self.move + self.b_move).reshape(batch, 3) / self.temp
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logp = torch.log_softmax(logits, dim=-1)

        if Brain._use_gumbel:
            u = torch.rand_like(logits).clamp(1e-20, 1.0)
            gumbel = -torch.log(-torch.log(u))
            move = (logits + gumbel).argmax(dim=-1)
        else:
            move = torch.multinomial(logp.exp(), 1, generator=self.gen).squeeze(1)

        return d_cell, move, logp.gather(1, move[:, None]).squeeze(1)
