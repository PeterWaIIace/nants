"""The ants' brains: one small MLP, reacting only to what is sensed.

Like Langton's ant, there is no memory. What the ant does depends purely on
the field around it.
"""

import torch

from .ant import CLOCK_DIM, COMPASS_DIM, FACING_DIM


class Brain:
    def __init__(
        self, batch, cell_dim, width=16, seed=0, scale=1.0, temp=1.0,
        device="cpu", shared=False, zero_out=False,
    ):
        # shared=False: one brain per batch slot (sampling many random ants)
        # shared=True:  one brain, batch slots are just repeated rollouts (training)
        gen = torch.Generator(device=device).manual_seed(seed)
        self.gen = gen
        self.batch = batch
        self.cell_dim = cell_dim
        self.temp = temp
        self.device = device

        def w(a, b):
            size = (1 if shared else batch, a, b)
            return torch.randn(size, generator=gen, device=device) * (scale / a**0.5)

        def b(n):
            return torch.zeros(1 if shared else batch, 1, n, device=device)

        self.w1 = w(3 * cell_dim + CLOCK_DIM + COMPASS_DIM + FACING_DIM, width)  # what it senses
        self.write = w(width, cell_dim)  # what to add to the cell below
        self.move = w(width, 3)  # turn left, straight on, turn right
        self.b1, self.b_write, self.b_move = b(width), b(cell_dim), b(3)

        if zero_out:  # NCA style: start as a no-op, let gradients build it up
            self.write = torch.zeros_like(self.write)

    def parameters(self):
        return [self.w1, self.write, self.move, self.b1, self.b_write, self.b_move]

    def train(self):
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    def __call__(self, sense):
        """sense (B,3C) -> d_cell (B,C), move (B,), logp of that move (B,)"""
        h = torch.relu(sense.unsqueeze(1) @ self.w1 + self.b1)
        d_cell = (h @ self.write + self.b_write).squeeze(1)  # linear out, unbounded

        logits = (h @ self.move + self.b_move).squeeze(1) / self.temp
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logp = torch.log_softmax(logits, dim=-1)
        move = torch.multinomial(logp.exp(), 1, generator=self.gen).squeeze(1)
        return d_cell, move, logp.gather(1, move[:, None]).squeeze(1)
