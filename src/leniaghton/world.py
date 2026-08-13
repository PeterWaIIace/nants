"""The fields the ants walk on, batched into one world."""

import math

import torch

from .ant import Ant


class Field:
    def __init__(self, batch, size, cell_dim, noise=0.0, gen=None, init=0.0, device="cpu"):
        """init is a scalar or a (C,) tensor: the value every cell starts at."""
        self.size = size
        shape = (batch, size, size, cell_dim)
        start = torch.as_tensor(init, dtype=torch.float32, device=device)
        self.cells = start + torch.randn(shape, generator=gen, device=device) * noise
        self.rows = torch.arange(batch, device=device)

    def get(self, pos):
        """The cell under every ant: (B, N, C). pos is (B, N, 2)."""
        return self.cells[self.rows[:, None], pos[..., 1], pos[..., 0]]

    def patch(self, pos):
        """The 3x3 neighbourhood around every ant: (B, N, 3, 3, C), y first."""
        span = torch.arange(-1, 2, device=pos.device)
        ys = (pos[..., 1, None] + span) % self.size  # (B, N, 3)
        xs = (pos[..., 0, None] + span) % self.size
        rows = self.rows[:, None, None, None]
        return self.cells[rows, ys[..., :, None], xs[..., None, :]]

    def add(self, pos, delta):
        """Add to the cells under the ants, summing where two ants coincide."""
        rows = self.rows[:, None].expand_as(pos[..., 0])
        self.cells.index_put_((rows, pos[..., 1], pos[..., 0]), delta, accumulate=True)

    def wrap(self, pos):
        return pos % self.size


class World:
    def __init__(
        self, brain, size, seed=0, noise=0.1, init=0.0, landmark=None, horizon=1500,
        ants=1, scatter=0,
    ):
        """landmark: [((dx, dy), values), ...] painted at the middle of the field."""
        dev = brain.device
        gen = torch.Generator(device=dev).manual_seed(seed)
        batch = brain.batch
        middle = size // 2

        self.field = Field(batch, size, brain.cell_dim, noise, gen, init, dev)
        for (dx, dy), values in landmark or []:
            mark = torch.as_tensor(values, dtype=torch.float32, device=dev)
            self.field.cells[:, middle + dy, middle + dx] = mark

        start = torch.full((batch, ants, 2), middle, device=dev)
        if scatter:  # drop them anywhere within a small circle around the seed
            angle = torch.rand((batch, ants), generator=gen, device=dev) * 2 * math.pi
            reach = scatter * torch.rand(
                (batch, ants), generator=gen, device=dev
            ).sqrt()
            start = start + torch.stack(
                [angle.cos() * reach, angle.sin() * reach], dim=-1
            ).round().long()

        self.ant = Ant(brain, start % size, horizon)
        if scatter:  # scattered ants face any which way; stacked ones all face up
            self.ant.heading = torch.randint(
                4, (batch, ants), generator=gen, device=dev
            )
            self.ant.start_heading = self.ant.heading.clone()

        if ants > 1:  # each world gets between 1 and `ants` of them
            crowd = torch.randint(1, ants + 1, (batch, 1), generator=gen, device=dev)
            rank = torch.arange(ants, device=dev).expand(batch, ants)
            self.ant.active = (rank < crowd).float()

    def step(self):
        self.ant.step(self.field)

    def run(self, steps):
        for _ in range(steps):
            self.step()
