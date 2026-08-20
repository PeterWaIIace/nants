"""The ants: a position, a heading and a clock, moved by their brains.

No memory. The ant reads its 3x3 neighbourhood with fixed sobel kernels in
world coordinates, and picks a turn: left, straight on, or right. It then
always steps forward, like a turmite.

Besides the field it senses the clock (how far into the run it is), where it
is relative to its starting cell, and which way it is facing.
"""

import math

import torch

# headings, clockwise from up, as (dx, dy)
RING = torch.tensor([(0, -1), (1, 0), (0, 1), (-1, 0)])
TURNS = torch.tensor([-1, 0, 1])  # left, straight on, right
LIMIT = 8.0  # how far past the picture range an ant is allowed to see

# set to torch.tensor([]) to take the clock away and leave the ant timeless
CLOCK_FREQS = torch.tensor([1.0, 2.0, 4.0, 8.0])
CLOCK_DIM = 2 * len(CLOCK_FREQS)
# Where the ant is relative to its start, described two ways: as (x, y) and as
# (distance, angle). Both get sines and cosines at several frequencies, so that
# nearby places look different instead of forming one smooth ramp.
COMPASS_FREQS = torch.tensor([1.0, 2.0, 4.0, 8.0])
CARTESIAN_DIM = 2 + 4 * len(COMPASS_FREQS)
POLAR_DIM = 1 + 4 * len(COMPASS_FREQS)
COMPASS_DIM = CARTESIAN_DIM + POLAR_DIM
FACING_DIM = 2  # which way it is pointing

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8
SOBEL_Y = SOBEL_X.T

# cos and sin of a whole number of quarter turns, for rotating into the start frame
QUARTER_COS = torch.tensor([1.0, 0.0, -1.0, 0.0])
QUARTER_SIN = torch.tensor([0.0, 1.0, 0.0, -1.0])


class Ant:
    """A batch of worlds, each holding one or more ants that share a field."""

    def __init__(self, brain, pos, horizon=1500, span=None):
        self.brain = brain
        self.pos = pos  # (B, N, 2) as [x, y]
        self.span = span  # what the compass divides by; defaults to half the field
        self.origin = pos.clone()  # where each one started, for the compass
        self.heading = torch.zeros(pos.shape[:2], dtype=torch.long, device=pos.device)
        self.start_heading = self.heading.clone()  # the frame everything is reported in

        # +1 as trained, -1 to swap left and right everywhere: what it senses
        # sideways, which way its lateral coordinate runs, and which way it turns.
        # A mirrored ant paints a mirrored picture.
        self.flip = torch.ones(pos.shape[:2], device=pos.device)

        # 1 for an ant that is present, 0 for an empty slot: it senses and moves
        # but writes nothing, so a batch can hold worlds with different crowds
        self.active = torch.ones(pos.shape[:2], device=pos.device)

        # angle, in radians, to turn the sobel kernels by before reading them.
        # 0 is how it was trained: gradients along the grid's own axes.
        self.twist = torch.zeros(pos.shape[:2], device=pos.device)
        self.horizon = horizon  # how long a run lasts, for the clock
        self.t = torch.tensor(0, device=pos.device, dtype=torch.float32)

        self.ring = RING.to(pos.device)
        self.turns = TURNS.to(pos.device)
        self.freqs = CLOCK_FREQS.to(pos.device)
        self.cfreqs = COMPASS_FREQS.to(pos.device)
        self.sobel = torch.stack([SOBEL_X, SOBEL_Y]).to(pos.device)
        self.cos = QUARTER_COS.to(pos.device)
        self.sin = QUARTER_SIN.to(pos.device)

    def to_ant_frame(self, x, y):
        """Turn world-axis components into each ant's own frame, as it faces now."""
        shape = self.heading.shape + (1,) * (x.dim() - 2)
        c = self.cos[self.heading].reshape(shape)
        s = self.sin[self.heading].reshape(shape)
        sideways = self.flip.reshape(shape)  # -1 swaps the ant's left and right
        return (x * c + y * s) * sideways, y * c - x * s

    def clock(self):
        """How far into the run we are, as sines and cosines: (B, N, CLOCK_DIM)"""
        phase = 2 * math.pi * self.freqs * (self.t / self.horizon)
        feats = torch.cat([phase.sin(), phase.cos()])
        return feats.expand(self.pos.shape[:2] + feats.shape)

    def compass(self, field):
        """Where each ant is relative to its start: (B, N, COMPASS_DIM)"""
        half = field.size // 2
        step = self.span or half  # pin this to keep the drawing the size it learned
        away = ((self.pos - self.origin + half) % field.size - half).float() / step
        x, y = self.to_ant_frame(away[..., 0], away[..., 1])

        offset = torch.stack([x, y], dim=-1)  # roughly -1 .. 1
        phase = math.pi * offset[..., None] * self.cfreqs  # (B, N, 2, F)
        cartesian = [offset, phase.sin().flatten(-2), phase.cos().flatten(-2)]

        radius = offset.norm(dim=-1, keepdim=True)
        turns = torch.atan2(y, x)[..., None] * self.cfreqs  # circular harmonics
        rings = math.pi * radius * self.cfreqs
        polar = [radius, turns.sin(), turns.cos(), rings.sin(), rings.cos()]

        return torch.cat(cartesian + polar, dim=-1)

    def facing(self):
        """Which way it points, as a turn count from where it started: (B, N, 2)"""
        pointing = self.ring[(self.heading - self.start_heading) % 4].float()
        return torch.stack([pointing[..., 0] * self.flip, pointing[..., 1]], dim=-1)

    def sense(self, field):
        """everything an ant knows: (B, N, 3C + CLOCK + COMPASS + FACING)"""
        # the field is unbounded, but nothing sane lives past LIMIT, and letting
        # a runaway value reach the brain turns the move logits into infinities
        patch = field.patch(self.pos).clamp(-LIMIT, LIMIT)  # (B, N, 3, 3, C)
        grad = torch.einsum("kij,bnijc->bnkc", self.sobel, patch)  # (B, N, 2, C)

        # sobel is steerable, so turning the pair of gradients by an angle is
        # the same as turning both kernels by it
        gx, gy = grad[..., 0, :], grad[..., 1, :]
        turn, lean = self.twist[..., None].cos(), self.twist[..., None].sin()
        gx, gy = gx * turn + gy * lean, gy * turn - gx * lean

        ahead, aside = self.to_ant_frame(gx, gy)

        parts = [
            patch[:, :, 1, 1],
            ahead,
            aside,
            self.clock(),
            self.compass(field),
            self.facing(),
        ]
        return torch.cat(parts, dim=-1)

    def step(self, field):
        sense = self.sense(field)
        flat = sense.reshape(-1, sense.shape[-1])  # one row per ant
        d_cell, move, logp = self.brain(flat)

        shape = self.pos.shape[:2]
        self.logp = logp.reshape(shape) * self.active  # for REINFORCE
        field.add(self.pos, d_cell.reshape(shape + (-1,)) * self.active[..., None])

        turn = self.turns[move.reshape(shape)] * self.flip.long()  # mirrored: left <-> right
        self.heading = (self.heading + turn) % 4
        self.pos = field.wrap(self.pos + self.ring[self.heading])
        self.t += 1
