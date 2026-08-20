"""ncpu-style logic gates as an ant task.

The inputs are drawn as discs to the left of the landmark, and the answer
belongs in a disc on the right. A bit is a shade: black for 0, grey for 1,
on the usual white field. The input discs are pinned — re-stamped after every
step — so the ants can read them forever but never erase them. To compute the
gate, the colony has to carry what it reads across the field and leave the
right shade in the output disc, with no memory but the field itself.
"""

import torch

from nants.world import World

# truth tables, in the block order the batch is arranged in
GATES = {
    "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "NAND": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "NOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
    "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
}

INK = {0: -1.0, 1: 0.0}  # black for 0, grey for 1; the background stays white
LEVELS = [-1.0, 0.0, 1.0]  # what a painted disc can read as: 0, 1, unpainted


def disc(r):
    """(dy, dx) offsets of a filled circle of radius r, for stamping a bit."""
    return [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dy * dy + dx * dx <= r * r
    ]


class GateTask:
    """Everything gate training needs that the gecko did not: per-world
    starting fields, pinned cells, one target per input pattern, and the
    output disc mask for the loss and the exact-match score."""

    def __init__(self, gate, landmark, size, batch, cell_dim, device,
                 r=2, io_dx=14, io_dy=5):
        self.gate = gate
        self.size = size
        self.patterns = sorted(GATES[gate])  # block g of the batch gets pattern g
        groups = len(self.patterns)
        assert batch % groups == 0, "each pattern needs an equal share of the batch"
        per_group = batch // groups

        mid = size // 2
        io_dx = min(io_dx, mid - r - 1)  # keep discs inside the field
        cells = disc(r)
        white = torch.tensor([1.0] * 3 + [0.0] * (cell_dim - 3), device=device)

        def shade(bit):  # a bit as a full cell: the shade, with scratch alive
            return torch.tensor([INK[bit]] * 3 + [1.0] * (cell_dim - 3),
                                dtype=torch.float32, device=device)

        def input_pos(i, n):  # inputs in one column, left of the landmark
            return mid + (2 * i - (n - 1)) * io_dy, mid - io_dx

        out_y, out_x = mid, mid + io_dx  # the answer, the same distance right

        self.init = white[None, None, None, :].expand(batch, size, size, cell_dim).clone()
        self.target = white[None, None, None, :].expand(groups, size, size, cell_dim).clone()
        self.out_mask = torch.zeros(groups, size, size, device=device)

        pin_b, pin_y, pin_x, pin_v = [], [], [], []
        for g, pattern in enumerate(self.patterns):
            # one starting screen per pattern: landmark plus its input discs
            screen = white[None, None, :].expand(size, size, cell_dim).clone()
            pinned = []
            for (dx, dy), values in landmark:
                y, x = mid + dy, mid + dx
                cell = torch.as_tensor(values, dtype=torch.float32, device=device)
                screen[y, x] = cell
                pinned.append((y, x, cell))
            for i, bit in enumerate(pattern):
                cy, cx = input_pos(i, len(pattern))
                for dy, dx in cells:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < size and 0 <= x < size:
                        screen[y, x] = shade(bit)
                        pinned.append((y, x, shade(bit)))

            lo, hi = g * per_group, (g + 1) * per_group
            self.init[lo:hi] = screen
            for b in range(lo, hi):
                for y, x, cell in pinned:
                    pin_b.append(b)
                    pin_y.append(y)
                    pin_x.append(x)
                    pin_v.append(cell)

            # the target is the same screen, plus the answer in the output disc
            answer = GATES[gate][pattern]
            self.target[g] = screen
            for dy, dx in cells:
                y, x = out_y + dy, out_x + dx
                if 0 <= y < size and 0 <= x < size:
                    self.target[g, y, x] = shade(answer)
                    self.out_mask[g, y, x] = 1.0

        self.pins = (
            torch.tensor(pin_b, device=device),
            torch.tensor(pin_y, device=device),
            torch.tensor(pin_x, device=device),
            torch.stack(pin_v),
        )

    def make_world(self, brain, seed, steps, ants, scatter):
        return World(
            brain, self.size, seed=seed, noise=0.0, init=self.init,
            horizon=steps, ants=ants, scatter=scatter, pins=self.pins,
        )

    def exact_match(self, world):
        """Accuracy per input pattern: the mean shade of the output disc,
        snapped to the nearest of black / grey / white, against the truth table."""
        field = world.field.cells[..., :3].mean(dim=-1)  # (B, H, W)
        grouped = field.view(len(self.patterns), -1, self.size, self.size)
        spots = self.out_mask.sum(dim=(1, 2))[:, None]
        shade = (grouped * self.out_mask[:, None]).sum(dim=(2, 3)) / spots

        levels = torch.tensor(LEVELS, device=shade.device)
        snapped = levels[(shade.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
        want = torch.tensor(
            [INK[GATES[self.gate][p]] for p in self.patterns], device=shade.device
        )[:, None]
        return (snapped == want).float().mean(dim=-1)
