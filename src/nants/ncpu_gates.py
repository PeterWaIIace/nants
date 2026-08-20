"""NCPU-style logic gates as an ant task.

Circles on a grey background: black for 0, white for 1, grey for empty.
Input circles are pinned — re-stamped every step — so ants can read them
but never erase them. The colony must carry information across the field
and leave the answer in the output circle, using only the field as memory.

The bits are randomised every call to `regenerate()`, so the colony
cannot memorise a static target — it must learn the general computation.
"""

import torch

from nants.world import World

GATES = {
    "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "NAND": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "NOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
    "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
}

# ncpu normalisation: grey=128 → 0.0, black=0 → -1.0, white=255 → 1.0
INK = {0: -1.0, 1: 1.0}   # black for 0, white for 1
EMPTY = 0.0                 # grey background
LEVELS = [-1.0, 0.0, 1.0]  # possible disc shades


def _circle_coords(r):
    """(dy, dx) offsets of a filled circle of radius r (for stamping bits)."""
    return [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dy * dy + dx * dx <= r * r
    ]


def _shade(bit, cell_dim, device="cpu"):
    """A full cell vector for one bit: R=G=B=sink value, scratch alive."""
    return torch.tensor(
        [INK[bit]] * 3 + [1.0] * (cell_dim - 3),
        dtype=torch.float32, device=device,
    )


def _empty_cell(cell_dim, device="cpu"):
    """Grey background cell."""
    return torch.tensor(
        [EMPTY] * 3 + [1.0] * (cell_dim - 3),
        dtype=torch.float32, device=device,
    )


class NCPUGateTask:
    """Everything gate training needs that nants did not: circle-based
    targets, random bits per call, pinned input circles, and the output
    disc mask for the loss and exact-match score.

    ``regenerate()`` draws new random bits for every input pattern and
    rebuilds the target, init, and pin tensors.  Call it once before
    training and again at the start of each epoch (or even every step,
    for a harder curriculum).
    """

    def __init__(
        self,
        gate,
        size,
        batch,
        cell_dim,
        device,
        r=4,
        input_bits=2,
        io_gap=6,
        seed=None,
    ):
        self.gate = gate
        self.size = size
        self.batch = batch
        self.cell_dim = cell_dim
        self.device = device
        self.r = r
        self.input_bits = input_bits
        self.io_gap = io_gap

        self.patterns = sorted(GATES[gate])  # all possible input patterns
        groups = len(self.patterns)
        assert batch % groups == 0, "batch must be divisible by number of input patterns"
        self.per_group = batch // groups

        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)

        self.cells = _circle_coords(r)
        self.empty = _empty_cell(cell_dim, device)

        # pre-allocate tensors (will be filled by regenerate)
        self.init = None
        self.target = None
        self.out_mask = None
        self.pins = None

        self.regenerate()

    def _input_positions(self, pattern):
        """Screen-space (y, x) centre for each input bit circle, left side."""
        mid = self.size // 2
        n = len(pattern)
        # stack vertically, centred, with a gap between circles
        total_h = n * (2 * self.r + self.io_gap) - self.io_gap
        start_y = mid - total_h // 2
        x = mid - self.io_gap
        centres = []
        for i in range(n):
            y = start_y + i * (2 * self.r + self.io_gap) + self.r
            centres.append((y, x))
        return centres

    def _output_positions(self, n_bits):
        """Screen-space (y, x) centre for each output bit circle, right side."""
        mid = self.size // 2
        total_h = n_bits * (2 * self.r + self.io_gap) - self.io_gap
        start_y = mid - total_h // 2
        x = mid + self.io_gap
        centres = []
        for i in range(n_bits):
            y = start_y + i * (2 * self.r + self.io_gap) + self.r
            centres.append((y, x))
        return centres

    def _stamp_circle(self, screen, cy, cx, bit, pinned):
        """Stamp one circle onto `screen` and append to `pinned` list."""
        shade = _shade(bit, self.cell_dim, screen.device)
        for dy, dx in self.cells:
            y, x = cy + dy, cx + dx
            if 0 <= y < self.size and 0 <= x < self.size:
                screen[y, x] = shade
                pinned.append((y, x, shade))

    def _stamp_input_circles(self, screen, pattern, pinned):
        """Draw input circles for `pattern` and record pins."""
        centres = self._input_positions(pattern)
        for i, bit in enumerate(pattern):
            self._stamp_circle(screen, *centres[i], bit, pinned)

    def _stamp_output_circles(self, screen, bits):
        """Draw output circles (for the target screen only, not pinned)."""
        centres = self._output_positions(len(bits))
        for i, bit in enumerate(bits):
            self._stamp_circle(screen, *centres[i], bit, pinned=[])

    def _truth_table(self, pattern):
        """Look up the correct output for a given input pattern."""
        gate_fn = GATES[self.gate]
        return gate_fn[pattern]

    def regenerate(self):
        """Re-draw all screens and pins with freshly randomised bits."""
        groups = len(self.patterns)
        B, S, C = self.batch, self.size, self.cell_dim
        device = self.device

        self.init = self.empty[None, None, None, :].expand(B, S, S, C).clone()
        self.target = self.empty[None, None, None, :].expand(groups, S, S, C).clone()
        self.out_mask = torch.zeros(groups, S, S, device=device)

        pin_b, pin_y, pin_x, pin_v = [], [], [], []

        for g, pattern in enumerate(self.patterns):
            # build one base screen: grey background + input circles
            screen = self.empty[None, None, :].expand(S, S, C).clone()
            pinned = []
            self._stamp_input_circles(screen, pattern, pinned)

            # fill every world-share with this same base screen
            lo = g * self.per_group
            hi = (g + 1) * self.per_group
            self.init[lo:hi] = screen
            for b in range(lo, hi):
                for y, x, cell in pinned:
                    pin_b.append(b)
                    pin_y.append(y)
                    pin_x.append(x)
                    pin_v.append(cell)

            # target = base screen + correct output
            answer = self._truth_table(pattern)
            self.target[g] = screen
            answer_bits = [answer]  # single-bit output for basic gates
            self._stamp_output_circles(self.target[g], answer_bits)
            centres = self._output_positions(len(answer_bits))
            for cy, cx in centres:
                for dy, dx in self.cells:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < S and 0 <= x < S:
                        self.out_mask[g, y, x] = 1.0

        self.pins = (
            torch.tensor(pin_b, device=device),
            torch.tensor(pin_y, device=device),
            torch.tensor(pin_x, device=device),
            torch.stack(pin_v).to(device),
        )

    def make_world(self, brain, seed, steps, ants, scatter):
        return World(
            brain,
            self.size,
            seed=seed,
            noise=0.0,
            init=self.init,
            horizon=steps,
            ants=ants,
            scatter=scatter,
            pins=self.pins,
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
            [INK[self._truth_table(p)] for p in self.patterns],
            device=shade.device,
        )[:, None]
        return (snapped == want).float().mean(dim=-1)
