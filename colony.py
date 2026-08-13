"""One field, many colonies.

Every colony is a crowd of ants on its own landmark, in its own starting pose.
They all share a single grid, so they can walk into each other's paintings.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import paint
import showcase
import train
from leniaghton.world import World

COLS = 3  # colonies across
ROWS = 3
SPACING = 48  # cells between one landmark and the next
ZOOM = 5
ARROW = 5


def grid_layout(ants, device):
    """A colony per cell of a COLS x ROWS grid, each in its own pose."""
    colonies = COLS * ROWS
    poses = [(h, f) for f in (1.0, -1.0) for h in range(4)]
    poses.insert(colonies // 2, (0, 1.0))

    middles = []
    for i in range(colonies):
        r, c = divmod(i, COLS)
        middles.append((SPACING // 2 + c * SPACING, SPACING // 2 + r * SPACING))

    return SPACING * COLS, middles, poses[:colonies]


def close_layout(ants, device):
    """Two colonies close enough that their paintings have to share ground.

    The gap between landmarks comes from the command line, in cells;
    a whole gecko is train.GECKO wide.
    """
    numbers = [int(a) for a in sys.argv if a.isdigit()]
    gap = numbers[0] if numbers else train.GECKO

    size = SPACING * 2
    middle = size // 2
    middles = [(middle - gap // 2, middle), (middle + gap - gap // 2, middle)]
    return size, middles, [(0, 1.0), (0, 1.0)]


def spread(middles, poses, ants, device):
    """Repeat each colony's start and pose once per ant in it."""
    spots, facing, flip = [], [], []
    for (x, y), (h, f) in zip(middles, poses):
        spots += [[x, y]] * ants
        facing += [h] * ants
        flip += [f] * ants

    return (
        torch.tensor(spots, device=device)[None],  # (1, colonies * ants, 2)
        torch.tensor(facing, device=device)[None],
        torch.tensor(flip, device=device, dtype=torch.float32)[None],
    )


def draw_arrows(sheet, spots, facings, alive, alpha):
    over = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(over)
    colour = tuple(showcase.ANT) + (alpha,)

    for (x, y), h, on in zip(spots, facings, alive):
        if not on:
            continue
        cx, cy = (x + 0.5) * ZOOM, (y + 0.5) * ZOOM
        dx, dy = showcase.ARROW_DIRS[h]

        tip = (cx + dx * ARROW, cy + dy * ARROW)
        pen.line([(cx - dx * ARROW * 0.3, cy - dy * ARROW * 0.3), tip], colour, 2)
        for side in (-1, 1):
            bx = tip[0] - (dx + side * dy) * ARROW * 0.4
            by = tip[1] - (dy - side * dx) * ARROW * 0.4
            pen.line([tip, (bx, by)], colour, 2)

    return Image.alpha_composite(sheet.convert("RGBA"), over).convert("RGB")


def picture(world, alpha):
    got = np.clip(world.field.cells[0, :, :, :3].cpu().numpy(), -1, 1)
    img = Image.fromarray(((got + 1) / 2 * 255).astype(np.uint8))
    img = img.resize((img.width * ZOOM, img.height * ZOOM), Image.NEAREST)

    ant = world.ant
    return draw_arrows(
        img,
        ant.pos[0].cpu().numpy(),
        ant.heading[0].cpu().numpy(),
        ant.active[0].cpu().numpy(),
        alpha,
    )


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]

    ck = showcase.match_clock(run / "best.pt", cell_dim=train.CELL_DIM)
    brain = train.Brain(
        1, cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    )
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved)
    print(f"{run}: brain from epoch {len(ck['losses']) - 1}, mse {min(ck['losses']):.4f}")

    close = "close" in sys.argv
    layout = close_layout if close else grid_layout
    size, middles, poses = layout(train.ANTS, train.DEVICE)

    # "lonely": lay both landmarks, but only give the first one any ants
    lonely = "lonely" in sys.argv
    homes = middles[:1] if lonely else middles
    spots, facing, flip = spread(homes, poses, train.ANTS, train.DEVICE)

    world = World(
        brain, size, seed=0, noise=0.0, init=train.WHITE,
        horizon=train.STEPS, ants=len(homes) * train.ANTS,
    )
    ant = world.ant
    ant.pos, ant.origin = spots.clone(), spots.clone()
    ant.heading, ant.start_heading = facing, facing.clone()
    ant.flip = flip
    ant.active = torch.ones_like(flip)  # every colony gets its full crowd
    ant.span = train.SIZE // 2  # keep the geckos the size they were trained at

    bare = "bare" in sys.argv  # start with nothing at all: ants on blank white
    if not bare:
        for (dx, dy), values in train.LANDMARK:  # one landmark under each colony
            mark = torch.tensor(values, device=train.DEVICE)
            for x, y in middles:
                world.field.cells[0, y + dy, x + dx] = mark

    slow = showcase.SLOW_STEPS // 5
    frames = []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            every = 5 if i < showcase.SLOW_STEPS else paint.EVERY
            if i % every == 0:
                frames.append(picture(world, 255 if len(frames) < slow else 128))

    timing = [showcase.SLOW_MS] * slow
    timing += [showcase.FAST_MS] * (len(frames) - slow)
    timing[0], timing[-1] = 1000, 2000

    tag = "lonely" if lonely else str(middles[1][0] - middles[0][0])
    suffix = "_bare" if bare else ""
    name = f"colony_close{tag}{suffix}.gif" if close else f"colony{suffix}.gif"
    still = name.replace(".gif", ".png")
    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    shown = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    shown[0].save(
        run / name, save_all=True, append_images=shown[1:],
        duration=timing, loop=0,
    )
    frames[-1].save(run / still)
    print(f"{len(middles)} colonies of {train.ANTS} on one {size}x{size} field")
    print(f"-> {run}/{still} and {name}")


if __name__ == "__main__":
    main()
