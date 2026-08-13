"""A grid of geckos: same brain, random starting place and facing, roomier field.

The starts are kept away from the edges so nothing wraps around and gets cut.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from leniaghton import ant as ant_module

SIZE = 48  # matches the field it was trained on
RUNS = 9
MARGIN = 15  # keep starts this far from the edge

STEPS = 6000  # as trained
SLOW_STEPS = 100  # record these first steps densely and play them slowly
SLOW_MS = 100  # 10 fps to begin with
FAST_MS = 33  # then 30 fps

ANT = [220, 30, 30]  # the ant, and the arrow showing which way it faces
ARROW = 6  # arrow length in enlarged pixels
ARROW_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # matches the ant's headings


def draw_arrows(sheet, zoom, spots, facings, alive, cols, alpha=255, gap=1):
    """An arrow on every living ant, drawn on the enlarged sheet so it can be seen."""
    pitch = SIZE + gap
    over = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(over)
    colour = tuple(ANT) + (alpha,)

    for i, (tile, heads, here) in enumerate(zip(spots, facings, alive)):
        r, c = divmod(i, cols)
        for (x, y), h, on in zip(tile, heads, here):
            if not on:
                continue
            cx = (c * pitch + x + 0.5) * zoom
            cy = (r * pitch + y + 0.5) * zoom
            dx, dy = ARROW_DIRS[h]

            tail = (cx - dx * ARROW * 0.3, cy - dy * ARROW * 0.3)
            tip = (cx + dx * ARROW, cy + dy * ARROW)
            pen.line([tail, tip], fill=colour, width=2)
            for side in (-1, 1):  # the two barbs
                bx = tip[0] - (dx + side * dy) * ARROW * 0.4
                by = tip[1] - (dy - side * dx) * ARROW * 0.4
                pen.line([tip, (bx, by)], fill=colour, width=2)

    return Image.alpha_composite(sheet.convert("RGBA"), over).convert("RGB")


def match_clock(path, cell_dim):
    """Older brains were trained with a clock. Restore whatever this one used."""
    ck = torch.load(path, weights_only=False)
    wanted = ck["w"][0].shape[1]
    fixed = 3 * cell_dim + ant_module.COMPASS_DIM + ant_module.FACING_DIM

    freqs = (wanted - fixed) // 2
    ant_module.CLOCK_FREQS = torch.tensor([2.0**i for i in range(freqs)])
    ant_module.CLOCK_DIM = 2 * freqs
    print(f"brain wants {wanted} inputs -> clock of {2 * freqs}")
    return ck


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]

    import train  # imported after the clock is known

    ck = match_clock(run / "best.pt", cell_dim=16)
    import importlib

    from leniaghton import brain as brain_module
    importlib.reload(brain_module)
    train.Brain = brain_module.Brain

    import paint
    from leniaghton.world import World

    brain = train.Brain(
        RUNS, cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    )
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved)
    print(f"{run}: brain from epoch {len(ck['losses']) - 1}, mse {min(ck['losses']):.4f}")

    def snapshot(world):
        """The pictures plus where every ant is and which way it points."""
        got = np.clip(world.field.cells[:, :, :, :3].cpu().numpy(), -1, 1)
        imgs = ((got + 1) / 2 * 255).astype(np.uint8)
        ant = world.ant
        return (
            imgs,
            ant.pos.cpu().numpy(),  # (worlds, ants, 2)
            ant.heading.cpu().numpy(),
            ant.active.cpu().numpy(),
        )


    world = World(
        brain, SIZE, seed=0, noise=0.0, init=train.WHITE, landmark=train.LANDMARK,
        horizon=train.STEPS, ants=train.ANTS, scatter=0,
    )
    # the eight ways to place a shape: four turns, each plain or mirrored.
    # the middle tile is the plain one it was trained as.
    poses = [(h, f) for f in (1.0, -1.0) for h in range(4)]
    poses.insert(RUNS // 2, (0, 1.0))

    ants = world.ant.pos.shape[1]
    facing = torch.tensor([h for h, _ in poses], device=train.DEVICE)
    flip = torch.tensor([f for _, f in poses], device=train.DEVICE)

    a = world.ant  # every ant in a world shares that world's starting pose
    a.heading = facing[:, None].expand(RUNS, ants).contiguous()
    a.start_heading = a.heading.clone()
    a.flip = flip[:, None].expand(RUNS, ants).contiguous()

    def shifts(world):
        """How far to roll each field so the finished painting sits dead centre."""
        got = np.clip(world.field.cells[:, :, :, :3].cpu().numpy(), -1, 1)
        ink = np.abs(got - 1.0).sum(-1)  # how far each cell is from white
        ink[ink < 0.6] = 0  # ignore faint smudges, weight by how solid the paint is
        origin = world.ant.origin[:, 0].cpu().numpy()

        rows = np.arange(SIZE)
        out = []
        for weight, (ox, oy) in zip(ink, origin):
            # roll to the origin first, so a painting that wrapped becomes whole
            first = (SIZE // 2 - oy, SIZE // 2 - ox)
            w = np.roll(weight, first, axis=(0, 1))
            total = w.sum() or 1.0

            middle_y = (w.sum(1) * rows).sum() / total
            middle_x = (w.sum(0) * rows).sum() / total
            out.append((
                first[0] + SIZE // 2 - round(middle_y),
                first[1] + SIZE // 2 - round(middle_x),
            ))
        return out

    snaps = []
    with torch.no_grad():
        for i in range(STEPS):
            world.step()
            # dense frames at the start, so the first steps are easy to follow
            every = 5 if i < SLOW_STEPS else paint.EVERY
            if i % every == 0:
                snaps.append(snapshot(world))

    roll = shifts(world)  # one shift per rollout, from the finished picture
    slow = SLOW_STEPS // 5  # the dense frames from the beginning

    frames = []
    for n, (imgs, spots, facings, alive) in enumerate(snaps):
        moved = np.stack([np.roll(i, r, axis=(0, 1)) for i, r in zip(imgs, roll)])
        here = [[((x + r[1]) % SIZE, (y + r[0]) % SIZE) for x, y in tile]
                for tile, r in zip(spots, roll)]

        # solid while the gif is slow, half faded once it speeds up
        fade = 255 if n < slow else 128
        frames.append(draw_arrows(
            paint.tile(moved), paint.ZOOM, here, facings, alive, cols=3, alpha=fade
        ))

    timing = [SLOW_MS] * slow + [FAST_MS] * (len(frames) - slow)
    timing[0] = 1000  # pause on the blank field before it starts
    timing[-1] = 2000  # and on the finished picture at the end

    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    frames = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    frames[0].save(
        run / "showcase.gif", save_all=True, append_images=frames[1:],
        duration=timing, loop=0,
    )
    frames[-1].convert("RGB").save(run / "showcase.png")
    print(f"-> {run}/showcase.png and showcase.gif")


if __name__ == "__main__":
    main()
