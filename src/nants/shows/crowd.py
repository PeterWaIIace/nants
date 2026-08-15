"""One gif showing everything the colony can do.

A three by three grid. The first square has one ant, the second two, and so on
up to nine in the last. Each square starts in its own pose, so the geckos come
out turned and mirrored. Halfway through, time freezes, a random piece of every
gecko is wiped away, and the colony carries on and repairs it.

The bar along the bottom is simulation time. It stops while the damage is done.

    uv run python crowd.py out/<run-folder>
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from nants.shows import showcase
from nants import train
from nants.world import World

COLS = 3
ROWS = 3
ZOOM = 8            # cells to screen pixels
PAD = 10            # screen pixels between cards
EDGE = 20           # margin round the whole thing, twice the gap
ARROW = 7           # arrow length in screen pixels
# paced exactly like the web demo: a pause, a cubic wind up, then a cruise
FPS = 30            # frames a second of video
HOLD = 0.4          # seconds standing still at the start
WINDUP = 4.0        # seconds spent winding up at the start
REWIND = 2.0        # a shorter wind up after the damage
CRUISE = 500        # simulation steps a second once wound up
CUT_AT = 0.5        # freeze and take the bite this far through
BITE = 12           # side of the square wiped out of each gecko
FREEZE = 8          # a beat on the wound, then the wind up takes over
BAR = 12            # height of the timer bar, in screen pixels

# the demo's palette
RED = (150, 46, 38)      # the ground, a muted brick
RED_DEEP = (112, 30, 24)  # shadows and the empty part of the timer
SAND = (252, 222, 185)   # the accent
MARK = (176, 3, 0)       # the ants, the one loud red left
ROUND = 6                # corner radius, in screen pixels
DROP = 4                 # hard shadow offset


def poses(panels):
    """Four turns, each plain or mirrored, with the plain one in the middle."""
    out = [(h, f) for f in (1.0, -1.0) for h in range(4)]
    out.insert(panels // 2, (0, 1.0))
    return out[:panels]


def bites(target, gen, panels):
    """A random square over the animal, one per panel."""
    ink = np.abs(target.cpu().numpy() - 1.0).sum(-1) > 0.15
    ys, xs = np.nonzero(ink)

    picks = []
    for _ in range(panels):
        y = int(torch.randint(int(ys.min()), int(ys.max()) - BITE, (1,), generator=gen))
        x = int(torch.randint(int(xs.min()), int(xs.max()) - BITE, (1,), generator=gen))
        picks.append((y, y + BITE, x, x + BITE))
    return picks


def rounded(card):
    """Round the corners of one square, so it sits like a card in the demo."""
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, card.width - 1, card.height - 1],
                                           radius=ROUND, fill=255)
    card.putalpha(mask)
    return card


def triangle(cx, cy, dx, dy, reach):
    """A little arrowhead pointing the way the ant is about to step."""
    sx, sy = -dy * reach * 0.62, dx * reach * 0.62
    return [
        (cx + dx * reach, cy + dy * reach),                 # the tip
        (cx - dx * reach * 0.6 + sx, cy - dy * reach * 0.6 + sy),
        (cx - dx * reach * 0.6 - sx, cy - dy * reach * 0.6 - sy),
    ]


def draw(world, along, frozen, fade=1.0):
    """Nine cards on the red, the ants on top, and the timer underneath."""
    got = np.clip(world.field.cells[:, :, :, :3].cpu().numpy(), -1, 1)
    imgs = ((got + 1) / 2 * 255).astype(np.uint8)

    side = train.SIZE * ZOOM
    pitch = side + PAD
    grid_wide = COLS * pitch - PAD + DROP
    grid_tall = ROWS * pitch - PAD + DROP
    wide = grid_wide + EDGE * 2
    tall = grid_tall + EDGE * 2 + BAR + PAD

    out = Image.new("RGB", (wide, tall), RED)
    shade = ImageDraw.Draw(out)

    for i, img in enumerate(imgs):
        r, c = divmod(i, COLS)
        x, y = EDGE + c * pitch, EDGE + r * pitch

        shade.rounded_rectangle(
            [x + DROP, y + DROP, x + side - 1 + DROP, y + side - 1 + DROP],
            radius=ROUND, fill=RED_DEEP,
        )
        card = Image.fromarray(img).resize((side, side), Image.NEAREST)
        out.paste(rounded(card.convert("RGBA")), (x, y), rounded(card.convert("RGBA")))

    # the ants: solid triangles, fading out as the simulation speeds up
    over = Image.new("RGBA", out.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(over)
    pos = world.ant.pos.cpu().numpy()
    facing = world.ant.heading.cpu().numpy()
    alive = world.ant.active.cpu().numpy()
    tint = MARK + (int(255 * fade),)

    for panel in range(len(imgs)):
        r, c = divmod(panel, COLS)
        for n in range(pos.shape[1]):
            if not alive[panel, n]:
                continue
            x, y = pos[panel, n]
            cx = EDGE + c * pitch + (x + 0.5) * ZOOM
            cy = EDGE + r * pitch + (y + 0.5) * ZOOM
            dx, dy = showcase.ARROW_DIRS[facing[panel, n]]
            pen.polygon(triangle(cx, cy, dx, dy, ARROW), fill=tint)

    out = Image.alpha_composite(out.convert("RGBA"), over).convert("RGB")
    pen = ImageDraw.Draw(out)

    # the timer: sand filling a dark red track, still while time is frozen
    top = EDGE + grid_tall + PAD
    left, right = EDGE, wide - EDGE - 1
    pen.rounded_rectangle([left, top, right, top + BAR - 1], radius=BAR // 2,
                          fill=RED_DEEP)
    run_to = left + max(int((right - left) * along), BAR)
    pen.rounded_rectangle([left, top, run_to, top + BAR - 1], radius=BAR // 2,
                          fill=SAND if not frozen else (255, 255, 255))
    return out


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]
    panels = COLS * ROWS

    ck = showcase.match_clock(run / "best.pt", cell_dim=train.CELL_DIM)
    brain = train.Brain(
        panels, cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    )
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved)
    print(f"{run}: brain from epoch {len(ck['losses']) - 1}, mse {min(ck['losses']):.4f}")

    target = train.target_image().to(train.DEVICE)
    world = World(
        brain, train.SIZE, seed=0, noise=0.0, init=train.WHITE,
        landmark=None, horizon=train.STEPS, ants=panels, scatter=0,  # blank field
    )
    ant = world.ant

    # square 1 gets one ant, square 2 gets two, and so on
    crowds = torch.arange(1, panels + 1, device=train.DEVICE)[:, None]
    ant.active = (torch.arange(panels, device=train.DEVICE) < crowds).float()

    # each square in its own pose: four turns, plain and mirrored
    turns = poses(panels)
    facing = torch.tensor([h for h, _ in turns], device=train.DEVICE)[:, None]
    flip = torch.tensor([f for _, f in turns], device=train.DEVICE)[:, None]
    ant.heading = facing.expand(panels, panels).contiguous()
    ant.start_heading = ant.heading.clone()
    ant.flip = flip.expand(panels, panels).contiguous()

    gen = torch.Generator().manual_seed(4)
    squares = bites(target, gen, panels)
    blank = torch.tensor(train.WHITE, device=train.DEVICE)
    cut = int(train.STEPS * CUT_AT)

    frames, timing = [], []
    clock = 0.0   # seconds of video so far
    from_when = 0.0  # where the wind up curve starts; reset after the damage
    owed = 0.0    # simulation steps still owed at the current rate
    done = 0

    with torch.no_grad():
        while done < train.STEPS:
            since = clock - from_when
            windup = REWIND if from_when else WINDUP
            if since < HOLD:
                rate = 0.0
            else:
                along = min((since - HOLD) / windup, 1.0)
                rate = CRUISE * along * along * along  # cubic, as in the demo

            owed += rate / FPS
            due = int(owed)
            owed -= due

            for _ in range(min(due, train.STEPS - done)):
                world.step()
                done += 1

                if done == cut:
                    part = done / train.STEPS
                    healthy = ((world.field.cells[:, :, :, :3] - target) ** 2).mean()

                    for panel, (top, bottom, left, right) in enumerate(squares):
                        world.field.cells[panel, top:bottom, left:right] = blank
                    hurt = ((world.field.cells[:, :, :, :3] - target) ** 2).mean()
                    frames += [draw(world, part, True, 1.0)] * FREEZE  # a beat on the wound
                    timing += [1000 // FPS] * FREEZE
                    print(f"  step {done}: mse {healthy:.4f} -> {hurt:.4f} after the bite")

                    # wind up again from a standstill, so the repair is watchable
                    clock += FREEZE / FPS
                    from_when = clock
                    owed = 0.0
                    break  # nothing else moves on the frame the bite lands

            # full strength when slow, a quarter at full tilt. rounded to a few
            # steps, so the ants keep landing on the same palette colours
            fade = 1 - 0.75 * min(rate / CRUISE, 1)
            fade = round(fade * 8) / 8
            frames.append(draw(world, done / train.STEPS, False, fade))
            timing.append(1000 // FPS)
            clock += 1 / FPS

    # only the upright squares can be scored against the upright target
    scores = ((world.field.cells[:, :, :, :3] - target) ** 2).mean(dim=(1, 2, 3))
    for i, (turn, mirror) in enumerate(turns):
        if turn == 0 and mirror > 0:
            print(f"  {i + 1} ants, upright: healed back to mse {scores[i]:.4f}")

    frames += [frames[-1]]
    timing += [33]
    timing[0], timing[-1] = 1000, 1000  # a second held at each end
    print(f"  {clock:.1f}s of video: {WINDUP}s winding up at the start, "
          f"{REWIND}s after the damage")

    # build the palette from frames spread through the run, not just the last one,
    # so the faded ants and the half painted geckos both get their own colours
    picks = [frames[i] for i in (0, len(frames) // 4, len(frames) // 2, -1)]
    montage = Image.new("RGB", (frames[0].width, frames[0].height * len(picks)))
    for i, frame in enumerate(picks):
        montage.paste(frame, (0, i * frames[0].height))
    palette = montage.convert("P", palette=Image.ADAPTIVE, colors=256)

    shown = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    shown[0].save(
        run / "crowd.gif", save_all=True, append_images=shown[1:],
        duration=timing, loop=0, optimize=True,
    )
    size = (run / "crowd.gif").stat().st_size / 1e6
    print(f"-> {run}/crowd.gif  ({len(frames)} frames, {shown[0].width}px wide, {size:.0f} MB)")


if __name__ == "__main__":
    main()
