"""Let the colony finish a gecko, cut a piece off, and keep the clock running.

Nothing in training ever showed the ants damage, so whatever happens next is
not something they were taught.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import paint
import showcase
import train
from leniaghton.world import World

ZOOM = 5
CUT_AT = 0.5  # do the damage this far through the run
PANELS = ["untouched", "head cut off", "tail cut off"]


def parts(target):
    """Boxes round the head and the tail, found from the target itself."""
    ink = np.abs(target.cpu().numpy() - 1.0).sum(-1) > 0.15
    ys, xs = np.nonzero(ink)
    side = 12

    head = (ys.min(), ys.min() + side, xs.min(), xs.min() + side)
    tail = (ys.max() - side, ys.max(), xs.max() - side, xs.max())
    return [None, head, tail]


def wipe(world, panel, box):
    """Erase a square back to blank white, scratch channels and all."""
    top, bottom, left, right = box
    blank = torch.tensor(train.WHITE, device=world.field.cells.device)
    world.field.cells[panel, top:bottom, left:right] = blank


def picture(world, target):
    got = np.clip(world.field.cells[:, :, :, :3].cpu().numpy(), -1, 1)
    imgs = ((got + 1) / 2 * 255).astype(np.uint8)

    pos = world.ant.pos.cpu().numpy()
    for n in range(pos.shape[1]):
        imgs[np.arange(len(imgs)), pos[:, n, 1], pos[:, n, 0]] = showcase.ANT

    strip = np.concatenate(list(imgs), axis=1)
    out = Image.fromarray(strip)
    return out.resize((out.width * ZOOM, out.height * ZOOM), Image.NEAREST)


def score(world, target):
    got = world.field.cells[:, :, :, :3]
    return ((got - target) ** 2).mean(dim=(1, 2, 3)).tolist()


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]

    ck = showcase.match_clock(run / "best.pt", cell_dim=train.CELL_DIM)
    brain = train.Brain(
        len(PANELS), cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    )
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved)
    print(f"{run}: brain from epoch {len(ck['losses']) - 1}, mse {min(ck['losses']):.4f}")

    target = train.target_image().to(train.DEVICE)
    world = World(
        brain, train.SIZE, seed=0, noise=0.0, init=train.WHITE,
        landmark=train.LANDMARK, horizon=train.STEPS, ants=train.ANTS, scatter=0,
    )
    world.ant.active = torch.ones_like(world.ant.active)  # full crowd everywhere
    boxes = parts(target)

    cut = int(train.STEPS * CUT_AT)
    frames, hold = [], []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            if i == cut:
                print("before the cut: " + ", ".join(
                    f"{n} {s:.4f}" for n, s in zip(PANELS, score(world, target))
                ))
                for panel, box in enumerate(boxes):
                    if box is not None:
                        wipe(world, panel, box)
                print("after  the cut: " + ", ".join(
                    f"{n} {s:.4f}" for n, s in zip(PANELS, score(world, target))
                ))
                hold = [picture(world, target)] * 25  # pause on the damage

            if i % paint.EVERY == 0:
                frames.append(picture(world, target))
                if hold:
                    frames += hold
                    hold = []

    print("at the end:     " + ", ".join(
        f"{n} {s:.4f}" for n, s in zip(PANELS, score(world, target))
    ))

    frames += [frames[-1]] * 30
    timing = [33] * len(frames)
    timing[0], timing[-1] = 1000, 2500

    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    shown = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    shown[0].save(
        run / "regen.gif", save_all=True, append_images=shown[1:],
        duration=timing, loop=0,
    )
    frames[-1].save(run / "regen.png")
    print(f"-> {run}/regen.png and regen.gif")


if __name__ == "__main__":
    main()
