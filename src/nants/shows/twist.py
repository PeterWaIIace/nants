"""Turn the ant's sobel kernels and see what the colony builds.

The ant was trained reading gradients along the grid's axes. Here the same
brain reads them turned by 45, 90 and 135 degrees, everything else unchanged.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nants.shows import paint
from nants.shows import showcase
from nants import train
from nants.world import World

ZOOM = 5
ANGLES = [0, 45, 90, 135]  # degrees to turn the kernels by


def picture(world):
    got = np.clip(world.field.cells[:, :, :, :3].cpu().numpy(), -1, 1)
    imgs = ((got + 1) / 2 * 255).astype(np.uint8)

    pos = world.ant.pos.cpu().numpy()
    for n in range(pos.shape[1]):
        imgs[np.arange(len(imgs)), pos[:, n, 1], pos[:, n, 0]] = showcase.ANT

    strip = np.concatenate(list(imgs), axis=1)
    out = Image.fromarray(strip)
    return out.resize((out.width * ZOOM, out.height * ZOOM), Image.NEAREST)


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]

    ck = showcase.match_clock(run / "best.pt", cell_dim=train.CELL_DIM)
    brain = train.Brain(
        len(ANGLES), cell_dim=train.CELL_DIM, width=128, seed=0,
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
    world.ant.active = torch.ones_like(world.ant.active)

    radians = torch.tensor(
        [math.radians(a) for a in ANGLES], device=train.DEVICE
    )
    world.ant.twist = radians[:, None].expand_as(world.ant.twist).contiguous()

    frames = []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            if i % paint.EVERY == 0:
                frames.append(picture(world))

    got = world.field.cells[:, :, :, :3]
    scores = ((got - target) ** 2).mean(dim=(1, 2, 3)).tolist()
    for angle, mse in zip(ANGLES, scores):
        print(f"  kernels turned {angle:3d} degrees -> mse {mse:.4f}")

    frames += [frames[-1]] * 30
    timing = [33] * len(frames)
    timing[0], timing[-1] = 1000, 2500

    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    shown = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    shown[0].save(
        run / "twist.gif", save_all=True, append_images=shown[1:],
        duration=timing, loop=0,
    )
    frames[-1].save(run / "twist.png")
    print(f"-> {run}/twist.png and twist.gif")


if __name__ == "__main__":
    main()
