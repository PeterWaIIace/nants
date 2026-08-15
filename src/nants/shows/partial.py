"""Start with a piece of the gecko already painted and ants standing in it.

Can a colony that wakes up inside an existing fragment finish the animal?
Three panels: ants in the middle of the fragment facing every which way, the
same but all facing up, and a control with ants on the gecko's true origin.
"""

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
PANELS = ["in the piece, random", "in the piece, all up", "at the origin (control)"]


def fragment(target):
    """Keep the head end of the gecko, drop the rest. Returns mask and centre."""
    ink = np.abs(target.cpu().numpy() - 1.0).sum(-1) > 0.15
    ys, xs = np.nonzero(ink)

    keep = np.zeros_like(ink)
    edge = ys.min() + (ys.max() - ys.min()) // 2  # the top half of the animal
    keep[: edge + 1] = ink[: edge + 1]

    ky, kx = np.nonzero(keep)
    return keep, (int(kx.mean()), int(ky.mean()))


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
        len(PANELS), cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    )
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved)
    print(f"{run}: brain from epoch {len(ck['losses']) - 1}, mse {min(ck['losses']):.4f}")

    target = train.target_image().to(train.DEVICE)
    keep, middle = fragment(target)
    print(f"fragment covers {keep.sum()} cells, its centre is {middle}")

    world = World(
        brain, train.SIZE, seed=0, noise=0.0, init=train.WHITE,
        horizon=train.STEPS, ants=train.ANTS, scatter=0,
    )
    ant = world.ant
    ant.active = torch.ones_like(ant.active)

    # paint the surviving piece into every panel
    mask = torch.tensor(keep, device=train.DEVICE)
    world.field.cells[:, :, :, :3][:, mask] = target[mask]

    home = torch.tensor(middle, device=train.DEVICE)
    ant.pos = ant.pos.clone()
    ant.pos[:2] = home  # panels 0 and 1 wake up inside the fragment
    ant.origin = ant.pos.clone()

    gen = torch.Generator(device=train.DEVICE).manual_seed(5)
    ant.heading = torch.zeros_like(ant.heading)
    ant.heading[0] = torch.randint(
        4, ant.heading[0].shape, generator=gen, device=train.DEVICE
    )
    ant.start_heading = ant.heading.clone()

    frames = []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            if i % paint.EVERY == 0:
                frames.append(picture(world))

    got = world.field.cells[:, :, :, :3]
    scores = ((got - target) ** 2).mean(dim=(1, 2, 3)).tolist()
    for name, mse in zip(PANELS, scores):
        print(f"  {name:26s} mse {mse:.4f}")

    frames += [frames[-1]] * 30
    timing = [33] * len(frames)
    timing[0], timing[-1] = 1500, 2500

    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    shown = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    shown[0].save(
        run / "partial.gif", save_all=True, append_images=shown[1:],
        duration=timing, loop=0,
    )
    frames[-1].save(run / "partial.png")
    print(f"-> {run}/partial.png and partial.gif")


if __name__ == "__main__":
    main()
