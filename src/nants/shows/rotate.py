"""Start the trained ant facing each of the four directions and see what it paints.

If its senses really are relative to its own starting pose, the picture should
come out rotated with it.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nants.shows import paint
from nants import train
from nants.world import World

FACES = ["up (as trained)", "right", "down", "left"]


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]
    target = train.target_image().to(train.DEVICE)
    brain, epoch, best = paint.load_brain(run, batch=4)
    print(f"{run}: brain from epoch {epoch}, mse {best:.4f}")

    world = World(
        brain, train.SIZE, seed=0, noise=0.0, init=train.WHITE,
        seed_cell=train.SEED, horizon=train.STEPS, ants=train.ANTS,
    )
    facing = torch.arange(4, device=train.DEVICE)
    world.ant.heading = facing.clone()
    world.ant.start_heading = facing.clone()

    frames = []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            if i % paint.EVERY == 0:
                frames.append(paint.tile(paint.pictures(world), cols=4))

    frames += [frames[-1]] * 20
    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    frames = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    frames[0].save(
        run / "rotate.gif", save_all=True, append_images=frames[1:],
        duration=67, loop=0,
    )

    paint.tile(paint.pictures(world), cols=4).save(run / "rotate.png")
    for face, img in zip(FACES, world.field.cells[:, :, :, :3]):
        err = ((img - target) ** 2).mean().item()
        print(f"  starting {face:16s} mse vs upright target {err:.4f}")


if __name__ == "__main__":
    main()
