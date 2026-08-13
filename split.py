"""One colony, one field, but half the ants start facing up and half facing down.

They share a starting cell, so both halves paint around the same origin: one
gecko the right way up, one upside down, on top of each other.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

import colony
import paint
import showcase
import train
from leniaghton.world import World

ANTS = 8  # half face up, half face down


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

    world = World(
        brain, train.SIZE, seed=0, noise=0.0, init=train.WHITE,
        landmark=train.LANDMARK, horizon=train.STEPS, ants=ANTS, scatter=0,
    )
    ant = world.ant
    ant.active = torch.ones_like(ant.active)

    facing = torch.tensor(
        [0] * (ANTS // 2) + [2] * (ANTS - ANTS // 2), device=train.DEVICE
    )
    ant.heading = facing[None].contiguous()  # 0 is up, 2 is down
    ant.start_heading = ant.heading.clone()
    print(f"{ANTS // 2} ants facing up, {ANTS - ANTS // 2} facing down")

    slow = showcase.SLOW_STEPS // 5
    frames = []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            every = 5 if i < showcase.SLOW_STEPS else paint.EVERY
            if i % every == 0:
                frames.append(colony.picture(world, 255 if len(frames) < slow else 128))

    timing = [showcase.SLOW_MS] * slow
    timing += [showcase.FAST_MS] * (len(frames) - slow)
    timing[0], timing[-1] = 1000, 2500

    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    shown = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    shown[0].save(
        run / "split.gif", save_all=True, append_images=shown[1:],
        duration=timing, loop=0,
    )
    frames[-1].save(run / "split.png")
    print(f"-> {run}/split.png and split.gif")


if __name__ == "__main__":
    main()
