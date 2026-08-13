"""Drop the trained ant anywhere, facing anywhere, and see what it paints."""

import sys
from pathlib import Path

import torch
from PIL import Image

import paint
import train
from leniaghton.world import World

RUNS = 9


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]
    target = train.target_image().to(train.DEVICE)
    brain, epoch, best = paint.load_brain(run, batch=RUNS)
    print(f"{run}: brain from epoch {epoch}, mse {best:.4f}")

    # no seed cell yet: the ants have not been placed
    world = World(
        brain, train.SIZE, seed=0, noise=0.0, init=train.WHITE, horizon=train.STEPS,
    )
    gen = torch.Generator(device=train.DEVICE).manual_seed(7)
    where = torch.randint(
        train.SIZE, (RUNS, 2), generator=gen, device=train.DEVICE
    )
    facing = torch.randint(4, (RUNS,), generator=gen, device=train.DEVICE)

    ant = world.ant
    ant.pos, ant.origin = where, where.clone()
    ant.heading, ant.start_heading = facing, facing.clone()
    world.field.set(ant.pos, torch.tensor(train.SEED, device=train.DEVICE).expand(RUNS, -1))

    frames = []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            if i % paint.EVERY == 0:
                frames.append(paint.tile(paint.pictures(world)))

    frames += [frames[-1]] * 20
    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    frames = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    frames[0].save(
        run / "scatter.gif", save_all=True, append_images=frames[1:],
        duration=67, loop=0,
    )
    paint.tile(paint.pictures(world)).save(run / "scatter.png")

    for i in range(RUNS):
        x, y = where[i].tolist()
        print(f"  ant {i}: start ({x:2d},{y:2d}) facing {facing[i].item()}")


if __name__ == "__main__":
    main()
