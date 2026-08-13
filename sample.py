"""Sample random brains and save tiled gifs."""

from pathlib import Path

from leniaghton.brain import Brain
from leniaghton.render import Recorder
from leniaghton.world import World

OUT = Path("out")
N_BRAINS = 24


def make_gif(name, steps, every):
    brain = Brain(N_BRAINS, cell_dim=4, width=32, seed=0, scale=0.25)
    world = World(brain, size=64, seed=0, noise=0.1, horizon=steps)

    rec = Recorder(world, cols=6, zoom=2)
    rec.record(steps, every)
    rec.save(OUT / name)
    print(f"{name}: {steps} steps, {len(rec.frames)} frames")


def main():
    OUT.mkdir(exist_ok=True)
    make_gif("sample.gif", steps=400, every=1)        # close-up: every step
    make_gif("long.gif", steps=10_000, every=50)      # overview: 200 frames
    make_gif("epic.gif", steps=200_000, every=1000)   # deep time: 200 frames


if __name__ == "__main__":
    main()
