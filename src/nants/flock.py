"""Train several creatures at once, side by side in one rollout.

Step time grows with the batch, so this is not free: measured on a 3090,
four creatures at 1024 runs each cost 42 s an epoch against 11 s for one.
The win comes from splitting one batch between them instead of widening it —
256 runs each, four creatures, 15 s an epoch, about a third of the time four
runs in a row would take.

Each creature owns an equal block of the batch and its own row of brain
weights, so nothing is shared between them but the clock. Every one starts
from a brain trained earlier, which is most of why this is quick: the walking
and the writing carry over, and only the shape has to be learned.

    uv run nants-flock out/<a-finished-run> [out/<flock-to-resume>]
"""

import sys
from datetime import datetime
from pathlib import Path

import torch

from nants import train

CREATURES = {
    "butterfly": "\N{BUTTERFLY}",
    "turtle": "\N{TURTLE}",
    "octopus": "\N{OCTOPUS}",
    "snail": "\N{SNAIL}",
}
PER = 256   # runs per creature; four of them share one batch of 1024
EPOCHS = 900


def targets_for(names):
    """One target picture per creature, stacked: (G, S, S, 3)."""
    pictures = []
    for name in names:
        train.EMOJI = CREATURES[name]
        pictures.append(train.target_image())
    return torch.stack(pictures).to(train.DEVICE)


def already_trained(name):
    """The newest earlier run of this creature, if one got far enough to save."""
    runs = sorted(p for p in Path("out").glob(f"*-{name}") if (p / "best.pt").exists())
    return runs[-1] if runs else None


def warm_start(brain, sources):
    """Fill each group's row from a brain trained earlier."""
    with torch.no_grad():
        for g, source in enumerate(sources):
            ck = torch.load(source / "best.pt", weights_only=False)
            for p, saved in zip(brain.parameters(), ck["w"]):
                p[g] = saved[0].to(train.DEVICE)
            print(f"  group {g} <- {source}  (mse {min(ck['losses']):.4f})", flush=True)


def save_one(brain, opt, losses, world, target, folder, g):
    """Write a creature's own checkpoint, in the shape the exporter expects."""
    folder.mkdir(parents=True, exist_ok=True)
    weights = [p[g : g + 1].detach().clone() for p in brain.parameters()]
    torch.save({"w": weights, "opt": opt.state_dict(), "losses": losses},
               folder / "best.pt")

    painted = world.field.cells[g * PER]
    train.save_png(TinyWorld(painted), target, folder / "best.png")


class TinyWorld:
    """save_png only wants one field, so hand it one."""

    def __init__(self, cells):
        self.field = type("F", (), {"cells": cells[None]})()


def main():
    if train.DEVICE == "cuda":
        train.speed_up()

    source = Path(sys.argv[1])
    run = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out") / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-flock")
    run.mkdir(parents=True, exist_ok=True)

    names = list(CREATURES)
    groups = len(names)
    train.BATCH = PER * groups
    print(f"{groups} creatures, {PER} runs each, batch {train.BATCH} -> {run}\n",
          flush=True)

    targets = targets_for(names)
    brain = train.Brain(
        train.BATCH, cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, groups=groups,
    ).train()
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)

    losses = train.load_ckpt(brain, opt, run / "brain.pt")
    if not losses:  # a fresh flock starts from whatever each creature has already
        warm_start(brain, [already_trained(name) or source for name in names])

    best = [float("inf")] * groups
    curves = [[] for _ in names]

    for e in range(len(losses), len(losses) + EPOCHS):
        scores, world = train.epoch(brain, targets, opt, seed=e)
        losses.append(scores.mean().item())

        for g, name in enumerate(names):
            curves[g].append(scores[g].item())
            if scores[g].item() < best[g]:
                best[g] = scores[g].item()
                save_one(brain, opt, curves[g], world, targets[g], run / name, g)

        if e % 25 == 0:
            line = f"epoch {e:5d}  " + "  ".join(
                f"{n} {s:.4f}" for n, s in zip(names, scores.tolist()))
            print(line, flush=True)
            with open(run / "train.log", "a") as log:
                log.write(line + "\n")
            train.save_curve(losses, targets[0], run / "loss.png")
            train.save_ckpt(brain, opt, losses, run / "brain.pt")

    print("\nall done")


if __name__ == "__main__":
    main()
