"""Carry a trained brain over to a run with different inputs.

The clocked brain reads 93 numbers; the clockless one reads 85. Everything
except the eight clock rows of the first layer is the same, so we keep the rest
and let training close the gap.
"""

import sys
from datetime import datetime
from pathlib import Path

import torch

from nants import train
from nants import ant as ant_module

CLOCKED = 8  # how many clock inputs the old brain had


def main():
    source = Path(sys.argv[1])
    ck = torch.load(source / "best.pt", weights_only=False)
    print(f"from {source}: epoch {len(ck['losses']) - 1}, mse {min(ck['losses']):.4f}")

    brain = train.Brain(
        train.BATCH, cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    ).train()
    wanted = brain.w1.shape[1]
    print(f"old brain read {ck['w'][0].shape[1]} numbers, the new one reads {wanted}")

    kept = []
    for name, saved in zip(["w1", "write", "move", "b1", "b_write", "b_move"], ck["w"]):
        if name == "w1":
            # rows are: cell, gradients, [clock], compass, facing
            before = 3 * train.CELL_DIM
            saved = torch.cat([saved[:, :before], saved[:, before + CLOCKED :]], dim=1)
            print(f"  w1: dropped the {CLOCKED} clock rows -> {tuple(saved.shape)}")
        kept.append(saved.to(train.DEVICE))

    with torch.no_grad():
        for p, saved in zip(brain.parameters(), kept):
            p.copy_(saved)

    run = Path("out") / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-noclock")
    run.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)  # fresh, the shapes moved
    train.save_ckpt(brain, opt, [], run / "brain.pt")

    print(f"-> {run}")
    print(f"carry on with:  uv run python train.py {run}")


if __name__ == "__main__":
    main()
