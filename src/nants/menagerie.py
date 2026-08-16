"""Train a colony for each of several creatures, one after another.

Each starts from an already trained brain rather than from nothing, so it only
has to learn the new shape, which takes a fraction of the time.

    uv run nants-menagerie out/<a-finished-run>

Give it the stamp of an earlier menagerie and it picks up where that one
stopped, rather than starting the whole set again:

    uv run nants-menagerie out/<a-finished-run> 20260816-063150
"""

import sys
from datetime import datetime
from pathlib import Path

import torch

from nants import train

CREATURES = {
    "ant": "\N{ANT}",
    "butterfly": "\N{BUTTERFLY}",
    "turtle": "\N{TURTLE}",
    "octopus": "\N{OCTOPUS}",
    "snail": "\N{SNAIL}",
}
EACH = 900  # epochs per creature, warm started


def borrow(source, into):
    """Put a trained brain into a fresh run folder, with the counters reset."""
    ck = torch.load(source / "best.pt", weights_only=False)

    brain = train.Brain(
        train.BATCH, cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    ).train()
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved.to(train.DEVICE))

    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)
    into.mkdir(parents=True, exist_ok=True)
    train.save_ckpt(brain, opt, [], into / "brain.pt")
    return min(ck["losses"])


def done_so_far(run):
    """How many epochs a half finished run already has behind it."""
    saved = run / "brain.pt"
    if not saved.exists():
        return None
    return len(torch.load(saved, weights_only=False)["losses"])


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]
    stamp = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime(
        "%Y%m%d-%H%M%S")

    print(f"starting every creature from {source}\n", flush=True)

    for name, emoji in CREATURES.items():
        run = Path("out") / f"{stamp}-{name}"
        already = done_so_far(run)

        if already is None:
            note = f"from a brain at mse {borrow(source, run):.4f}"
        elif already >= EACH:
            print(f"=== {name} {emoji} -> {run}   done already", flush=True)
            continue
        else:
            note = f"carrying on from epoch {already}"

        print(f"=== {name} {emoji} -> {run}   ({note})", flush=True)

        train.EMOJI = emoji
        train.EPOCHS = EACH - (already or 0)
        sys.argv = ["menagerie", str(run)]
        train.main()

    print("\nall done")


if __name__ == "__main__":
    main()
