"""Train the ants to compute a logic gate, ncpu style.

    uv run python starter/my_gates.py

Inputs are pinned discs left of the landmark, the output disc is on the
right. Stop whenever the accuracy line hits 1.00 on all four patterns.
"""

import sys

from nants import train

# ---- the things worth changing -------------------------------------------

train.TARGET = "gates"
train.GATE = "AND"          # AND, OR, NAND, NOR, XOR, XNOR
train.STEPS = 2000          # a walk to the inputs and back is ~60 cells
train.EPOCHS = 4000

# smaller and faster, if you have no gpu or no patience:
# train.SIZE, train.STEPS, train.BATCH = 32, 800, 256

# ---- and the part that runs -----------------------------------------------


def main():
    print(f"growing {train.GATE} on a {train.SIZE}x{train.SIZE} field "
          f"with up to {train.ANTS} ants, on {train.DEVICE}")

    run = train.new_run_dir()
    sys.argv = ["my_gates", str(run)]
    train.main()

    print(f"\ndone. look in {run}")


if __name__ == "__main__":
    main()
