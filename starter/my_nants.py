"""Grow your own picture. Change the emoji, run the file, wait.

    uv run python starter/my_nants.py

It writes a folder under out/ holding the loss curve, the picture as it
stands, and the best brain so far. You can watch those files while it runs,
and stop it whenever you are happy: the best brain is already saved.
"""

import sys

from nants import train
from nants.shows import crowd

# ---- the things worth changing -------------------------------------------

train.EMOJI = "\N{LIZARD}"  # try "\N{OCTOPUS}", "\N{MAPLE LEAF}", "\N{ROCKET}"
train.GECKO = 28            # how many cells across the drawing should be
train.ANTS = 8              # up to this many ants share a field
train.EPOCHS = 3000         # stop earlier by pressing ctrl-c

# smaller and faster, if you have no gpu or no patience:
# train.SIZE, train.GECKO, train.STEPS, train.BATCH = 32, 20, 3000, 256

# ---- and the part that runs -----------------------------------------------


def main():
    print(f"growing {train.EMOJI} on a {train.SIZE}x{train.SIZE} field "
          f"with up to {train.ANTS} ants, on {train.DEVICE}")

    run = train.new_run_dir()
    sys.argv = ["my_nants", str(run)]  # the scripts take the run folder this way
    train.main()

    print("\nrendering the gif")
    crowd.main()
    print(f"\ndone. look in {run}")


if __name__ == "__main__":
    main()
