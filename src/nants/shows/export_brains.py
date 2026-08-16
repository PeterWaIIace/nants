"""Write every trained creature out as weights.js, for the browser demo.

One brain per creature, base64 float32, all in one file so the page can hold
several colonies at once.

    uv run nants-brains ../nants-website \\
        gecko=out/20260810-145808 ant=out/20260816-063150-ant

With no pairs given it takes the newest run it can find for each creature.
"""

import base64
import sys
from pathlib import Path

import numpy as np

from nants import train
from nants.shows import showcase

NAMES = ["w1", "write", "move", "b1", "b_write", "b_move"]
EMOJI = {
    "gecko": "\N{LIZARD}",
    "ant": "\N{ANT}",
    "butterfly": "\N{BUTTERFLY}",
    "turtle": "\N{TURTLE}",
    "octopus": "\N{OCTOPUS}",
    "snail": "\N{SNAIL}",
}


def newest(creature):
    """The best run of this creature: a flock folder or a run of its own."""
    found = [p for p in Path("out").glob(f"*-{creature}") if (p / "best.pt").exists()]
    found += [p for p in Path("out").glob(f"*-flock/{creature}")
              if (p / "best.pt").exists()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def pack(run):
    ck = showcase.match_clock(run / "best.pt", cell_dim=train.CELL_DIM)
    parts = []
    for name, tensor in zip(NAMES, ck["w"]):
        flat = tensor.detach().cpu().numpy().squeeze(0).astype(np.float32)
        packed = base64.b64encode(flat.tobytes()).decode()
        parts.append(f'    {name}: {{ shape: {list(flat.shape)}, data: "{packed}" }},')
    return parts, min(ck["losses"]), len(ck["losses"]) - 1


def main():
    where = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    pairs = [a.split("=", 1) for a in sys.argv[2:] if "=" in a]
    runs = {name: Path(folder) for name, folder in pairs}
    for creature in EMOJI:
        if creature not in runs and (found := newest(creature)):
            runs[creature] = found

    lines = ["// every trained creature, exported by export_brains.py",
             "const BRAINS = {"]
    for creature, run in runs.items():
        parts, mse, epochs = pack(run)
        print(f"{creature:10s} {str(run):42s} epoch {epochs:5d}  mse {mse:.4f}")
        lines.append(f'  {creature}: {{ emoji: "{EMOJI[creature]}", mse: {mse:.4f},')
        lines += parts
        lines.append("  },")
    lines.append("};")

    (where / "weights.js").write_text("\n".join(lines) + "\n")
    print(f"\n-> {where / 'weights.js'}")


if __name__ == "__main__":
    main()
