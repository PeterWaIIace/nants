"""Write the trained brain out as weights.js, for the browser demo to read.

The numbers go in as base64 float32, which keeps the file small. The demo
itself lives on the website branch; pass the folder to write into.

    uv run nants-weights out/<run-folder> ../nants-website
"""

import base64
import sys
from pathlib import Path

import numpy as np
import torch

from nants import train
from nants.shows import showcase

NAMES = ["w1", "write", "move", "b1", "b_write", "b_move"]


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]

    ck = showcase.match_clock(run / "best.pt", cell_dim=train.CELL_DIM)
    print(f"{run}: epoch {len(ck['losses']) - 1}, mse {min(ck['losses']):.4f}")

    lines = ["// the trained ant, exported by export_weights.py", "const WEIGHTS = {"]
    for name, tensor in zip(NAMES, ck["w"]):
        flat = tensor.detach().cpu().numpy().squeeze(0).astype(np.float32)
        packed = base64.b64encode(flat.tobytes()).decode()
        lines.append(f'  {name}: {{ shape: {list(flat.shape)}, data: "{packed}" }},')
        print(f"  {name:8s} {tuple(flat.shape)}")

    lines.append("};")
    where = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out = where / "weights.js"
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out} ({out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
