"""Write the trained brain into demo/weights.js so a browser can read it.

The numbers go in as base64 float32, which keeps the file small and loads
without a web server (a plain file:// page cannot fetch json).
"""

import base64
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import showcase  # noqa: E402
import train  # noqa: E402

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
    out = Path(__file__).parent / "weights.js"
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out} ({out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
