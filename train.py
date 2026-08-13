"""Train one ant to paint a gecko.

Two gradient paths:
  - what it writes  -> normal backprop through d_cell
  - where it walks  -> REINFORCE, since the move is a discrete sample
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from leniaghton.ant import LIMIT
from leniaghton.brain import Brain
from leniaghton.world import World

SIZE = 48  # the field
BATCH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 6000
CHUNK = 100  # truncate backprop every CHUNK steps
POLICY_W = 1.0  # weight of the REINFORCE term against the image loss
CLIP = 1.0  # largest gradient norm we let through
ANTS = 8  # up to this many ants share a field; each world gets 1 to 8
EPOCHS = 8000
GECKO = 28  # the gecko itself spans this many cells, the rest is breathing room
EMOJI = "\N{LIZARD}"


TARGET = "gecko"  # "green" / "square" diagnostics, or "gecko" for the real thing
GREEN = torch.tensor([-1.0, 0.2, -0.6])


def square_target():
    """A filled green square in the middle: the simplest localized picture."""
    img = torch.ones(SIZE, SIZE, 3)  # white
    lo, hi = SIZE // 2 - 6, SIZE // 2 + 6
    img[lo:hi, lo:hi] = GREEN
    return img


def target_image():
    if TARGET == "green":
        return GREEN.expand(SIZE, SIZE, 3).clone()  # no localizing needed at all
    if TARGET == "square":
        return square_target()

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", size=109
    )
    img = Image.new("RGB", (136, 136), "white")
    ImageDraw.Draw(img).text((14, 14), EMOJI, font=font, embedded_color=True)

    # the glyph sits off to one side of its box, so crop to the drawing itself
    ink = np.asarray(img).min(-1) < 250
    ys, xs = np.nonzero(ink)
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    wide = img.width >= img.height  # keep it in proportion
    side = (GECKO, round(GECKO * img.height / img.width)) if wide else (
        round(GECKO * img.width / img.height), GECKO
    )
    rgb = np.asarray(img.resize(side, Image.LANCZOS), dtype=np.float32) / 255.0

    canvas = torch.ones(SIZE, SIZE, 3)  # white, with the gecko centred in it
    top, left = (SIZE - side[1]) // 2, (SIZE - side[0]) // 2
    canvas[top : top + side[1], left : left + side[0]] = torch.tensor(rgb * 2 - 1)
    return canvas


CELL_DIM = 16  # first 3 channels are the picture, the rest are the ant's scratch space
WHITE = [1.0, 1.0, 1.0] + [0.0] * (CELL_DIM - 3)
SCATTER = 0  # 0 stacks every ant on the landmark facing up; >0 scatters them


def mark(colour):
    return list(colour) + [1.0] * (CELL_DIM - 3)  # visible colour, scratch alive


# Three cells the ants can all see and agree on: the corner is the origin, red
# points up, green points right. Being three different colours, it fixes the
# axes and the handedness, which a plain L would not.
LANDMARK = [
    ((0, 0), mark([-1.0, -1.0, -1.0])),  # black corner
    ((0, -1), mark([1.0, -1.0, -1.0])),  # red, the y axis
    ((1, 0), mark([-1.0, 1.0, -1.0])),  # green, the x axis
]


def epoch(brain, target, opt, seed):
    """One rollout, graded more and more strictly as the picture fills in."""
    world = World(
        brain, SIZE, seed=seed, noise=0.0, init=WHITE, landmark=LANDMARK, horizon=STEPS, ants=ANTS, scatter=SCATTER
    )
    chunks = STEPS // CHUNK
    weights = torch.arange(1, chunks + 1, dtype=torch.float32)
    weights = weights / weights.sum()  # late chunks matter most

    opt.zero_grad()
    for k in range(chunks):
        logps = []
        for _ in range(CHUNK):
            world.step()
            logps.append(world.ant.logp)

        painted = world.field.cells[:, :, :, :3]
        per_run = ((painted - target) ** 2).mean(dim=(1, 2, 3))

        # advantage in raw mse units, so it stays comparable to the image loss
        adv = -(per_run - per_run.mean()).detach()
        # sum over steps and over ants: they share the one picture being scored
        walked = torch.stack(logps).sum(dim=(0, 2))
        reinforce = -(walked * adv).mean()

        (weights[k] * (per_run.mean() + POLICY_W * reinforce)).backward()

        # detach for truncated backprop, and stop any runaway before it reaches inf
        world.field.cells = world.field.cells.detach().clamp(-LIMIT, LIMIT)

    # one update per rollout, clipped, and skipped outright if it went bad
    size = torch.nn.utils.clip_grad_norm_(brain.parameters(), CLIP)
    if torch.isfinite(size):
        opt.step()
    opt.zero_grad()

    return per_run.mean().item(), world  # score of the finished picture


def save_png(world, target, path):
    got = world.field.cells[0, :, :, :3].detach().cpu().numpy()
    both = np.concatenate([got, target.cpu().numpy()], axis=1)
    img = ((np.clip(both, -1, 1) + 1) / 2 * 255).astype(np.uint8)
    Image.fromarray(img).resize((SIZE * 8 * 2, SIZE * 8), Image.NEAREST).save(path)


def save_curve(losses, target, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, label="ant")
    blank = torch.tensor(WHITE[:3], device=target.device)
    ax.axhline(
        float(((blank - target) ** 2).mean().cpu()),
        color="grey", ls="--", label="write nothing",
    )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mse (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def new_run_dir():
    """out/<datetime>/ so runs sort by age. Pass a folder to resume one."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return Path("out") / datetime.now().strftime("%Y%m%d-%H%M%S")


def write_config(run):
    settings = {
        "target": TARGET, "size": SIZE, "gecko": GECKO, "batch": BATCH, "steps": STEPS,
        "chunk": CHUNK, "policy_w": POLICY_W, "cell_dim": CELL_DIM, "device": DEVICE,
        "ants": ANTS,
    }
    lines = [f"{k} = {v}" for k, v in settings.items()]
    (run / "config.txt").write_text("\n".join(lines) + "\n")


def save_ckpt(brain, opt, losses, path):
    weights = [p.detach().clone() for p in brain.parameters()]
    torch.save({"w": weights, "opt": opt.state_dict(), "losses": losses}, path)


def load_ckpt(brain, opt, path):
    """Resume if a checkpoint exists. Returns the loss history so far."""
    if not path.exists():
        return []

    ck = torch.load(path, weights_only=False)
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved)
    opt.load_state_dict(ck["opt"])
    print(f"resumed from epoch {len(ck['losses'])}")
    return ck["losses"]


def speed_up():
    """Fuse the sensing into fewer, larger gpu kernels: about 1.5x per epoch.

    Only `sense` is worth compiling. The brain samples its move with a
    generator, which defeats the compiler and comes out slower.
    """
    from leniaghton import ant as ant_module

    ant_module.Ant.sense = torch.compile(ant_module.Ant.sense)


def main():
    if DEVICE == "cuda":
        speed_up()

    run = new_run_dir()
    frames, ckpts = run / "frames", run / "checkpoints"
    for folder in (run, frames, ckpts):
        folder.mkdir(parents=True, exist_ok=True)
    write_config(run)
    print(f"run: {run}", flush=True)

    target = target_image().to(DEVICE)
    brain = Brain(
        BATCH, cell_dim=CELL_DIM, width=128, seed=0,
        device=DEVICE, shared=True, zero_out=True,
    ).train()
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)

    losses = load_ckpt(brain, opt, run / "brain.pt")
    best = min(losses, default=float("inf"))

    for e in range(len(losses), len(losses) + EPOCHS):
        loss, world = epoch(brain, target, opt, seed=e)
        losses.append(loss)

        if loss < best:  # keep the best brain we ever had, whatever happens later
            best = loss
            save_ckpt(brain, opt, losses, run / "best.pt")
            save_png(world, target, run / "best.png")
        if e % 25 == 0:
            line = f"epoch {e:5d}  mse {loss:.4f}"
            print(line, flush=True)
            with open(run / "train.log", "a") as log:
                log.write(line + "\n")
            save_png(world, target, run / "train.png")
            save_curve(losses, target, run / "loss.png")
            save_ckpt(brain, opt, losses, run / "brain.pt")
        if e % 100 == 0:
            save_png(world, target, frames / f"{e:06d}.png")
        if e % 200 == 0:
            save_ckpt(brain, opt, losses, ckpts / f"{e:06d}.pt")

    save_curve(losses, target, run / "loss.png")
    save_ckpt(brain, opt, losses, run / "brain.pt")


if __name__ == "__main__":
    main()
