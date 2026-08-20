"""Train one ant to paint a gecko.

Two gradient paths:
  - what it writes  -> normal backprop through d_cell
  - where it walks  -> REINFORCE, since the move is a discrete sample
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from nants.ant import LIMIT
from nants.brain import Brain
from nants.world import World

SIZE = 48  # the field
BATCH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 6000
CHUNK = 200  # truncate backprop every CHUNK steps
POLICY_W = 1.0  # weight of the REINFORCE term against the image loss
CLIP = 1.0  # largest gradient norm we let through
ANTS = 8  # up to this many ants share a field; each world gets 1 to 8
EPOCHS = 8000
GECKO = 28  # the gecko itself spans this many cells, the rest is breathing room
EMOJI = "\N{LIZARD}"


TARGET = "ncpu"   # "green" / "square" / "gecko", "gates" or "ncpu" for logic gates
GATE = "XOR"      # which truth table to grow when TARGET == "gates" or "ncpu"
GATE_R = 2        # radius of one bit's disc, in cells
IO_DX = 14        # input and output discs sit this far left/right of the landmark
IO_DY = 5         # vertical spacing between input discs
MASK_W = 0.8      # ncpu's combined loss: 0.8 on the output disc, 0.2 on the field
# the emoji font, overridable so the code runs on a machine without it installed
FONT = os.environ.get("NANTS_FONT",
                      "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf")
GREEN = torch.tensor([-1.0, 0.2, -0.6])

# ncpu-style circle targets (loaded lazily only when TARGET == "ncpu")
NCPU_TASK = None


def _get_ncpu_task():
    """Lazily build and cache the NCPUGateTask so its state persists across epochs."""
    global NCPU_TASK
    if NCPU_TASK is None:
        from nants.ncpu_gates import NCPUGateTask
        NCPU_TASK = NCPUGateTask(
            gate=GATE,
            size=SIZE,
            batch=BATCH,
            cell_dim=CELL_DIM,
            device=DEVICE,
            r=4,
            input_bits=2,
            io_gap=6,
        )
    return NCPU_TASK


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

    font = ImageFont.truetype(FONT, size=109)
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


def epoch(brain, opt, seed, task=None):
    """One rollout, graded more and more strictly as the picture fills in.

    When *task* is given its ``regenerate()`` is called first so every
    epoch sees freshly randomised bits (for ncpu-style dynamic targets).
    """
    if task is not None:
        task.regenerate()
        target = task.target[..., :3]       # picture channels only, (G, H, W, 3)
        world = task.make_world(
            brain, seed=seed, steps=STEPS, ants=ANTS, scatter=SCATTER,
        )
    else:
        world = World(
            brain, SIZE, seed=seed, noise=0.0, init=WHITE, landmark=LANDMARK,
            horizon=STEPS, ants=ANTS, scatter=SCATTER,
        )
        target = target_image().to(DEVICE)  # static target, recomputed each call

    chunks = STEPS // CHUNK
    weights = torch.arange(1, chunks + 1, dtype=torch.float32, device=DEVICE)
    weights = weights / weights.sum()  # late chunks matter most

    opt.zero_grad()
    for k in range(chunks):
        logps = []
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            for _ in range(CHUNK):
                world.step()
                logps.append(world.ant.logp)

        painted = world.field.cells[:, :, :, :3]
        if target.dim() == 4:  # one target per group of runs, trained side by side
            groups = target.shape[0]
            spread = painted.view(groups, -1, *painted.shape[1:])
            diff2 = ((spread - target[:, None]) ** 2).mean(dim=4)  # (G, n, H, W)
            if task is not None:
                # ncpu's combined loss: 0.8 on the output disc, 0.2 on the field
                mask = task.out_mask  # (G, H, W)
                spots = mask.sum(dim=(1, 2))[:, None]  # (G, 1)
                per_run = MASK_W * (diff2 * mask[:, None]).sum(dim=(2, 3)) / spots \
                        + (1 - MASK_W) * diff2.mean(dim=(2, 3))
            else:
                per_run = diff2.mean(dim=(2, 3))
        else:
            per_run = ((painted - target) ** 2).mean(dim=(1, 2, 3))

        # advantage in raw mse units, so it stays comparable to the image loss.
        # the baseline is per group: a harder picture must not drag the others.
        adv = -(per_run - per_run.mean(dim=-1, keepdim=True)).detach()
        # sum over steps and over ants: they share the one picture being scored
        walked = torch.stack(logps).sum(dim=(0, 2)).view(per_run.shape)
        reinforce = -(walked * adv).mean()

        (weights[k] * (per_run.mean() + POLICY_W * reinforce)).backward()

        # detach for truncated backprop, and stop any runaway before it reaches inf
        world.field.cells = world.field.cells.detach().clamp(-LIMIT, LIMIT)

    # one update per rollout, clipped, and skipped outright if it went bad
    size = torch.nn.utils.clip_grad_norm_(brain.parameters(), CLIP)
    if torch.isfinite(size):
        opt.step()
    opt.zero_grad()

    # the score of the finished picture, one number per group
    return per_run.detach().mean(dim=-1), world


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
    if TARGET == "gates":
        settings.update(gate=GATE, gate_r=GATE_R, io_dx=IO_DX,
                        io_dy=IO_DY, mask_w=MASK_W)
    elif TARGET == "ncpu":
        settings.update(gate=GATE, mask_w=MASK_W, circle_r=4, io_gap=6,
                        input_bits=2)
    lines = [f"{k} = {v}" for k, v in settings.items()]
    (run / "config.txt").write_text("\n".join(lines) + "\n")


def save_ckpt(brain, opt, losses, path):
    weights = [p.detach().clone() for p in brain.parameters()]
    torch.save({"w": weights, "opt": opt.state_dict(), "losses": losses}, path)


_prev_ckpt_thread = None


def save_ckpt_async(brain, opt, losses, path):
    """Save checkpoint in a background thread so I/O never stalls training."""
    global _prev_ckpt_thread
    if _prev_ckpt_thread is not None:
        _prev_ckpt_thread.join()

    weights = [p.detach().clone() for p in brain.parameters()]
    state = opt.state_dict()
    loss_copy = list(losses)

    def _save():
        torch.save({"w": weights, "opt": state, "losses": loss_copy}, path)

    _prev_ckpt_thread = threading.Thread(target=_save, daemon=True)
    _prev_ckpt_thread.start()


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
    """Fuse the entire step (sense + brain + write + move) into fewer GPU kernels.

    Compiling Ant.step gives ~2-3x over the baseline.  The brain uses the
    gumbel-max trick instead of multinomial so the compiler can fuse the
    sampling into the same graph.
    """
    from nants.brain import Brain as BrainClass
    BrainClass._use_gumbel = True

    from nants import ant as ant_module
    ant_module.Ant.step = torch.compile(ant_module.Ant.step)


def main():
    if DEVICE == "cuda":
        speed_up()

    run = new_run_dir()
    frames, ckpts = run / "frames", run / "checkpoints"
    for folder in (run, frames, ckpts):
        folder.mkdir(parents=True, exist_ok=True)
    write_config(run)
    print(f"run: {run}", flush=True)

    task = None
    if TARGET == "gates":
        from nants import gates
        task = gates.GateTask(
            GATE, LANDMARK, SIZE, BATCH, CELL_DIM, DEVICE,
            r=GATE_R, io_dx=IO_DX, io_dy=IO_DY,
        )
        vis_target = task.target[0, ..., :3]
    elif TARGET == "ncpu":
        task = _get_ncpu_task()
        vis_target = task.target[0, ..., :3]
    else:
        task = None
        vis_target = target_image().to(DEVICE)

    brain = Brain(
        BATCH, cell_dim=CELL_DIM, width=128, seed=0,
        device=DEVICE, shared=True, zero_out=True,
    ).train()
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)

    losses = load_ckpt(brain, opt, run / "brain.pt")
    best = min(losses, default=float("inf"))

    pbar = tqdm(range(len(losses), len(losses) + EPOCHS), initial=len(losses),
                total=len(losses) + EPOCHS, desc="training")
    for e in pbar:
        scores, world = epoch(brain, opt, seed=e, task=task)
        loss = scores.mean().item()
        losses.append(loss)

        desc = f"mse {loss:.4f}"
        if task is not None:
            acc = task.exact_match(world)
            desc += f"  acc {acc.mean().item():.3f}"
        pbar.set_description(desc)

        if loss < best:  # keep the best brain we ever had, whatever happens later
            best = loss
            save_ckpt_async(brain, opt, losses, run / "best.pt")
            save_png(world, vis_target, run / "best.png")
        if e % 25 == 0:
            line = f"epoch {e:5d}  mse {loss:.4f}"
            if task is not None:
                line += f"  acc {acc.mean().item():.3f}  " \
                        + " ".join(f"{a.item():.2f}" for a in acc)
            with open(run / "train.log", "a") as log:
                log.write(line + "\n")
            save_png(world, vis_target, run / "train.png")
            save_curve(losses, vis_target, run / "loss.png")
            save_ckpt_async(brain, opt, losses, run / "brain.pt")
        if e % 100 == 0:
            save_png(world, vis_target, frames / f"{e:06d}.png")
        if e % 200 == 0:
            save_ckpt(brain, opt, losses, ckpts / f"{e:06d}.pt")

    pbar.close()
    save_curve(losses, vis_target, run / "loss.png")
    save_ckpt(brain, opt, losses, run / "brain.pt")


if __name__ == "__main__":
    main()
