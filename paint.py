"""Watch the trained ant paint, as gifs."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import train
from leniaghton.world import World

BATCH = 9  # nine rollouts of the same brain, to see how much they vary
EVERY = 10  # one frame every EVERY steps
ZOOM = 6


def pictures(world):
    """Every rollout's visible channels as uint8 images, ant marked red."""
    got = np.clip(world.field.cells[:, :, :, :3].cpu().numpy(), -1, 1)
    imgs = ((got + 1) / 2 * 255).astype(np.uint8)

    pos = world.ant.pos.cpu().numpy()  # (B, N, 2)
    for n in range(pos.shape[1]):
        imgs[np.arange(len(imgs)), pos[:, n, 1], pos[:, n, 0]] = [255, 50, 50]
    return imgs


def tile(imgs, cols=3, gap=1):
    size = imgs.shape[1]
    rows = -(-len(imgs) // cols)
    pitch = size + gap

    sheet = np.full((rows * pitch, cols * pitch, 3), 40, dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r * pitch : r * pitch + size, c * pitch : c * pitch + size] = img

    out = Image.fromarray(sheet)
    return out.resize((out.width * ZOOM, out.height * ZOOM), Image.NEAREST)


def beside_target(img, target):
    goal = ((np.clip(target.cpu().numpy(), -1, 1) + 1) / 2 * 255).astype(np.uint8)
    both = np.concatenate([img, goal], axis=1)
    out = Image.fromarray(both)
    return out.resize((out.width * ZOOM * 2, out.height * ZOOM * 2), Image.NEAREST)


def load_brain(run, batch):
    brain = train.Brain(
        batch, cell_dim=train.CELL_DIM, width=128, seed=0,
        device=train.DEVICE, shared=True,
    )
    ck = torch.load(run / "best.pt", weights_only=False)
    with torch.no_grad():
        for p, saved in zip(brain.parameters(), ck["w"]):
            p.copy_(saved)
    return brain, len(ck["losses"]) - 1, min(ck["losses"])


def rollout(brain, target, run, name, render):
    world = World(
        brain, train.SIZE, seed=0, noise=0.0, init=train.WHITE,
        landmark=train.LANDMARK, horizon=train.STEPS, ants=train.ANTS,
        scatter=train.SCATTER,
    )
    frames = []
    with torch.no_grad():
        for i in range(train.STEPS):
            world.step()
            if i % EVERY == 0:
                frames.append(render(world, target))

    frames += [frames[-1]] * 20  # hold on the finished picture

    # one palette for the whole gif, or unchanged pixels shimmer frame to frame
    palette = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    frames = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]

    frames[0].save(
        run / name, save_all=True, append_images=frames[1:], duration=67, loop=0
    )
    print(f"{name}: {len(frames)} frames")


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p for p in Path("out").iterdir() if (p / "best.pt").exists()
    )[-1]
    target = train.target_image().to(train.DEVICE)

    brain, epoch, best = load_brain(run, batch=1)
    print(f"{run}: best brain from epoch {epoch}, mse {best:.4f}")
    rollout(brain, target, run, "paint.gif",
            lambda w, t: beside_target(pictures(w)[0], t))

    brain, _, _ = load_brain(run, batch=BATCH)
    rollout(brain, target, run, "paint_many.gif", lambda w, t: tile(pictures(w)))


if __name__ == "__main__":
    main()
