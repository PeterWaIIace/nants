# 🐜 Neural Ants — NAnts

A colony of ants with a tiny brain each. None of them has ever seen a gecko.
Together they draw one anyway.

**[Play with it in the browser →](https://ichko.github.io/nants/)**

Every ant walks a wrapping 48×48 grid. At each step it reads its own 3×3 patch,
how far it has walked from where it woke up, and a clock. From that it decides
what to add to the cell beneath it and whether to turn left, carry straight on,
or turn right — then steps forward. No memory between steps, no map, and no
signalling between ants.

## Setup

```bash
git clone https://github.com/ichko/nants
cd nants
uv sync
```

## Grow your own

```bash
uv run python starter/my_nants.py
```

Open [`starter/my_nants.py`](starter/my_nants.py), change the emoji at the top,
and run it again to grow something else. It writes a folder under `out/` with
the picture as it stands, the loss curve, and the best brain so far — all
watchable while it runs. Stop with ctrl-c whenever you like.

See [`starter/README.md`](starter/README.md) for what each file is, how to carry
a run on, and what to change if you have no gpu.

### Logic gates

```bash
uv run python starter/my_gates.py
```

Train the ants to compute a Boolean gate (AND, OR, XOR, …) instead of
painting a picture. Inputs are pinned discs left of the landmark, the answer
belongs in a disc on the right. Open
[`starter/my_gates.py`](starter/my_gates.py), change the gate, and watch the
accuracy climb.

## The demonstrations

Each takes a run folder and writes a gif into it.

```bash
uv run nants-crowd out/<datetime>
```

| command | what it shows |
|---|---|
| `nants-crowd` | one to nine ants a square, with a bite taken out halfway |
| `nants-showcase` | the eight poses: four turns, each plain or mirrored |
| `nants-colony` | many colonies on one field, and how close they can sit |
| `nants-regen` | cutting the head or the tail off, and the repair |
| `nants-twist` | the sobel kernels turned 45, 90 and 135 degrees |
| `nants-partial` | waking up inside a fragment somebody else painted |
| `nants-scatter` | dropped anywhere on the field, facing anywhere |
| `nants-weights` | export a brain as `weights.js` for the browser |

## How it is trained

Two gradients from the same rollout: **what it writes**, by ordinary backprop
truncated every 100 steps, and **where it walks**, by REINFORCE, since a turn is
a discrete choice. 1024 colonies run in parallel, one optimizer step each,
gradients clipped, and the best brain kept whenever the loss improves.

## What it can and cannot do

It builds the picture at any crowd size from one ant to nine, in any of eight
poses, starting anywhere on the field, and it repairs damage it was never
trained on. Colonies side by side ignore each other and both come out clean.

It cannot join a structure it did not start. An ant waking up inside somebody
else's half-drawn gecko builds its own around itself instead, because every ant
paints relative to its own dead reckoning and nothing it reads can move that
anchor.

Three ablations, measured rather than argued:

- **the clock is decorative** — dropping its eight inputs costs nothing
  (0.0007 against 0.0006), and its trained weights were already far smaller than
  every other input's
- **the seed is decorative too** — a blank field with no landmark grows the same
  geckos
- **perception is not** — turning the sobel kernels degrades the picture
  smoothly with the angle

## Layout

```
src/nants/
  ant.py        what an ant senses and how it moves
  brain.py      the network: 93 in, 128 hidden, 19 out
  world.py      the field, and the ants that share it
  train.py      the training loop
  gates.py      ncpu-style logic gates as a task
  shows/        the demonstrations above
starter/        one file to grow your own
```

The browser demo lives on the [`website`](../../tree/website) branch, published
by `.github/workflows/pages.yml`.
