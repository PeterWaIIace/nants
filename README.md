# nants

A colony of memoryless ants that paints a picture.

Every ant walks a wrapping 48×48 grid. At each step it reads its own 3×3
neighbourhood through fixed sobel kernels, how far it has walked from where it
woke up, and a clock. From that alone it decides what to add to the cell
beneath it and whether to turn left, carry straight on, or turn right — then it
steps forward, like a turmite. There is no memory between steps, no map, and no
signalling between ants.

Trained against a gecko, a colony of eight reaches mse 0.0006.

**[Try it in the browser](https://ichko.github.io/nants/)** — drop ants, erase
pieces, and watch the field's sixteen channels underneath.

## Running it

```bash
uv sync
uv run nants-train                    # start a run, writes to out/<datetime>/
uv run nants-train out/<run-folder>   # carry one on
```

A run folder holds `train.log`, a loss curve, the picture as it stands,
`best.pt` for the best brain so far, and checkpoints every 200 epochs.

Then any of the demonstrations, each taking a run folder:

```bash
uv run nants-crowd out/<run-folder>
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
| `nants-paint` | a single run, beside the target |
| `nants-sample` | untrained brains, for comparison |
| `nants-weights` | export a brain as `weights.js` for the browser |

## How it is trained

Two gradients, from the same rollout:

- **what it writes** — ordinary backprop through the added value, truncated
  every 100 steps
- **where it walks** — REINFORCE, since the turn is a discrete sample, scored
  against how the other rollouts in the batch did

1024 rollouts run in parallel on the gpu, one optimizer step each, gradients
clipped, and the best brain kept whenever the loss improves.

## What it can and cannot do

It builds the picture at any crowd size from one ant to nine, in any of the
eight poses, starting anywhere on the field, and it repairs damage it was never
trained on. Colonies placed side by side ignore each other and both come out
clean.

It cannot join a structure it did not start: an ant that wakes up inside
somebody else's half-painted gecko builds its own around itself instead. Nor
can it colonise a landmark left for it. Both follow from the same fact — every
ant paints relative to its own dead reckoning, and nothing it reads from the
field can move that anchor.

Three ablations worth knowing, all measured rather than argued:

- **the clock is decorative.** Dropping its eight inputs entirely costs
  nothing (0.0007 against 0.0006); the trained weights on those rows were
  already three to nine times smaller than every other input's.
- **the landmark is decorative too.** Starting on a blank field with no seed
  at all produces the same geckos.
- **perception is not.** Turning the sobel kernels degrades the picture
  smoothly with the angle, so the local reading really is doing work.

## Layout

```
src/nants/
  ant.py        what an ant senses and how it moves
  brain.py      the small network, one hidden layer of 128
  world.py      the field, and the ants that share it
  train.py      the training loop
  warmstart.py  carry a brain over to a run with different inputs
  shows/        the demonstrations in the table above
```

The browser demo lives on the [`website`](../../tree/website) branch and is
published from there by `.github/workflows/pages.yml`.
