# Grow your own

```bash
uv sync
uv run python starter/my_nants.py
```

Open `my_nants.py` and change the emoji at the top to grow something else.
Everything else is already wired up.

While it runs, the run folder under `out/` fills with:

| file | what it is |
|---|---|
| `train.png` | the picture as it stands, beside the target |
| `loss.png` | the error over time, on a log scale |
| `best.pt` | the best brain so far, saved whenever it improves |
| `train.log` | the same numbers as text |

Stop it with ctrl-c whenever you like; the best brain is already on disk. To
carry on later, point the trainer back at the same folder:

```bash
uv run nants-train out/<datetime>
```

Then render whichever demonstration you want from that run:

```bash
uv run nants-crowd out/<datetime>       # one to nine ants, with damage halfway
uv run nants-showcase out/<datetime>    # the eight poses
uv run nants-regen out/<datetime>       # cut the head off, watch it grow back
```

## If it is slow

Training runs 1024 colonies in parallel, which wants a gpu. Without one,
uncomment the smaller settings in `my_nants.py` — a 32×32 field with 256
colonies still grows a recognisable shape, in far less time.
