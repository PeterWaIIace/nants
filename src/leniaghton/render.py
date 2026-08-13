"""Turn a batched run into a tiled gif."""

import numpy as np
from matplotlib import colormaps
from PIL import Image

VIRIDIS = colormaps["viridis"]


def frames_of(world):
    """Every world in the batch as RGB float arrays, ants marked red."""
    cells = world.field.cells[:, :, :, 0].cpu().numpy()
    imgs = VIRIDIS(np.clip((cells + 1) / 2, 0, 1))[..., :3]

    pos = world.ant.pos.cpu().numpy()
    imgs[np.arange(len(imgs)), pos[:, 1], pos[:, 0]] = [1.0, 0.2, 0.2]
    return imgs


class Recorder:
    def __init__(self, world, cols=4, zoom=4, gap=1):
        self.world = world
        self.cols = cols
        self.zoom = zoom
        self.gap = gap
        self.frames = []

    def snap(self):
        tiles = frames_of(self.world)
        n, size = len(tiles), tiles.shape[1]
        rows = -(-n // self.cols)
        pitch = size + self.gap

        sheet = np.full((rows * pitch, self.cols * pitch, 3), 0.15)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, self.cols)
            sheet[r * pitch : r * pitch + size, c * pitch : c * pitch + size] = tile

        img = Image.fromarray((sheet * 255).astype(np.uint8))
        self.frames.append(
            img.resize((img.width * self.zoom, img.height * self.zoom), Image.NEAREST)
        )

    def record(self, steps, every=10):
        for i in range(steps):
            self.world.step()
            if i % every == 0:
                self.snap()

    def save(self, path, ms=67):  # ~15 fps
        self.frames[0].save(
            path, save_all=True, append_images=self.frames[1:], duration=ms, loop=0
        )
