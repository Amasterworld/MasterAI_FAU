import numpy as np
import matplotlib.pyplot as plt


class Checker:

    # constructor, default size = 8 and two colors, they are constrast (like black and white)

    def __init__(self, resolution, tile_size):
        # if resolution % (2*tile_size) != 0:
        #    raise ValueError("Resolution must be divisible by 2*tile_size.")
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution), dtype=np.uint8)

    def draw(self):
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError("Resolution must be evenly divisible by 2 * tile_size")

        tiles_per_dim = self.resolution // self.tile_size
        row_indices, col_indices = np.indices((tiles_per_dim, tiles_per_dim))
        is_odd_tile = (row_indices + col_indices) % 2 == 1
        test = np.repeat(is_odd_tile, self.tile_size, axis=0)

        pattern = np.repeat(np.repeat(is_odd_tile, self.tile_size, axis=0), self.tile_size, axis=1)

        self.output = pattern.astype(int)

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Circle:

    def __init__(self, resolution, radius, position) -> None:
        if len(position) != 2:
            raise ValueError("length of tuple position must be 2")

        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self) -> None:
        x = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, x)
        # calculate the distance from the center of the given circle to all points
        dist_squared = (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2
        # if the distance from the center of circle to points <= radius, assign them to 1, otherwise 0
        mask = dist_squared <= self.radius ** 2

        self.output = mask.astype(int)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution, 3))

        x = np.linspace(0.0, 1.0, self.resolution)
        # print(x)
        y = np.linspace(0.0, 1.0, self.resolution)
        xx, yy = np.meshgrid(x, y)

        self.output[:, :, 0] = xx
        self.output[:, :, 1] = yy
        self.output[:, :, 2] = 1 - xx
        # print(self.output)

        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.axis('off')
        plt.show()
