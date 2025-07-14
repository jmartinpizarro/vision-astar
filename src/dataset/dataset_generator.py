"""
Synthetic grid dataset generator script.
"""

import random

from SyntheticGridGenerator import SyntheticGridGenerator
from config.dataset_generator_config import DATASET_GENERATION_ALGORITHMS


def main():
    # the idea is to generate as much as images as possible in the 10x10 grid using pygame for rendering and saving
    # them in the corresponding folder. The algorithm will be based on iterating until needed for saving as many
    # matrix as possible. Then, pygame will render them and saved them as images

    grids = generate_dataset()

    for grid in grids:
        transform_matrix_into_image(grid)
        export_as_image(grid)

    return 0


def generate_dataset(max_iters: int = 10000) -> set[list[list[int]]]:
    """
    Generates the grid dataset as pure numerical matrix
    :param max_iters:
    :return: a set with grids randomly generated
    """
    # set for not having duplicated grids in our dataset
    grids: set[list[list[int]]] = set()
    generator = SyntheticGridGenerator()

    for i in range(0, max_iters, 1):
        # populate the parameters for the generator
        width = random.randint(3, 10)
        height = random.randint(3, 10)
        prob = random.randint(0, 2)
        algorithm = DATASET_GENERATION_ALGORITHMS[prob]
        generator.width = width
        generator.height = height
        generator.algorithm = algorithm

        generator.generate_grid()

        # we don't want repeated grids in our set!
        if generator.grid not in grids:
            grids.add(generator.grid)

        generator.clear()

    return grids


def transform_matrix_into_image(matrix: list[list[int]]) -> int:
    return 0


def export_as_image(matrix: list[list[int]]) -> int:
    return 0


if __name__ == "__main__":
    main()
