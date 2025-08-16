#!usr/bin/env python3
"""
Script for generating a non-deterministic synthetic dataset based on grids
"""

import os
import time
import argparse
import logging

import src.synthetic_dataset.utils as utils


def main():
    logging.info("Init of Dataset generation...\n")

    start = time.time()

    # generate img
    for grid_size in range(3, 11, 1):
        for i in range(1, utils.NUM_IMAGES + 1):
            grid = utils.generate_random_grid(grid_size)
            clean_img = utils.draw_grid(grid, grid_size)
            noisy_img = utils.apply_noise(clean_img)
            filename, matrix = utils.save_grid_and_matrix(
                utils.global_counter, grid, noisy_img
            )
            matrix_inline = ""
            for row in grid:
                for elem in row:
                    matrix_inline += elem
                matrix_inline += ","
            utils.csv_lines.append(
                f'{filename},"{matrix}",{grid_size},"{matrix_inline}"'
            )
            utils.global_counter += 1

    # save csv
    with open(os.path.join(utils.OUTPUT_DIR, "dataset.csv"), "w") as f:
        f.write("\n".join(utils.csv_lines))

    end = time.time()

    logging.info(
        "Dataset generation has finished\n"
        f"Total time elapsed {end - start:.2f}\n"
        f"Data can be found in the {utils.OUTPUT_DIR} folder\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision Astar Synthetic Dataset Generator Script"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
