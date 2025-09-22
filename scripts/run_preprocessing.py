#!/usr/bin/env python3
"""
Script for preprocessing the data. Takes into account if the user wants
to process data for the CellNet or other networks (as a parameter)
"""

import time
import argparse
import logging
from pathlib import Path

from src.preprocessing.cellnet_preprocessing.preprocessing import preprocess_dataset


def main(input_path, output_path, neural_network):
    logging.info(
        f"Init of preprocessing...\n"
        f"You are going to transform data for the {neural_network} network.\n"
        f"Data will be saved in: {output_path}"
    )

    t_start = time.time()
    preprocess_dataset(input_path, output_path, neural_network)
    t_end = time.time()

    logging.info(
        f"Preprocessing has finished...\nTime elapsed: {t_end - t_start:.2f} seconds\n"
        f"Data has been saved in: {output_path}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Astar Data preprocessing")
    parser.add_argument("--input", required=True, help="Raw data route")
    parser.add_argument("--output", required=True, help="Outout for preprocessed data")
    parser.add_argument(
        "--network",
        required=True,
        help="Depending on the type of network, preprocessing will be different",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main(Path(args.input), Path(args.output), Path(args.network))
