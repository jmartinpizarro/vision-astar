#!/usr/bin/env python3
"""
Script that runs the main algorithm to solve
grid problems
"""

import os
import time
import logging
import argparse
from pathlib import Path
from typing import List

import torch
import PIL
from PIL import Image

from models.cellNet import CellNet
from datasets.utils import CELLNET_TRANSFORMER_IMAGE, LABEL_ENCODING_STR_INT

from src.algorithm.ida_astar import ida_astar
from src.algorithm.State import State

inv_label_mapping = {v: k for k, v in LABEL_ENCODING_STR_INT.items()}


def main(file, network, size):
    # arguments are alaways a path object
    file = str(file)
    network = str(network)
    size = int(str(size))

    logging.info(
        "\n\n"
        "-- Vision Astar Grid Solver --\n\n"
        "\t Config:\n"
        f"\t\t --file: {file}\n"
        f"\t\t --network: {network}\n"
    )

    # initilization of the problem
    # specific case for cellNet
    if network == "cellNet":
        if 0 >= size >= 10:
            logging.error("Size can only be an integer in range [0, 10]")

        model = load_model(network)

    print_model_state_dict(model)

    # start the solver
    start = time.time()

    matrix = cellNet_predict_cells(model, file, size)
    
    origin, goal = None, None
    
    for row in range(size):
        for col in range(size): # square grid!
            cell = matrix[row][col]
            if cell == "O":
                origin = (row, col)
            if cell == "G":
                goal = (row, col)
            

    init = State(2, matrix, origin[0], origin[1])
    goal = State(2, matrix, goal[0], goal[1])

    path = ida_astar(init, goal)

    end = time.time()

    if path == -1:
        logging.error("Seems that no possible solution could be calculated\n\n")
        return 0

    print("\n\nA solution was calculated for the problem:\n")

    print(" -> ".join(f"({s.x},{s.y})" for s in path[0]))

    print(
        f"A total of {len(path) - 1} movements were need to achieved the final solution\n"
    )

    return 0


def load_model(network: str):
    "Loads the model and returns an object to do inference"

    match network:
        case "cellNet":
            model = CellNet()
            model.load_state_dict(
                torch.load(f"checkpoints/{network}_best.pth", weights_only=True)
            )

    return model


def print_model_state_dict(model):
    "Prints the entire state dict of the model loaded"

    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print("\t", param_tensor, " -> ", model.state_dict()[param_tensor].size())


def cellNet_predict_cells(model, image: PIL.Image, size: int) -> List[str]:
    "Given an image route and the size of the grid, it returns an array of the contents of the cell"

    model.eval()
    matrix = []

    with Image.open(f"{image}") as im:
        width, height = im.size
        cell_size = width / size

        for row in range(size):
            row_cells = []
            for col in range(size):
                # Bounding box
                left = int(round(col * cell_size))
                upper = int(round(row * cell_size))
                right = int(round(left + cell_size))
                lower = int(round(upper + cell_size))
                box = (left, upper, right, lower)

                # crop
                sub_img = im.crop(box)
                # transformer resizes, normalises and to_tensor
                cell_input = CELLNET_TRANSFORMER_IMAGE(sub_img).unsqueeze(0)

                # inference
                with torch.no_grad():
                    logits = model(cell_input)
                    label_idx = logits.argmax(dim=1).item()
                    label = inv_label_mapping[label_idx]

                row_cells.append(label)
            matrix.append(row_cells)

    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Astar Solver Script")

    parser.add_argument(
        "--file", required=True, help="File of the image (.jpeg) to solve"
    )
    parser.add_argument(
        "--network",
        required=True,
        help="Network to solve the problem. If using CellNet, an optional parameter --size must be used. This parameter can only be: cellNet, gridNet, convHeadNet, vitGrid",
    )
    parser.add_argument(
        "--size",
        required=False,
        help="The size of the grid. Because of the limitations of the cellnet network, the size of grid must be given by the user",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    main(Path(args.file), Path(args.network), Path(args.size))
