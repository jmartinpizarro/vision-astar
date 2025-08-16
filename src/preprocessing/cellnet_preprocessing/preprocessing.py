"Preprocessing for the CellNet"

import os
import shutil
import logging

import pandas as pd
from PIL import Image

pd.set_option("display.max_columns", None)


def preprocess_dataset(input_path: str, output_path: str, neural_network: str) -> int:
    """
    Just for the CellNet. At this point, obtaining cells inputs from the images
    Return 0 if everything went correct, -1 otherwise
    """

    try:
        if os.path.exists(output_path) and os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.makedirs(f"{output_path}/images")
    except FileNotFoundError:
        logging.error(
            f"[preprocess_dataset]::An error has ocurred when creating the output path in {output_path}. Ensure that the path exists.\n"
        )
        return -1

    # Data Processing Algorithm
    # AS images are 300x300, it is not necessary to take into account
    # tons of params. In fact, it is possible to make all the necessary
    # cuts in the image using just the size of the image and the
    # width (number of colums) of the matrix

    # load dataset
    try:
        images_metadata = pd.read_csv(f"{input_path}/dataset.csv")
    except FileNotFoundError:
        logging.error(
            f"[preprocess_dataset]::CSV file not found in {input_path}/dataset.csv"
        )
        return -1

    output = {"fileName": [], "label": []}

    # 1. for each image, save the fileName and the size
    for index, row in images_metadata.iterrows():
        fileName, size, matrix_str = row["filename"], row["size"], row["matrix_inline"]

        # each image has a 300x300 size, it is possible to know the
        # size of each cell just doing dividing
        # total image size / size of matrix
        with Image.open(f"{input_path}/images/{fileName}") as im:
            width, height = im.size

            cell_size = width / size

            # now it is possible to go through the entire image, iterating
            # from left to right, upper to lower part of it until all cells
            # has been processed

            # iterate through the total number of cells
            matrix_str = list(matrix_str)
            matrix_str = [ch for ch in matrix_str if ch != ","]
            c = 1
            while c <= size * size:
                # awesome way of calculating row and col of a matrix!
                row = (c - 1) // size
                col = (c - 1) % size

                # define bounding box of the new image
                left_origin = col * cell_size
                upper_origin = row * cell_size
                right_offset = left_origin + cell_size
                lower_offset = upper_origin + cell_size
                box = (left_origin, upper_origin, right_offset, lower_offset)

                label = matrix_str[c - 1]

                # crop and save
                sub_img = im.crop(box)

                # transform it into a 32x32 image size
                sub_img = sub_img.resize((32, 32))

                sub_img.save(f"{output_path}/images/{fileName}_{c}.jpeg")

                output["fileName"].append(f"{fileName}_{c}.jpeg")
                output["label"].append(label)

                c += 1

    df = pd.DataFrame(output)
    df.to_csv(f"{output_path}/preprocessed_dataset.csv")
    return 0
