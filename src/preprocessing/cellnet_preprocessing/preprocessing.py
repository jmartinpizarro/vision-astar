"Preprocessing for the CellNet"

import logging

import pandas as pd

import src.preprocessing.cellnet_preprocessing.utils as utils

pd.set_option("display.max_columns", None)


def preprocess_dataset(input_path: str, output_path: str, neural_network: str) -> int:
    """
    Just for the CellNet. At this point, obtaining cells inputs from the images
    Return 0 if everything went correct, -1 otherwise
    """

    utils.generate_output_folders(output_path)

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

        utils.process_image(output, fileName, size, matrix_str, input_path, output_path)

    df = pd.DataFrame(output)
    df.to_csv(f"{output_path}/preprocessed_dataset.csv", index=False)
    return 0
