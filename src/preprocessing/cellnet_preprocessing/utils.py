"Utils for the CellNet preprocessing methodology"

import os
import shutil
import logging
from typing import Dict, List

from PIL import Image, ImageFilter

rotation_angles = [-45, -30, -15, 15, 30, 45]

def generate_output_folders(output_path: str) -> int:
    """
    Generates the output folders required for the preprocessing
    script to run correctly

    @param output_path: str -> name of the folder where the raw
    data is going to be preprocessed

    @returns 0 if all went correctly. FileNotFoundError otherwise
    """

    try:
        if os.path.exists(output_path) and os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.makedirs(f"{output_path}/images")
    except FileNotFoundError:
        logging.error(
            f"[preprocess_dataset]::An error has ocurred when creating the output path in {output_path}. Ensure that the path exists.\n"
        )

    return 0


def process_image(
    output: Dict[List[str], List[str]],
    fileName: str,
    size: int,
    matrix_str: str,
    input_path: str,
    output_path: str,
) -> int:
    """
    Process an entire image depending on its size,
    cropping it.

    @param output: dictionary with image's metadata
    @param fileName: str -> name of the input image
    @param size: int -> size of the square-grid
    @param matrix_str: str -> a simpler way of representing the matrix in a string; used for generating the labels
    @param input_path: str -> input path of the raw data folder
    @param output_path: str -> output path

    @returns the 0 if everything went correct, -1 otherwise
    """

    # each image has a 300x300 size, it is possible to know the
    # size of each cell just doing dividing
    # total image size / size of matrix
    with Image.open(f"{input_path}/images/{fileName}") as im:
        width, height = im.size
        cell_size = width / size

        # now it is possible to go through the entire image, iterating
        # from left to right, upper to lower part of it until all cells
        # has been processed iterate through the total number of cells

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
            
            # remove .jpeg extension
            name, ext = os.path.splitext(fileName)
            fileName = name

            sub_img.save(f"{output_path}/images/{fileName}_{c}.jpeg")
            
            output["fileName"].append(f"{fileName}_{c}.jpeg")
            output["label"].append(label)
            
            # after doing studies on the dataset, it is clearly seen that
            # the dataset is unbalanced. Thus, for the images with labels
            # that are not " ", more actions would be apply such as 
            # blur and rotation
            
            if label != " ":
                for angle in rotation_angles:
                    
                    # rotate
                    rotated = sub_img.rotate(angle=angle)
                    
                    rotated.save(f"{output_path}/images/{fileName}_{c}_{angle}.jpeg")
                    output["fileName"].append(f"{fileName}_{c}_{angle}.jpeg")
                    output["label"].append(label)
                    
                    # add blur
                    blured = rotated.filter(filter=ImageFilter.GaussianBlur(radius=0.5))
                    
                    blured.save(f"{output_path}/images/{fileName}_{c}_{angle}_BLUR.jpeg")
                    output["fileName"].append(f"{fileName}_{c}_{angle}_BLUR.jpeg")
                    output["label"].append(label)

            c += 1

    return 0
