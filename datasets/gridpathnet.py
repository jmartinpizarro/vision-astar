"""
File for the Dataset class definition of the GridPathNet dataset.
Abstracted so the same class can be used when training a CellNet or
other versions or Vision Astar
"""

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GridPathNet(Dataset):
    def __init__(
        self,
        preprocessed_route: str,
        label_mode: str,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    ):
        """
        Constructor of the dataset

        @param csv_route: str -> route to the preprocessed data
        @param label_mode: str -> depending on the value of it, the __getitem() method
        will return the label in a string, int or matrix (for generalising the class)
        @param transform -> transform that will be applied to the data. By default,
        it resizes, transforms into a tensor and then normalises in the range [-1, 1]
        """
        self.preprocessed_route = preprocessed_route
        self.label_mode = label_mode
        self.data = pd.read_csv(f"{preprocessed_route}/preprocessed_dataset.csv")
        self.transform = transform

        if self.label_mode == "int":
            self.label_mapper = {"O": 0, "G": 1, "X": 2, " ": 3, "F": 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Returns the image with the corresponding transform applied

        @param idx: int -> index of the DataFrame that we want to read
        @returns a transformed image
        """
        image_name = self.data.iloc[idx, 0]  # name of the file
        image_route = f"{self.preprocessed_route}/images/{image_name}"
        image = Image.open(image_route).convert("RGB")

        label = self.data.iloc[idx, 1]  # label of the image

        if self.label_mode == "int":
            label = self.label_mapper[label]

        # TODO: matrix format when i have decided
        # how to approach the other preprocessing

        if self.transform:
            image = self.transform(image)

        return image, label
