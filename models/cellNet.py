"""
Definition of the CellNet structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 12, 5
        )  # 32 - 5 = 27 -> 27 / 1 (default) -> 27 + 1 = 28 --> (12, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)  # (12, 14, 14)
        self.conv2 = nn.Conv2d(
            12, 24, 5
        )  # 14 - 5 = 9 -> 9 / 1 -> 9 + 1 = 10 --> (24, 10, 10)

        # after that, another pool layer is applied to the conv2 layer
        # getting (24, 5, 5)

        # apply fully connected layer, also named as Flatten (24 * 5 * 5)
        # generate 128 output features
        self.fc1 = nn.Linear(24 * 5 * 5, 128)

        # last layer is the Classifier(). Expected outputs are only four: " ", "G", "O", "X"
        self.fc2 = nn.Linear(128, 4)

        # to reduce overfitting (first iterations had a val_acc = 0.99), Dropout
        # layers are implemented, one layer after a conv layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # in order to be able to generate non-linear relations an activation function
        # ReLU is applied. It is important to know that is really useful to apply an
        # activation function after doing linear operations, such as convolutions
        # or fully connections

        # dropout should go after pooling, as we activate (or not) neurons that resume
        # data. It wouldn't have any sense to do it before pooling

        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
