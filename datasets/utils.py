"""
File for declaring utils used in the trainer of neural networks
"""

from typing import Tuple
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import config.cellnet as cellNet_config  # to ensure reproducibility

CELLNET_TRANSFORMER_IMAGE = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

LABEL_ENCODING_STR_INT = {"O": 0, "G": 1, "X": 2, " ": 3, "F": 4}


def generate_data_loaders(
    dataset: Dataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Given a dataset, it generates the train and test split DataLoader

    @param dataset: a generic dataset filled with the data

    @returns a tuple (train_split, test_split, val_split) filled with the dataloaders
    """

    # need the train and test split
    # for the moment, we will use the standard 0.8-0.2 train-test split
    # could be possible to use sklearn.model_selection.train_test_split, but not for today
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_remaining = n_total - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val

    train_split, val_split, test_split = random_split(
        dataset, [n_train, n_val, n_test], generator=cellNet_config.seed
    )

    train_loader = DataLoader(train_split, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_split, batch_size=32, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_split, batch_size=32, shuffle=False, num_workers=2)

    return (train_loader, val_loader, test_loader)
