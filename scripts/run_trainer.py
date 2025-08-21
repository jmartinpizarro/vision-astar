#!/usr/bin/env python3
"""
Script for training the neural network (all of them, depending on the input).
"""

import argparse
import logging
from pathlib import Path

import matplotlib
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from models.cellNet import CellNet
from datasets.gridpathnet import GridPathNet
import datasets.utils as dataset_utils

matplotlib.use("qtagg")

PREPROCESSED_ROUTE = "data/cellnet_preprocessed_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(network, config):
    logging.info(
        "[trainer]::Training of the network is going to start:\n\n"
        "\tConfig:\n\n"
        f"\tModel will be trained using {str(device).upper()}\n"
        f"\t--network: {network}\n"
        f"\t--config: {config}\n\n"
    )

    # arguments in main are a Path() object!
    network = str(network)
    config = str(config)

    net = None

    if str(network) == "cellnet":
        net = CellNet()

    else:
        raise ValueError("The other CNNs are not implemented yet")

    net = net.to(device)

    dataset = GridPathNet(PREPROCESSED_ROUTE, "int")

    # data split declaration
    train_split, val_split, test_split = dataset_utils.generate_data_loaders(dataset)

    criterion = nn.CrossEntropyLoss()
    # TODO: wanna research more about dynamic learning rate
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    logging.info("[run_trainer]::Training has started\n")

    # training the network is based on two phases: training and
    # evaluation.

    for epoch in range(5):  # loop through the train dataset n times
        # train
        net.train()
        running_loss = 0.0

        for index, (inputs, labels) in enumerate(
            tqdm(train_split, desc=f"Train E{epoch + 1}")
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            # zero the param gradients
            optimizer.zero_grad()

            # forward + backward + optimise
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (index + 1) % 500 == 0:
                avg = running_loss / 500
                logging.info(f"[{epoch + 1}, {index + 1}] loss: {avg:.4f}")
                running_loss = 0.0

        # eval
        net.eval()

        best_val_loss = float("inf")
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # for inference
            for inputs, labels in tqdm(val_split, desc=f"Val E{epoch + 1}"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / total
        val_acc = correct / total
        logging.info(
            f"E{epoch + 1} Validation loss: {val_loss:.4f}, acc: {val_acc:.4f}\n"
        )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), "./checkpoints/cellnet_best.pth")
            logging.info(f"Saved best model (val_loss={val_loss:.4f})\n\n")

    logging.info("[run_trainer]::Training has finished\n\n")

    # for the sabe of checking the output, try the test_dataset
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_split:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    logging.info(f"[run_trainer]::Precision with test_split is: {accuracy}\n")

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Astar Vision Script")
    parser.add_argument("--network", required=True, help="Neural Network To Train")
    parser.add_argument(
        "--config",
        required=True,
        help="Route to .yaml file with the hyperparameters configuration",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    main(Path(args.network), Path(args.config))
