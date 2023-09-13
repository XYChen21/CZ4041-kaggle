import sys
sys.path.append("d:/Coding/CZ4041/CZ4041-kaggle")   # Set root

import torch
import numpy as np
import torch.nn as nn

from torch import optim
from src.config import Config
from utils.dataloader import create_train_val_data_loader
from models.enc_both import VisionTransformer

# REMOVE!
CUDA = torch.cuda.is_available()
assert CUDA == True

TRAIN_IMAGE_DIR = "./data/train"
TRAIN_RELATIONSHIP_FILE = (
    "./data/train-relationships/train_relationships_processed.csv"
)

def load_weights():
    pass


def train(model, criterion, optimizer, train_loader, valid_loader):
    train_counter = []
    train_loss_history = []
    train_iteration_number = 0

    valid_counter = []
    valid_loss_history = []
    valid_iteration_number = 0

    valid_loss_min = np.Inf  # set initial "min" to infinity

    train_class_correct = list(0 for i in range(2))
    train_class_total = list(0 for i in range(2))

    valid_class_correct = list(0 for i in range(2))
    valid_class_total = list(0 for i in range(2))

    for epoch in range(0, Config.train_number_epochs):
        valid_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader, 0):
            row, img0, img1, label = data
            row, img0, img1, label = row.cuda(), img0.cuda(), img1.cuda(), label.cuda()

            optimizer.zero_grad()
            output = model(img0, img1)

            # TODO: Confirm this with xy
            # _, pred = torch.max(output, 1)
            pred = (output >= 0.5).float()

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()

            correct = pred.eq(label.view_as(pred))
            for j in range(len(label)):
                target = label[j].data
                train_class_correct[target] += correct[j].item()
                train_class_total[target] += 1

            if i % 30 == 0:
                print(
                    "Epoch number {}\n Current loss {}\n".format(epoch + 1, loss.item())
                )
                train_iteration_number += 30
                train_counter.append(train_iteration_number)
                train_loss_history.append(loss.item())

                for i in range(2):
                    if train_class_total[i] > 0:
                        print(
                            "\nTraining Accuracy of %5s: %2d%% (%2d/%2d)"
                            % (
                                str(i),
                                100 * train_class_correct[i] / train_class_total[i],
                                np.sum(train_class_correct[i]),
                                np.sum(train_class_total[i]),
                            )
                        )

                print(
                    "\nTraining Accuracy (Overall): %2d%% (%2d/%2d)"
                    % (
                        100.0 * np.sum(train_class_correct) / np.sum(train_class_total),
                        np.sum(train_class_correct),
                        np.sum(train_class_total),
                    )
                )

        model.eval()
        for i, data in enumerate(valid_loader, 0):
            row, img0, img1, label = data
            row, img0, img1, label = row.cuda(), img0.cuda(), img1.cuda(), label.cuda()

            output = model(img0, img1)

            # TODO: Confirm this with xy
            # _, pred = torch.max(output, 1)
            pred = (output >= 0.5).float()

            loss = criterion(output, label)

            correct = pred.eq(label.view_as(pred))
            for j in range(len(label)):
                target = label[j].data
                valid_class_correct[target] += correct[j].item()
                valid_class_total[target] += 1

            valid_loss += loss.item()

            if i % 30 == 0:
                print(
                    "Epoch number {}\n Current loss {}\n".format(epoch + 1, loss.item())
                )
                valid_iteration_number += 30
                valid_counter.append(valid_iteration_number)
                valid_loss_history.append(loss.item())

                for i in range(2):
                    if train_class_total[i] > 0:
                        print(
                            "\nValdiation Accuracy of %5s: %2d%% (%2d/%2d)"
                            % (
                                str(i),
                                100 * valid_class_correct[i] / valid_class_total[i],
                                np.sum(valid_class_correct[i]),
                                np.sum(valid_class_total[i]),
                            )
                        )

                print(
                    "\nValdiation Accuracy (Overall): %2d%% (%2d/%2d)"
                    % (
                        100.0 * np.sum(valid_class_correct) / np.sum(valid_class_total),
                        np.sum(valid_class_correct),
                        np.sum(valid_class_total),
                    )
                )

        if valid_loss <= valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, valid_loss
                )
            )
            torch.save(model.state_dict(), "/checkpoints/model.pt")
            valid_loss_min = valid_loss


if __name__ == "__main__":
    enc_both = VisionTransformer(224, 16, 768, 6, 8, 1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(enc_both.parameters(), lr=0.008, momentum=0.9)

    train_load, valid_load = create_train_val_data_loader(
        TRAIN_IMAGE_DIR, TRAIN_RELATIONSHIP_FILE
    )

    train(
        model=enc_both,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_load,
        valid_loader=valid_load,
    )
