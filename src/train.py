import sys

sys.path.append("d:/Coding/CZ4041/CZ4041-kaggle")  # Set root

import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from src.config import Config
from utils.dataloader import create_train_val_data_loader

# Importing models
from models.classifier.linear import LinearClassifier
from transformers import ViTFeatureExtractor, ViTModel

# OPTIONAL: For visualizing training metrics
from torch.utils.tensorboard import SummaryWriter

# REMOVE!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}!")


# TODO: Remove hardcoding, also transforms already performed
def load_encoder():
    model_name_or_path = "google/vit-base-patch16-224-in21k"
    # feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
    model = ViTModel.from_pretrained(model_name_or_path)
    return model


# TODO: Is this function needed?
def load_classifier():
    return LinearClassifier(input_size=768)


# TODO: Is this function needed?
def combine_embeddings(embedding1, embedding2):
    return (embedding1 - embedding2) ** 2


def train_classifier(
    encoder, classifier, criterion, optimizer, train_loader, valid_loader
):
    # train_counter = []
    # train_loss_history = []
    train_iteration_number = 0

    # valid_counter = []
    # valid_loss_history = []
    valid_iteration_number = 0

    valid_loss_min = np.Inf  # set initial "min" to infinity

    train_class_correct = list(0 for i in range(2))
    train_class_total = list(0 for i in range(2))

    valid_class_correct = list(0 for i in range(2))
    valid_class_total = list(0 for i in range(2))

    for epoch in tqdm(range(Config.number_of_epochs)):
        valid_loss = 0.0

        classifier.train()
        for i, data in tqdm(enumerate(train_loader, 0)):
            row, img1, img2, label = data
            row, img1, img2, label = (
                row.to(device),
                img1.to(device),
                img2.to(device),
                label.to(device),
            )

            optimizer.zero_grad()

            # TODO: Confirm this with xy
            # Using the CLS token embedding
            x1 = encoder(img1).last_hidden_state[:, 0, :]
            x2 = encoder(img2).last_hidden_state[:, 0, :]
            x_combined = combine_embeddings(x1, x2)

            output = classifier(x_combined)
            _, pred = torch.max(output, 1)
            #pred = (output >= 0.5).float()

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()

            correct = pred.eq(label.view_as(pred))
            for j in range(len(label)):
                target = label[j].data
                train_class_correct[target] += correct[j].item()
                train_class_total[target] += 1

            # OPTIONAL: For visualizing training metrics
            writer.add_scalar("Loss/train", loss, epoch)

            if i % 30 == 0:
                print(
                    "Epoch number {}\n Current loss {}\n".format(epoch + 1, loss.item())
                )
                train_iteration_number += 30
                # train_counter.append(train_iteration_number)
                # train_loss_history.append(loss.item())

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

        classifier.eval()
        for i, data in tqdm(enumerate(valid_loader, 0)):
            row, img1, img2, label = data
            row, img1, img2, label = (
                row.to(device),
                img1.to(device),
                img2.to(device),
                label.to(device),
            )

            # TODO: Confirm this with xy
            # Using the CLS token embedding
            x1 = encoder(img1).last_hidden_state[:, 0, :]
            x2 = encoder(img2).last_hidden_state[:, 0, :]
            x_combined = combine_embeddings(x1, x2)

            output = classifier(x_combined)
            _, pred = torch.max(output, 1)
            #pred = (output >= 0.5).float()

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
                # valid_counter.append(valid_iteration_number)
                # valid_loss_history.append(loss.item())

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
            torch.save(classifier.state_dict(), "./checkpoints/Linear_Classifier_1.pt")
            valid_loss_min = valid_loss
    
    # OPTIONAL: For visualizing training metrics
    writer.close()


if __name__ == "__main__":
    encoder = load_encoder().to(device)
    classifier = load_classifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        classifier.parameters(), lr=Config.learning_rate, momentum=Config.momentum
    )

    train_loader, valid_loader = create_train_val_data_loader(
        Config.train_image_dir,
        Config.train_relationship_file,
        Config.train_test_split_ratio,
    )
    
    
    # OPTIONAL: For visualizing training metrics
    writer = SummaryWriter()

    train_classifier(
        encoder=encoder,
        classifier=classifier,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    writer.close()