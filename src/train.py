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
    model = LinearClassifier(input_size=768)
    return model


def combine_embeddings(x1, x2):
    return torch.pow(torch.sub(x1, x2), 2)


def train_classifier(
    encoder, classifier, criterion, optimizer, train_loader, valid_loader
):
    # train_counter = []
    # train_loss_history = []
    # train_iteration_number = 0
    # valid_counter = []
    # valid_loss_history = []
    # valid_iteration_number = 0
    valid_loss_min = np.Inf  # set initial "min" to infinity

    train_class_correct = list(0 for i in range(2))
    train_class_total = list(0 for i in range(2))
    valid_class_correct = list(0 for i in range(2))
    valid_class_total = list(0 for i in range(2))

    encoder.to(device)
    classifier.to(device)

    for epoch in tqdm(range(Config.number_of_epochs)):
        train_loss = 0.0
        valid_loss = 0.0

        classifier.train()
        for i, data in enumerate(train_loader):
            row, img1, img2, label = data
            row, img1, img2, label = (
                row.to(device),
                img1.to(device),
                img2.to(device),
                label.to(device),
            )

            optimizer.zero_grad()

            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)

            start1.record()
            # TODO: Confirm this with xy
            # Using the CLS token embedding
            x1 = encoder(img1).last_hidden_state[:, 0, :]
            x2 = encoder(img2).last_hidden_state[:, 0, :]
            x_combined = combine_embeddings(x1, x2)
            end1.record()
            torch.cuda.synchronize()
            print(f"Encoder Time {start1.elapsed_time(end1)} ms")

            start2 = torch.cuda.Event(enable_timing=True)
            end2 = torch.cuda.Event(enable_timing=True)

            start2.record()
            output = classifier(x_combined)
            _, pred = torch.max(output, 1)
            # pred = (output >= 0.5).float()
            end2.record()
            torch.cuda.synchronize()
            print(f"Forward Time {start2.elapsed_time(end2)} ms")

            start3 = torch.cuda.Event(enable_timing=True)
            end3 = torch.cuda.Event(enable_timing=True)

            start3.record()
            loss = criterion(output, label)
            loss.backward()  # Investigate why this takes way too long.
            end3.record()
            torch.cuda.synchronize()
            print(f"Backward Time {start3.elapsed_time(end3)} ms")

            start4 = torch.cuda.Event(enable_timing=True)
            end4 = torch.cuda.Event(enable_timing=True)

            start4.record()
            optimizer.step()
            end4.record()
            torch.cuda.synchronize()
            print(f"Optimizer Step Time {start4.elapsed_time(end4)} ms")

            correct = pred.eq(label.view_as(pred))
            for j in range(len(label)):
                target = label[j].data
                train_class_correct[target] += correct[j].item()
                train_class_total[target] += 1

            train_loss += loss.item()

            if i % (Config.batch_size - 1) == 0:
                print(
                    "Epoch number {}\n Current loss {}\n".format(epoch + 1, loss.item())
                )
                # train_iteration_number += 30
                # train_counter.append(train_iteration_number)
                # train_loss_history.append(loss.item())

                for i in range(2):
                    if train_class_total[i] > 0:
                        print(
                            "Training Accuracy of %5s: %2d%% (%2d/%2d)"
                            % (
                                str(i),
                                100 * train_class_correct[i] / train_class_total[i],
                                np.sum(train_class_correct[i]),
                                np.sum(train_class_total[i]),
                            )
                        )

                print(
                    "\nTraining Accuracy (Overall): %2d%% (%2d/%2d)\n"
                    % (
                        100.0 * np.sum(train_class_correct) / np.sum(train_class_total),
                        np.sum(train_class_correct),
                        np.sum(train_class_total),
                    )
                )

        writer.add_scalar("Loss/train", train_loss, epoch)  # OPTIONAL

        classifier.eval()
        for i, data in enumerate(valid_loader):
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
            # pred = (output >= 0.5).float()

            loss = criterion(output, label)

            correct = pred.eq(label.view_as(pred))
            for j in range(len(label)):
                target = label[j].data
                valid_class_correct[target] += correct[j].item()
                valid_class_total[target] += 1

            valid_loss += loss.item()

            if i % (Config.batch_size - 1) == 0:
                print(
                    "Epoch number {}\n Current loss {}\n".format(epoch + 1, loss.item())
                )
                # valid_iteration_number += 30
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

        writer.add_scalar("Loss/valid", valid_loss, epoch)  # OPTIONAL

    writer.close()  # OPTIONAL


if __name__ == "__main__":
    encoder = load_encoder()
    classifier = load_classifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=Config.learning_rate)

    train_loader, valid_loader = create_train_val_data_loader(
        Config.train_image_dir,
        Config.train_relationship_file,
        Config.train_test_split_ratio,
    )

    writer = SummaryWriter()  # OPTIONAL

    train_classifier(
        encoder=encoder,
        classifier=classifier,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    writer.close()  # OPTIONAL
