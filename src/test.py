import sys

sys.path.append("d:/Coding/CZ4041/CZ4041-kaggle")  # Set root

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from src.config import Config
from utils.dataloader import create_test_dataloader

# Importing models
from models.classifier.linear import LinearClassifier
from transformers import ViTFeatureExtractor, ViTModel

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
def load_classifier(path_to_model_weights: str):
    model = LinearClassifier(input_size=768)
    model.load_state_dict(torch.load(path_to_model_weights))
    return model


def combine_embeddings(x1, x2):
    return torch.pow(torch.sub(x1, x2), 2)


def create_submission(path_to_template: str, path_to_save: str, predictions):
    template = pd.read_csv(path_to_template)

    # Remember to save as floats as metric is AUC
    for row, pred in predictions.items():
        template.loc[row, "is_related"] = pred.astype(float)

    template.to_csv(path_or_buf=path_to_save, index=False)
    return


def test_classifier(encoder, classifier, test_loader):
    predictions = {}

    encoder.to(device)
    classifier.to(device)

    classifier.eval()
    for i, data in tqdm(enumerate(test_loader)):
        row, img1, img2 = data
        row, img1, img2 = row.to(device), img1.to(device), img2.to(device)

        x1 = encoder(img1).last_hidden_state[:, 0, :]
        x2 = encoder(img2).last_hidden_state[:, 0, :]
        x_combined = combine_embeddings(x1, x2)

        output = classifier(x_combined)
        _, pred = torch.max(output, 1)

        for i in range(len(row)):
            predictions[row[i].item] = pred[i].item()

    return predictions


if __name__ == "__main__":
    path_to_model_weights = "./checkpoints/Linear_Classifier_1"
    path_to_template = "./data/submissions/sample_submission.csv"
    path_to_save = "./data/submissions/Submission_1"

    encoder = load_encoder()
    classifier = load_classifier(path_to_model_weights)

    test_loader = create_test_dataloader(
        Config.test_image_dir,
        Config.test_relationship_file,
    )

    predictions = test_classifier(
        encoder=encoder, classifier=classifier, test_loader=test_loader
    )

    create_submission(
        path_to_template=path_to_template,
        path_to_save=path_to_save,
        predictions=predictions,
    )
