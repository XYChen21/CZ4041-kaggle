import sys

sys.path.append("d:/Coding/CZ4041/CZ4041-kaggle")  # Set root

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

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
def load_classifier(path_to_weights: str):
    model = LinearClassifier(input_size=768)
    model.load_state_dict(torch.load(path_to_weights))
    return model


# TODO: Complete
def create_submission(path_to_template: str, path_to_save: str, predictions):
    template = pd.read_csv(path_to_template)
    pass


def test_classifier(encoder, classifier, test_loader):
    predictions = []

    classifier.eval()
    for i, data in tqdm(enumerate(test_loader, 0)):
        row, img1, img2 = data
        row, img1, img2 = row.to(device), img1.to(device), img2.to(device)

        output1 = vggnet(img0, img1)
        output = net(output1)
        # output= net(img0,img1)
        _, pred = torch.max(output, 1)

        # count = 0
        # for item in row:
        #     sample_submission.loc[item, "is_related"] = pred[count].item()
        #     count += 1

    return predictions


if __name__ == "__main__":
    encoder = load_encoder().to(device)
    classifier = load_classifier().to(device)

    test_loader = create_test_dataloader(
        Config.test_image_dir,
        Config.test_relationship_file,
    )

    predictions = test_classifier(
        encoder=encoder, classifier=classifier, test_loader=test_loader
    )

    create_submission()
