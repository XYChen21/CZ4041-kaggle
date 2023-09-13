import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from src.config import Config
from utils.dataset import FamilyDataset, FamilyTestDataset
from torch.utils.data import DataLoader


def create_train_val_data_loader(
    train_image_dir: str, train_relationship_file: str, split_ratio: float = 0.8
):
    df = pd.read_csv(train_relationship_file)
    train_df, valid_df = np.split(df, [int(split_ratio * len(df))])

    train_transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = FamilyDataset(
        relations=train_df, data_dir=train_image_dir, transform=train_transform
    )

    valid_dataset = FamilyDataset(
        relations=valid_df, data_dir=train_image_dir, transform=valid_transform
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=Config.num_workers,
        batch_size=Config.batch_size,
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=True,
        num_workers=Config.num_workers,
        batch_size=Config.batch_size,
    )

    return train_loader, valid_loader


def create_test_dataloader(test_image_dir: str, test_relationship_file: str):
    df = pd.read_csv(test_relationship_file)

    transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    test_dataset = FamilyTestDataset(
        relations=df, root_dir=test_image_dir, transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        num_workers=Config.num_workers,
        batch_size=Config.batch_size,
    )

    return test_loader