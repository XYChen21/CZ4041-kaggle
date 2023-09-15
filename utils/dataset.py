import os
import random

from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset


class FamilyDataset(Dataset):
    """Family Dataset."""

    def __init__(self, relations, data_dir, transform=None):
        """
        Args:
            relations (pandas.DataFrame): Data frame with the image directories and labels.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.relations = relations
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.relations)

    def __getpair__(self, idx) -> Tuple[str, str]:
        pair = (
            os.path.join(
                self.data_dir, self.relations.iloc[idx, 0], self.relations.iloc[idx, 1]
            ),
            os.path.join(
                self.data_dir, self.relations.iloc[idx, 2], self.relations.iloc[idx, 3]
            ),
        )
        return pair

    def __getlabel__(self, idx) -> int:
        return self.relations.iloc[idx, 4]

    def __getitem__(self, idx) -> Tuple[int, Image.Image, Image.Image, int]:
        pair = self.__getpair__(idx)
        label = self.__getlabel__(idx)

        first = random.choice(os.listdir(pair[0]))
        second = random.choice(os.listdir(pair[1]))

        img1 = Image.open(pair[0] + "/" + first)
        img2 = Image.open(pair[1] + "/" + second)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return idx, img1, img2, label


class FamilyTestDataset(Dataset):
    """Family Dataset."""

    def __init__(self, relations, data_dir, transform=None):
        """
        Args:
            relations (string): Data frame with the image paths.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.relations = relations
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.relations)

    def __getpair__(self, idx) -> Tuple[str, str]:
        pair = (
            os.path.join(self.data_dir, self.relations.iloc[idx, 0].split("-")[0]),
            os.path.join(self.data_dir + self.relations.iloc[idx, 0].split("-")[1]),
        )
        return pair

    def __getlabel__(self, idx) -> int:
        return self.relations.iloc[idx, 1]

    def __getitem__(self, idx) -> Tuple[int, Image.Image, Image.Image]:
        pair = self.__getpair__(idx)

        img1 = Image.open(pair[0])
        img2 = Image.open(pair[1])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return idx, img1, img2
