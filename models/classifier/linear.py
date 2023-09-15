import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_size:int):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.fc1(x)
        return x
