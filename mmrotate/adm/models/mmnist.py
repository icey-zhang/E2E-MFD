import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetEncoder(nn.Module):
    def __init__(self):
        super(LeNetEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 10, 5, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(320, 50)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.relu(x)


class CLSTaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 10))

    def forward(self, x):

        assert (x != 0).sum() != 0

        return F.log_softmax(self.fc(x), dim=1)
