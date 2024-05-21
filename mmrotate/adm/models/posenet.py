import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PoseNetEncoder(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(PoseNetEncoder, self).__init__()
        self.base_model = models.resnet34(pretrained=True)
        self.dropout_rate = dropout_rate

        feat_in = self.base_model.fc.in_features
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        nn.init.kaiming_normal_(self.fc_last.weight)
        if self.fc_last.bias is not None:
            nn.init.constant_(self.fc_last.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        if self.dropout_rate:
            x = F.dropout(x, p=self.dropout_rate)

        return x


class RegressionHead(nn.Module):
    def __init__(self, n_outputs, n_inputs=2048):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(n_inputs, n_outputs, bias=True)
        nn.init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class OrientationHead(RegressionHead):
    def __init__(self, n_outputs=4):
        super(OrientationHead, self).__init__(n_outputs)

    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=1)


class PositionHead(RegressionHead):
    def __init__(self, n_outputs=3):
        super(PositionHead, self).__init__(n_outputs)
