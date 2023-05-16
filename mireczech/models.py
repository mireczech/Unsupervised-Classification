
import torch
import torch.nn as nn
import numpy as np
import torchvision

class SimCLRModel(nn.Module):
    def __init__(self):
        super(SimCLRModel, self).__init__()

        resnet = torchvision.models.resnet34()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])


    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)

        return h
