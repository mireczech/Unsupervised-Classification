
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
        return self.backbone(x).flatten(start_dim=1)


class ClusteringModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(ClusteringModel, self).__init__()

        self.backbone = backbone
        self.cluster_head = nn.ModuleList([nn.Linear(feature_dim, num_classes)])
            
    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out
