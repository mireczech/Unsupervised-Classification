
import os
import csv
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tf
import torchvision
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ImageDatasetWithFolderStructure(torchvision.datasets.ImageFolder):
    def __init__(
            self, 
            root,
            split,
            transform,
            size,
    ):
        super(ImageDatasetWithFolderStructure, self).__init__(root=os.path.join(root, split), transform=None)

        assert split in ('train', 'val', 'test')

        self.transform = transform
        self.split = split
        self.resize = tf.Resize(size)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out


class MatekDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            root='mireczech/data/matek',
            split='train', 
            transform=None,
    ):
        ImageDatasetWithFolderStructure.__init__(self, root, split, transform, size=128)

class IsicDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            root='mireczech/data/isic',
            split='train', 
            transform=None,
    ):
        ImageDatasetWithFolderStructure.__init__(self, root, split, transform, size=128)

class RetinopathyDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            root='mireczech/data/retinopathy',
            split='train', 
            transform=None,
    ):
        ImageDatasetWithFolderStructure.__init__(self, root, split, transform, size=128)

class JurkatDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            root='mireczech/data/jurkat',
            split='train', 
            transform=None,
    ):
        ImageDatasetWithFolderStructure.__init__(self, root, split, transform, size=64)

class Cifar10Dataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            root='mireczech/data/cifar10',
            split='train', 
            transform=None,
    ):
        ImageDatasetWithFolderStructure.__init__(self, root, split, transform, size=32)
