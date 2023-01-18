import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class ImageDataset(torch.utils.data.Dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    def __init__(self, config):
        self.data = datasets.MNIST(config.dataroot, train=True, download=True, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    @classmethod
    def input_shape(cls):
        return [1, 28, 28]

    @classmethod
    def normalize(cls, x):
        return cls.transform(x)

    @classmethod
    def denormalize(cls, x):
        return cls.reverse_transform(x.cpu())
