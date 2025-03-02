# data_loader.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as dt

class DataLoader:
    def __init__(self, data_folder, batch_size=32, image_size=64):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.image_size = image_size

    def get_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = datasets.ImageFolder(root=self.data_folder, transform=transform)
        data_loader = dt(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return data_loader
