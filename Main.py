import json
import os
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.optim as optim


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = dir_list
                self.cls_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = path_dir.split('/')[-1]

            for name_file in file_list:
                fail_path = os.path.join(path_dir, name_file)
                self.data_list.append((fail_path, self.cls_idx[cls]))

            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset
    def __getitem__(self, item):
        fail_path, target = self.data_list[item]
        sample = np.array(Image.open(fail_path))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


train_dataset = MyDataset('/archive/train')

print(train_dataset.len_dataset)


