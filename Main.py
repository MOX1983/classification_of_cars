import json
import os
from PIL import Image
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.optim import Adam
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

            cls = path_dir.split('\\')[-1]

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


train_dataset = MyDataset('.\\archive\\train')
test_dataset = MyDataset('.\\archive\\valid')

# img, one_hot_pos = train_dataset[5116]
# cls = train_dataset.classes[one_hot_pos]
# print(f"класс {cls}")
# plt.imshow(img)
# plt.show()

train_data, valid_data = random_split(train_dataset, [0.8, 0.2])
print(len(train_data))
print(len(valid_data))

trein_loder = DataLoader(train_data, batch_size=16, shuffle=True)
vajid_loder = DataLoader(valid_data, batch_size=16, shuffle=False)
test_loder = DataLoader(test_dataset, batch_size=16, shuffle=False)