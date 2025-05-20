import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import transforms



class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

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
                if os.path.isfile(fail_path):
                    try:
                        with Image.open(fail_path) as img:
                            img.verify()
                        self.data_list.append((fail_path, self.cls_idx[cls]))
                    except:
                        print(f"[Предупреждение] Файл битый и не будет загружен: {fail_path}")
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, item):
        fail_path, target = self.data_list[item]
        try:
            image = Image.open(fail_path).convert("RGBA").convert("RGB")
        except Exception as e:
            print(f"[Ошибка] Не удалось открыть: {fail_path}\n{e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        if self.transform is not None:
            image = self.transform(image)

        return image, target

class MyModel(nn.Module):
    def __init__(self, in_channels, out):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, (3, 3)),   # 128 -> 126
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),  #126 -> 124
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.layer = nn.Sequential(
            nn.Linear(64 * 124 * 124, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out)
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        out = self.layer(x)
        return out

