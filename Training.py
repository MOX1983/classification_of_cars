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
from tqdm import tqdm

from Main import MyDataset, MyModel



train_dataset = MyDataset('.\\archive\\train')
test_dataset = MyDataset('.\\archive\\valid')

train_data, valid_data = random_split(train_dataset, [0.8, 0.2])

trein_loder = DataLoader(train_data, batch_size=16, shuffle=True)
vajid_loder = DataLoader(valid_data, batch_size=16, shuffle=False)
test_loder = DataLoader(test_dataset, batch_size=16, shuffle=False)


model = MyModel(3, 2)

loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)



