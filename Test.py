import os
from PIL import Image

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from Main import MyDataset, MyModel

pathT = '.\\Datasets\\valid\\truck\\'
# num = '10167'
pathC = '.\\Datasets\\valid\\car\\'
# num = '05118'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = MyModel(3, 2)
model.load_state_dict(torch.load('model.pt'))

test = {pathT : ['05117', '06513', '07602', '10105'], pathC : ['05138', '05143', '06784', '06912']}
for key, val in test.items():
    for i in val:
        img = Image.open(os.path.join(key, f'{i}.jpeg')).convert('RGB')
        img_t = transform(img).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(img_t)
            predicted_class = output.argmax(dim=1).item()

        clss = ["car", "truck"]
        print(f"Predicted class: {clss[predicted_class]}")
        plt.imshow(img)
        plt.show()