from PIL import Image

import torch
import torchvision.transforms as transforms

from Main import MyDataset, MyModel

def run(path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    model = MyModel(3, 5)
    model.load_state_dict(torch.load('best_model.pt'))

    img = Image.open(path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_t)
        predicted_class = output.argmax(dim=1).item()

    clss = ['bike', 'bus', 'car', 'motorbike', 'truck']


    return "Predicted class: " + clss[predicted_class]


for i in [".\\Datasets\\valid\\bike\\4290.jpg",
          ".\\Datasets\\valid\\bus\\bus-32_jpg.rf.e497b8fe649df811691e65d05a28ff0d.jpg",
          ".\\Datasets\\valid\\car\\03444.jpeg",
          ".\\Datasets\\valid\\motorbike\\46_sym_elite_50_vmse_1568719361305_11139.jpg",
          ".\\Datasets\\valid\\truck\\Image_013471_jpg.rf.4fee3b4ad830a3e9bb36d8237d148cdd.jpg"]:
    print(run(i))