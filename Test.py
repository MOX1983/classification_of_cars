from PIL import Image

import torch
import torchvision.transforms as transforms

from traing.Main import MyModel

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
