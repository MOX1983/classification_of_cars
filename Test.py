from PIL import Image

import torch
import torchvision.transforms as transforms

from Main import MyDataset, MyModel

def run(path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    model = MyModel(3, 2)
    model.load_state_dict(torch.load('model.pt'))

    img = Image.open(path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_t)
        predicted_class = output.argmax(dim=1).item()

    clss = ["car", "truck"]
    # print(f"Predicted class: {clss[predicted_class]}")
    # plt.imshow(img)
    # plt.show()
    return "Predicted class: " + clss[predicted_class]


# path = '.\\Datasets\\valid\\car\\05392.jpeg'
# run(path)


# path = '.\\Datasets\\valid\\Truck\\'
# num = 'Datacluster Truck (135).jpg'


# test = {pathT : ['Datacluster Truck (75).jpg', 'Datacluster Truck (94).jpg', 'Datacluster Truck (31).jpg', 'Datacluster Truck (42).jpg'],
#         pathC : ['05138.jpeg', '05143.jpeg', '06784.jpeg', '06912.jpeg']}
# for key, val in test.items():
#     for i in val:
#         img = Image.open(os.path.join(key, i)).convert('RGB')
#         img_t = transform(img).unsqueeze(0)
#
#         model.eval()
#         with torch.no_grad():
#             output = model(img_t)
#             predicted_class = output.argmax(dim=1).item()
#
#         clss = ["car", "truck"]
#         print(f"Predicted class: {clss[predicted_class]}")
#         plt.imshow(img)
#         plt.show()