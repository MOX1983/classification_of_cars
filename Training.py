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



train_dataset = MyDataset('.\\Datasets\\train')
test_dataset = MyDataset('.\\Datasets\\valid')

train_data, valid_data = random_split(train_dataset, [0.8, 0.2])

trein_loder = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loder = DataLoader(valid_data, batch_size=16, shuffle=False)
test_loder = DataLoader(test_dataset, batch_size=16, shuffle=False)


model = MyModel(3, 2)

loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# i = torch.rand([16, 3, 128, 128], dtype=torch.float32) # это проверка модели
#
# out = model(i)
# print(out.shape)

EPOCHS = 15
train_loss = []
train_acc = []
val_loss = []
val_acc = []


for epoch in range(EPOCHS):

    model.train()
    runn_train_loss = []
    true_answer = 0
    train_loop = tqdm(trein_loder, leave=False)
    for x, targets in train_loop:

        pred = model(x)
        loss = loss_model(pred, targets)

        opt.zero_grad()
        loss.backward()

        opt.step()

        runn_train_loss.append(loss.item())
        res_train_loss = sum(runn_train_loss)/len(runn_train_loss)

        true_answer += (pred.argmax(dim=1) == targets).sum().item()

        train_loop.set_description(f"epoch [{epoch+1}/{EPOCHS}, train loss= {res_train_loss:.4f}]")

    runn_traing_acc = true_answer / len(train_data)

    train_loss.append(res_train_loss)
    train_acc.append(runn_traing_acc)


    model.eval()
    with  torch.no_grad():
        runn_val_loss = []
        true_answer = 0
        for x, targets in valid_loder:

            pred = model(x)
            loss = loss_model(pred, targets)

            runn_val_loss.append(loss.item())
            res_val_loss = sum(runn_train_loss) / len(runn_train_loss)

            true_answer += (pred.argmax(dim=1) == targets).sum().item()


        runn_val_loss = true_answer / len(train_data)

        val_loss.append(runn_val_loss)
        val_acc.append(res_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, train loss= {res_train_loss:.4f},"
          f" train acc={runn_traing_acc:.4f},"
          f" val loss={runn_val_loss:.4f},"
          f" val acc={res_val_loss}")

st = model.state_dict()
torch.save(st, 'model.pt')
