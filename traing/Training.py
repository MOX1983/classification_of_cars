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

def train():
    train_dataset = MyDataset('../Datasets/train')

    train_data, valid_data = random_split(train_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)


    model = MyModel(3, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 30
    best_val_loss = float('inf')

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct = 0.0, 0

        train_loop = tqdm(train_loader, desc=f"[{epoch+1}/{EPOCHS}] Epoch", leave=False)
        for images, labels in train_loop:
            images, labels = images, labels

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = correct / len(train_data)

        model.eval()
        val_running_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images, labels
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_running_loss / len(valid_loader)
        avg_val_acc = val_correct / len(valid_data)

        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print(f" Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Acc = {avg_train_acc:.4f} | "
              f"Val Loss = {avg_val_loss:.4f}, Acc = {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '../best_model2.pt')
            print("Save best_model2")


if __name__ == '__main__':
    train()
