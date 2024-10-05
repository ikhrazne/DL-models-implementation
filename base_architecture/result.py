import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_image():
    pass


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=10,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=5)

        self.linear1 = nn.Linear(320, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = F.dropout(x)

        x = x.view(-1, 320)

        x = torch.relu(self.linear1(x))
        x = F.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return torch.softmax(x, dim=1)


def read_data(task="train", batch_size=25):
    df = pd.read_csv(rf"../digit-recognizer (1)/train.csv")

    train_df, test_df = train_test_split(df, test_size=0.2)
    # print(train_df)

    if task == "test":
        train_df = test_df

    labels = torch.tensor(train_df["label"].values)

    df = train_df.drop(["label"], axis=1)

    train_images = torch.reshape(torch.tensor(df.values), (-1, 1, 28, 28)).to(torch.float).to(device)

    # train_images = train_images.unsqueeze(0)

    dataloader = DataLoader(TensorDataset(train_images, labels),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1
                            )

    return dataloader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = read_data(batch_size=100)

    model = Net().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    loss = nn.CrossEntropyLoss()

    EPOCH = 20

    model.train()
    for epoch in range(EPOCH):
        epoch_loss = 0
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss_result = loss(output, y)
            epoch_loss += loss_result.item()
            loss_result.backward()
            optimizer.step()

        print(f"the average loss of the {epoch} is: " + str(epoch_loss / 25))

    test_dataloader = read_data("test")

    model.eval()

    correct = 0

    with torch.no_grad():
        for test_X, test_y in test_dataloader:
            test_X.to(device)
            test_y.to(device)
            output = model(test_X)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred).sum().item())

    print("the number of correct number is: " + str(correct) + " from ")
