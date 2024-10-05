
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd


class ConvulotionModel(nn.Module):

    def __init__(self):
        super(ConvulotionModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1)
        )
        # self.layer2 = nn.Conv1d()
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def build_matrix(input_tensor):
    results = torch.tensor([])
    for i in range(len(input_tensor)):
        input_matrix = input_tensor[i].reshape([28, 28])
        # input_matrix = input_matrix.unsqueeze(0)
        # input_matrix = input_matrix.unsqueeze(0)
        results = torch.cat((results, input_matrix))
    return results


if __name__ == '__main__':


    device = torch.device("cpu")
    print("Using CPU")
    epochs = 10

    data = pd.read_csv(r'digit-recognizer (1)/train.csv', sep=',')

    label_tensor = torch.tensor(data["label"].values).to(device)

    input_tensor = torch.tensor(data.drop("label", axis=1).values).type(torch.float32).reshape(-1, 1, 28, 28).to(device)

    model = ConvulotionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()
        print("Epoch: ", epoch, "Loss: ", loss.item())

    # input_matrix = build_matrix(input_tensor)
    # print(input_matrix.shape)
    # X_train = input_tensor.reshape(-1, 1, 28, 28).float()


