import pandas as pd
import torch
import yfinance
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


TICKER = "AAPL"


class Net(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True)

        self.linear1 = nn.Linear(hidden_size, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.lstm(x)
        x = F.sigmoid(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


def get_stock_prices():
    result = yfinance.Ticker(TICKER)
    data = result.history()
    return data


def preprocess(df: pd.DataFrame):
    result = {
        "input": [],
        "label": []
    }

    return df


if __name__ == "__main__":
    data = get_stock_prices()

    data = data.drop(["Dividends", "Stock Splits"], axis=1)

    model = Net(input_size=10, hidden_size=200)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train_dataloader = None

    model.train()

    for epoch in range(100):

        epoch_loss = 0

        for X, y in train_dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_function(output, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"the average loss of this {epoch} epoch is: ")

    model.test()
    test_df = pd.DataFrame()

    correct = 0
    with torch.no_grad():

        for index, row in test_df.iterrows():
            y = torch.tensor(row["label"].values)
            x = torch.tensor(row.drop(["label"], axis=1))

            output = model(x)

            if output == y:
                correct += 1

    print(f"the correct predictions are: {correct} / {test_df.__len__()}")
    print(f"the percentage is: " + str((correct / test_df.__len__()) * 100))