import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


def plot_losses(train_losses, val_losses, title: str = "Cox") -> None:

    train_losses = torch.stack(train_losses) / train_losses[0]
    val_losses = torch.stack(val_losses) / val_losses[0]

    plt.plot(train_losses, label="training")
    plt.plot(val_losses, label="validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Normalized loss")
    plt.title(title)
    plt.yscale("log")
    plt.show()
    # plt.clf()


class Custom_dataset(Dataset):
    """ "Custom dataset for the GSBG2 brain cancer dataset"""

    # defining values in the constructor
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # Getting data size/length
    def __len__(self):
        return len(self.df)

    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # Targets
        event = torch.tensor(sample["cens"]).bool()
        time = torch.tensor(sample["time"]).float()
        # Predictors
        x = torch.tensor(sample.drop(["cens", "time"]).values).float()
        return x, (event, time)
