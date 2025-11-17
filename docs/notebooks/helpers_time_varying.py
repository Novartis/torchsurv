import torch
import matplotlib.pyplot as plt


class GroupedDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.groups = [g for _, g in df.groupby("id")]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        sample = self.groups[idx]
        # Targets
        event = torch.unique_consecutive(
            torch.tensor(sample["event_at_time"].values).bool()
        )
        time = torch.unique_consecutive(torch.tensor(sample["time"].values).float())
        id = torch.tensor(sample["id"].values).long()
        start = torch.tensor(sample["start"].values).float()

        # Predictors
        x = torch.tensor(
            sample.drop(
                ["event", "event_at_time", "time", "start", "stop", "id"], axis=1
            ).values
        ).float()
        return x, (event, time), id, start


def collate_fn(batch):

    xs, ets, ids, starts = zip(*batch)  # unzip

    # xs, stops get cats (because the dim differ by id)
    x = torch.cat(xs, dim=0)
    ids = torch.cat(ids, dim=0)
    starts = torch.cat(starts, dim=0)

    # event, time get default stacking
    events, times = zip(*ets)
    events = torch.stack(events, dim=0)
    times = torch.stack(times, dim=0)

    return x, (events, times), ids, starts


def expand_log_hz(id, start, time, log_hz_short):

    ids = torch.unique_consecutive(id)
    n = len(ids)

    log_hz = torch.zeros((n, len(time)), device=log_hz_short.device)

    for idx, _id in enumerate(ids):
        id_idx = id == _id

        # Which row within this id to use
        time_idx = torch.searchsorted(start[id_idx], time, right=True) - 1

        # Slice out the correct hazard value
        value = log_hz_short[id_idx][time_idx].squeeze()

        # Repeat the value across the entire row (your original behavior)
        log_hz[idx, :] = value

    return log_hz


def expand_log_hz_survival(id, start, time, log_hz_long):

    ids = torch.unique_consecutive(id)
    n = len(ids)

    log_hz = torch.zeros((n, len(time)), device=log_hz_long.device)

    for idx, _id in enumerate(ids):
        id_idx = id == _id

        # Which row within this id to use
        time_idx = torch.searchsorted(start[id_idx], time, right=True) - 1

        # Columns indices
        cols = torch.arange(log_hz.shape[1])

        # Slice out the correct hazard value
        value = log_hz_long[id_idx][time_idx.squeeze(), cols].squeeze()

        # Repeat the value across the entire row (your original behavior)
        log_hz[idx, :] = value

    return log_hz


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
