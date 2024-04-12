from typing import Tuple

import lifelines
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Local module
from torchsurv.loss.momentum import Momentum


class LitSurvival(L.LightningModule):
    """Survival Model Fitter"""

    def __init__(self, backbone, loss, batch_size: int = 64, dataname: str = "lung"):
        super().__init__()
        self.backbone = backbone
        self.loss = loss
        self.batch_size = batch_size
        self.dataname = dataname

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # Can be anything
        x, y, t = batch
        params = self(x)
        loss = self.loss(params, y, t)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def setup(self, stage: str):
        self.dataset = SurvivalDataset(self.dataname)

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )


class LitSurvivalTwins(LitSurvival):
    """Survival Model Fitter"""

    def __init__(self, steps: int = 5, **kw):
        super(LitSurvivalTwins, self).__init__(**kw)
        self.momentum = Momentum(
            backbone=self.backbone,
            loss=self.loss,
            steps=steps,
            batchsize=self.batch_size,
            rate=0.999,
        )

    def forward(self, x, y, t):
        return self.momentum(x, y, t)

    def training_step(self, batch, batch_idx):
        # Can be anything
        x, y, t = batch
        loss = self(x, y, t)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


class SimpleLinearNNOneParameter(L.LightningModule):
    """Neural network with output = bias + weight * x"""

    def __init__(self, input_size: int):
        super(SimpleLinearNNOneParameter, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class SimpleLinearNNTwoParameters(L.LightningModule):
    """Neural network with output1 = bias_2 + weight_2 * x and output2 = bias_1 + 0 * x"""

    def __init__(self, input_size: int):
        super(SimpleLinearNNTwoParameters, self).__init__()
        self.linear1 = nn.Linear(input_size, 1)
        self.linear2 = nn.Linear(input_size, 1)
        self.freeze_linear2_weights()

    def forward(self, x: torch.Tensor):
        output1 = self.linear1(x)
        output2 = self.linear2(torch.zeros_like(x))  # Use zeros as input for output2
        return torch.hstack((output1, output2))

    def freeze_linear2_weights(self):
        # Set requires_grad to False for the weights parameters of linear2
        for param_name, param in self.linear2.named_parameters():
            if "weight" in param_name:
                param.requires_grad = False
            elif "bias" in param_name:
                param.requires_grad = True


class SurvivalDataset(Dataset):
    def __init__(self, name: str = "lung"):
        self.name = name
        self.df = (
            lifelines.datasets.load_lung()
            if "lung" in name
            else lifelines.datasets.load_gbsg2()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if "lung" in self.name:
            x = torch.tensor(row[["sex", "age"]].values, dtype=torch.float32)
            y = torch.tensor(row["status"], dtype=torch.float32)
            t = torch.tensor(row["time"], dtype=torch.float32)
        else:
            x = torch.tensor(row[["age", "tsize"]].tolist(), dtype=torch.float32)
            t = torch.tensor(row["time"], dtype=torch.float32)
            y = torch.tensor(row["cens"], dtype=torch.float32)

        return x, y, t


class SurvivalDataGenerator:
    def __init__(
        self,
        test_ties_time_event: bool = False,
        test_ties_time_censoring: bool = False,
        test_ties_time_event_censoring: bool = False,
        train_ties_time_event: bool = False,
        train_ties_time_censoring: bool = False,
        train_ties_time_event_censoring: bool = False,
        ties_score_events: bool = False,
        ties_score_censoring: bool = False,
        ties_score_event_censoring: bool = False,
        test_event_at_last_time: bool = False,
        test_no_censoring: bool = False,
        test_all_censored: bool = False,
        train_no_censoring: bool = False,
        train_all_censored: bool = False,
        test_max_time_gt_train_max_time: bool = False,
        test_max_time_in_new_time: bool = False,
    ):
        """Simulate survival data with specified characteristics.

        Args:
            test_ties_time_event (bool, optional):
                Whether there should be at least one tie in event times in the test set.
                Defaults to False.
            test_ties_time_censoring (bool, optional):
                Whether there should be at least one tie in censoring times in the test set.
                Defaults to False.
            test_ties_time_event_censoring (bool, optional):
                Whether there should be at least one tie between the event time and censoring time in the test set.
                Defaults to False.
            train_ties_time_event (bool, optional):
                Whether there should be at least one tie in event times in the train set.
                Defaults to False.
            train_ties_time_censoring (bool, optional):
                Whether there should be at least one tie in censoring times in the train set.
                Defaults to False.
            train_ties_time_event_censoring (bool, optional):
                Whether there should be at least one tie between the event time and censoring time in the train set.
                Defaults to False.
            ties_score_events (bool, optional):
                Whether there should be at least one tie in the risk score associated to patients with event.
                Defaults to False.
            ties_score_censoring (bool, optional):
                Whether there should be at least one tie in the risk score associated to patients with censoring.
                Defaults to False.
            ties_score_event_censoring (bool, optional):
                Whether there should be at least one tie in the risk score associated to a patient with event and a patient with censoring.
                Defaults to False.
            test_event_at_last_time (bool, optional):
                Whether the last time should be associated to an event in test set.
                Defaults to False.
            test_no_censoring (bool, optional):
                Whether there should be no patients censored in test set.
                Defaults to False.
            test_all_censored (bool, optional):
                Whether all patient should be censored in test set.
                Defaults to False.
            train_no_censoring (bool, optional):
                Whether there should be no patients censored in train set.
                Defaults to False.
            train_all_censored (bool, optional):
                Whether all patient should be censored in train set.
                Defaults to False.
            test_max_time_gt_train_max_time (bool, optional):
                Whether the maximum time in the test set should be greater than that in the train set.
                Defaults to False.
            test_max_time_in_new_time (bool, optional):
                Whether the maximum time in the test set should be included in the evaluation times.
                Defaults to False.
        """

        # Create internal attributes states
        self.test_ties_time_event = test_ties_time_event
        self.test_ties_time_censoring = test_ties_time_censoring
        self.test_ties_time_event_censoring = test_ties_time_event_censoring
        self.train_ties_time_event = train_ties_time_event
        self.train_ties_time_censoring = train_ties_time_censoring
        self.train_ties_time_event_censoring = train_ties_time_event_censoring
        self.test_event_at_last_time = test_event_at_last_time
        self.test_no_censoring = test_no_censoring
        self.test_all_censored = test_all_censored
        self.train_no_censoring = train_no_censoring
        self.train_all_censored = train_all_censored
        self.test_max_time_gt_train_max_time = test_max_time_gt_train_max_time
        self.test_max_time_in_new_time = test_max_time_in_new_time
        self.ties_score_events = ties_score_events
        self.ties_score_censoring = ties_score_censoring
        self.ties_score_event_censoring = ties_score_event_censoring

        # generate simulated survival data
        self._generate_input()

        # given simulated data, evaluate value of conditions.
        self._evaluate_conditions()

        # check that the conditions set by the flags are met.
        self._check_conditions()

    def get_input(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Returns simulated data as tensors.

        Returns (Tuple, torch.tensor):
            time-to-event or censoring in train set,
            event indicator in train test,
            time-to-event or censoring in test set,
            event indicator in test test,
            estimated risk score,
            evaluation time
        """
        return (
            self.train_time,
            self.train_event,
            self.test_time,
            self.test_event,
            self.estimate,
            self.new_time,
        )

    def get_input_array(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """Returns simulated data as np array.

        Returns (Tuple, np.array):
            Array containing the event indicator, and time-to-event or censoring of train set,
            Array containing the event indicator, and time-to-event or censoring of test set,
            Array containing estimated risk score,
            Array containing evaluation time.
        """

        # convert simulated input to arrays
        self._convert_to_arrays()

        return (
            self.y_train_array,
            self.y_test_array,
            self.estimate_array,
            self.new_time_array,
        )

    def _generate_input(self):

        # random maximum time in observational period
        tmax = torch.randint(5, 500, (1,)).item()

        # random number samples in train and number of samples in test
        n_train = torch.randint(20, 101, (1,)).item()
        n_test = torch.randint(20, 101, (1,)).item()

        # generate random train and test time-to-event or censoring and event indicator
        self._generate_data(tmax, n_train, n_test)

        # generate random estimate
        self._generate_estimate()

        # generate random evaluation time
        self._generate_new_time()

    def _generate_data(self, tmax: int, n_train: int, n_test: int):

        # time-to-event or censoring in train
        train_time = torch.randint(1, tmax + 1, (n_train,)).float()

        # event status (1: event, 0: censored) in train
        train_event = torch.randint(0, 1 + 1, (n_train,)).float()
        while train_event.sum() == 0:
            train_event = torch.randint(0, 1 + 1, (n_train,))

        # enforce conditions
        train_time, train_event = self._enforce_conditions_data(
            train_time, train_event, "train"
        )

        # order by time and create internal attributes
        index = torch.sort(train_time)[1]
        self.train_time = train_time[index]
        self.train_event = train_event[index].bool()

        # time-to-event or censoring in test
        test_time = torch.randint(1, tmax + 1, (n_test,)).float()

        # event status (1: event, 0: censored) in test
        test_event = torch.randint(0, 1 + 1, (len(test_time),)).float()

        # order by time
        index_test = torch.sort(test_time)[1]
        test_time = test_time[index_test]
        test_event = test_event[index_test]

        # enforce conditions
        test_time, test_event = self._enforce_conditions_data(
            test_time, test_event, "test"
        )

        # order by time and create internal attributes
        index = torch.sort(test_time)[1]
        self.test_time = test_time[index]
        self.test_event = test_event[index].bool()

    def _enforce_conditions_data(
        self, time: torch.tensor, event: torch.tensor, dataset_type: str
    ) -> Tuple[torch.tensor, torch.tensor]:

        # if test max time should be greater than train max time
        if dataset_type == "test":
            if self.test_max_time_gt_train_max_time:
                time[torch.where(event == 1.0)[0][0]] = self.train_time.max() + 1
            else:
                index_time = (time > self.train_time.min()) & (
                    time < self.train_time.max()
                )
                time = time[index_time]
                event = event[index_time]

        # if there should be ties in two event times
        if (dataset_type == "train" and self.train_ties_time_event) or (
            dataset_type == "test" and self.test_ties_time_event
        ):
            time[torch.where(event == 1.0)[0][0]] = time[
                torch.where(event == 1.0)[0][1]
            ]

        # if there should be ties in two censoring times
        if (dataset_type == "train" and self.train_ties_time_censoring) or (
            dataset_type == "test" and self.test_ties_time_censoring
        ):
            time[torch.where(event == 0.0)[0][0]] = time[
                torch.where(event == 0.0)[0][1]
            ]

        # if there should be a tie in an event time and a censoring time
        if (dataset_type == "train" and self.train_ties_time_event_censoring) or (
            dataset_type == "test" and self.test_ties_time_event_censoring
        ):
            time[torch.where(event == 1.0)[0][0]] = time[
                torch.where(event == 0.0)[0][0]
            ]

        # if there should be an event at the last time
        if dataset_type == "test" and self.test_event_at_last_time:
            event[-1] = 1

        # if there should be no censoring
        if (dataset_type == "train" and self.train_no_censoring) or (
            dataset_type == "test" and self.test_no_censoring
        ):
            event = event.fill_(1.0)

        # if all patients should be censored
        if (dataset_type == "train" and self.train_all_censored) or (
            dataset_type == "test" and self.test_all_censored
        ):
            event = event.fill_(0.0)

        return time, event

    def _generate_estimate(self):

        # random risk score for observations in test
        estimate = torch.randn(len(self.test_event))

        # enforce conditions risk score
        self.estimate = self._enforce_conditions_estimate(estimate)

    def _enforce_conditions_estimate(self, estimate: torch.tensor) -> torch.tensor:

        # if there should be ties in risk score associated to patients with event
        if self.ties_score_events:
            estimate[torch.where(self.test_event == 1.0)[0][0]] = estimate[
                torch.where(self.test_event == 1.0)[0][1]
            ]

        # if there should be ties in risk score associated to patients with censoring
        if self.ties_score_censoring:
            estimate[torch.where(self.test_event == 0.0)[0][0]] = estimate[
                torch.where(self.test_event == 0.0)[0][1]
            ]

        # if there should be ties in risk score associated to patients with event and with censoring
        if self.ties_score_event_censoring:
            estimate[torch.where(self.test_event == 1.0)[0][0]] = estimate[
                torch.where(self.test_event == 0.0)[0][0]
            ]

        return estimate

    def _generate_new_time(self):

        if torch.all(self.test_event == False):
            # if all patients are censored in test, no evaluation time
            new_time = torch.tensor([])

        else:
            # random number of evaluation time
            n = torch.randint(low=1, high=int(len(self.test_time) / 2), size=(1,))

            # generate random evaluation time within test event time
            new_time = torch.unique(
                torch.randint(
                    low=self.test_time[self.test_event == 1].min().long(),
                    high=self.test_time[self.test_event == 1].max().long(),
                    size=(n,),
                ).float()
            )

        # enforce conditions
        self.new_time = self._enforce_conditions_time(new_time)

    def _enforce_conditions_time(self, new_time: torch.tensor) -> torch.tensor:

        # if the test max time should be included in evaluation time
        if self.test_max_time_in_new_time:
            new_time = torch.cat(
                [new_time, torch.tensor([self.test_time.max()])], dim=0
            )

        return new_time

    def _evaluate_conditions(self):

        # are there ties in event times
        self.has_train_ties_time_event = self._has_ties(
            self.train_time[self.train_event == 1]
        )
        self.has_test_ties_time_event = self._has_ties(
            self.test_time[self.test_event == 1]
        )

        # are there ties in censoring times
        self.has_train_ties_time_censoring = self._has_ties(
            self.train_time[self.train_event == 0]
        )
        self.has_test_ties_time_censoring = self._has_ties(
            self.test_time[self.test_event == 0]
        )

        # are there ties in event and censoring times
        self.has_test_ties_time_event_censoring = self._has_ties(
            self.test_time[self.test_event == 1],
            self.test_time[self.test_event == 0],
        )
        self.has_train_ties_time_event_censoring = self._has_ties(
            self.train_time[self.train_event == 1],
            self.train_time[self.train_event == 0],
        )

        # is last time is event
        self.has_test_event_at_last_time = (self.test_event[-1] == 1.0).item()

        # is there no censoring
        self.has_train_no_censoring = torch.all(self.train_event).item()
        self.has_test_no_censoring = torch.all(self.test_event).item()

        # is all patients all censored
        self.has_train_all_censored = torch.all(self.train_event == False).item()
        self.has_test_all_censored = torch.all(self.test_event == False).item()

        # is test time greater than train time
        self.has_test_max_time_gt_train_max_time = (
            self.test_time.max() > self.train_time.max()
        ).item()

        # is max test time included in evaluation time
        self.has_test_max_time_in_new_time = torch.any(
            self.new_time == self.test_time.max()
        ).item()

        # is there ties in risk score associated to patients with event
        self.has_ties_score_events = self._has_ties(
            self.estimate[self.test_event == 1.0]
        )

        # is there ties in risk score associated to patients with censoring
        self.has_ties_score_event_censoring = self._has_ties(
            self.estimate[self.test_event == 1.0],
            self.estimate[self.test_event == 0.0],
        )

        # is there ties in risk score associated to patients with event and censoring
        self.has_ties_score_censoring = self._has_ties(
            self.estimate[self.test_event == 0.0]
        )

    def _check_conditions(self):
        """Compare condition evaluated on simulated to condition required."""

        self._check_condition(self.test_ties_time_event, self.has_test_ties_time_event)
        self._check_condition(
            self.test_ties_time_censoring, self.has_test_ties_time_censoring
        )
        self._check_condition(
            self.test_ties_time_event_censoring, self.has_test_ties_time_event_censoring
        )
        self._check_condition(
            self.train_ties_time_event, self.has_train_ties_time_event
        )
        self._check_condition(
            self.train_ties_time_censoring, self.has_train_ties_time_censoring
        )
        self._check_condition(
            self.train_ties_time_event_censoring,
            self.has_train_ties_time_event_censoring,
        )
        self._check_condition(
            self.test_event_at_last_time, self.has_test_event_at_last_time
        )
        self._check_condition(self.test_no_censoring, self.has_test_no_censoring)
        self._check_condition(self.train_no_censoring, self.has_train_no_censoring)
        self._check_condition(self.train_all_censored, self.has_train_all_censored)
        self._check_condition(self.test_all_censored, self.has_test_all_censored)
        self._check_condition(
            self.test_max_time_gt_train_max_time,
            self.has_test_max_time_gt_train_max_time,
        )
        self._check_condition(
            self.test_max_time_in_new_time, self.has_test_max_time_in_new_time
        )
        self._check_condition(self.ties_score_events, self.has_ties_score_events)
        self._check_condition(
            self.has_ties_score_event_censoring, self.has_ties_score_event_censoring
        )
        self._check_condition(
            self.has_ties_score_censoring, self.has_ties_score_censoring
        )

    def _check_condition(self, condition, flag):
        if condition == True and flag == False:
            raise ValueError("Condition is not met.")

    def _has_ties(self, tensor, tensor2=None):
        # check if there are ties within tensor or with tensor2 if specified
        if tensor2 is None:
            return len(tensor) > len(torch.unique(tensor))
        else:
            cat, counts = torch.cat([tensor, tensor2]).unique(return_counts=True)
            intersection = cat[torch.where(counts.gt(1))]
            return intersection.numel() > 0

    def _convert_to_arrays(self):
        # train time and survival as numpy array
        self.y_train_array = np.array(
            list(zip(self.train_event.numpy(), self.train_time.numpy())),
            dtype=[("survival", "?"), ("futime", "<i8")],
        )

        # test time and survival as numpy array
        self.y_test_array = np.array(
            list(zip(self.test_event.numpy(), self.test_time.numpy())),
            dtype=[("survival", "?"), ("futime", "<i8")],
        )

        # risk score and times as numpy array
        self.estimate_array = self.estimate.numpy()
        self.new_time_array = self.new_time.numpy()


class DataBatchContainer:
    def __init__(self):
        self.batches = []

    def add_batch(self, batch):
        self.batches.append(batch)

    def generate_batches(self, n_batch: int, flags_to_set: list):
        """Simulate a set of n_batch batches.

        Args:
            n_batch (int): Number of batches.
            flags_to_set (list): List of flags.
        """

        # increase number of batches if there are more flags than required number of batches
        if len(flags_to_set) > n_batch:
            n_batch = len(flags_to_set)

        for i in range(n_batch):

            if i >= len(flags_to_set):
                # simulate data without flag
                self.generate_one_batch()
            else:
                # simulate data given one flag
                self.generate_one_batch(flags_to_set[i])

    def generate_one_batch(self, flag_to_set: str = None, **kwargs):
        """Simulate one batch.

        Args:
            flag_to_set (str, optional): Name of condition. Defaults to None.
        """

        if flag_to_set is not None:
            kwargs[flag_to_set] = True

        data_batch = SurvivalDataGenerator(**kwargs)
        self.add_batch(data_batch.get_input() + data_batch.get_input_array())


def conditions_ci(output):
    # Lower < Upper, lower >= 0, upper <= 1
    return all([output[0] <= output[1], output[0] >= 0, output[1] <= 1])


def conditions_p_value(output):
    # p-value must be within [0-1]
    return all([output <= 1, output >= 0])


if __name__ == "__main__":
    # generate random survival data
    data = SurvivalDataGenerator(train_ties_time_event=True)
    data = SurvivalDataGenerator(test_ties_time_event=True)
    data = SurvivalDataGenerator(train_ties_time_censoring=True)
    data = SurvivalDataGenerator(test_ties_time_censoring=True)
    data = SurvivalDataGenerator(train_ties_time_event_censoring=True)
    data = SurvivalDataGenerator(test_ties_time_event_censoring=True)
    data = SurvivalDataGenerator(test_event_at_last_time=True)
    data = SurvivalDataGenerator(test_no_censoring=True)
    data = SurvivalDataGenerator(train_no_censoring=True)
    data = SurvivalDataGenerator(test_all_censored=True)
    data = SurvivalDataGenerator(train_all_censored=True)
    data = SurvivalDataGenerator(test_max_time_gt_train_max_time=True)
    data = SurvivalDataGenerator(test_max_time_in_new_time=True)
    data = SurvivalDataGenerator(ties_score_events=True)
    data = SurvivalDataGenerator(ties_score_event_censoring=True)
    data = SurvivalDataGenerator(ties_score_censoring=True)

    # batch of randomly generate data
    batch_container = DataBatchContainer()
    batch_container.generate_batches(
        n_batch=10, flags_to_set=["train_ties_time_event", "test_ties_time_event"]
    )
    batches = batch_container.batches
