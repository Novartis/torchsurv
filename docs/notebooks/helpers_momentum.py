import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex


class LitMNIST(L.LightningModule):

    def __init__(
        self,
        backbone,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = "."
        self.learning_rate = 1e-3
        self.num_workers = 2

        # Model evaluation
        self.cindex = ConcordanceIndex()

        # Momentum
        self.model = backbone

    def forward(
        self,
        x,
    ):
        return self.model(x)

    # Lightning related behaviors:

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y[y == 0] = 10  # Offset 0 to prevent log(0)
        log_hz = self(x)
        loss = neg_partial_log_likelihood(
            log_hz, torch.ones_like(y, device=y.device).bool(), y.float()
        )
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y[y == 0] = 10  # Offset 0 to prevent log(0)
        log_hz = self(x)
        loss = neg_partial_log_likelihood(
            log_hz, torch.ones_like(y, device=y.device).bool(), y.float()
        )
        cindex = self.cindex(
            log_hz, torch.ones_like(y, device=y.device).bool(), y.float()
        )
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "cindex",
            cindex,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


class LitMomentum(L.LightningModule):

    def __init__(self, backbone):
        super().__init__()
        # Set our init args as class attributes
        self.data_dir = "."
        self.learning_rate = 1e-3
        self.num_workers = 2

        # Model evaluation
        self.cindex = ConcordanceIndex()

        # Momentum
        self.model = backbone

    # Lightning related behaviors:

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x, event, time):
        return self.model(x, event, time)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y[y == 0] = 10  # Offset 0 to prevent log(0)
        loss = self(x, torch.ones_like(y, device=y.device).bool(), y.float())
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y[y == 0] = 10  # Offset 0 to prevent log(0)
        loss = self(x, torch.ones_like(y, device=y.device).bool(), y.float())
        log_hz_k = self.model.target(x)
        cindex = self.cindex(
            log_hz_k, torch.ones_like(y, device=y.device).bool(), y.float()
        )
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "cindex",
            cindex,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, transforms=None, num_workers: int = 2):
        super().__init__()
        self.data_dir = "."
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transforms)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
