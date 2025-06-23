import unittest

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex

# set seed for reproducibility
torch.manual_seed(42)


class LitMNIST(LightningModule):
    def __init__(self, hidden_size=128, learning_rate=5e-4):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = "."
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = 256
        self.num_workers = 4

        # Model optimizisation
        self.loss = neg_partial_log_likelihood
        self.cindex = ConcordanceIndex()

        # Hardcode some dataset specific attributes
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0,), std=(1,)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y[y == 0] = 10.0  # Offset 0 to prevent log(0)
        params = self(x)
        loss = self.loss(params, torch.ones_like(y, device=y.device).bool(), y)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y[y == 0] = 10  # Offset 0 to prevent log(0)
        params = self(x)
        loss = self.loss(params, torch.ones_like(y, device=y.device).bool(), y)
        cindex = self.cindex(
            params, torch.ones_like(y, device=y.device).bool(), y.float()
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

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
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
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


class TestMetrics(unittest.TestCase):
    def test_training(self):
        model = LitMNIST()
        trainer = Trainer(
            logger=False,
            enable_checkpointing=False,
            accelerator="auto",
            fast_dev_run=2,
        )
        trainer.fit(model)
        results = trainer.validate(model)[0]
        values = list(results.values())
        names = list(results.keys())
        self.assertTrue(names == ["val_loss_epoch", "cindex_epoch"])
        self.assertTrue(values[0] > 0)  # Loss
        self.assertTrue(all([values[1] <= 1, values[1] >= 0]))  # Cindex


if __name__ == "__main__":
    unittest.main()
