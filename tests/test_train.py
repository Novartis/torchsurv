import unittest

import lightning as L
import torch
from loguru import logger
from utils import (
    LitSurvival,
    LitSurvivalTwins,
    SimpleLinearNNOneParameter,
    SimpleLinearNNTwoParameters,
)

from torchsurv.loss.weibull import neg_log_likelihood as weibull

# set seed for reproducibility
torch.manual_seed(42)

trainer = L.Trainer(max_epochs=2, log_every_n_steps=5)


class TestLitTraining(unittest.TestCase):
    BATCH_N = 10

    def run_training(self, trainer, model):
        logger.critical(f"Loading {model.dataname} dataset")
        trainer.fit(model)
        return model

    def test_one_param(self):
        """One parameter Weibull"""
        model = self.run_training(
            trainer,
            LitSurvival(
                backbone=SimpleLinearNNOneParameter(input_size=2),
                loss=weibull,
                dataname="lung",
                batch_size=248 // self.BATCH_N,
            ),
        )
        with torch.no_grad():
            params = model(torch.randn(1, 2))
        self.assertEqual(params.size(), (1, 1))

    def test_two_params(self):
        """Two parameters Weibull"""
        model = self.run_training(
            trainer,
            LitSurvival(
                backbone=SimpleLinearNNTwoParameters(input_size=2),
                loss=weibull,
                dataname="gbsb",
                batch_size=686 // self.BATCH_N,
            ),
        )
        with torch.no_grad():
            params = model(torch.randn(1, 2))
        self.assertEqual(params.size(), (1, 2))

    def test_twins(self):
        """Two parameters Weibull"""
        model = self.run_training(
            trainer,
            LitSurvivalTwins(
                backbone=SimpleLinearNNOneParameter(input_size=2),
                loss=weibull,
                steps=2,
                dataname="gbsb",
                batch_size=686 // self.BATCH_N,
            ),
        )

        with torch.no_grad():
            params = model.momentum.infer(torch.randn(4, 2))
        self.assertEqual(params.size(), (4, 1))


if __name__ == "__main__":
    unittest.main()
