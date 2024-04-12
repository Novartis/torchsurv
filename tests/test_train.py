import unittest

# Global
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

# Set a seed for reproducibility
seed_value = 45
torch.manual_seed(seed_value)
BATCH_N = 10


trainer = L.Trainer(max_epochs=2, log_every_n_steps=5)


class TestLitTraining(unittest.TestCase):
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
                batch_size=248 // BATCH_N,
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
                batch_size=686 // BATCH_N,
            ),
        )
        with torch.no_grad():
            params = model(torch.randn(1, 2))
        self.assertEqual(params.size(), (1, 2))

    def test_twins(self):
        """Two parameters Weibull"""
        model = LitSurvivalTwins(
            backbone=SimpleLinearNNOneParameter(input_size=2),
            loss=weibull,
            steps=2,
            dataname="gbsb",
            batch_size=686 // BATCH_N,
        )
        res = self.run_training(
            trainer,
            model,
        )

        with torch.no_grad():
            x = torch.randn(1, 2)
            # logger.info(f"No training: {res.twinnets.backbone(x)}")
            # logger.info(f"Student (Q): {res.twinnets.encoder_q(x)}")
            # logger.info(f"Teacher (K): {res.twinnets.encoder_k(x)}")

        # Model parameters (sanity checks)
        # for params in model.twinnets.backbone.parameters():
        #     logger.info(f"No training: {params}")
        # for params in model.twinnets.encoder_q.parameters():
        #     logger.info(f"Student (Q): {params}")
        # for params in model.twinnets.encoder_k.parameters():
        #     logger.info(f"Teacher (K): {params}")


if __name__ == "__main__":
    unittest.main()
