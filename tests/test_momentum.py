import os
import unittest

import torch
from torch import nn

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.momentum import Momentum
from torchsurv.loss.weibull import neg_log_likelihood

# set seed for reproducibility
torch.manual_seed(42)


class TestMometum(unittest.TestCase):
    def test_momentum_weibull(self):
        model = Momentum(
            backbone=nn.Sequential(nn.Linear(8, 2)),  # Weibull expect two outputs
            loss=neg_log_likelihood,
        )
        x = torch.randn((3, 8))
        results = model.infer(x)
        self.assertEqual(results.size(), (3, 2))
        self.assertEqual(results.requires_grad, False)
        self.assertEqual(model.online.training, True)
        self.assertEqual(model.target.training, False)
        self.assertTrue(torch.equal(results, model.target(x)))

    def test_momentum_cox(self):
        model = Momentum(
            backbone=nn.Sequential(nn.Linear(8, 1)),  # Cox expect one outputs
            loss=neg_partial_log_likelihood,
        )
        x = torch.randn((3, 8))
        results = model.infer(x)
        self.assertEqual(results.size(), (3, 1))
        self.assertEqual(results.requires_grad, False)
        self.assertEqual(model.online.training, True)
        self.assertEqual(model.target.training, False)
        self.assertTrue(torch.equal(results, model.target(x)))


if __name__ == "__main__":
    unittest.main()
