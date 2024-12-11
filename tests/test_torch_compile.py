"""

Tests for torch.compile

References:
    - https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    - https://github.com/pytorch/pytorch/issues/122094

"""

# global modules
import unittest

import torch

# Local modules
from torchsurv.loss.cox import neg_partial_log_likelihood as cox
from torchsurv.loss.weibull import neg_log_likelihood as weibull

# set seed for reproducibility
torch.manual_seed(42)

N = 512


class TestTorchCompile(unittest.TestCase):
    """
    Tests using torch.compile with cox
    """

    def test_cox_equivalence(self):
        """
        whether the compiled version of cox evaluates to the same value
        """

        # random data and parameters
        log_hz = torch.randn(N)
        event = torch.randint(low=0, high=2, size=(N,)).bool()
        time = torch.randint(low=1, high=100, size=(N,))

        # compiled version of cox
        ccox = torch.compile(cox)

        loss_cox = cox(log_hz, event, time)
        loss_ccox = ccox(log_hz, event, time)

        self.assertTrue(torch.allclose(loss_cox, loss_ccox, rtol=1e-3, atol=1e-3))

    def test_weibull_equivalence(self):
        """
        whether the compiled version of weibull evaluates to the same value
        """

        # random data and parameters
        log_hz = torch.randn(N)
        event = torch.randint(low=0, high=2, size=(N,)).bool()
        time = torch.randint(low=1, high=100, size=(N,))

        # compiled version of weibull
        cweibull = torch.compile(weibull)

        loss_weibull = weibull(log_hz, event, time)
        loss_cweibull = cweibull(log_hz, event, time)

        self.assertTrue(
            torch.allclose(loss_weibull, loss_cweibull, rtol=1e-3, atol=1e-3)
        )


if __name__ == "__main__":

    unittest.main()
