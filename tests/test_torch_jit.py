import os
import unittest

import torch
import torch._dynamo

from torchsurv.loss.cox import neg_partial_log_likelihood as cox
from torchsurv.loss.weibull import neg_log_likelihood as weibull

# set seed for reproducibility
torch.manual_seed(42)

# Disable TorchScript JIT
os.environ["PYTORCH_JIT"] = "0"
torch._dynamo.config.suppress_errors = True


class TestTorchCompile(unittest.TestCase):
    """
    Tests using torch.compile with cox
    """

    N = 512

    def test_cox_equivalence(self):
        """
        whether the compiled version of cox evaluates to the same value
        """

        # random data and parameters
        log_hz = torch.randn(self.N)
        event = torch.randint(low=0, high=2, size=(self.N,)).bool()
        time = torch.randint(low=1, high=100, size=(self.N,))

        ccox = torch.compile(cox)  # scripted version of cox
        scox = torch.jit.script(cox)  # compiled version of cox

        loss_cox = cox(log_hz, event, time, ties_method="efron")
        loss_scox = scox(log_hz, event, time, ties_method="efron")
        loss_ccox = ccox(log_hz, event, time, ties_method="efron")

        self.assertTrue(
            torch.allclose(loss_cox, loss_scox, rtol=1e-3, atol=1e-3),
            msg="scripted failed",
        )
        self.assertTrue(
            torch.allclose(loss_cox, loss_ccox, rtol=1e-3, atol=1e-3),
            msg="compiled failed",
        )

    def test_weibull_equivalence(self):
        """
        whether the compiled version of weibull evaluates to the same value
        """

        # random data and parameters
        log_hz = torch.randn(self.N)
        event = torch.randint(low=0, high=2, size=(self.N,)).float()
        time = torch.randint(low=1, high=100, size=(self.N,))

        cweibull = torch.compile(weibull)  # scripted version of cox
        sweibull = torch.jit.script(weibull)  # compiled version of cox

        loss_weibull = weibull(log_hz, event, time)
        loss_cweibull = cweibull(log_hz, event, time)
        loss_sweibull = sweibull(log_hz, event, time)

        self.assertTrue(
            torch.allclose(loss_weibull, loss_sweibull, rtol=1e-3, atol=1e-3),
            msg="scripted failed",
        )
        self.assertTrue(
            torch.allclose(loss_weibull, loss_cweibull, rtol=1e-3, atol=1e-3),
            msg="compiled failed",
        )


if __name__ == "__main__":
    unittest.main()
