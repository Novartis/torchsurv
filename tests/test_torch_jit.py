import os

import torch

from torchsurv.loss.cox import neg_partial_log_likelihood as cox
from torchsurv.loss.weibull import neg_log_likelihood_weibull as weibull
from torchsurv.metrics.cindex import ConcordanceIndex

# set seed for reproducibility
torch.manual_seed(42)

# Disable TorchScript JIT
os.environ["PYTORCH_JIT"] = "0"


class TestTorchCompile:
    """
    Tests using torch.compile with cox
    """

    N = 512

    def test_cox_equivalence(self):
        """
        whether the compiled version of cox evaluates to the same value
        """

        # random data and parameters
        log_hz = torch.randn(self.N, dtype=torch.float)
        event = torch.randint(low=0, high=2, size=(self.N,), dtype=torch.bool)
        time = torch.randint(low=1, high=100, size=(self.N,), dtype=torch.float)

        ccox = torch.compile(cox, backend="eager")  # compiled version of cox
        scox = torch.jit.script(cox)  # scripted version of cox

        loss_cox = cox(log_hz, event, time, ties_method="efron")
        loss_scox = scox(log_hz, event, time, ties_method="efron")
        loss_ccox = ccox(log_hz, event, time, ties_method="efron")

        assert torch.allclose(loss_cox, loss_scox, rtol=1e-3, atol=1e-3), "scripted failed"
        assert torch.allclose(loss_cox, loss_ccox, rtol=1e-3, atol=1e-3), "compiled failed"

    def test_weibull_equivalence(self):
        """
        whether the compiled version of weibull evaluates to the same value
        """

        # random data and parameters
        log_hz = torch.randn(self.N, dtype=torch.float)
        event = torch.randint(low=0, high=2, size=(self.N,), dtype=torch.bool)
        time = torch.randint(low=1, high=100, size=(self.N,), dtype=torch.float)

        cweibull = torch.compile(weibull, backend="eager")  # compiled version of weibull
        sweibull = torch.jit.script(weibull)  # scripted version of weibull

        loss_weibull = weibull(log_hz, event, time)
        loss_cweibull = cweibull(log_hz, event, time)
        loss_sweibull = sweibull(log_hz, event, time)

        assert torch.allclose(loss_weibull, loss_sweibull, rtol=1e-3, atol=1e-3), "scripted failed"
        assert torch.allclose(loss_weibull, loss_cweibull, rtol=1e-3, atol=1e-3), "compiled failed"

    def test_cindex_jit(self):
        """
        whether torch.compile round-trip for ConcordanceIndex returns finite values
        """

        log_hz = torch.randn(self.N, dtype=torch.float)
        event = torch.randint(low=0, high=2, size=(self.N,), dtype=torch.bool)
        time = torch.randint(low=1, high=100, size=(self.N,), dtype=torch.float)

        cindex = ConcordanceIndex()

        result = cindex(log_hz, event, time)
        compiled_cindex = torch.compile(cindex.__call__, backend="eager")
        result_compiled = compiled_cindex(log_hz, event, time)

        assert torch.isfinite(result), f"cindex result not finite: {result}"
        assert torch.isfinite(result_compiled), f"compiled cindex result not finite: {result_compiled}"
