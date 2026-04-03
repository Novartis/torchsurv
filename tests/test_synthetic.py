import pytest
import torch
from torch import nn

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.tools import make_synthetic_data


class TestSyntheticData:
    def test_shapes_and_dtypes(self):
        batch = make_synthetic_data(n=128, m=6, rho=0.75, seed=12)

        assert set(batch) == {"x", "event", "time", "log_risk", "beta"}
        assert batch["x"].shape == (128, 6)
        assert batch["event"].shape == (128,)
        assert batch["time"].shape == (128,)
        assert batch["log_risk"].shape == (128,)
        assert batch["beta"].shape == (6,)
        assert batch["x"].dtype == torch.float32
        assert batch["event"].dtype == torch.bool
        assert batch["time"].dtype == torch.float32
        assert batch["log_risk"].dtype == torch.float32
        assert batch["beta"].dtype == torch.float32
        assert torch.all(batch["time"] >= 0.0)
        assert batch["event"].any()

    def test_reproducible_with_seed(self):
        batch_a = make_synthetic_data(n=64, m=4, rho=0.5, censoring_rate=0.2, seed=7)
        batch_b = make_synthetic_data(n=64, m=4, rho=0.5, censoring_rate=0.2, seed=7)

        for key in batch_a:
            assert torch.equal(batch_a[key], batch_b[key])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n": 0, "m": 4, "rho": 0.5}, "n"),
            ({"n": 32, "m": 0, "rho": 0.5}, "m"),
            ({"n": 32, "m": 4, "rho": -0.1}, "rho"),
            ({"n": 32, "m": 4, "rho": 1.1}, "rho"),
            ({"n": 32, "m": 4, "rho": 0.5, "censoring_rate": -0.1}, "censoring_rate"),
            ({"n": 32, "m": 4, "rho": 0.5, "censoring_rate": 1.0}, "censoring_rate"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            make_synthetic_data(**kwargs)

    def test_signal_strength_controls_concordance(self):
        cindex = ConcordanceIndex()
        high_signal = []
        low_signal = []

        for seed in range(5):
            high_batch = make_synthetic_data(n=256, m=8, rho=1.0, seed=seed)
            low_batch = make_synthetic_data(n=256, m=8, rho=0.0, seed=seed)
            high_estimate = high_batch["x"] @ high_batch["beta"]
            low_estimate = low_batch["x"] @ low_batch["beta"]

            high_signal.append(cindex(high_estimate, high_batch["event"], high_batch["time"], instate=False).item())
            low_signal.append(cindex(low_estimate, low_batch["event"], low_batch["time"], instate=False).item())

        high_mean = sum(high_signal) / len(high_signal)
        low_mean = sum(low_signal) / len(low_signal)

        assert high_mean > low_mean + 0.2
        assert 0.4 < low_mean < 0.6

    def test_high_signal_dataset_is_trainable(self):
        batch = make_synthetic_data(n=256, m=8, rho=1.0, seed=0)
        model = nn.Linear(8, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        cindex = ConcordanceIndex()

        losses = []
        with torch.no_grad():
            initial_cindex = cindex(model(batch["x"]).squeeze(), batch["event"], batch["time"], instate=False).item()

        for _ in range(80):
            optimizer.zero_grad()
            loss = neg_partial_log_likelihood(model(batch["x"]), batch["event"], batch["time"])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        with torch.no_grad():
            final_cindex = cindex(model(batch["x"]).squeeze(), batch["event"], batch["time"], instate=False).item()

        assert losses[-1] < losses[0]
        assert final_cindex > initial_cindex + 0.1
