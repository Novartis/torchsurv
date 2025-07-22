import torch
from hypothesis import given
from hypothesis import strategies as st
from torchsurv.loss.cox import neg_partial_log_likelihood


@given(
    log_hazard=st.lists(st.floats(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    n_samples=st.integers(min_value=10, max_value=100),
)
def test_cox_loss_properties(log_hazard, n_samples):
    # Ensure log_hazard matches n_samples
    if len(log_hazard) != n_samples:
        return  # Skip invalid cases

    # Convert log_hazard to a tensor
    log_hazard_tensor = torch.tensor(log_hazard, dtype=torch.float32)

    # Generate dummy survival times and event indicators
    survival_times = torch.arange(1, n_samples + 1, dtype=torch.float32)
    event_indicators = torch.ones(n_samples, dtype=torch.float32)

    # Compute the loss
    loss = neg_partial_log_likelihood(log_hazard_tensor, survival_times, event_indicators)

    # Assert the loss is finite
    assert torch.isfinite(loss), "Loss should be finite"
