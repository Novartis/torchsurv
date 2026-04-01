from __future__ import annotations

import pytest
import torch

from torchsurv.loss import competing_risks
from torchsurv.loss.cox import neg_partial_log_likelihood as cox


def test_competing_risks_loss_matches_sum_of_binary_cox_losses() -> None:
    log_hz = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.5, 0.2],
            [0.2, -0.1],
            [0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    event = torch.tensor([1, 2, 0, 1, 2], dtype=torch.long)
    time = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0], dtype=torch.float32)

    observed = competing_risks.neg_partial_log_likelihood(log_hz, event, time, ties_method="efron", reduction="sum")
    expected = cox(log_hz[:, 0], event == 1, time, ties_method="efron", reduction="sum") + cox(
        log_hz[:, 1], event == 2, time, ties_method="efron", reduction="sum"
    )

    assert torch.allclose(observed, expected)


def test_competing_risks_loss_matches_stratified_binary_cox_losses() -> None:
    log_hz = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.5, 0.2],
            [0.2, -0.1],
            [0.0, 0.5],
            [0.7, -0.4],
        ],
        dtype=torch.float32,
    )
    event = torch.tensor([1, 2, 0, 1, 2, 1], dtype=torch.long)
    time = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0, 2.0], dtype=torch.float32)
    strata = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    observed = competing_risks.neg_partial_log_likelihood(
        log_hz,
        event,
        time,
        ties_method="breslow",
        reduction="sum",
        strata=strata,
    )
    expected = cox(log_hz[:, 0], event == 1, time, ties_method="breslow", reduction="sum", strata=strata) + cox(
        log_hz[:, 1], event == 2, time, ties_method="breslow", reduction="sum", strata=strata
    )

    assert torch.allclose(observed, expected)


def test_all_censored_batch_returns_zero_loss() -> None:
    log_hz = torch.randn(5, 2)
    event = torch.zeros(5, dtype=torch.long)
    time = torch.arange(1, 6, dtype=torch.float32)

    loss = competing_risks.neg_partial_log_likelihood(log_hz, event, time)

    assert loss.item() == 0.0


@pytest.mark.parametrize(
    ("log_hz", "event", "match"),
    [
        pytest.param(torch.randn(4), torch.tensor([1, 0, 1, 0]), "shape", id="log_hz_not_2d"),
        pytest.param(torch.randn(4, 2), torch.tensor([1, 3, 0, 1]), "cause label", id="cause_exceeds_heads"),
        pytest.param(torch.randn(4, 2), torch.tensor([1, -1, 0, 1]), "non-negative", id="negative_cause_label"),
    ],
)
def test_invalid_competing_risks_inputs_raise(log_hz: torch.Tensor, event: torch.Tensor, match: str) -> None:
    time = torch.arange(1, 5, dtype=torch.float32)
    with pytest.raises(ValueError, match=match):
        competing_risks.neg_partial_log_likelihood(log_hz, event, time)


def test_baseline_curves_match_hand_computed_toy_example() -> None:
    log_hz = torch.zeros((4, 2), dtype=torch.float32)
    event = torch.tensor([1, 2, 0, 1], dtype=torch.long)
    time = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    baseline = competing_risks.baseline_cumulative_incidence_function(log_hz, event, time)

    expected_hazard = torch.tensor(
        [
            [0.25, 0.0],
            [0.0, 1.0 / 3.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_survival = torch.tensor(
        [
            torch.exp(torch.tensor(-0.25)),
            torch.exp(torch.tensor(-0.25 - 1.0 / 3.0)),
            torch.exp(torch.tensor(-0.25 - 1.0 / 3.0)),
            torch.exp(torch.tensor(-0.25 - 1.0 / 3.0 - 1.0)),
        ],
        dtype=torch.float32,
    )
    expected_cif = torch.tensor(
        [
            [1.0 - torch.exp(torch.tensor(-0.25)), 0.0],
            [
                1.0 - torch.exp(torch.tensor(-0.25)),
                torch.exp(torch.tensor(-0.25)) * (1.0 - torch.exp(torch.tensor(-1.0 / 3.0))),
            ],
            [
                1.0 - torch.exp(torch.tensor(-0.25)),
                torch.exp(torch.tensor(-0.25)) * (1.0 - torch.exp(torch.tensor(-1.0 / 3.0))),
            ],
            [
                1.0
                - torch.exp(torch.tensor(-0.25))
                + torch.exp(torch.tensor(-0.25 - 1.0 / 3.0)) * (1.0 - torch.exp(torch.tensor(-1.0))),
                torch.exp(torch.tensor(-0.25)) * (1.0 - torch.exp(torch.tensor(-1.0 / 3.0))),
            ],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(baseline["baseline_hazard"], expected_hazard, atol=1e-6)
    assert torch.allclose(baseline["baseline_survival"], expected_survival, atol=1e-6)
    assert torch.allclose(baseline["baseline_cif"], expected_cif, atol=1e-6)
    assert torch.allclose(baseline["baseline_survival"], 1 - baseline["baseline_cif"].sum(dim=1), atol=1e-6)


def test_cif_predictions_are_monotone_and_match_survival_identity() -> None:
    log_hz = torch.zeros((4, 2), dtype=torch.float32)
    event = torch.tensor([1, 2, 0, 1], dtype=torch.long)
    time = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    baseline = competing_risks.baseline_cumulative_incidence_function(log_hz, event, time)

    new_log_hz = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    new_time = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)

    cif = competing_risks.cumulative_incidence_function(baseline, new_log_hz, new_time)
    survival = competing_risks.survival_function(baseline, new_log_hz, new_time)

    assert torch.all(cif[:, 1:, :] >= cif[:, :-1, :])
    assert torch.allclose(survival, 1 - cif.sum(dim=2), atol=1e-6)
    assert torch.all(cif[1, :, 0] >= cif[0, :, 0])
    assert torch.all(survival[1] <= survival[0])


def test_baseline_and_prediction_support_multiple_strata() -> None:
    log_hz = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, -0.2],
            [0.0, 0.0],
            [0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    event = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    time = torch.tensor([1.0, 2.0, 1.5, 3.0], dtype=torch.float32)
    strata = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    baseline = competing_risks.baseline_cumulative_incidence_function(log_hz, event, time, strata=strata)
    cif = competing_risks.cumulative_incidence_function(
        baseline,
        torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
        torch.tensor([1.5, 3.0], dtype=torch.float32),
        new_strata=torch.tensor([0, 1], dtype=torch.long),
    )

    assert set(baseline.keys()) == {0, 1}
    assert cif.shape == (2, 2, 2)
