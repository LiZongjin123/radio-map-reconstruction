import torch
from pytest import raises

from radio_map_reconstruction.sampler import Sampler, straight_through_top_k


def test_hard_top_k_supports_heterogeneous_budgets_and_row_major_ties():
    scores = torch.tensor(
        [
            [[[5.0, 5.0], [5.0, 5.0]]],
            [[[1.0, 4.0], [3.0, 2.0]]],
        ]
    )
    valid_receiving_area = torch.tensor(
        [
            [[[True, True], [False, True]]],
            [[[True, False], [True, True]]],
        ]
    )

    learned_sampling_mask = straight_through_top_k(
        scores,
        valid_receiving_area,
        torch.tensor([2, 1]),
        temperature=0.1,
        tolerance=1e-6,
        max_iterations=64,
    )

    assert torch.equal(
        learned_sampling_mask,
        torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
    )
    assert torch.equal(
        learned_sampling_mask.sum(dim=(1, 2, 3)), torch.tensor([2.0, 1.0])
    )


def test_straight_through_top_k_has_hard_values_and_soft_score_gradients():
    scores = torch.tensor(
        [[[[0.5, -0.5], [1.0, -1.0]]]], requires_grad=True
    )

    learned_sampling_mask = straight_through_top_k(
        scores,
        torch.ones_like(scores, dtype=torch.bool),
        torch.tensor([2]),
        temperature=0.1,
        tolerance=1e-6,
        max_iterations=64,
    )
    learned_sampling_mask.sum().backward()

    assert torch.equal(
        learned_sampling_mask.detach(),
        torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
    )
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) > 0


def test_straight_through_top_k_reports_sample_specific_invalid_inputs():
    scores = torch.zeros(2, 1, 2, 2)
    valid_receiving_area = torch.ones_like(scores, dtype=torch.bool)

    with raises(ValueError, match=r"sample 0.*temperature.*0"):
        straight_through_top_k(
            scores,
            valid_receiving_area,
            torch.tensor([1, 1]),
            temperature=0,
            tolerance=1e-6,
            max_iterations=64,
        )

    with raises(ValueError, match=r"sample 1.*K.*0"):
        straight_through_top_k(
            scores,
            valid_receiving_area,
            torch.tensor([1, 0]),
            temperature=0.1,
            tolerance=1e-6,
            max_iterations=64,
        )

    with raises(ValueError, match=r"sample 1.*K.*5.*Valid Receiving Area.*4"):
        straight_through_top_k(
            scores,
            valid_receiving_area,
            torch.tensor([1, 5]),
            temperature=0.1,
            tolerance=1e-6,
            max_iterations=64,
        )

    valid_receiving_area[1] = False
    with raises(ValueError, match=r"sample 1.*empty Valid Receiving Area"):
        straight_through_top_k(
            scores,
            valid_receiving_area,
            torch.tensor([1, 1]),
            temperature=0.1,
            tolerance=1e-6,
            max_iterations=64,
        )


def test_straight_through_top_k_stays_finite_for_extreme_finite_scores():
    extreme = torch.finfo(torch.float32).max / 2
    scores = torch.tensor(
        [[[[extreme, -extreme], [extreme, -extreme]]]], requires_grad=True
    )

    learned_sampling_mask = straight_through_top_k(
        scores,
        torch.ones_like(scores, dtype=torch.bool),
        torch.tensor([2]),
        temperature=0.1,
        tolerance=1e-6,
        max_iterations=64,
    )
    learned_sampling_mask.sum().backward()

    assert torch.isfinite(learned_sampling_mask).all()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_sampler_uses_building_then_transmitter_input_and_sixteen_base_channels():
    sampler = Sampler()
    observed_inputs = []
    hook = sampler.score_model.register_forward_pre_hook(
        lambda _module, inputs: observed_inputs.append(inputs[0].detach().clone())
    )
    building_map = torch.ones(1, 1, 16, 16)
    transmitter_map = torch.full((1, 1, 16, 16), 2.0)

    sampling_score_map = sampler(building_map, transmitter_map)
    hook.remove()

    assert sampling_score_map.shape == (1, 1, 16, 16)
    assert sampler.score_model.in_channels == 2
    assert sampler.score_model.input_block.residual[0].out_channels == 16
    assert torch.equal(observed_inputs[0][:, 0:1], building_map)
    assert torch.equal(observed_inputs[0][:, 1:2], transmitter_map)
