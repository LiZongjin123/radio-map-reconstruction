from math import log

import torch
from torch import Tensor, cat
from torch.nn import Module

from radio_map_reconstruction.model import ResUnet


class Sampler(Module):
    """K-independent Ranked Sampling Policy producing Sampling Score Maps."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.score_model = ResUnet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )

    def forward(self, building_map: Tensor, transmitter_map: Tensor) -> Tensor:
        if building_map.shape != transmitter_map.shape:
            raise ValueError(
                "building_map and transmitter_map must have the same shape"
            )
        if building_map.ndim != 4 or building_map.shape[1] != 1:
            raise ValueError(
                "building_map and transmitter_map must each have shape "
                "(batch, 1, height, width)"
            )
        return self.score_model(cat((building_map, transmitter_map), dim=1))


class _StraightThroughMask(torch.autograd.Function):
    @staticmethod
    def forward(context, hard_mask: Tensor, soft_mask: Tensor) -> Tensor:
        del context
        del soft_mask
        return hard_mask

    @staticmethod
    def backward(context, gradient: Tensor) -> tuple[None, Tensor]:
        del context
        return None, gradient


class _TemperatureSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(
        context,
        scores: Tensor,
        threshold: Tensor,
        temperature: float,
        gradient_limit: float,
    ) -> Tensor:
        output = torch.sigmoid((scores - threshold) / temperature)
        context.save_for_backward(output)
        context.temperature = temperature
        context.gradient_limit = gradient_limit
        return output

    @staticmethod
    def backward(context, gradient: Tensor) -> tuple[Tensor, None, None, None]:
        (output,) = context.saved_tensors
        score_gradient = gradient * output * (1 - output) / context.temperature
        score_gradient = torch.nan_to_num(
            score_gradient,
            nan=0.0,
            posinf=context.gradient_limit,
            neginf=-context.gradient_limit,
        ).clamp(-context.gradient_limit, context.gradient_limit)
        return score_gradient, None, None, None


def _validate_top_k_inputs(
    scores: Tensor,
    valid_receiving_area: Tensor,
    sample_counts: Tensor,
    *,
    temperature: float,
    tolerance: float,
    max_iterations: int,
) -> None:
    if scores.ndim != 4 or scores.shape[1] != 1:
        raise ValueError("scores must have shape (batch, 1, height, width)")
    if valid_receiving_area.shape != scores.shape:
        raise ValueError("valid_receiving_area must have the same shape as scores")
    if sample_counts.ndim != 1 or sample_counts.shape[0] != scores.shape[0]:
        raise ValueError("sample_counts must contain one K per batch sample")
    if sample_counts.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("sample_counts must contain integers")
    if temperature <= 0:
        raise ValueError(
            f"sample 0: temperature must be positive; got {temperature}"
        )
    if tolerance <= 0:
        raise ValueError(f"sample 0: tolerance must be positive; got {tolerance}")
    if max_iterations < 1:
        raise ValueError(
            "sample 0: max_iterations must be at least 1; "
            f"got {max_iterations}"
        )

    flat_valid_receiving_area = valid_receiving_area.bool().flatten(start_dim=1)
    for sample_index, sample_count in enumerate(sample_counts.tolist()):
        valid_count = int(flat_valid_receiving_area[sample_index].sum().item())
        if valid_count == 0:
            raise ValueError(
                f"sample {sample_index}: empty Valid Receiving Area"
            )
        if sample_count < 1:
            raise ValueError(
                f"sample {sample_index}: K must be at least 1; got {sample_count}"
            )
        if sample_count > valid_count:
            raise ValueError(
                f"sample {sample_index}: K {sample_count} exceeds "
                f"Valid Receiving Area size {valid_count}"
            )


def _soft_top_k(
    scores: Tensor,
    valid_receiving_area: Tensor,
    sample_counts: Tensor,
    *,
    temperature: float,
    tolerance: float,
    max_iterations: int,
) -> Tensor:
    flat_scores = scores.flatten(start_dim=1)
    flat_valid_receiving_area = valid_receiving_area.bool().flatten(start_dim=1)
    flat_soft_mask = torch.zeros_like(flat_scores)
    work_dtype = torch.float64 if scores.dtype != torch.float64 else scores.dtype

    for sample_index, sample_count in enumerate(sample_counts.tolist()):
        valid_positions = torch.nonzero(
            flat_valid_receiving_area[sample_index], as_tuple=False
        ).squeeze(1)
        valid_scores = flat_scores[sample_index, valid_positions].to(work_dtype)

        padding = temperature * max(
            20.0,
            log(valid_scores.numel() / tolerance) + 2.0,
        )
        ranked_indices = torch.argsort(
            valid_scores.detach(), descending=True, stable=True
        )
        selection_boundary_score = valid_scores.detach()[
            ranked_indices[sample_count - 1]
        ]
        score_scale = valid_scores.detach().abs().max().clamp_min(1)
        normalized_differences = (
            valid_scores / score_scale
            - selection_boundary_score / score_scale
        )
        normalized_padding = padding / score_scale
        far_above = normalized_differences.detach() > normalized_padding
        far_below = normalized_differences.detach() < -normalized_padding
        near_selection_boundary = ~(far_above | far_below)

        centered_scores = torch.zeros_like(valid_scores)
        centered_scores[far_above] = padding
        centered_scores[far_below] = -padding
        centered_scores[near_selection_boundary] = (
            valid_scores[near_selection_boundary] - selection_boundary_score
        )

        with torch.no_grad():
            lower = centered_scores.min() - padding
            upper = centered_scores.max() + padding
            target = float(sample_count)

            for _ in range(max_iterations):
                threshold = (lower + upper) / 2
                soft_sum = torch.sigmoid(
                    (centered_scores - threshold) / temperature
                ).sum()
                if abs(soft_sum.item() - target) <= tolerance:
                    break
                if soft_sum > target:
                    lower = threshold
                else:
                    upper = threshold

            threshold = ((lower + upper) / 2).detach()

        gradient_limit = torch.finfo(scores.dtype).max
        soft_values = _TemperatureSigmoid.apply(
            centered_scores,
            threshold,
            temperature,
            gradient_limit,
        ).to(scores.dtype)
        flat_soft_mask[sample_index, valid_positions] = soft_values

    return flat_soft_mask.view_as(scores)


def straight_through_top_k(
    scores: Tensor,
    valid_receiving_area: Tensor,
    sample_counts: Tensor,
    *,
    temperature: float,
    tolerance: float,
    max_iterations: int,
) -> Tensor:
    _validate_top_k_inputs(
        scores,
        valid_receiving_area,
        sample_counts,
        temperature=temperature,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    hard_mask = torch.zeros_like(scores)
    flat_scores = scores.flatten(start_dim=1)
    flat_valid_receiving_area = valid_receiving_area.bool().flatten(start_dim=1)
    flat_hard_mask = hard_mask.flatten(start_dim=1)

    for sample_index, sample_count in enumerate(sample_counts.tolist()):
        valid_positions = torch.nonzero(
            flat_valid_receiving_area[sample_index], as_tuple=False
        ).squeeze(1)
        valid_scores = flat_scores[sample_index, valid_positions]
        ranked_indices = torch.argsort(valid_scores, descending=True, stable=True)
        selected_positions = valid_positions[ranked_indices[:sample_count]]
        flat_hard_mask[sample_index, selected_positions] = 1

    soft_mask = _soft_top_k(
        scores,
        valid_receiving_area,
        sample_counts,
        temperature=temperature,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return _StraightThroughMask.apply(hard_mask, soft_mask)
