from pytest import approx
from torch import tensor

from radio_map_reconstruction.loss import RadioMapLoss


def test_loss_macro_averages_each_sample_without_clipping_predictions():
    outputs = tensor(
        [
            [[[2.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
        ]
    )
    targets = {
        "gain": tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
        "mask": tensor(
            [
                [[[True, False], [False, False]]],
                [[[True, True], [True, True]]],
            ]
        ),
    }

    loss = RadioMapLoss()(outputs, targets)

    assert loss.item() == approx(0.75)
