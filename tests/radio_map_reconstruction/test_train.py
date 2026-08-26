from pytest import approx
from torch import Tensor, tensor
from torch.nn import Module, Parameter
from torch.optim import SGD
from torch.utils.data import DataLoader

from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.train import CONFIG, eval_one_epoch, train_one_epoch


class PassThroughModel(Module):
    def __init__(self):
        super().__init__()
        self.anchor = Parameter(tensor(0.0))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs[:, :1] + self.anchor * 0


def test_train_epoch_loss_weights_samples_equally_across_uneven_batches(monkeypatch):
    monkeypatch.setitem(CONFIG, "device", "cpu")
    samples = [
        (
            tensor([[[prediction]]]),
            {
                "gain": tensor([[[0.0]]]),
                "mask": tensor([[[True]]]),
            },
        )
        for prediction in (1.0, 1.0, 3.0)
    ]
    dataloader = DataLoader(samples, batch_size=2, shuffle=False)
    model = PassThroughModel()
    optimizer = SGD(model.parameters(), lr=0.0)

    epoch_loss = train_one_epoch(
        model, dataloader, optimizer, RadioMapLoss(), epoch=0, epoch_num=1
    )

    assert epoch_loss == approx(11 / 3)


def test_eval_epoch_reports_unclipped_loss_and_clipped_rmse_by_sample_count(
    monkeypatch,
):
    monkeypatch.setitem(CONFIG, "device", "cpu")
    sample_counts = (10, 20, 30, 50, 75, 100, 125, 150, 175, 200)
    samples = []
    for prediction in (2.0, 0.0):
        for sample_count in sample_counts:
            inputs = tensor(0.0).new_zeros((4, 1, 200))
            inputs[0] = prediction
            inputs[3, 0, :sample_count] = 1
            samples.append(
                (
                    inputs,
                    {
                        "gain": tensor(0.0).new_zeros((1, 1, 200)),
                        "mask": tensor(True).new_ones((1, 1, 200)),
                    },
                )
            )

    loss, rmse, rmse_by_sample_count = eval_one_epoch(
        PassThroughModel(),
        DataLoader(samples, batch_size=6, shuffle=False),
        RadioMapLoss(),
        epoch=0,
        epoch_num=1,
    )

    assert loss == approx(2.0)
    assert rmse == approx(0.5)
    assert rmse_by_sample_count == approx(
        {sample_count: 0.5 for sample_count in sample_counts}
    )
