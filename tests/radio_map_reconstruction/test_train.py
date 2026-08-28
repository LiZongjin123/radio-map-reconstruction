import csv

import radio_map_reconstruction.train as train_module
from pytest import approx, raises
from torch import Tensor, load, tensor
from torch.nn import Module, Parameter
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
from torch.utils.data import DataLoader

from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.train import (
    CONFIG,
    eval_one_epoch,
    run_training,
    train_one_epoch,
)
from radio_map_reconstruction.util import delete_reconstructor_run

EVALUATION_SAMPLE_COUNTS = (10, 20, 30, 50, 75, 100, 125, 150, 175, 200)


class PassThroughModel(Module):
    def __init__(self):
        super().__init__()
        self.anchor = Parameter(tensor(0.0))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs[:, :1] + self.anchor * 0


class ConstantPredictionModel(Module):
    def __init__(self, initial_prediction: float):
        super().__init__()
        self.prediction = Parameter(tensor(initial_prediction))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.prediction.expand(inputs.shape[0], 1, 1, inputs.shape[-1])


def make_orchestration_loaders() -> tuple[DataLoader, DataLoader]:
    train_inputs = tensor(0.0).new_zeros((4, 1, 200))
    train_targets = {
        "gain": tensor(0.2).expand(1, 1, 200),
        "mask": tensor(True).expand(1, 1, 200),
    }
    train_loader = DataLoader([(train_inputs, train_targets)], batch_size=1)

    val_samples = []
    for index, sample_count in enumerate(EVALUATION_SAMPLE_COUNTS):
        inputs = tensor(0.0).new_zeros((4, 1, 200))
        inputs[3, 0, :sample_count] = 1
        val_samples.append(
            (
                inputs,
                {
                    "gain": tensor(1.0 if index == 9 else 0.0).expand(1, 1, 200),
                    "mask": tensor(True).expand(1, 1, 200),
                },
            )
        )
    return train_loader, DataLoader(val_samples, batch_size=5, shuffle=False)


def test_train_epoch_loss_weights_samples_equally_across_uneven_batches(monkeypatch):
    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
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
    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
    samples = []
    for map_index in range(2):
        for count_index, sample_count in enumerate(EVALUATION_SAMPLE_COUNTS):
            prediction = count_index / 10 if map_index == 0 else 2.0
            inputs = tensor(0.0).new_zeros((4, 1, 200))
            inputs[0] = 9.0 if map_index == 0 else prediction
            inputs[0, 0, 0] = prediction
            inputs[3, 0, :sample_count] = 1
            valid_receiving_area = tensor(False).new_zeros((1, 1, 200))
            valid_receiving_area[..., : 1 if map_index == 0 else 200] = True
            samples.append(
                (
                    inputs,
                    {
                        "gain": tensor(0.0).new_zeros((1, 1, 200)),
                        "mask": valid_receiving_area,
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

    expected_rmse_by_sample_count = {
        sample_count: (count_index / 10 + 1.0) / 2
        for count_index, sample_count in enumerate(EVALUATION_SAMPLE_COUNTS)
    }
    assert loss == approx(2.1425)
    assert rmse == approx(0.725)
    assert rmse == approx(sum(expected_rmse_by_sample_count.values()) / 10)
    assert rmse_by_sample_count == approx(expected_rmse_by_sample_count)


def test_training_writes_history_and_selects_best_average_rmse_without_clearing_run_dir(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def run_once(initial_prediction: float) -> None:
        train_loader, val_loader = make_orchestration_loaders()
        model = ConstantPredictionModel(initial_prediction)
        optimizer = SGD(model.parameters(), lr=0.25)
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=RadioMapLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=2,
            run_dir=tmp_path,
        )

    run_once(-0.2)
    run_once(-0.1)

    with (tmp_path / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert list(rows[0]) == [
        "epoch",
        "train_loss",
        "val_loss",
        "val_mean_per_sample_normalized_rmse",
        "learning_rate",
    ]
    assert len(rows) == 2
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert [float(row["train_loss"]) for row in rows] == approx([0.09, 0.0225])
    assert [float(row["val_loss"]) for row in rows] == approx([0.0925, 0.09015625])
    assert [
        float(row["val_mean_per_sample_normalized_rmse"]) for row in rows
    ] == approx([0.14, 0.17])
    assert [float(row["learning_rate"]) for row in rows] == approx([0.25, 0.125])

    best = load(tmp_path / "best.pt", weights_only=True)
    latest = load(tmp_path / "latest.pt", weights_only=True)
    assert best["model_state_dict"]["prediction"].item() == approx(0.05, abs=1e-7)
    assert latest["model_state_dict"]["prediction"].item() == approx(
        0.0875, abs=1e-7
    )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_train_owns_and_cleans_only_the_reconstructor_run_subtree(
    monkeypatch, tmp_path
):
    run_root = tmp_path / "run"
    reconstructor_run_dir = run_root / "reconstructor"
    sampler_sentinel = run_root / "sampler" / "keep.txt"
    legacy_checkpoint = run_root / "best.pt"
    stale_reconstructor_artifact = reconstructor_run_dir / "stale.txt"
    for path in (sampler_sentinel, legacy_checkpoint, stale_reconstructor_artifact):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep", encoding="utf-8")

    monkeypatch.setitem(
        CONFIG["reconstructor"], "run_dir", str(reconstructor_run_dir)
    )
    monkeypatch.setitem(CONFIG["runtime"], "run_dir", str(run_root))
    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
    monkeypatch.setattr(train_module, "RadioDataset", lambda part: part)
    monkeypatch.setattr(train_module, "DataLoader", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        train_module, "ResUnet", lambda **kwargs: ConstantPredictionModel(0.0)
    )
    monkeypatch.setattr(train_module.swanlab, "init", lambda **kwargs: None)
    monkeypatch.setattr(train_module.swanlab, "log", lambda metrics: None)
    monkeypatch.setattr(train_module.swanlab, "finish", lambda: None)
    captured_run_dirs = []

    def capture_run_directory(**kwargs):
        captured_run_dirs.append(kwargs["run_dir"])

    monkeypatch.setattr(train_module, "run_training", capture_run_directory)

    train_module.train()

    assert captured_run_dirs == [reconstructor_run_dir]
    assert not stale_reconstructor_artifact.exists()
    assert sampler_sentinel.read_text(encoding="utf-8") == "keep"
    assert legacy_checkpoint.read_text(encoding="utf-8") == "keep"


def test_reconstructor_cleanup_rejects_the_run_root(monkeypatch, tmp_path):
    run_root = tmp_path / "run"
    sampler_sentinel = run_root / "sampler" / "keep.txt"
    legacy_checkpoint = run_root / "best.pt"
    sampler_sentinel.parent.mkdir(parents=True)
    sampler_sentinel.write_text("sampler", encoding="utf-8")
    legacy_checkpoint.write_text("legacy", encoding="utf-8")

    monkeypatch.setitem(CONFIG["runtime"], "run_dir", str(run_root))
    monkeypatch.setitem(CONFIG["reconstructor"], "run_dir", str(run_root))
    monkeypatch.setitem(
        CONFIG["sampler"], "run_dir", str(run_root / "sampler")
    )

    with raises(ValueError, match="must be an isolated child"):
        delete_reconstructor_run()

    assert sampler_sentinel.read_text(encoding="utf-8") == "sampler"
    assert legacy_checkpoint.read_text(encoding="utf-8") == "legacy"


def test_reconstructor_cleanup_rejects_a_sampler_descendant(
    monkeypatch, tmp_path
):
    run_root = tmp_path / "run"
    sampler_run_dir = run_root / "sampler"
    overlapping_reconstructor_dir = sampler_run_dir / "reconstructor"
    sampler_artifact = overlapping_reconstructor_dir / "sampler-owned.txt"
    sampler_artifact.parent.mkdir(parents=True)
    sampler_artifact.write_text("sampler", encoding="utf-8")

    monkeypatch.setitem(CONFIG["runtime"], "run_dir", str(run_root))
    monkeypatch.setitem(
        CONFIG["reconstructor"],
        "run_dir",
        str(overlapping_reconstructor_dir),
    )
    monkeypatch.setitem(CONFIG["sampler"], "run_dir", str(sampler_run_dir))

    with raises(ValueError, match="must be an isolated child"):
        delete_reconstructor_run()

    assert sampler_artifact.read_text(encoding="utf-8") == "sampler"
