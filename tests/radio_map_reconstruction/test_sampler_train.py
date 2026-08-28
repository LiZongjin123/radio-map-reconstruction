import csv
from copy import deepcopy

import torch
import radio_map_reconstruction.sampler_train as sampler_train_module
from torch import Tensor, tensor
from torch.nn import Module, Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.sampler_train import CONFIG, run_sampler_training


EVALUATION_SAMPLE_COUNTS = (10, 20, 30, 50, 75, 100, 125, 150, 175, 200)


class TinySampler(Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = Parameter(tensor(0.1))

    def forward(self, building_map: Tensor, transmitter_map: Tensor) -> Tensor:
        positions = torch.linspace(
            0, 1, building_map.shape[-1], device=building_map.device
        ).reshape(1, 1, 1, -1)
        return self.scale * positions.expand_as(building_map) + transmitter_map * 0


class RecordingReconstructor(Module):
    def __init__(self) -> None:
        super().__init__()
        self.sparse_weight = Parameter(tensor(0.5))
        self.observed_inputs: list[Tensor] = []

    def forward(self, inputs: Tensor) -> Tensor:
        self.observed_inputs.append(inputs.detach().clone())
        return inputs[:, :1] * self.sparse_weight


def make_sample(sample_count: int | None = None):
    dense_target = torch.linspace(0.1, 0.9, 202).reshape(1, 1, 202)
    transmitter_map = torch.zeros_like(dense_target)
    building_map = torch.zeros_like(dense_target)
    fixed_sampling_mask = torch.zeros_like(dense_target)
    if sample_count is not None:
        fixed_sampling_mask[..., :sample_count] = 1
    inputs = torch.cat(
        (
            dense_target * fixed_sampling_mask,
            transmitter_map,
            building_map,
            fixed_sampling_mask,
        )
    )
    return inputs, {
        "gain": dense_target,
        "mask": torch.ones_like(dense_target, dtype=torch.bool),
    }


def test_complete_sampler_training_seam(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
    torch.manual_seed(7)
    sampler = TinySampler()
    reconstructor = RecordingReconstructor()
    reconstructor_before = deepcopy(reconstructor.state_dict())
    sampler_before = deepcopy(sampler.state_dict())
    train_loader = DataLoader(
        [make_sample(), make_sample()], batch_size=2, shuffle=False
    )
    val_loader = DataLoader(
        [make_sample(sample_count) for sample_count in EVALUATION_SAMPLE_COUNTS],
        batch_size=5,
        shuffle=False,
    )
    optimizer = SGD(sampler.parameters(), lr=0.5)
    scheduler = ExponentialLR(optimizer, gamma=0.5)

    run_sampler_training(
        sampler=sampler,
        reconstructor=reconstructor,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=RadioMapLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_num=1,
        run_dir=tmp_path,
        sampler_model_config={"base_channels": 16},
        sampler_training_config={
            "minimum_sample_count": 10,
            "maximum_sample_count": 200,
            "temperature": 0.1,
            "bisection_tolerance": 1e-6,
            "bisection_max_iterations": 64,
        },
    )

    training_inputs = reconstructor.observed_inputs[0]
    training_sample_counts = training_inputs[:, 3].sum(dim=(1, 2))
    assert training_sample_counts.shape == (2,)
    assert torch.all((10 <= training_sample_counts) & (training_sample_counts <= 200))
    assert training_sample_counts[0] != training_sample_counts[1]
    dense_target = make_sample()[1]["gain"].expand(2, -1, -1, -1)
    assert torch.equal(training_inputs[:, 0:1], training_inputs[:, 3:4] * dense_target)
    assert torch.count_nonzero(training_inputs[:, 1]) == 0
    assert torch.count_nonzero(training_inputs[:, 2]) == 0

    assert reconstructor.state_dict() == reconstructor_before
    assert sampler.state_dict()["scale"] != sampler_before["scale"]

    with (tmp_path / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert set(rows[0]) == {
        "epoch",
        "train_loss",
        "val_loss",
        "val_mean_per_sample_normalized_rmse",
        "learning_rate",
    }

    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert latest.keys() == best.keys() == {
        "sampler_model_config",
        "sampler_training_config",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "training_state",
    }
    assert latest["epoch"] == 1
    assert "best_val_mean_per_sample_normalized_rmse" in latest["training_state"]
    assert all("reconstructor" not in key for key in latest)


def test_train_sampler_loads_fixed_reconstructor_and_owns_only_sampler_run(
    monkeypatch, tmp_path
):
    run_root = tmp_path / "run"
    reconstructor_dir = run_root / "reconstructor"
    sampler_dir = run_root / "sampler"
    reconstructor_best = reconstructor_dir / "best.pt"
    reconstructor_sentinel = reconstructor_dir / "keep.txt"
    stale_sampler_artifact = sampler_dir / "stale.txt"
    for path in (
        reconstructor_best,
        reconstructor_sentinel,
        stale_sampler_artifact,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep", encoding="utf-8")

    monkeypatch.setitem(CONFIG["runtime"], "run_dir", str(run_root))
    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
    monkeypatch.setitem(CONFIG["reconstructor"], "run_dir", str(reconstructor_dir))
    monkeypatch.setitem(CONFIG["sampler"], "run_dir", str(sampler_dir))
    monkeypatch.setattr(sampler_train_module, "RadioDataset", lambda part: part)
    monkeypatch.setattr(sampler_train_module, "DataLoader", lambda **kwargs: kwargs)
    loaded_paths = []
    frozen_reconstructor = RecordingReconstructor()
    monkeypatch.setattr(
        sampler_train_module,
        "load_reconstructor_checkpoint",
        lambda path: loaded_paths.append(path) or frozen_reconstructor,
    )
    created_sampler_configs = []
    monkeypatch.setattr(
        sampler_train_module,
        "Sampler",
        lambda **kwargs: created_sampler_configs.append(kwargs) or TinySampler(),
    )
    monkeypatch.setattr(sampler_train_module.swanlab, "init", lambda **kwargs: None)
    monkeypatch.setattr(sampler_train_module.swanlab, "log", lambda metrics: None)
    monkeypatch.setattr(sampler_train_module.swanlab, "finish", lambda: None)
    captured_run_dirs = []
    monkeypatch.setattr(
        sampler_train_module,
        "run_sampler_training",
        lambda **kwargs: captured_run_dirs.append(kwargs["run_dir"]),
    )

    sampler_train_module.train_sampler()

    assert loaded_paths == [reconstructor_best]
    assert created_sampler_configs == [CONFIG["sampler"]["model"]]
    assert captured_run_dirs == [sampler_dir]
    assert not stale_sampler_artifact.exists()
    assert reconstructor_best.read_text(encoding="utf-8") == "keep"
    assert reconstructor_sentinel.read_text(encoding="utf-8") == "keep"
