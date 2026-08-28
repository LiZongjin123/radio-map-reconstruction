from pathlib import Path

from yaml import safe_load


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_configuration_has_one_section_for_each_runtime_concern():
    with (PROJECT_ROOT / "config.yml").open(encoding="utf-8") as config_file:
        config = safe_load(config_file)

    assert set(config) == {
        "runtime",
        "dataset",
        "reconstructor",
        "sampler",
        "evaluation",
        "experiment_logging",
    }
    assert config["runtime"]["seed"] == 42
    assert "seed" not in config["evaluation"]
    assert config["runtime"]["run_dir"] == "run"
    assert config["reconstructor"]["run_dir"] == "run/reconstructor"
    assert config["sampler"]["run_dir"] == "run/sampler"
