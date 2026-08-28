from pathlib import Path

from yaml import safe_load


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yml"

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = safe_load(config_file)


def project_path(configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def reconstructor_run_dir() -> Path:
    return project_path(CONFIG["reconstructor"]["run_dir"])


def run_root_dir() -> Path:
    return project_path(CONFIG["runtime"]["run_dir"])


def sampler_run_dir() -> Path:
    return project_path(CONFIG["sampler"]["run_dir"])
