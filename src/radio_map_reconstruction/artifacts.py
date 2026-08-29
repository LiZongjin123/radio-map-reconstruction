from pathlib import Path


RECONSTRUCTOR_RUN_PATH = Path("run") / "reconstructor"
COARSE_RECONSTRUCTOR_RUN_PATH = Path("run") / "coarse_reconstructor"
SAMPLER_RUN_PATH = Path("run") / "sampler"
EVALUATION_RUN_PATH = Path("run") / "evaluation"
EVALUATION_FIGURE_CASES = (
    (0, 10),
    (1, 50),
    (2, 100),
    (3, 200),
)
SAMPLING_DIAGNOSTIC_CASES = (
    (4, 10),
    (5, 50),
    (6, 100),
    (7, 200),
)
