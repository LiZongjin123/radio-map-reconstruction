from pathlib import Path


RECONSTRUCTOR_RUN_PATH = Path("run") / "reconstructor"
COARSE_RECONSTRUCTOR_RUN_PATH = Path("run") / "coarse_reconstructor"
EVALUATION_BUNDLE_CASES = (
    (0, 10),
    (1, 50),
    (2, 100),
    (3, 200),
)
EVALUATION_BUNDLE_SAMPLE_COUNTS = tuple(
    sample_count for _, sample_count in EVALUATION_BUNDLE_CASES
)
