import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from ninflam.cli import load_config
from ninflam.pipeline.train import run_training_pipeline


if __name__ == "__main__":
    run_training_pipeline(load_config())
