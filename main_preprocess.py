import sys
from pathlib import Path

# Get project root
ROOT = Path(__file__).resolve().parent

# Add src to sys.path BEFORE any imports
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ninflam.cli import load_config, run_preprocessing_pipeline

if __name__ == "__main__":
    run_preprocessing_pipeline(load_config())
