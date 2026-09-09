import argparse
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from ninflam.core.logging_config import configure_logging
from ninflam.pipeline.train import run_training_pipeline
from ninflam.pre_pipeline import NInflamPipeline


def load_config(config_dir: Optional[Path] = None):
    root = Path(__file__).resolve().parents[2]
    config_dir = config_dir or root / "configs"

    cfg = OmegaConf.load(config_dir / "config.yaml")
    cfg.data = OmegaConf.load(config_dir / "data" / "data.yaml")
    cfg.split = OmegaConf.load(config_dir / "split" / "split.yaml")
    cfg.preprocessing = OmegaConf.load(config_dir / "preprocessing" / "preprocessing.yaml")
    cfg.io = OmegaConf.load(config_dir / "io" / "io.yaml")
    return cfg


def run_preprocessing_pipeline(cfg=None) -> None:
    cfg = cfg or load_config()
    configure_logging(str(cfg.get("log_level", "INFO")))
    NInflamPipeline(cfg).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro-inflammation package pipeline")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train"],
        default="preprocess",
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    cfg = load_config()
    configure_logging(str(cfg.get("log_level", "INFO")))

    if args.mode == "preprocess":
        NInflamPipeline(cfg).run()
    else:
        run_training_pipeline(cfg)


if __name__ == "__main__":
    main()
