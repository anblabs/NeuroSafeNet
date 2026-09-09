from ninflam.cli import load_config
from ninflam.pre_pipeline import NInflamPipeline


def run_preprocessing_pipeline(cfg=None) -> None:
    cfg = cfg or load_config()
    NInflamPipeline(cfg).run()

