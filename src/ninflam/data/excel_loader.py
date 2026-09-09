import os
import logging
import pandas as pd
from pathlib import Path
from ninflam.core.exceptions import DataRootError
from ninflam.data.loader_base import BaseLoader

log = logging.getLogger(__name__)

class ExcelLoader(BaseLoader):
    def __init__(self, file_name: str, sheet: int):
        self.file_name = file_name
        self.sheet = sheet

    def load(self) -> pd.DataFrame:
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[3]
        data_root = ROOT / "Data"
        path = data_root / self.file_name

        log.info("Loading Excel: %s (sheet=%s)", path, self.sheet)
        return pd.read_excel(path, sheet_name=self.sheet)
