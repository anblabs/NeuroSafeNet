from pathlib import Path

class ArtifactManager:
    def __init__(self, outdir: str):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> str:
        return str(self.outdir / name)
