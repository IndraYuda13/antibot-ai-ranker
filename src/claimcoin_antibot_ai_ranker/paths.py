from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class SourcePaths:
    claimcoin_root: Path = PROJECTS_ROOT / "claimcoin-autoclaim"
    solver_root: Path = PROJECTS_ROOT / "antibot-image-solver"

    @property
    def db_path(self) -> Path:
        return self.claimcoin_root / "state" / "claimcoin.sqlite3"

    @property
    def label_root(self) -> Path:
        return self.solver_root / "state" / "antibot-labeling"
