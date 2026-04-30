from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class SourcePaths:
    # Generic ranker, current local adapter defaults to Boskuu's live faucet dataset.
    source_root: Path = Path(os.environ.get("ANTIBOT_RANKER_SOURCE_ROOT", PROJECTS_ROOT / "claimcoin-autoclaim"))
    solver_root: Path = Path(os.environ.get("ANTIBOT_RANKER_SOLVER_ROOT", PROJECTS_ROOT / "antibot-image-solver"))
    db_name: str = os.environ.get("ANTIBOT_RANKER_DB_NAME", "claimcoin.sqlite3")
    case_prefix: str = os.environ.get("ANTIBOT_RANKER_CASE_PREFIX", "claimcoin")

    @property
    def db_path(self) -> Path:
        return self.source_root / "state" / self.db_name

    @property
    def label_root(self) -> Path:
        return self.solver_root / "state" / "antibot-labeling"
