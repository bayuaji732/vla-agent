from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from agent.config import get_settings
from agent.models import Trajectory

logger = logging.getLogger(__name__)
cfg = get_settings()


class TrajectoryLogger:
    def __init__(self) -> None:
        self._dir = Path(cfg.trajectory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, trajectory: Trajectory) -> Path:
        filename = (
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
            f"{trajectory.status.value}_"
            f"{trajectory.id[:8]}.json"
        )
        path = self._dir / filename
        path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Trajectory saved → %s", path)
        return path

    def load(self, path: str | Path) -> Trajectory:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Trajectory(**data)

    def list_trajectories(self) -> list[Path]:
        return sorted(self._dir.glob("*.json"), reverse=True)