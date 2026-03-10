from __future__ import annotations

from pathlib import Path


def ensure_parent(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
