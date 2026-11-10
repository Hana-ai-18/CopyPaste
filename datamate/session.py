"""Helpers for hosted notebooks with a hard session limit (Kaggle: 12 h) - nothing here touches the model or the answers."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

RUN_DIRS = ("public_test_out", "train_eval", "exemplar_run")


def remaining_budget_s(session_hours: float, started_at: float, reserve_s: float = 1800.0, now: float | None = None, floor_s: float = 600.0) -> int:
    """Seconds the agent may still work so that it ends GRACEFULLY (merge + save) before the session is killed."""
    left = session_hours * 3600 - ((now if now is not None else time.time()) - started_at) - reserve_s
    return int(max(floor_s, left))


def import_previous_outputs(input_root: str | Path, work: str | Path) -> list[str]:
    """A new session starts with an empty /kaggle/working. If the previous notebook's output is attached as input data
    (Add Data -> Notebook Output), copy its run folders back so that --resume continues where it stopped. Never overwrites."""
    root, work = Path(input_root), Path(work)
    copied: list[str] = []
    if not root.exists():
        return copied
    for name in RUN_DIRS:
        for src in root.rglob(name):
            if not src.is_dir() or src.parent.name in RUN_DIRS:
                continue
            dst = work / name
            if dst.exists():
                continue
            shutil.copytree(src, dst)
            copied.append(f"{src} -> {dst}")
            break
    return copied
