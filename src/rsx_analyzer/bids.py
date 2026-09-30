"""Small, typed helpers for discovering and matching BIDS-like derivatives."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

ENTITY_PATTERN = re.compile(r"(?P<key>sub|ses|task|run)-(?P<value>[^_]+)")
RUN_KEY = ("subject", "session", "task", "run")


def parse_bids_entities(filename: str | Path) -> dict[str, str]:
    """Parse subject/session/task/run, representing absent ses/run as ``n/a``."""
    entities = dict(ENTITY_PATTERN.findall(Path(filename).name))
    if "sub" not in entities or "task" not in entities:
        raise ValueError(f"Filename must contain subject and task entities: {filename}")
    return {"subject": entities["sub"], "session": entities.get("ses", "n/a"),
            "task": entities["task"], "run": entities.get("run", "n/a")}


def discover_files(
    folder: str | Path, pattern: str, task_filter: str | None = None
) -> pd.DataFrame:
    """Discover unique derivative files and return their BIDS entity table."""
    paths = sorted(Path(folder).rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {folder}")
    records: list[dict[str, Any]] = []
    for path in paths:
        entities = parse_bids_entities(path.name)
        if task_filter is not None and entities["task"].casefold() != task_filter.casefold():
            continue
        records.append({**entities, "file_path": path})
    if not records:
        raise FileNotFoundError(f"No {pattern!r} files for task={task_filter!r} under {folder}")
    frame = pd.DataFrame(records)
    duplicates = frame.duplicated(list(RUN_KEY), keep=False)
    if duplicates.any():
        raise ValueError("Multiple files share a subject/session/task/run key:\n" +
                         frame.loc[duplicates, list(RUN_KEY) + ["file_path"]].to_string(index=False))
    return frame


def run_key(record: dict[str, Any] | pd.Series) -> tuple[str, str, str, str]:
    """Return the canonical key used to match connectivity and motion runs."""
    return tuple(str(record[column]) for column in RUN_KEY)  # type: ignore[return-value]
