"""Small, typed helpers for discovering and matching BIDS-like derivatives."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

ENTITY_PATTERN = re.compile(r"(?P<key>sub|ses|task|run)-(?P<value>[^_]+)")
EXTENDED_ENTITY_PATTERN = re.compile(
    r"(?P<key>sub|ses|task|run|space|res|desc)-(?P<value>[^_]+)"
)
RUN_KEY = ("subject", "session", "task", "run")


def parse_bids_entities(filename: str | Path) -> dict[str, str]:
    """Parse subject/session/task/run, representing absent ses/run as ``n/a``."""
    entities = dict(ENTITY_PATTERN.findall(Path(filename).name))
    if "sub" not in entities or "task" not in entities:
        raise ValueError(f"Filename must contain subject and task entities: {filename}")
    session = entities.get("ses", "n/a")
    run = entities.get("run", "n/a")
    session = f"{int(session):02d}" if session.isdigit() else session
    run = f"{int(run):02d}" if run.isdigit() else run
    return {
        "subject": entities["sub"],
        "session": session,
        "task": entities["task"],
        "run": run,
    }


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


def discover_linc_qc_files(
    folder: str | Path, task_filter: str | None = None,
) -> pd.DataFrame:
    """Read BIDS entities and motion metrics from LINC QC CSV files.

    CSVs may contain a single run or multiple run rows. Entities can be
    supplied in columns or in the CSV filename; columns take precedence.
    """
    paths = sorted(Path(folder).rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No LINC QC CSV files found under {folder}")

    aliases = {
        "subject": ("sub", "subject", "subject_id", "participant_id"),
        "session": ("ses", "session"),
        "task": ("task",),
        "run": ("run",),
        "space": ("space",),
        "res": ("res", "resolution"),
        "desc": ("desc", "description"),
    }
    metric_columns = ("mean_fd", "num_censored_volumes", "num_retained_volumes")
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            table = pd.read_csv(path, dtype=str)
        except pd.errors.EmptyDataError:
            filename_entities = dict(EXTENDED_ENTITY_PATTERN.findall(path.name))
            if "sub" not in filename_entities or "task" not in filename_entities:
                raise ValueError(
                    f"Empty LINC QC CSV {path} must encode subject and task in its filename"
                )
            task = filename_entities["task"]
            if task_filter is not None and task.casefold() != task_filter.casefold():
                continue
            empty_entities = {
                "subject": filename_entities["sub"],
                "session": filename_entities.get("ses", "n/a"),
                "task": task,
                "run": filename_entities.get("run", "n/a"),
            }
            for entity in ("session", "run"):
                if empty_entities[entity].isdigit():
                    empty_entities[entity] = f"{int(empty_entities[entity]):02d}"
            records.append({
                **empty_entities,
                "sub": f"sub-{empty_entities['subject']}",
                "ses": f"ses-{empty_entities['session']}" if empty_entities["session"] != "n/a" else "n/a",
                "task_entity": f"task-{empty_entities['task']}",
                "run_entity": f"run-{empty_entities['run']}" if empty_entities["run"] != "n/a" else "n/a",
                **{entity: filename_entities[entity] for entity in ("space", "res", "desc")
                   if entity in filename_entities},
                **{column: float("nan") for column in metric_columns},
                "file_path": path,
                "read_error": "Empty LINC QC CSV file",
            })
            continue
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise ValueError(f"Could not read LINC QC CSV {path}: {error}") from error
        normalized_columns = {str(column).strip().casefold(): column for column in table.columns}
        missing_metrics = [
            column for column in metric_columns if column not in normalized_columns
        ]
        if missing_metrics:
            raise ValueError(f"LINC QC CSV {path} is missing columns: {missing_metrics}")
        filename_entities = dict(EXTENDED_ENTITY_PATTERN.findall(path.name))
        for row_number, row in table.iterrows():
            entities: dict[str, str] = {}
            for entity, candidates in aliases.items():
                value: object | None = None
                for candidate in candidates:
                    source_column = normalized_columns.get(candidate)
                    if source_column is not None and pd.notna(row[source_column]):
                        value = row[source_column]
                        break
                if value is None:
                    value = filename_entities.get(
                        {"subject": "sub", "session": "ses"}.get(entity, entity)
                    )
                if value is not None:
                    normalized = str(value).strip()
                    prefix = {
                        "subject": "sub-", "session": "ses-", "task": "task-",
                        "run": "run-", "space": "space-", "res": "res-", "desc": "desc-",
                    }.get(entity)
                    if prefix and normalized.casefold().startswith(prefix):
                        normalized = normalized[len(prefix):]
                    if entity in {"session", "run"} and normalized.isdigit():
                        normalized = f"{int(normalized):02d}"
                    entities[entity] = normalized
            if not entities.get("subject") or not entities.get("task"):
                raise ValueError(
                    f"LINC QC CSV {path}, row {row_number + 2}: subject and task "
                    "must be present in columns or filename"
                )
            entities.setdefault("session", "n/a")
            entities.setdefault("run", "n/a")
            if task_filter is not None and entities["task"].casefold() != task_filter.casefold():
                continue
            metrics: dict[str, object] = {}
            for column in metric_columns:
                source_column = normalized_columns[column]
                try:
                    metrics[column] = float(row[source_column])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"LINC QC CSV {path}, row {row_number + 2}: invalid {column}"
                    ) from error
                if not pd.notna(metrics[column]):
                    raise ValueError(
                        f"LINC QC CSV {path}, row {row_number + 2}: missing {column}"
                    )
            records.append({
                **{column: entities[column] for column in RUN_KEY},
                "sub": f"sub-{entities['subject']}",
                "ses": f"ses-{entities['session']}" if entities["session"] != "n/a" else "n/a",
                "task_entity": f"task-{entities['task']}",
                "run_entity": f"run-{entities['run']}" if entities["run"] != "n/a" else "n/a",
                **{column: entities[column] for column in ("space", "res", "desc") if column in entities},
                **metrics,
                "file_path": path,
                "read_error": None,
            })
    if not records:
        raise FileNotFoundError(f"No LINC QC CSV rows for task={task_filter!r} under {folder}")
    frame = pd.DataFrame(records)
    duplicates = frame.duplicated(list(RUN_KEY), keep=False)
    if duplicates.any():
        raise ValueError(
            "Multiple LINC QC CSV rows share a subject/session/task/run key:\n"
            + frame.loc[duplicates, list(RUN_KEY) + ["file_path"]].to_string(index=False)
        )
    return frame.reset_index(drop=True)


def run_key(record: dict[str, Any] | pd.Series) -> tuple[str, str, str, str]:
    """Return the canonical key used to match connectivity and motion runs."""
    return tuple(str(record[column]) for column in RUN_KEY)  # type: ignore[return-value]
