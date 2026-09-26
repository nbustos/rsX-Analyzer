"""Read node labels and network assignments from XCP-D atlas TSV sidecars."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AtlasMetadata:
    """Atlas nodes in the order defined by the TSV's integer ``index``."""

    names: tuple[str, ...]
    indices: tuple[int, ...]
    networks: dict[str, tuple[str | None, ...]]


def load_atlas_tsv(path: str | Path) -> AtlasMetadata:
    """Load an XCP-D-compatible atlas TSV or an AtlasPack 4S TSV.

    Node names are read from ``name`` (XCP-D schema) or ``label`` (AtlasPack
    schema). Columns whose names identify network assignments are retained
    when present; missing values remain unassigned.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"Atlas TSV has no header: {path}")

        name_column = next((column for column in ("name", "label") if column in fields), None)
        if "index" not in fields or name_column is None:
            raise ValueError(
                f"Atlas TSV must contain 'index' and either 'name' or 'label': {path}"
            )

        records = []
        for line_number, row in enumerate(reader, start=2):
            try:
                index = int(row["index"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid atlas index at {path}:{line_number}: {row.get('index')!r}"
                ) from error
            name = (row.get(name_column) or "").strip()
            if not name:
                raise ValueError(f"Missing node name at {path}:{line_number}")
            records.append((index, name, row))

    if not records:
        raise ValueError(f"Atlas TSV contains no nodes: {path}")

    records.sort(key=lambda item: item[0])
    indices = tuple(record[0] for record in records)
    if len(set(indices)) != len(indices):
        raise ValueError(f"Atlas TSV contains duplicate indices: {path}")

    network_columns = [
        column for column in fields
        if column.casefold().startswith("network_label")
        or column.casefold() == "network_id"
    ]
    networks = {
        column: tuple((row.get(column) or "").strip() or None for _, _, row in records)
        for column in network_columns
    }
    return AtlasMetadata(
        names=tuple(record[1] for record in records),
        indices=indices,
        networks=networks,
    )
