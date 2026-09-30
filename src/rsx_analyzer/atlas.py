"""Read node labels and network assignments from XCP-D atlas TSV sidecars."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Mapping


def _normalize_assignment(value: str | None) -> str | None:
    assignment = (value or "").strip()
    return None if assignment.casefold() in {"", "n/a", "na", "none"} else assignment


@dataclass(frozen=True)
class AtlasMetadata:
    """Atlas nodes in the order defined by the TSV's integer ``index``."""

    names: tuple[str, ...]
    indices: tuple[int, ...]
    networks: dict[str, tuple[str | None, ...]]


def derive_network_assignments(
    atlas: AtlasMetadata,
    source_column: str,
    atlas_names: Mapping[str, str],
    assignments: Mapping[str, str],
    target_column: str | None = None,
) -> AtlasMetadata:
    """Return an atlas with derived assignments for named parcels.

    Existing non-null assignments are retained; ``assignments`` is applied to
    parcels identified by ``atlas_names`` (for example subcortex/cerebellum).
    """
    if source_column not in atlas.networks:
        raise KeyError(f"Unknown atlas network column: {source_column}")
    target = target_column or source_column
    values = list(atlas.networks[source_column])
    for i, name in enumerate(atlas.names):
        category = atlas_names.get(name)
        if category in assignments:
            values[i] = assignments[category]
    networks = dict(atlas.networks)
    networks[target] = tuple(values)
    return replace(atlas, networks=networks)


assign_derived_networks = derive_network_assignments


def load_atlas_tsv(path: str | Path) -> AtlasMetadata:
    """Load an XCP-D-compatible atlas TSV or an AtlasPack 4S TSV.

    Node names are read from ``name`` (XCP-D schema) or ``label`` (AtlasPack
    schema).     Network and community assignment columns are retained when present;
    blank and ``n/a`` assignments remain unassigned.
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
        if column.casefold().startswith(("network_label", "community"))
    ]
    networks = {
        column: tuple(
            _normalize_assignment(row.get(column))
            for _, _, row in records
        )
        for column in network_columns
    }
    return AtlasMetadata(
        names=tuple(record[1] for record in records),
        indices=indices,
        networks=networks,
    )
