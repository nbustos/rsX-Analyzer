"""Optional HDF5 motion-QC readers and inventories."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

MOTION_FIELDS = ("remaining_frame_mean_FD", "remaining_seconds",
                 "remaining_total_frame_count", "skip", "threshold", "total_frame_count")


def read_motion_metrics(hdf5_path: str | Path, fd_threshold: float) -> dict[str, float]:
    """Read and validate scalar DCAN motion metrics for one FD threshold."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("read_motion_metrics requires optional dependency 'h5py'") from exc
    path = Path(hdf5_path)
    group_path = f"dcan_motion/fd_{fd_threshold:g}"
    with h5py.File(path, "r") as handle:
        if group_path not in handle:
            raise KeyError(f"Missing {group_path!r} in motion file {path}")
        group = handle[group_path]
        missing = [field for field in MOTION_FIELDS if field not in group]
        if missing:
            raise KeyError(f"Missing {missing} in {path}:{group_path}")
        values: dict[str, float] = {}
        for field in MOTION_FIELDS:
            value = np.asarray(group[field][()])
            if value.size != 1:
                raise ValueError(f"Expected scalar {field!r} in {path}:{group_path}")
            values[field] = float(value.reshape(-1)[0])
    if not np.isclose(values["threshold"], fd_threshold):
        raise ValueError(f"Stored FD threshold does not match {fd_threshold:g} in {path}:{group_path}")
    total, remaining = values["total_frame_count"], values["remaining_total_frame_count"]
    if total < 0 or remaining < 0 or remaining > total:
        raise ValueError(f"Invalid frame counts in {path}:{group_path}")
    return {"fd_threshold": values["threshold"], "mean_fd_remaining": values["remaining_frame_mean_FD"],
            "seconds_remaining": values["remaining_seconds"], "frames_remaining": remaining,
            "frames_total": total, "frames_censored": total - remaining,
            "fraction_remaining": remaining / total if total else np.nan, "skip": values["skip"]}


def inventory_hdf5(path: str | Path) -> dict[str, Any]:
    """Return a lightweight dataset/group inventory without loading arrays."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("inventory_hdf5 requires optional dependency 'h5py'") from exc
    groups: list[str] = []
    datasets: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group):
                groups.append(name)
            else:
                datasets[name] = tuple(obj.shape)
        handle.visititems(visit)
    return {"path": Path(path), "groups": tuple(groups), "datasets": datasets}


inventory_motion_file = inventory_hdf5


def summarize_motion_thresholds(
    paths: list[str | Path], thresholds: tuple[float, ...] = (0.3, 0.4, 0.5)
) -> pd.DataFrame:
    """Read each file at each threshold, returning one row per file/threshold."""
    rows = []
    for path in paths:
        for threshold in thresholds:
            try:
                rows.append({"file_path": Path(path), **read_motion_metrics(path, threshold)})
            except (OSError, KeyError, ValueError) as exc:
                rows.append({"file_path": Path(path), "fd_threshold": threshold,
                             "error": str(exc)})
    return pd.DataFrame(rows)


def inventory_motion_files(paths: list[str | Path]) -> pd.DataFrame:
    """Inventory HDF5 files while retaining unreadable-file errors."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            inventory = inventory_hdf5(path)
            rows.append({"file_path": Path(path), "readable": True,
                         "groups": inventory["groups"], "datasets": inventory["datasets"],
                         "error": None})
        except OSError as exc:
            rows.append({"file_path": Path(path), "readable": False,
                         "groups": (), "datasets": {}, "error": str(exc)})
    return pd.DataFrame(rows)
