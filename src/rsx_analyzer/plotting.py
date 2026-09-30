"""Optional, small plotting primitives for notebook reports."""
from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import pandas as pd


def plot_threshold_comparison(
    frame: pd.DataFrame, x: str = "fd_threshold", y: str = "remaining_seconds",
    *, ax: object | None = None, **kwargs: object
) -> object:
    """Plot remaining-duration distributions and run points by FD threshold."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_threshold_comparison requires 'matplotlib'") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    groups = [
        (threshold, values.dropna().to_numpy(dtype=float))
        for threshold, values in frame.groupby(x, sort=True)[y]
    ]
    groups = [(threshold, values) for threshold, values in groups if values.size]
    if not groups:
        raise ValueError("No non-missing duration data are available to plot")
    positions = np.arange(1, len(groups) + 1)
    ax.boxplot(
        [values for _, values in groups],
        positions=positions,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#a9c7df", "edgecolor": "#315f7d"},
        medianprops={"color": "#24465c", "linewidth": 1.5},
    )
    for position, (_, values) in zip(positions, groups):
        jitter = np.linspace(-0.18, 0.18, len(values))
        ax.scatter(position + jitter, values, color="#29465b", alpha=0.2, s=10, **kwargs)
    ax.set_xticks(positions, [str(threshold) for threshold, _ in groups])
    return ax


def plot_feature_distribution(frame: pd.DataFrame, columns: Sequence[str],
                              *, ax: object | None = None) -> object:
    """Plot feature distributions for quick QC."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_feature_distribution requires 'matplotlib'") from exc
    if ax is None:
        _, ax = plt.subplots()
    frame[list(columns)].plot(kind="box", ax=ax)
    return ax


def plot_feature_motion_associations(
    associations: pd.DataFrame, *, top_n: int = 15, ax: object | None = None
) -> object:
    """Plot the strongest feature/remaining-duration rank associations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_feature_motion_associations requires 'matplotlib'") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    selected = associations.head(top_n).sort_values("spearman_rho")
    ax.barh(selected["feature"], selected["spearman_rho"], color="#6289a8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman correlation with remaining seconds")
    ax.set_ylabel("Connectivity feature")
    return ax


def plot_triple_networks(
    values: pd.DataFrame, *, ax: object | None = None,
    order: Sequence[str] | None = None,
) -> object:
    """Plot long-form 9/19-network triple-system values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_triple_networks requires 'matplotlib'") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    order = list(order) if order is not None else list(values["feature"].unique())
    for offset, (partition, group) in enumerate(values.groupby("partition", sort=False)):
        positions = [order.index(item) + (offset - 0.5) * 0.22 for item in group["feature"]]
        ax.scatter(positions, group["value"], label=partition, alpha=0.65)
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
    ax.set_ylabel("Connectivity")
    ax.legend()
    return ax


def plot_outlier_summary(
    feature_summary: pd.DataFrame,
    subject_report: pd.DataFrame,
    top_n: int = 20,
    axes: Sequence[object] | None = None,
) -> tuple[object, object]:
    """Plot the two outlier-screening bar charts side by side.

    Left: features with the most robust outlier flags (``n_outlier_rows``),
    restricted to features with at least one flag. Right: subjects with the
    most flagged connectivity-feature values (``n_flagged_values``). Mirrors
    the notebook's outlier-screening report plots.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_outlier_summary requires 'matplotlib'") from exc
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(16, 7))
    feature_ax, subject_ax = axes

    top_features = feature_summary.loc[feature_summary["n_outlier_rows"] > 0].head(top_n)
    top_features = top_features.sort_values("n_outlier_rows")
    feature_ax.barh(top_features["feature"], top_features["n_outlier_rows"], color="#c27a62")
    feature_ax.set_xlabel("Flagged subject/session values")
    feature_ax.set_ylabel("Connectivity feature")
    feature_ax.set_title("Features with the most robust outlier flags")

    top_subjects = subject_report.head(top_n).sort_values("n_flagged_values")
    subject_column = "SubjectID" if "SubjectID" in top_subjects else top_subjects.columns[0]
    subject_ax.barh(top_subjects[subject_column], top_subjects["n_flagged_values"], color="#6289a8")
    subject_ax.set_xlabel("Flagged feature values across sessions")
    subject_ax.set_ylabel("Subject ID")
    subject_ax.set_title("Subjects with the most connectivity-feature outliers")

    return feature_ax, subject_ax
