"""Run matching, missingness, and QC summary tables."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import pandas as pd
from .bids import RUN_KEY


def match_runs(connectivity: pd.DataFrame, motion: pd.DataFrame) -> pd.DataFrame:
    """Outer-join run tables on subject/session/task/run and expose missingness."""
    left, right = connectivity.copy(), motion.copy()
    merged = left.merge(right, on=list(RUN_KEY), how="outer", suffixes=("_conn", "_motion"), indicator=True)
    merged["connectivity_missing"] = merged["_merge"].eq("right_only")
    merged["motion_missing"] = merged["_merge"].eq("left_only")
    return merged.drop(columns="_merge")


def missingness_summary(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Summarize missing counts and fractions for selected columns."""
    cols = list(columns) if columns is not None else list(frame.columns)
    return pd.DataFrame({"column": cols,
                         "missing_count": [int(frame[c].isna().sum()) for c in cols],
                         "missing_fraction": [float(frame[c].isna().mean()) for c in cols]})


def threshold_summary(frame: pd.DataFrame, threshold_column: str = "fd_threshold",
                      value_column: str = "seconds_remaining") -> pd.DataFrame:
    """Aggregate retained duration by motion threshold."""
    return (frame.groupby(threshold_column, dropna=False)[value_column]
            .agg(["count", "mean", "median", "min", "max"]).reset_index())


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize connectivity/motion coverage by session and run."""
    required = set(RUN_KEY) | {"connectivity_present", "motion_present", "motion_readable"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing coverage columns: {sorted(missing)}")
    rows = []
    for (session, run), group in frame.groupby(["session", "run"], sort=True, dropna=False):
        conn, motion, readable = (
            group["connectivity_present"], group["motion_present"], group["motion_readable"]
        )
        rows.append({
            "session": session, "run": run,
            "n_connectivity_subjects": group.loc[conn, "subject"].nunique(),
            "n_motion_subjects": group.loc[motion, "subject"].nunique(),
            "n_subjects_with_both_files": group.loc[conn & motion, "subject"].nunique(),
            "n_subjects_with_both_readable": group.loc[conn & readable, "subject"].nunique(),
            "n_connectivity_only": group.loc[conn & ~motion, "subject"].nunique(),
            "n_motion_only": group.loc[~conn & motion, "subject"].nunique(),
            "n_motion_unreadable": group.loc[motion & ~readable, "subject"].nunique(),
        })
    return pd.DataFrame(rows)


def feature_motion_associations(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    motion_column: str = "remaining_seconds",
) -> pd.DataFrame:
    """Calculate Pearson/Spearman run-level associations and FDR q-values."""
    try:
        from scipy.stats import pearsonr, spearmanr
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:
        raise ImportError(
            "feature_motion_associations requires optional dependencies 'scipy' and 'statsmodels'"
        ) from exc
    rows = []
    for feature in feature_columns:
        paired = frame[[motion_column, feature]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(paired) < 3 or paired.nunique().min() < 2:
            continue
        pearson_r, pearson_p = pearsonr(paired[motion_column], paired[feature])
        spearman_rho, spearman_p = spearmanr(paired[motion_column], paired[feature])
        rows.append({"feature": feature, "n_runs": len(paired),
                     "pearson_r": pearson_r, "pearson_p": pearson_p,
                     "spearman_rho": spearman_rho, "spearman_p": spearman_p})
    if not rows:
        raise ValueError("No variable connectivity features could be tested against remaining seconds")
    result = pd.DataFrame(rows)
    result["spearman_q"] = multipletests(result["spearman_p"], method="fdr_bh")[1]
    result["abs_spearman_rho"] = result["spearman_rho"].abs()
    return result.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)
