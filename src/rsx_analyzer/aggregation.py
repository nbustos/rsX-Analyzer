"""Pure feature aggregation and robust screening helpers."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd


def average_runs(frame: pd.DataFrame, by: Sequence[str], numeric_only: bool = True) -> pd.DataFrame:
    """Average repeated runs by identifiers, preserving non-numeric identifiers."""
    keys = list(by)
    if numeric_only:
        values = frame.select_dtypes(include="number").columns.difference(keys)
        out = frame.groupby(keys, dropna=False, as_index=False)[list(values)].mean()
        return out
    return frame.groupby(keys, dropna=False, as_index=False).mean(numeric_only=True)


def summarize_multiple_runs(
    merged_runs: pd.DataFrame,
    run_count_column: str = "n_runs_included",
    subject_column: str = "SubjectID",
    session_column: str = "Session",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize subjects with multiple matched runs overall and by session."""
    required = {run_count_column, subject_column, session_column}
    missing = required - set(merged_runs.columns)
    if missing:
        raise KeyError(f"Missing multiple-run summary columns: {sorted(missing)}")
    subject_counts = merged_runs[
        [subject_column, session_column, run_count_column]
    ].copy()
    multiple = subject_counts.loc[subject_counts[run_count_column] >= 2].reset_index(drop=True)
    by_session = subject_counts.groupby(session_column, dropna=False).agg(
        n_subjects=(subject_column, "nunique"),
        n_subjects_with_multiple_runs=(run_count_column, lambda values: int((values >= 2).sum())),
    ).reset_index()
    return by_session, multiple


def weighted_feature_mean(data: pd.DataFrame, feature_weights: Mapping[str, float]) -> pd.Series:
    """Compute row-wise weighted means while ignoring missing feature values."""
    columns = list(feature_weights)
    missing = [c for c in columns if c not in data]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    values, weights = data[columns], pd.Series(feature_weights, dtype=float)
    valid = values.notna()
    denominator = valid.mul(weights, axis="columns").sum(axis=1).replace(0, np.nan)
    return values.fillna(0).mul(weights, axis="columns").sum(axis=1) / denominator


def robust_outlier_mask(values: pd.Series, z_threshold: float = 3.5) -> pd.Series:
    """Flag modified-z-score outliers using median absolute deviation."""
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if not np.isfinite(mad):
        return pd.Series(False, index=values.index)
    if mad == 0:
        return numeric.ne(median).fillna(False)
    return ((0.67448975 * (numeric - median).abs() / mad) > z_threshold).fillna(False)


def robust_outlier_screen(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    z_threshold: float = 3.5,
    *,
    subject_column: str = "SubjectID",
    session_column: str = "Session",
) -> dict[str, pd.DataFrame]:
    """Return notebook-compatible robust scores, flags, and summary reports.

    MAD is preferred; when it is zero or unavailable, the IQR/1.349 scale is
    used. Features with no positive finite scale are reported as unscorable.
    """
    columns = list(feature_columns)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    values = frame[columns].replace([np.inf, -np.inf], np.nan).apply(
        pd.to_numeric, errors="coerce"
    )
    medians = values.median()
    mads = values.sub(medians, axis="columns").abs().median()
    iqrs = values.quantile(0.75) - values.quantile(0.25)
    scales = mads.copy()
    factors = pd.Series(0.6745, index=columns, dtype=float)
    fallback = scales.eq(0) | scales.isna()
    scales.loc[fallback] = iqrs.loc[fallback] / 1.349
    factors.loc[fallback] = 1.0
    scorable = scales.index[scales.gt(0) & np.isfinite(scales)].tolist()
    unscorable = sorted(set(columns) - set(scorable))
    scores = values[scorable].sub(medians[scorable], axis="columns").div(
        scales[scorable], axis="columns"
    ).mul(factors[scorable], axis="columns")
    mask = scores.abs().gt(z_threshold)
    summary = pd.DataFrame(
        {
            "feature": scorable,
            "n_outlier_rows": mask.sum().to_numpy(),
            "n_values": values[scorable].notna().sum().to_numpy(),
        }
    )
    summary["percent_outlier"] = 100 * summary["n_outlier_rows"] / summary["n_values"]
    summary = summary.sort_values(
        ["n_outlier_rows", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    records: list[dict[str, object]] = []
    for row, column in zip(*np.where(mask.to_numpy())):
        record = {
            subject_column: frame.iloc[row][subject_column]
            if subject_column in frame
            else row,
            session_column: frame.iloc[row][session_column]
            if session_column in frame
            else None,
            "feature": scorable[column],
            "value": values.iloc[row][scorable[column]],
            "robust_z": scores.iloc[row, column],
        }
        records.append(record)
    flags = pd.DataFrame(
        records,
        columns=[subject_column, session_column, "feature", "value", "robust_z"],
    )
    if flags.empty:
        subjects = pd.DataFrame(
            columns=[
                subject_column, "n_flagged_sessions", "flagged_sessions",
                "n_flagged_features", "n_flagged_values", "max_abs_robust_z",
                "flagged_features",
            ]
        )
    else:
        subjects = flags.groupby(subject_column).agg(
            n_flagged_sessions=(session_column, "nunique"),
            flagged_sessions=(session_column, lambda x: ", ".join(sorted(x.astype(str).unique()))),
            n_flagged_features=("feature", "nunique"),
            n_flagged_values=("feature", "size"),
            max_abs_robust_z=("robust_z", lambda x: x.abs().max()),
            flagged_features=("feature", lambda x: ", ".join(sorted(x.unique()))),
        ).reset_index().sort_values(
            ["n_flagged_values", "max_abs_robust_z", subject_column],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return {
        "scores": scores,
        "mask": mask,
        "feature_summary": summary,
        "feature_flags": flags,
        "subject_report": subjects,
        "unscorable": pd.DataFrame({"unscorable_feature": unscorable}),
    }


def screen_outliers(frame: pd.DataFrame, columns: Sequence[str], z_threshold: float = 3.5) -> pd.DataFrame:
    """Return a boolean outlier table, one column per feature."""
    return pd.DataFrame({column: robust_outlier_mask(frame[column], z_threshold) for column in columns},
                        index=frame.index)


def macro_network_feature(data: pd.DataFrame, subnetworks_a: Sequence[str],
                          subnetworks_b: Sequence[str] | None = None,
                          network_counts: Mapping[str, int] | None = None,
                          suffix: str = "_19net") -> pd.Series:
    """Collapse network block features using parcel-pair weights."""
    b = list(subnetworks_a if subnetworks_b is None else subnetworks_b)
    a = list(subnetworks_a)
    counts = network_counts or {name: 1 for name in set(a + b)}
    weights: dict[str, int] = {}
    within = a == b
    for i, name_a in enumerate(a):
        candidates = b[i:] if within else b
        for name_b in candidates:
            if within and name_a == name_b:
                feature = f"{name_a}_within{suffix}"
                weight = counts[name_a] * (counts[name_a] - 1)
            else:
                first, second = sorted((name_a, name_b))
                feature = f"{first}_X_{second}{suffix}"
                weight = counts[name_a] * counts[name_b]
            weights[feature] = weight
    return weighted_feature_mean(data, weights)


def triple_network_summary(
    data: pd.DataFrame,
    network19_counts: Mapping[str, int],
    systems: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    network9_suffix: str = "_9net",
    network19_suffix: str = "_19net",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form weighted 9/19-network triple-system values and summary."""
    labels = list(systems)
    records: list[dict[str, object]] = []
    for label, mapping in systems.items():
        for network in mapping["network9"]:
            records.append({"partition": "9-network", "feature": f"{label} within",
                            "value": data[f"{network}_within{network9_suffix}"]})
        records.append({"partition": "19-network", "feature": f"{label} within",
                        "value": macro_network_feature(
                            data, mapping["network19"], network_counts=network19_counts,
                            suffix=network19_suffix,
                        )})
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            left9, right9 = systems[left]["network9"][0], systems[right]["network9"][0]
            first, second = sorted((left9, right9))
            records.append({"partition": "9-network", "feature": f"{left} x {right}",
                            "value": data[f"{first}_X_{second}{network9_suffix}"]})
            records.append({"partition": "19-network", "feature": f"{left} x {right}",
                            "value": macro_network_feature(
                                data, systems[left]["network19"],
                                systems[right]["network19"],
                                network19_counts,
                                network19_suffix,
                            )})
    rows: list[pd.DataFrame] = []
    for record in records:
        value = record["value"]
        series = value.reset_index(drop=True) if isinstance(value, pd.Series) else pd.Series(value)
        rows.append(pd.DataFrame({
            "partition": record["partition"],
            "feature": record["feature"],
            "value": series,
        }))
    long = pd.concat(rows, ignore_index=True)
    summary = long.groupby(["partition", "feature"])["value"].agg(
        n="count", mean="mean", median="median", std="std"
    ).reset_index()
    return long, summary
