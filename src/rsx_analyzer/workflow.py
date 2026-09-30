"""Compact orchestration of the rs-X1 discovery-to-master-dataframe workflow.

These functions compose the smaller, pure building blocks in
:mod:`rsx_analyzer` (BIDS discovery, atlas network derivation, connectivity
feature extraction, motion QC readout, and run averaging) into the exact
sequence of steps used by ``rs-X1.ipynb``. Everything here takes explicit
paths/dataframes and returns typed dataclasses; no notebook globals are
imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .atlas import AtlasMetadata, load_atlas_tsv
from .bids import RUN_KEY, discover_files
from .connectivity import summarize_connectivity_partitions
from .motion import read_motion_metrics
from .runs import coverage_summary as _coverage_summary, feature_motion_associations
from .runs import threshold_summary as _threshold_summary
from .aggregation import average_runs

KEY_COLUMNS: tuple[str, ...] = RUN_KEY

DEFAULT_ROI_SEEDS: tuple[str, ...] = (
    "LH_Hippocampus", "RH_Hippocampus", "LH_Amygdala", "RH_Amygdala",
)
DEFAULT_BILATERAL_PAIRS: dict[str, tuple[str, str]] = {
    "Bilateral_Hippocampus": ("LH_Hippocampus", "RH_Hippocampus"),
    "Bilateral_Amygdala": ("LH_Amygdala", "RH_Amygdala"),
}
DEFAULT_NETWORK_COLUMNS: dict[str, tuple[str, str]] = {
    "network_label_9network": ("network_label", "_9net"),
    "network_label_19network": ("network_label_17network", "_19net"),
}
DEFAULT_SUBCORTICAL_ATLAS_NAMES: frozenset[str] = frozenset(
    {"CIT168Subcortical", "ThalamusHCP", "SubcorticalHCP"}
)
DEFAULT_MOTION_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5)
MOTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "fd_threshold", "mean_fd_remaining", "frames_remaining", "frames_total",
    "frames_censored", "fraction_remaining", "skip",
)


@dataclass(frozen=True)
class AtlasNetworkConfig:
    """Rules for deriving 9/19-network (or other) partitions from a 4S atlas."""

    network_columns: Mapping[str, tuple[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_NETWORK_COLUMNS)
    )
    roi_seeds: tuple[str, ...] = DEFAULT_ROI_SEEDS
    bilateral_pairs: Mapping[str, tuple[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_BILATERAL_PAIRS)
    )
    subcortical_atlas_names: frozenset[str] = DEFAULT_SUBCORTICAL_ATLAS_NAMES
    cerebellum_atlas_name: str = "Cerebellum"


@dataclass(frozen=True)
class DiscoveryResult:
    """BIDS entity tables for discovered connectivity matrices and motion files."""

    conn_files: pd.DataFrame
    motion_files: pd.DataFrame


@dataclass(frozen=True)
class ConnectivityLoadResult:
    """Run-level connectivity feature table and the computed feature names."""

    connectivity_df: pd.DataFrame
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class MotionLoadResult:
    """Readable motion metrics plus an explicit inventory of unreadable files."""

    motion_df: pd.DataFrame
    motion_load_errors: pd.DataFrame


@dataclass(frozen=True)
class RunLevelResult:
    """File-coverage inventory, missingness report, and per-session/run coverage."""

    run_level_df: pd.DataFrame
    missingness_report: pd.DataFrame
    coverage_summary: pd.DataFrame


@dataclass(frozen=True)
class MasterDatasetResult:
    """Matched, readable subject/session/run feature table (``master_df``)."""

    master_df: pd.DataFrame
    feature_columns: tuple[str, ...]
    motion_feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowResult:
    """Everything produced by :func:`run_workflow`, mirroring the notebook."""

    atlas: AtlasMetadata
    discovery: DiscoveryResult
    connectivity: ConnectivityLoadResult
    motion: MotionLoadResult
    run_level: RunLevelResult
    motion_threshold_data: pd.DataFrame
    threshold_table: pd.DataFrame
    master: MasterDatasetResult
    motion_associations: pd.DataFrame
    merged_runs: pd.DataFrame


def derive_analysis_atlas(
    atlas_tsv_path: str | Path, config: AtlasNetworkConfig = AtlasNetworkConfig(),
) -> AtlasMetadata:
    """Load a 4S atlas TSV and fill subcortex/cerebellum parcels for each partition.

    Requires an ``atlas_name`` column and validates that every parcel receives
    a network assignment for each configured partition, matching the
    notebook's 9/19-network derivation.
    """
    path = Path(atlas_tsv_path)
    atlas = load_atlas_tsv(path)
    atlas_table = pd.read_csv(path, sep="\t").sort_values("index")
    if tuple(atlas_table["label"].astype(str)) != atlas.names:
        raise ValueError(f"Atlas TSV node labels/order do not match {path}")
    if "atlas_name" not in atlas_table.columns:
        raise ValueError(
            "A Schaefer 4S atlas TSV with an atlas_name column is required for network features"
        )
    atlas_names = atlas_table["atlas_name"].astype(str)
    derived: dict[str, tuple[str | None, ...]] = {}
    for output_column, (source_column, _) in config.network_columns.items():
        if source_column not in atlas.networks:
            raise ValueError(f"Atlas is missing required network assignment {source_column!r}")
        source_assignments = atlas.networks[source_column]
        assignments: list[str | None] = []
        for assignment, atlas_name in zip(source_assignments, atlas_names):
            if assignment is not None:
                assignments.append(assignment)
            elif atlas_name in config.subcortical_atlas_names:
                assignments.append("Subcort")
            elif atlas_name == config.cerebellum_atlas_name:
                assignments.append("Cerebellum")
            else:
                assignments.append(None)
        if any(assignment is None for assignment in assignments):
            raise ValueError(f"Could not assign every parcel in {output_column}")
        derived[output_column] = tuple(assignments)
    return replace(atlas, networks={**atlas.networks, **derived})


def discover_run_files(
    conn_mats_path: str | Path, motion_path: str | Path, task_filter: str | None = "rest",
) -> DiscoveryResult:
    """Discover connectivity matrices and motion HDF5 files by BIDS entities."""
    conn_files = discover_files(conn_mats_path, "*_conmat.tsv", task_filter)
    motion_files = discover_files(motion_path, "*.hdf5", task_filter)
    return DiscoveryResult(conn_files=conn_files, motion_files=motion_files)


def load_connectivity_features(
    conn_files: pd.DataFrame,
    atlas: AtlasMetadata,
    config: AtlasNetworkConfig = AtlasNetworkConfig(),
) -> ConnectivityLoadResult:
    """Read each connectivity matrix and compute its network/ROI feature row.

    Raises if a matrix's parcel labels or order do not exactly match ``atlas``.
    """
    # config.network_columns maps derived output column -> (source column,
    # suffix); connectivity summaries must read the *derived* (fully filled)
    # output column, so build a column->(column, suffix) mapping for it.
    feature_network_columns = {
        output_column: (output_column, suffix)
        for output_column, (_, suffix) in config.network_columns.items()
    }
    rows: list[dict[str, object]] = []
    for record in conn_files.to_dict("records"):
        matrix_df = pd.read_csv(record["file_path"], sep="\t", index_col=0)
        matrix_names = tuple(map(str, matrix_df.columns))
        row_names = tuple(map(str, matrix_df.index))
        if matrix_names != atlas.names or row_names != atlas.names:
            raise ValueError(
                f"Matrix parcel labels/order do not match the atlas: {record['file_path']}"
            )
        matrix = matrix_df.to_numpy(dtype=float)
        features = summarize_connectivity_partitions(
            matrix, atlas, feature_network_columns,
            roi_names=config.roi_seeds, bilateral_pairs=config.bilateral_pairs,
        )
        rows.append({
            **{column: record[column] for column in KEY_COLUMNS},
            "conn_mat_path": str(record["file_path"]),
            **features,
        })
    connectivity_df = pd.DataFrame(rows)
    feature_columns = tuple(
        column for column in connectivity_df.columns
        if column not in (*KEY_COLUMNS, "conn_mat_path")
    )
    return ConnectivityLoadResult(connectivity_df=connectivity_df, feature_columns=feature_columns)


def load_motion_metrics(
    motion_files: pd.DataFrame, fd_threshold: float = 0.3,
) -> MotionLoadResult:
    """Read DCAN motion metrics at one FD threshold, keeping unreadable files.

    Unreadable HDF5 files (``OSError``) are retained in ``motion_load_errors``
    with their BIDS keys, path, and error message rather than being silently
    dropped from the file inventory.
    """
    motion_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for record in motion_files.to_dict("records"):
        try:
            metrics = read_motion_metrics(record["file_path"], fd_threshold)
        except OSError as error:
            error_rows.append({
                **{column: record[column] for column in KEY_COLUMNS},
                "motion_hdf5_path": str(record["file_path"]),
                "error": str(error),
            })
            continue
        motion_rows.append({
            **{column: record[column] for column in KEY_COLUMNS},
            **metrics,
        })
    return MotionLoadResult(
        motion_df=pd.DataFrame(motion_rows),
        motion_load_errors=pd.DataFrame(error_rows),
    )


def build_run_level_table(
    conn_files: pd.DataFrame,
    motion_files: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    motion_df: pd.DataFrame,
) -> RunLevelResult:
    """Merge file inventories with readable features and classify file status.

    Unreadable/unmatched files remain visible via ``connectivity_present``,
    ``motion_present``, and ``motion_readable`` flags and a ``file_status``
    label, matching the notebook's coverage/missingness reporting.
    """
    run_level_df = conn_files[[*KEY_COLUMNS, "file_path"]].rename(
        columns={"file_path": "conn_mat_path"}
    ).merge(
        motion_files[[*KEY_COLUMNS, "file_path"]].rename(columns={"file_path": "motion_hdf5_path"}),
        on=list(KEY_COLUMNS), how="outer", validate="one_to_one",
    )
    run_level_df = run_level_df.merge(
        connectivity_df.drop(columns="conn_mat_path"),
        on=list(KEY_COLUMNS), how="left", validate="one_to_one",
    ).merge(
        motion_df, on=list(KEY_COLUMNS), how="left", validate="one_to_one",
    )
    run_level_df["connectivity_present"] = run_level_df["conn_mat_path"].notna()
    run_level_df["motion_present"] = run_level_df["motion_hdf5_path"].notna()
    run_level_df["motion_readable"] = (
        run_level_df["fd_threshold"].notna() if "fd_threshold" in run_level_df else False
    )
    run_level_df["file_status"] = np.select(
        [
            run_level_df["connectivity_present"] & run_level_df["motion_readable"],
            run_level_df["connectivity_present"] & run_level_df["motion_present"],
            run_level_df["connectivity_present"],
            run_level_df["motion_readable"],
            run_level_df["motion_present"],
        ],
        [
            "both files readable", "both files; motion unreadable", "connectivity only",
            "motion only", "motion only; unreadable",
        ],
        default="unclassified",
    )
    missingness_report = run_level_df.loc[
        run_level_df["file_status"] != "both files readable",
        [*KEY_COLUMNS, "file_status", "conn_mat_path", "motion_hdf5_path"],
    ].sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    coverage = _coverage_summary(run_level_df).sort_values(["session", "run"]).reset_index(drop=True)
    return RunLevelResult(
        run_level_df=run_level_df, missingness_report=missingness_report, coverage_summary=coverage,
    )


def build_master_dataframe(
    connectivity_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    feature_columns: Sequence[str],
    motion_feature_columns: Sequence[str] = MOTION_FEATURE_COLUMNS,
) -> MasterDatasetResult:
    """Inner-join readable connectivity/motion rows into the master feature table.

    Columns are ordered as ``SubjectID, Session, Run, remaining_seconds``, the
    motion feature columns, then computed connectivity features.
    """
    matched = connectivity_df.merge(
        motion_df, on=list(KEY_COLUMNS), how="inner", validate="one_to_one",
    )
    master_df = matched.rename(columns={
        "subject": "SubjectID", "session": "Session", "run": "Run",
        "seconds_remaining": "remaining_seconds",
    })[[
        "SubjectID", "Session", "Run", "remaining_seconds",
        *motion_feature_columns, *feature_columns,
    ]]
    return MasterDatasetResult(
        master_df=master_df,
        feature_columns=tuple(feature_columns),
        motion_feature_columns=tuple(motion_feature_columns),
    )


def average_matched_runs(master_df: pd.DataFrame, feature_columns: Sequence[str],
                         motion_feature_columns: Sequence[str] = MOTION_FEATURE_COLUMNS) -> pd.DataFrame:
    """Average matched, readable runs within subject/session (``df_merged_runs``).

    Adds ``n_runs_included`` and ``runs_included`` (sorted, comma-joined run
    labels) alongside the averaged numeric columns.
    """
    run_mean_columns = ["remaining_seconds", *motion_feature_columns, *feature_columns]
    grouped = master_df.groupby(["SubjectID", "Session"], as_index=False)
    averaged = grouped[run_mean_columns].mean()
    run_counts = master_df.groupby(["SubjectID", "Session"]).size().rename("n_runs_included")
    run_lists = master_df.groupby(["SubjectID", "Session"])["Run"].agg(
        lambda runs: ", ".join(sorted(runs.astype(str).unique()))
    ).rename("runs_included")
    return averaged.merge(
        run_counts, on=["SubjectID", "Session"], validate="one_to_one",
    ).merge(
        run_lists, on=["SubjectID", "Session"], validate="one_to_one",
    )


def run_workflow(
    conn_mats_path: str | Path,
    motion_path: str | Path,
    atlas_tsv_path: str | Path,
    *,
    task_filter: str | None = "rest",
    fd_threshold: float = 0.3,
    motion_thresholds: tuple[float, ...] = DEFAULT_MOTION_THRESHOLDS,
    config: AtlasNetworkConfig = AtlasNetworkConfig(),
) -> WorkflowResult:
    """Run the full discovery-to-master-dataframe workflow with explicit inputs.

    Combines, in order: BIDS discovery, atlas network derivation, connectivity
    feature extraction, motion QC readout (retaining unreadable-file
    inventory), run-level coverage/missingness, a multi-threshold remaining
    duration table, the matched-readable master dataframe, run-level
    feature/motion associations, and multi-run averaging.
    """
    atlas = derive_analysis_atlas(atlas_tsv_path, config)
    discovery = discover_run_files(conn_mats_path, motion_path, task_filter)
    connectivity = load_connectivity_features(discovery.conn_files, atlas, config)
    motion = load_motion_metrics(discovery.motion_files, fd_threshold)
    run_level = build_run_level_table(
        discovery.conn_files, discovery.motion_files, connectivity.connectivity_df, motion.motion_df,
    )
    unreadable_paths = set(motion.motion_load_errors["motion_hdf5_path"]) if not motion.motion_load_errors.empty else set()
    readable_motion_files = discovery.motion_files.loc[
        discovery.motion_files["file_path"].map(lambda path: str(path) not in unreadable_paths)
    ]
    threshold_rows = []
    for record in readable_motion_files.to_dict("records"):
        for threshold in motion_thresholds:
            metrics = read_motion_metrics(record["file_path"], threshold)
            threshold_rows.append({
                **{column: record[column] for column in KEY_COLUMNS},
                "fd_threshold": threshold, "remaining_seconds": metrics["seconds_remaining"],
            })
    motion_threshold_df = pd.DataFrame(threshold_rows)
    threshold_table = _threshold_summary(
        motion_threshold_df, threshold_column="fd_threshold", value_column="remaining_seconds",
    )
    master = build_master_dataframe(
        connectivity.connectivity_df, motion.motion_df, connectivity.feature_columns,
    )
    motion_associations = feature_motion_associations(
        master.master_df, master.feature_columns, motion_column="remaining_seconds",
    )
    merged_runs = average_matched_runs(master.master_df, master.feature_columns, master.motion_feature_columns)
    return WorkflowResult(
        atlas=atlas, discovery=discovery, connectivity=connectivity, motion=motion,
        run_level=run_level, motion_threshold_data=motion_threshold_df,
        threshold_table=threshold_table, master=master,
        motion_associations=motion_associations, merged_runs=merged_runs,
    )
