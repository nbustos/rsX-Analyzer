#!/usr/bin/env python
"""Command-line version of rs-X1.ipynb: optional Setup, load, QC, outliers,
triple-network summaries, and Excel/PDF export.

Example:
    python scripts/run_rsx.py --conn-mats RESULTS/atlases/NIFTI/4S356/conn_mats \
        --motion RESULTS/motion/linc_qc --output-dir out

Add --xcpd-dir and --results-dir to run the Setup step first (the
--conn-mats/--motion defaults then point into the new RESULTS directory).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from rsx_analyzer import (
    DEFAULT_ATLASES, export_analysis_results_pdf, export_excel, organize_xcpd_outputs,
    plot_triple_networks, robust_outlier_screen, run_workflow, summarize_multiple_runs,
    triple_network_summary,
)

TRIPLE_NETWORKS = {
    "Control (CEN)": {"network9": ("Cont",), "network19": ("ContA", "ContB", "ContC")},
    "Default (DMN)": {"network9": ("Default",), "network19": ("DefaultA", "DefaultB", "DefaultC")},
    "Salience (SN)": {"network9": ("SalVentAttn",), "network19": ("SalVentAttnA", "SalVentAttnB")},
}


def step(msg):
    print(f"[rs-X1] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xcpd-dir", type=Path, help="XCP-D output dir; enables the Setup step")
    p.add_argument("--results-dir", type=Path, help="RESULTS dir to create/use (required with --xcpd-dir)")
    p.add_argument("--setup-atlases", nargs="+", default=list(DEFAULT_ATLASES))
    p.add_argument("--setup-only", action="store_true", help="Stop after Setup")
    p.add_argument("--conn-mats", type=Path, help="Connectivity matrix dir")
    p.add_argument("--motion", type=Path, help="Motion dir named dcan_qc or linc_qc")
    p.add_argument("--atlas", default="4S356", help="Analysis atlas name (default 4S356)")
    p.add_argument("--atlas-tsv", type=Path, help="Atlas dseg TSV (default atlases/atlas-<atlas>Parcels_dseg.tsv)")
    p.add_argument("--task", default="rest")
    p.add_argument("--fd-threshold", type=float, default=0.3)
    p.add_argument("--min-retained", type=float, default=240)
    p.add_argument("--outlier-z", type=float, default=3.5)
    p.add_argument("--verbose", action="store_true", help="Also print the full tables")
    p.add_argument("--output-dir", type=Path, help="Where to write the xlsx/pdf (default: current dir)")
    args = p.parse_args(argv)
    if args.xcpd_dir and not args.results_dir:
        p.error("--results-dir is required with --xcpd-dir")
    if args.results_dir:
        args.conn_mats = args.conn_mats or args.results_dir / "atlases" / "NIFTI" / args.atlas / "conn_mats"
        args.motion = args.motion or args.results_dir / "motion" / "linc_qc"
    if not args.setup_only and not (args.conn_mats and args.motion):
        p.error("--conn-mats and --motion are required (or give --results-dir)")
    return args


def run_setup(args):
    step(f"Setup: organizing {args.xcpd_dir} -> {args.results_dir} ...")
    result = organize_xcpd_outputs(args.xcpd_dir, args.results_dir, atlases=args.setup_atlases)
    counts = result.manifest["status"].value_counts().to_dict()
    step(f"Setup done: {len(result.manifest)} files ({counts}); "
         f"{result.manifest['subject'].nunique()} subjects, sessions: {result.n_subjects.to_dict()}")
    if args.verbose:
        print(result.summary.to_string(index=False))
        print(result.audit.to_string(index=False))
    out = args.output_dir or Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "rs-X1_setup_audit.xlsx") as writer:
        result.manifest.to_excel(writer, sheet_name="manifest", index=False)
        result.audit.to_excel(writer, sheet_name="audit", index=False)
        result.subject_presence.to_excel(writer, sheet_name="subject_presence")
    step(f"Setup audit saved: {out / 'rs-X1_setup_audit.xlsx'}")


def main(argv=None):
    args = parse_args(argv)
    if args.xcpd_dir:
        run_setup(args)
    if args.setup_only:
        return 0

    atlas_tsv = args.atlas_tsv or ROOT / "atlases" / f"atlas-{args.atlas}Parcels_dseg.tsv"
    out = args.output_dir or Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    excel_path, pdf_path = out / "rs-X1_analysis.xlsx", out / "rs-X1_results.pdf"

    step("1/5 Loading connectivity and motion data ...")
    analysis = run_workflow(args.conn_mats, args.motion, atlas_tsv, task_filter=args.task,
                            fd_threshold=args.fd_threshold, minimum_retained_value=args.min_retained)
    atlas = analysis.atlas
    feature_columns = list(analysis.connectivity.feature_columns)
    master_df, merged = analysis.master.master_df, analysis.merged_runs
    motion_df = analysis.motion.motion_df
    n_missing = len(analysis.run_level.missingness_report)
    step(f"1/5 Loaded: {len(analysis.connectivity.connectivity_df)} matrices, {len(motion_df)} motion files "
         f"({analysis.motion_mode}), {len(feature_columns)} features, {n_missing} unmatched/unreadable files")
    errors = analysis.motion.motion_load_errors
    if not errors.empty:
        step(f"WARNING: {len(errors)} unreadable motion files")

    step("2/5 QC: merging runs and testing motion associations ...")
    by_session, multi = summarize_multiple_runs(merged)
    top = analysis.motion_associations.iloc[0]["feature"] if len(analysis.motion_associations) else "n/a"
    step(f"2/5 QC done: {len(master_df)} matched runs -> {len(merged)} merged subject/session rows; "
         f"{multi['SubjectID'].nunique()} subjects with multiple runs; top motion-associated feature: {top}")
    if args.verbose:
        for name, table in (("Coverage", analysis.run_level.coverage_summary),
                            ("Missingness", analysis.run_level.missingness_report),
                            ("Thresholds", analysis.threshold_table),
                            ("Multiple runs", by_session)):
            print(f"\n{name}:\n{table.to_string()}")

    step("3/5 Screening feature outliers ...")
    outliers = robust_outlier_screen(merged, feature_columns, z_threshold=args.outlier_z)
    report = outliers["subject_report"]
    step(f"3/5 Outliers done: {len(report)} flagged subjects, {len(outliers['unscorable'])} unscorable features")
    if args.verbose:
        print(report.to_string())

    step("4/5 Computing triple-network summaries ...")
    counts = pd.Series(atlas.networks["network_label_19network"]).value_counts().to_dict()
    triple_df, triple_summary = triple_network_summary(merged, counts, TRIPLE_NETWORKS)
    order = [f"{n} within" for n in TRIPLE_NETWORKS] + [
        f"{a} x {b}" for i, a in enumerate(TRIPLE_NETWORKS) for b in list(TRIPLE_NETWORKS)[i + 1:]]
    fig = plot_triple_networks(triple_df, order=order).figure
    fig.axes[0].set_title("Triple-network connectivity by atlas resolution")
    fig.tight_layout()
    step(f"4/5 Triple-network done: {len(triple_summary)} summary rows")
    if args.verbose:
        print(triple_summary.to_string())

    step("5/5 Writing Excel and PDF ...")
    export_excel({"master": master_df, "merged_runs": merged, "QC": report, "Trinetwork": triple_summary},
                 excel_path, figure=fig, figure_sheet="Trinetwork",
                 figure_title="Triple-network connectivity by atlas resolution",
                 figure_subtitle="19-network summary uses parcel-pair-weighted means.")
    export_analysis_results_pdf(
        pdf_path, atlas_name=atlas_tsv.name, n_atlas_parcels=len(atlas.names),
        fd_threshold=args.fd_threshold, n_conn_files=len(analysis.discovery.conn_files),
        n_motion_files=len(analysis.discovery.motion_files), n_motion_readable=len(motion_df),
        n_master_rows=len(master_df), n_merged_run_rows=len(merged), n_features=len(feature_columns),
        n_flagged_subjects=len(report), threshold_summary=analysis.threshold_table,
        motion_associations=analysis.motion_associations,
        outlier_feature_summary=outliers["feature_summary"], outlier_subject_report=report,
        triple_summary=triple_summary, triple_network_df=triple_df, triple_feature_order=order,
        outlier_z_threshold=args.outlier_z, motion_mode=analysis.motion_mode)
    step(f"5/5 Done. Excel: {excel_path} | PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
