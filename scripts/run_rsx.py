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
    result = organize_xcpd_outputs(args.xcpd_dir, args.results_dir, atlases=args.setup_atlases)
    print("Setup summary:\n", result.summary.to_string(index=False))
    print("\nSubjects traversed per session:\n", result.n_subjects.to_string())
    print("\nAudit (subjects per file type by session):\n", result.audit.to_string(index=False))
    out = args.output_dir or Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "rs-X1_setup_audit.xlsx") as writer:
        result.manifest.to_excel(writer, sheet_name="manifest", index=False)
        result.audit.to_excel(writer, sheet_name="audit", index=False)
        result.subject_presence.to_excel(writer, sheet_name="subject_presence")
    print(f"Setup audit workbook: {out / 'rs-X1_setup_audit.xlsx'}")


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

    analysis = run_workflow(args.conn_mats, args.motion, atlas_tsv, task_filter=args.task,
                            fd_threshold=args.fd_threshold, minimum_retained_value=args.min_retained)
    atlas = analysis.atlas
    feature_columns = list(analysis.connectivity.feature_columns)
    master_df, merged = analysis.master.master_df, analysis.merged_runs
    motion_df = analysis.motion.motion_df
    print(f"Atlas: {atlas_tsv.name} ({len(atlas.names)} parcels); motion mode: {analysis.motion_mode}")
    print(f"Loaded {len(analysis.connectivity.connectivity_df)} matrices and {len(motion_df)} readable motion files.")
    print(f"Features: {len(feature_columns)}; matched runs: {len(master_df)}; merged rows: {len(merged)}")
    print("\nSubject coverage by session and run:\n", analysis.run_level.coverage_summary.to_string())
    print("\nMissingness report:\n", analysis.run_level.missingness_report.to_string())
    if not analysis.motion.motion_load_errors.empty:
        print("\nUnreadable motion files:\n", analysis.motion.motion_load_errors.to_string())
    print("\nThreshold summary:\n", analysis.threshold_table.to_string())
    print("\nTop motion-associated features:\n",
          analysis.motion_associations[["feature", "n_runs", "spearman_rho", "spearman_q"]].head(20).to_string())
    by_session, multi = summarize_multiple_runs(merged)
    print("\nSubjects with multiple runs by session:\n", by_session.to_string())

    outliers = robust_outlier_screen(merged, feature_columns, z_threshold=args.outlier_z)
    report = outliers["subject_report"]
    print(f"\nFlagged {len(report)} subjects; unscorable features: {len(outliers['unscorable'])}")
    print(report.to_string())

    counts = pd.Series(atlas.networks["network_label_19network"]).value_counts().to_dict()
    triple_df, triple_summary = triple_network_summary(merged, counts, TRIPLE_NETWORKS)
    order = [f"{n} within" for n in TRIPLE_NETWORKS] + [
        f"{a} x {b}" for i, a in enumerate(TRIPLE_NETWORKS) for b in list(TRIPLE_NETWORKS)[i + 1:]]
    print("\nTriple-network summary:\n", triple_summary.to_string())
    fig = plot_triple_networks(triple_df, order=order).figure
    fig.axes[0].set_title("Triple-network connectivity by atlas resolution")
    fig.tight_layout()

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
    print(f"\nExcel workbook: {excel_path}\nPrint-ready PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
