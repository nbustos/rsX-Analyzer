"""Organize XCP-D ``func`` outputs into a RESULTS directory layout.

Layout::

    RESULTS/atlases/{CIFTI,NIFTI}/<atlas>/{conn_mats,reho,timeseries}
    RESULTS/motion/{dcan_qc,linc_qc}

Atlas files with ``space-fsLR`` go to CIFTI; every other space goes to NIFTI.
Motion files are format-independent and live at the top level.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import pandas as pd

DEFAULT_ATLASES = ("4S356", "Gordon", "HCP", "Tian")
FORMATS = ("CIFTI", "NIFTI")
ATLAS_KINDS = ("conn_mats", "reho", "timeseries")
MOTION_KINDS = ("dcan_qc", "linc_qc")
MANIFEST_COLUMNS = ["subject", "session", "format", "atlas", "kind", "ext", "source",
                    "destination", "status"]

_DATA_SUFFIXES = (".tsv", ".json", ".nii", ".nii.gz")


@dataclass
class OrganizeResult:
    manifest: pd.DataFrame
    summary: pd.DataFrame
    results_dir: Path
    audit: pd.DataFrame | None = None
    subject_presence: pd.DataFrame | None = None
    n_subjects: pd.Series | None = None


def _file_ext(name: str) -> str:
    return name.split(".", 1)[1] if "." in name else ""


def audit_manifest(manifest: pd.DataFrame, subjects_by_session: dict | None = None):
    """Count subjects holding each file type, per session.

    Returns ``(audit, subject_presence, n_subjects)``. ``audit`` has one row per
    format/atlas/kind/ext with the number of unique subjects per session and
    in ``all_sessions``; ``subject_presence`` is a subject x file-type matrix of
    the number of sessions with that file type; ``n_subjects`` is the number of
    subjects with a func directory in each session (reference denominator).
    """
    keys = ["format", "atlas", "kind", "ext"]
    n_subjects = pd.Series(
        {ses: len(subs) for ses, subs in (subjects_by_session or {}).items()},
        dtype="int64", name="n_subjects_traversed")
    if manifest.empty:
        return pd.DataFrame(columns=keys + ["all_sessions"]), pd.DataFrame(), n_subjects
    audit = (manifest.groupby(keys + ["session"])["subject"].nunique()
             .unstack("session", fill_value=0))
    audit["all_sessions"] = manifest.groupby(keys)["subject"].nunique()
    audit = audit.reset_index()
    audit.columns.name = None
    label = manifest[keys].astype(str).agg("/".join, axis=1).rename("file_type")
    presence = (manifest.assign(file_type=label)
                .groupby(["subject", "file_type"])["session"].nunique()
                .unstack("file_type", fill_value=0))
    return audit, presence, n_subjects


def _entity(name: str, key: str) -> str | None:
    match = re.search(rf"(?:^|_){key}-([A-Za-z0-9]+)", name)
    return match.group(1) if match else None


def _match_atlas(label: str | None, atlases) -> str | None:
    """Map a file's atlas label (e.g. ``4S356Parcels``) to a requested atlas."""
    if label is None:
        return None
    for atlas in atlases:
        if label == atlas or label == f"{atlas}Parcels":
            return atlas
    return None


def classify_func_file(path: Path, atlases=DEFAULT_ATLASES):
    """Return ``(format, atlas, kind)`` for a func file or ``None`` to skip it.

    ``atlas`` is ``None`` for motion files.
    """
    name = path.name
    if name.endswith("desc-dcan_qc.hdf5"):
        kind, atlas = "dcan_qc", None
    elif name.endswith("desc-linc_qc.csv") or name.endswith("desc-linc_qc.tsv"):
        kind, atlas = "linc_qc", None
    else:
        atlas = _match_atlas(_entity(name, "atlas"), atlases)
        if atlas is None or not name.endswith(_DATA_SUFFIXES):
            return None
        if "reho" in name:
            kind = "reho"
        elif "timeseries" in name:
            kind = "timeseries"
        elif re.search(r"conmat|relmat|pconn|coverage", name):
            kind = "conn_mats"
        else:
            return None
    fmt = "CIFTI" if _entity(name, "space") == "fsLR" else "NIFTI"
    if name.endswith(".hdf5") and _entity(name, "space") is None:
        fmt = "NIFTI"
    return fmt, atlas, kind


def _same_file(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size


def organize_xcpd_outputs(xcpd_dir, results_dir, atlases=DEFAULT_ATLASES,
                          dry_run: bool = False) -> OrganizeResult:
    """Copy designated-atlas, ReHo, timeseries, conn_mat and motion files.

    Traverses ``sub-*/[ses-*/]func``. Existing identical files are skipped, so the
    function is safe to re-run; existing files with different content are not
    overwritten (status ``conflict``). Nothing is written when ``dry_run``.
    """
    xcpd_dir, results_dir = Path(xcpd_dir).expanduser(), Path(results_dir).expanduser()
    if not xcpd_dir.is_dir():
        raise FileNotFoundError(f"XCP-D directory not found: {xcpd_dir}")
    atlases = tuple(atlases)

    if not dry_run:
        for kind in MOTION_KINDS:
            (results_dir / "motion" / kind).mkdir(parents=True, exist_ok=True)
        for fmt in FORMATS:
            for atlas in atlases:
                for kind in ATLAS_KINDS:
                    (results_dir / "atlases" / fmt / atlas / kind).mkdir(parents=True, exist_ok=True)

    rows = []
    subjects_by_session: dict[str, set] = {}
    for sub_dir in sorted(p for p in xcpd_dir.glob("sub-*") if p.is_dir()):
        ses_dirs = sorted(p for p in sub_dir.glob("ses-*") if p.is_dir()) or [sub_dir]
        for ses_dir in ses_dirs:
            session = ses_dir.name if ses_dir != sub_dir else "n/a"
            func_dir = ses_dir / "func"
            if not func_dir.is_dir():
                continue
            subjects_by_session.setdefault(session, set()).add(sub_dir.name)
            for src in sorted(p for p in func_dir.iterdir() if p.is_file()):
                cls = classify_func_file(src, atlases)
                if cls is None:
                    continue
                fmt, atlas, kind = cls
                dest_dir = (results_dir / "motion" / kind if atlas is None
                            else results_dir / "atlases" / fmt / atlas / kind)
                dest = dest_dir / src.name
                if dest.exists():
                    status = "skipped" if _same_file(src, dest) else "conflict"
                else:
                    status = "would_copy" if dry_run else "copied"
                    if not dry_run:
                        shutil.copy2(src, dest)
                rows.append({"subject": sub_dir.name, "session": session, "format": "n/a" if atlas is None else fmt,
                             "atlas": atlas or "motion", "kind": kind,
                             "ext": _file_ext(src.name),
                             "source": str(src), "destination": str(dest),
                             "status": status})

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    summary = (manifest.groupby(["format", "atlas", "kind", "status"]).size()
               .rename("n_files").reset_index()) if not manifest.empty else \
        pd.DataFrame(columns=["format", "atlas", "kind", "status", "n_files"])
    audit, presence, n_subjects = audit_manifest(manifest, subjects_by_session)
    return OrganizeResult(manifest, summary, results_dir, audit, presence, n_subjects)
