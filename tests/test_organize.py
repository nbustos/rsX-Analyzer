import tempfile
import unittest
from pathlib import Path

from rsx_analyzer import organize_xcpd_outputs

PRE = "sub-1244_ses-02_task-rest_run-02"
NII = "space-MNI152NLin2009cAsym"


def _touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class OrganizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.src, self.out = root / "xcpd", root / "RESULTS"
        func = self.src / "sub-1244" / "ses-02" / "func"
        for atlas in ("Gordon", "4S356Parcels", "Glasser"):
            for kind in ("measure-pearsoncorrelation_conmat.tsv", "reho.tsv", "timeseries.tsv",
                         "reho.json"):
                _touch(func / f"{PRE}_{NII}_atlas-{atlas}_{kind}")
        _touch(func / f"{PRE}_space-fsLR_atlas-Gordon_den-91k_stat-mean_timeseries.ptseries.nii")
        _touch(func / f"{PRE}_{NII}_desc-linc_qc.csv")
        _touch(func / f"{PRE}_desc-dcan_qc.hdf5")
        _touch(func / f"{PRE}_{NII}_desc-denoised_bold.nii.gz")
        _touch(self.src / "sub-1244.html")
        _touch(self.src / "test" / "ses-01" / "func" / "x.tsv")

    def test_layout_and_skips(self):
        res = organize_xcpd_outputs(self.src, self.out, atlases=["4S356", "Gordon"])
        o = self.out
        self.assertEqual(len(list((o / "atlases/NIFTI/Gordon/conn_mats").iterdir())), 1)
        self.assertEqual(len(list((o / "atlases/NIFTI/Gordon/reho").iterdir())), 1)
        self.assertEqual(len(list((o / "atlases/NIFTI/4S356/timeseries").iterdir())), 1)
        self.assertFalse(any((o / "atlases").rglob("*.json")) or any((o / "atlases").rglob("*.nii")))
        self.assertFalse((o / "atlases/NIFTI/Glasser").exists())
        self.assertEqual(len(list((o / "motion/linc_qc").iterdir())), 1)
        self.assertEqual(len(list((o / "motion/dcan_qc").iterdir())), 1)
        self.assertTrue((o / "atlases/CIFTI/4S356/reho").is_dir())
        self.assertFalse(res.manifest["source"].str.contains("denoised|/test/").any())

    def test_audit_counts_subjects_per_session(self):
        _touch(self.src / "sub-9999" / "ses-01" / "func"
               / f"sub-9999_ses-01_task-rest_{NII}_atlas-Gordon_reho.tsv")
        res = organize_xcpd_outputs(self.src, self.out, atlases=["Gordon"])
        a = res.audit
        row = a[(a.format == "NIFTI") & (a.atlas == "Gordon") & (a.kind == "reho")
                & (a.ext == "tsv")].iloc[0]
        self.assertEqual(row["ses-01"], 1)
        self.assertEqual(row["ses-02"], 1)
        self.assertEqual(row["all_sessions"], 2)
        self.assertEqual(res.n_subjects["ses-01"], 1)
        self.assertEqual(res.n_subjects["ses-02"], 1)
        self.assertEqual(res.subject_presence.loc["sub-1244"].max(), 1)

    def test_dry_run_and_idempotent(self):
        dry = organize_xcpd_outputs(self.src, self.out, dry_run=True)
        self.assertFalse(self.out.exists())
        self.assertTrue((dry.manifest["status"] == "would_copy").all())
        organize_xcpd_outputs(self.src, self.out)
        again = organize_xcpd_outputs(self.src, self.out)
        self.assertTrue((again.manifest["status"] == "skipped").all())

    def test_conflict_not_overwritten(self):
        organize_xcpd_outputs(self.src, self.out)
        dest = next((self.out / "atlases/NIFTI/Gordon/reho").glob("*.tsv"))
        dest.write_text("different")
        res = organize_xcpd_outputs(self.src, self.out)
        self.assertIn("conflict", set(res.manifest["status"]))
        self.assertEqual(dest.read_text(), "different")


if __name__ == "__main__":
    unittest.main()
