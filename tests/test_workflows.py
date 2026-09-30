import unittest

import numpy as np
import pandas as pd

from rsx_analyzer import (
    average_runs,
    macro_network_feature,
    missingness_summary,
    parse_bids_entities,
    robust_outlier_mask,
    weighted_feature_mean,
)
from rsx_analyzer import load_atlas_tsv, summarize_connectivity_partitions
from pathlib import Path
import tempfile


class WorkflowHelperTests(unittest.TestCase):
    def test_partition_summary_adds_bilateral_roi_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.tsv"
            path.write_text(
                "index\tname\tnetwork_label\n1\tLH_Hippocampus\tLimbic\n"
                "2\tRH_Hippocampus\tLimbic\n3\tCortex\tVisual\n",
                encoding="utf-8",
            )
            atlas = load_atlas_tsv(path)
        matrix = np.array([[1, 0.2, 0.4], [0.2, 1, 0.6], [0.4, 0.6, 1.]])
        features = summarize_connectivity_partitions(
            matrix, atlas, {"nine": ("network_label", "_9net")},
            roi_names=("LH_Hippocampus", "RH_Hippocampus"),
            bilateral_pairs={"Bilateral_Hippocampus": (
                "LH_Hippocampus", "RH_Hippocampus"
            )},
        )
        self.assertEqual(features["Bilateral_Hippocampus_to_Limbic_9net"], 0.2)

    def test_parse_entities_defaults_missing_run(self):
        self.assertEqual(
            parse_bids_entities("sub-01_ses-A_task-rest_desc-denoised_conmat.tsv"),
            {"subject": "01", "session": "A", "task": "rest", "run": "n/a"},
        )

    def test_weighted_mean_ignores_missing_values(self):
        frame = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 5.0]})
        result = weighted_feature_mean(frame, {"a": 1, "b": 3})
        np.testing.assert_allclose(result, [2.5, 5.0])

    def test_average_runs_and_missingness(self):
        frame = pd.DataFrame({"subject": ["01", "01"], "feature": [1.0, 3.0]})
        averaged = average_runs(frame, ["subject"])
        self.assertEqual(averaged.loc[0, "feature"], 2.0)
        summary = missingness_summary(pd.DataFrame({"feature": [1.0, np.nan]}), ["feature"])
        self.assertEqual(summary.loc[0, "missing_count"], 1)

    def test_macro_network_feature_uses_pair_weights(self):
        frame = pd.DataFrame({"A_within_19net": [1.0], "A_X_B_19net": [3.0]})
        result = macro_network_feature(frame, ["A"], ["B"], {"A": 2, "B": 3})
        self.assertEqual(result.iloc[0], 3.0)

    def test_robust_outlier_mask(self):
        mask = robust_outlier_mask(pd.Series([1.0, 1.0, 1.0, 100.0]))
        self.assertTrue(mask.iloc[-1])


if __name__ == "__main__":
    unittest.main()
