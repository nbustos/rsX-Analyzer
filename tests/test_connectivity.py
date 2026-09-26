import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsx_analyzer import load_atlas_tsv, summarize_connectivity


class AtlasMetadataTests(unittest.TestCase):
    def test_loads_xcpd_and_atlaspack_columns_in_index_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.tsv"
            path.write_text(
                "index\tlabel\tnetwork_label\tnetwork_label_17network\n"
                "2\tRH_Amygdala\tSubcort\tSubcort\n"
                "1\tLH_Amygdala\tLimbicB\tLimbicB\n",
                encoding="utf-8",
            )
            atlas = load_atlas_tsv(path)

        self.assertEqual(atlas.names, ("LH_Amygdala", "RH_Amygdala"))
        self.assertEqual(atlas.indices, (1, 2))
        self.assertEqual(
            atlas.networks["network_label_17network"], ("LimbicB", "Subcort")
        )

    def test_loads_xcpd_name_column_without_networks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.tsv"
            path.write_text("index\tname\n1\tparcel-a\n", encoding="utf-8")
            atlas = load_atlas_tsv(path)

        self.assertEqual(atlas.names, ("parcel-a",))
        self.assertEqual(atlas.networks, {})


class ConnectivitySummaryTests(unittest.TestCase):
    def setUp(self):
        self.atlas = load_atlas_tsv_from_text(
            "index\tname\tnetwork_label\n"
            "1\tROI_A\tAlpha\n"
            "2\tA2\tAlpha\n"
            "3\tROI_B\tBeta\n"
        )
        self.matrix = np.array(
            [
                [1.0, 0.2, 0.4],
                [0.2, 1.0, 0.6],
                [0.4, 0.6, 1.0],
            ]
        )

    def test_network_and_roi_summaries(self):
        features = summarize_connectivity(
            self.matrix,
            self.atlas,
            network_column="network_label",
            roi_names=("ROI_A",),
            suffix="_7net",
        )

        self.assertEqual(features["Alpha_within_7net"], 0.2)
        self.assertEqual(features["Alpha_X_Beta_7net"], 0.5)
        self.assertEqual(features["ROI_A_to_Alpha_7net"], 0.2)
        self.assertEqual(features["ROI_A_to_Beta_7net"], 0.4)

    def test_single_node_network_has_nan_within_mean(self):
        features = summarize_connectivity(
            self.matrix, self.atlas, network_column="network_label"
        )
        self.assertTrue(np.isnan(features["Beta_within"]))

    def test_rejects_matrix_atlas_mismatch(self):
        with self.assertRaisesRegex(ValueError, "atlas has 3"):
            summarize_connectivity(self.matrix[:2, :2], self.atlas, "network_label")

    def test_rejects_unknown_roi(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            summarize_connectivity(
                self.matrix, self.atlas, "network_label", roi_names=("missing",)
            )


def load_atlas_tsv_from_text(text):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "atlas.tsv"
        path.write_text(text, encoding="utf-8")
        return load_atlas_tsv(path)


if __name__ == "__main__":
    unittest.main()
