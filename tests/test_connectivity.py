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

    def test_unassigned_nodes_are_not_treated_as_a_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.tsv"
            path.write_text(
                "index\tlabel\tnetwork_label\tnetwork_id\n"
                "1\tcortical\tVisual\t1\n"
                "2\tsubcortical\tn/a\tn/a\n",
                encoding="utf-8",
            )
            atlas = load_atlas_tsv(path)

        self.assertEqual(atlas.networks["network_label"], ("Visual", None))
        self.assertNotIn("network_id", atlas.networks)

    def test_bundled_atlas_tables_match_their_declared_node_counts(self):
        atlas_directory = Path(__file__).parents[1] / "atlases"
        expected_counts = {
            "atlas-4S156Parcels_dseg.tsv": 156,
            "atlas-4S256Parcels_dseg.tsv": 256,
            "atlas-4S356Parcels_dseg.tsv": 356,
            "atlas-4S456Parcels_dseg.tsv": 456,
            "atlas-4S556Parcels_dseg.tsv": 556,
            "atlas-4S656Parcels_dseg.tsv": 656,
            "atlas-4S756Parcels_dseg.tsv": 756,
            "atlas-4S856Parcels_dseg.tsv": 856,
            "atlas-4S956Parcels_dseg.tsv": 956,
            "atlas-4S1056Parcels_dseg.tsv": 1056,
            "atlas-Glasser_dseg.tsv": 360,
            "atlas-Gordon_dseg.tsv": 333,
            "atlas-HCP_dseg.tsv": 19,
            "atlas-Tian_dseg.tsv": 50,
        }
        self.assertEqual(
            {path.name for path in atlas_directory.glob("*.tsv")},
            set(expected_counts),
        )
        for filename, expected_count in expected_counts.items():
            with self.subTest(atlas=filename):
                atlas = load_atlas_tsv(atlas_directory / filename)
                self.assertEqual(len(atlas.names), expected_count)
                self.assertEqual(len(set(atlas.indices)), expected_count)

        four_s = load_atlas_tsv(atlas_directory / "atlas-4S356Parcels_dseg.tsv")
        self.assertEqual(len(four_s.networks["network_label"]), 356)
        self.assertEqual(len({v for v in four_s.networks["network_label"] if v}), 7)
        self.assertEqual(
            len({v for v in four_s.networks["network_label_17network"] if v}), 17
        )
        glasser = load_atlas_tsv(atlas_directory / "atlas-Glasser_dseg.tsv")
        self.assertEqual(
            set(glasser.networks),
            {"community_yeo", "community_mesulam", "community_economo"},
        )
        self.assertIn("community", load_atlas_tsv(
            atlas_directory / "atlas-Gordon_dseg.tsv"
        ).networks)
        self.assertEqual(
            load_atlas_tsv(atlas_directory / "atlas-HCP_dseg.tsv").networks, {}
        )
        self.assertEqual(
            load_atlas_tsv(atlas_directory / "atlas-Tian_dseg.tsv").networks, {}
        )


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
