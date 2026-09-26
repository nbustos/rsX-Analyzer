# rsX-Analyzer

Scientific analysis tools for functional-connectivity outputs from
[XCP-D](https://github.com/PennLINC/xcp_d). The current prototype is based on
the Schaefer 4S356 analysis notebook; reusable code is being developed to work
from each atlas's own label table rather than assuming a fixed parcel count or
row range.

## XCP-D compatibility

The initial target is **XCP-D 26.2.0**. That release supports these built-in
atlases:

- Combined 4S atlases: `4S156Parcels`, `4S256Parcels`, `4S356Parcels`,
  `4S456Parcels`, `4S556Parcels`, `4S656Parcels`, `4S756Parcels`,
  `4S856Parcels`, `4S956Parcels`, and `4S1056Parcels`
- Cortical atlases: `Glasser`, `Gordon`, `MIDB`, and `MyersLabonte`
- Subcortical atlases: `Tian` and `HCP`

XCP-D writes atlas label tables as TSV sidecars in its derivatives'
`sourcedata/atlases/` directory. These tables provide the node order and
labels. 4S label tables from
[PennLINC/AtlasPack](https://github.com/PennLINC/AtlasPack) additionally
provide 7-network and 17-network assignments for cortical parcels. Not every
atlas defines the same network schemes; the analysis code uses a network
assignment only when it is present in that atlas's metadata. No parcel order
or network is inferred from hard-coded row numbers.

See [`docs/atlas-metadata.md`](docs/atlas-metadata.md) for metadata sources and
expected TSV schema.

## Prototype

[`notebooks/ANX_CBT_REST.ipynb`](notebooks/ANX_CBT_REST.ipynb) preserves the
original exploratory workflow, with machine-specific file paths replaced by
configurable paths. Its saved outputs have been cleared; run it against your
own data.

## Reusable code

The `rsx_analyzer` package loads XCP-D-style atlas TSVs and computes
within-/between-network connectivity and ROI-to-network means. For example:

```python
import pandas as pd

from rsx_analyzer import load_atlas_tsv, summarize_connectivity

atlas = load_atlas_tsv("atlas-4S356Parcels_dseg.tsv")
matrix = pd.read_csv("sub-01_atlas-4S356Parcels_connectivity.tsv",
                     sep="\t", index_col=0).to_numpy()
features = summarize_connectivity(
    matrix,
    atlas,
    network_column="network_label_17network",
    roi_names=("LH_Amygdala", "RH_Amygdala"),
)
```

The chosen atlas TSV must match the matrix's node order. Matrices and atlas
tables should come from the same XCP-D run/release. Check the atlas TSV's
available network columns before requesting a network scheme.

## Development

Install the package and test dependencies, then run the test suite:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests
```

Do not commit participant-level data, derived matrices, spreadsheets, or
notebook outputs. Keep analysis data outside this repository.
