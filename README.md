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

The atlas TSV/JSON sidecars examined for this project are included under
[`atlases/`](atlases/). The network assignments they provide differ by atlas;
see [`docs/atlas-metadata.md`](docs/atlas-metadata.md) before choosing a
network feature set.

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

The chosen atlas TSV must match the matrix's node order. The bundled tables
are metadata, not segmentation images; use atlas metadata from the same
XCP-D run/release as the matrix. Check the TSV's available assignment columns
before requesting a network scheme. HCP and Tian label tables have no network
assignment column, so network summaries cannot be inferred for them.

The `rs-X1.ipynb` workflow supports two motion-QC input formats. Set the motion
directory's final folder name to `dcan_qc` for DCAN HDF5 inputs or `linc_qc`
for LINC CSV inputs. LINC CSVs must include `mean_fd`,
`num_censored_volumes`, and `num_retained_volumes`; subject/session/task/run
entities may be columns in the CSV or BIDS entities in its filename. Optional
`space`, `res`, and `desc` metadata are preserved. For each subject/session,
the workflow ranks matched runs by remaining seconds (DCAN) or
`num_retained_volumes` (LINC), highest first, with run label as a deterministic
tie-breaker. It selects at most the top two runs. By default, every selected
run must meet the minimum of 240 seconds/retained volumes for the
subject/session to appear in `df_merged_runs`; if either of two selected runs
falls below 240, that session is omitted. A session with only one matched run
is included when that run meets the same minimum. This threshold is configurable
through `run_workflow(..., minimum_retained_value=...)`. The `master` table
still contains all matched readable runs, including runs/sessions excluded
from `df_merged_runs`. In LINC merged summaries, `num_censored_volumes` and
`num_retained_volumes` are summed over the selected runs; `mean_fd`, other
motion metrics, and connectivity features are averaged.

## Development

Requires Python 3.10 or newer (check with `python --version`; the base
Anaconda Python 3.9 is too old, so create an environment first, e.g.
`conda create -n rsx python=3.12 && conda activate rsx`). Run the commands
below from the repository root. The notebook also needs
`python -m pip install -e '.[notebook]'`. Install the package and test dependencies, then run the test suite:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests
```

Do not commit participant-level data, derived matrices, spreadsheets, or
notebook outputs. Keep analysis data outside this repository.

## Setup: organizing XCP-D outputs

`organize_xcpd_outputs(xcpd_dir, results_dir, atlases=DEFAULT_ATLASES, dry_run=False)`
traverses `sub-*/ses-*/func` and copies atlas, ReHo, timeseries, connectivity and
motion-QC files into:

```
RESULTS/atlases/{CIFTI,NIFTI}/<atlas>/{conn_mats,reho,timeseries}
RESULTS/motion/{dcan_qc,linc_qc}
```

- Default atlases: `4S356, Gordon, HCP, Tian` (`4S356` also matches `atlas-4S356Parcels`).
- `space-fsLR` files go to `CIFTI`; other spaces go to `NIFTI`.
- Connectivity files (`conmat`/`relmat`/`pconn`) and `coverage` files go to `conn_mats`.
- Files are copied, never moved; identical files are skipped, differing ones are reported as `conflict`.
- The return value has a per-file `manifest` and a `summary` count table.

The result also includes an audit: `audit` (unique subjects per format/atlas/kind/extension by
session plus `all_sessions`), `n_subjects` (subjects with a `func` dir per session) and
`subject_presence` (subject x file-type matrix of sessions holding that file type).

## Command-line script

`scripts/run_rsx.py` runs the same workflow as `rs-X1.ipynb` without Jupyter
(tables are printed; the Excel workbook and PDF report are written to `--output-dir`):

```bash
python scripts/run_rsx.py --conn-mats RESULTS/atlases/NIFTI/4S356/conn_mats \
    --motion RESULTS/motion/linc_qc --output-dir out
# Optional Setup first (also writes rs-X1_setup_audit.xlsx):
python scripts/run_rsx.py --xcpd-dir /path/to/xcp_d --results-dir RESULTS --output-dir out
python scripts/run_rsx.py --help
```
