# Atlas metadata and XCP-D 26.2.0

XCP-D atlas metadata is version-specific. This project targets XCP-D
`26.2.0`; its built-in atlas catalogue is defined in
[`xcp_d/utils/atlas.py`](https://github.com/PennLINC/xcp_d/blob/26.2.0/xcp_d/utils/atlas.py).

| Atlas family | Atlas labels |
| --- | --- |
| Combined 4S | `4S156Parcels`, `4S256Parcels`, `4S356Parcels`, `4S456Parcels`, `4S556Parcels`, `4S656Parcels`, `4S756Parcels`, `4S856Parcels`, `4S956Parcels`, `4S1056Parcels` |
| Cortical | `Glasser`, `Gordon`, `MIDB`, `MyersLabonte` |
| Subcortical | `Tian`, `HCP` |

The sidecars provided for this project are in [`../atlases/`](../atlases/).
XCP-D writes them below `sourcedata/atlases/`. The TSV provides node labels
and parcel order; XCP-D's loader expects `index` and `name`. These resources
use either `name` or `label` as the node-name column.

## Assignment logic in the supplied TSVs

| Atlas table(s) | Nodes | Assignment column(s) | Interpretation |
| --- | ---: | --- | --- |
| `atlas-4S{156,256,356,456,556,656,756,856,956,1056}Parcels_dseg.tsv` | Atlas label in filename | `network_label` | Seven Schaefer/Yeo cortical networks: `Cont`, `Default`, `DorsAttn`, `Limbic`, `SalVentAttn`, `SomMot`, `Vis`. The 56 added subcortical/cerebellar nodes are `n/a` and are left unassigned. |
| Same 4S tables | Same | `network_label_17network` | Seventeen cortical subdivisions (including `ContA-C`, `DefaultA-C`, and the other subdivisions encoded in the TSV); the same 56 non-cortical nodes are `n/a`. |
| `atlas-Glasser_dseg.tsv` | 360 | `community_yeo` | Seven Yeo communities. |
| Same Glasser table | Same | `community_mesulam` | Four Mesulam classes: heteromodal, idiotypic, paralimbic, unimodal. |
| Same Glasser table | Same | `community_economo` | Seven von Economo classes as named in the TSV. |
| `atlas-Gordon_dseg.tsv` | 333 | `community` | Gordon's 13 communities, named in the TSV. |
| `atlas-HCP_dseg.tsv` | 19 | None | Node labels only; no network assignment is supplied. |
| `atlas-Tian_dseg.tsv` | 50 | None | Node labels only; no network assignment is supplied. |

The 4S tables combine Schaefer cortical parcels with 56 subcortical/cerebellar
parcels, represented in `atlas_name` as `CIT168Subcortical` (28),
`ThalamusHCP` (14), `SubcorticalHCP` (4), and `Cerebellum` (10). These
non-cortical parcels are not given Schaefer network assignments in the
provided TSVs. Do not infer a cortical network for them or silently label them
as a shared `n/a` network.

The loader accepts both node-name columns and exposes `network_label*` and
`community*` assignment columns. Blank, `n/a`, and `na` values are treated as
unassigned. It does not treat unique parcel labels, numeric network indices,
`network_id`, or color columns as network assignments. The matrix node order
must match the atlas TSV order/index. Never reuse the prototype's fixed
4S356 row ranges for other atlases.

The supplied 4S JSON descriptions identify their license as CC BY-ND; these
source TSV/JSON files are included without alteration. The other supplied JSON
descriptions do not specify a `License` field, so this repository does not
assert an additional license for them. Consult their `ReferencesAndLinks` and
the corresponding atlas sources for terms of use.

Sources:

- [XCP-D 26.2.0 atlas selection and discovery](https://github.com/PennLINC/xcp_d/blob/26.2.0/xcp_d/utils/atlas.py)
- [XCP-D 26.2.0 output documentation](https://github.com/PennLINC/xcp_d/blob/26.2.0/docs/outputs.rst)
- [XCP-D 26.2.0 base image atlas-resource setup](https://github.com/PennLINC/xcp_d/blob/26.2.0/Dockerfile.base)
- [PennLINC AtlasPack source TSVs](https://github.com/PennLINC/AtlasPack)

The bundled files are TSV/JSON lookup and description sidecars, not atlas
segmentation images or time series. Verify they match the XCP-D outputs used
for an analysis.
