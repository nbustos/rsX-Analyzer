# Atlas metadata and XCP-D 26.2.0

XCP-D atlas metadata is version-specific. This project targets XCP-D
`26.2.0`; its built-in atlas catalogue is defined in
[`xcp_d/utils/atlas.py`](https://github.com/PennLINC/xcp_d/blob/26.2.0/xcp_d/utils/atlas.py).

| Atlas family | Atlas labels |
| --- | --- |
| Combined 4S | `4S156Parcels`, `4S256Parcels`, `4S356Parcels`, `4S456Parcels`, `4S556Parcels`, `4S656Parcels`, `4S756Parcels`, `4S856Parcels`, `4S956Parcels`, `4S1056Parcels` |
| Cortical | `Glasser`, `Gordon`, `MIDB`, `MyersLabonte` |
| Subcortical | `Tian`, `HCP` |

XCP-D writes atlas sidecars below
`sourcedata/atlases/`. The TSV is the source of node labels and parcel order;
XCP-D's loader expects `index` and `name`. The 4S source project,
[PennLINC/AtlasPack](https://github.com/PennLINC/AtlasPack), also publishes
TSVs using `index` and `label`, with `network_label` and
`network_label_17network` assignments for 4S cortical parcels.

The atlas TSV loader accepts both name-column conventions and retains
`network_label*` and `network_id` columns when present. Use only network
assignments actually provided for an atlas. In particular, do not infer that unrelated atlas
parcels share Schaefer/Yeo network assignments, and do not reuse the
prototype's hard-coded 4S356 row ranges for other atlases.

Sources:

- [XCP-D 26.2.0 atlas selection and discovery](https://github.com/PennLINC/xcp_d/blob/26.2.0/xcp_d/utils/atlas.py)
- [XCP-D 26.2.0 output documentation](https://github.com/PennLINC/xcp_d/blob/26.2.0/docs/outputs.rst)
- [XCP-D 26.2.0 base image atlas-resource setup](https://github.com/PennLINC/xcp_d/blob/26.2.0/Dockerfile.base)
- [PennLINC AtlasPack source TSVs](https://github.com/PennLINC/AtlasPack)

Atlas TSVs are currently read from the user's XCP-D derivatives rather than
vendored here. This keeps this initial code-only repository small and avoids
mistaking a newer atlas table for the pinned release's metadata.
