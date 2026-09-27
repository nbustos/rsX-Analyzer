# Bundled atlas metadata

This directory contains the atlas TSV label tables and their JSON description
sidecars supplied for analysis development against XCP-D 26.2.0. These are
metadata files only; they do not include the NIfTI/CIFTI segmentation images.

Each atlas has a `*_dseg.tsv` node table and a matching `*_dseg.json`
description. Preserve their indices and row ordering when pairing them with
timeseries or connectivity matrices. Assignment fields and their distinct
meanings are summarized in [`../docs/atlas-metadata.md`](../docs/atlas-metadata.md).

The supplied 4S source metadata declares CC BY-ND and is included without
modification. The other supplied JSON descriptions do not state a license;
consult their `ReferencesAndLinks` entries and the original atlas sources
before reuse or redistribution.
