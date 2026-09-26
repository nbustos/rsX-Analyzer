"""Atlas-aware summary features for square functional-connectivity matrices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .atlas import AtlasMetadata


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def summarize_connectivity(
    matrix: np.ndarray,
    atlas: AtlasMetadata,
    network_column: str | None = None,
    roi_names: Sequence[str] = (),
    suffix: str = "",
) -> dict[str, float]:
    """Compute network-pair and optional ROI-to-network mean connectivity.

    Within-network means exclude the matrix diagonal. Between-network means
    include all entries in the corresponding rectangular block. ROI names must
    match atlas node labels exactly; each ROI is excluded from its own network
    mean.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Connectivity matrix must be square; got {values.shape}")
    if values.shape[0] != len(atlas.names):
        raise ValueError(
            f"Matrix has {values.shape[0]} nodes but atlas has {len(atlas.names)}"
        )
    if network_column is None and roi_names:
        raise ValueError("A network_column is required to compute ROI-to-network features")
    if network_column is not None and network_column not in atlas.networks:
        available = ", ".join(sorted(atlas.networks)) or "none"
        raise ValueError(
            f"Atlas has no network column {network_column!r}; available: {available}"
        )

    result: dict[str, float] = {}
    if network_column is None:
        return result

    assignments = atlas.networks[network_column]
    network_names = sorted({name for name in assignments if name is not None})
    node_indices = {
        name: np.array([assignment == name for assignment in assignments], dtype=bool)
        for name in network_names
    }

    for i, network_a in enumerate(network_names):
        mask_a = node_indices[network_a]
        for j in range(i, len(network_names)):
            network_b = network_names[j]
            mask_b = node_indices[network_b]
            block = values[np.ix_(mask_a, mask_b)]
            if i == j:
                block = block[~np.eye(block.shape[0], dtype=bool)]
                key = f"{network_a}_within{suffix}"
            else:
                key = f"{network_a}_X_{network_b}{suffix}"
            result[key] = _finite_mean(block)

    name_to_index = {name: index for index, name in enumerate(atlas.names)}
    for roi in roi_names:
        if roi not in name_to_index:
            raise ValueError(f"ROI {roi!r} is not present in the atlas")
        roi_index = name_to_index[roi]
        roi_network = assignments[roi_index]
        for network in network_names:
            selected = node_indices[network].copy()
            if network == roi_network:
                selected[roi_index] = False
            result[f"{roi}_to_{network}{suffix}"] = _finite_mean(
                values[roi_index, selected]
            )

    return result
