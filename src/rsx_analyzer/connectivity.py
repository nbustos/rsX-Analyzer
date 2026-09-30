"""Atlas-aware summary features for square functional-connectivity matrices."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from .atlas import AtlasMetadata


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def summarize_connectivity(
    matrix: np.ndarray, atlas: AtlasMetadata, network_column: str | None = None,
    roi_names: Sequence[str] = (), suffix: str = "",
) -> dict[str, float]:
    """Compute within/between-network and optional ROI-to-network means."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Connectivity matrix must be square; got {values.shape}")
    if values.shape[0] != len(atlas.names):
        raise ValueError(f"Matrix has {values.shape[0]} nodes but atlas has {len(atlas.names)}")
    if network_column is None and roi_names:
        raise ValueError("A network_column is required to compute ROI-to-network features")
    if network_column is None:
        return {}
    if network_column not in atlas.networks:
        available = ", ".join(sorted(atlas.networks)) or "none"
        raise ValueError(f"Atlas has no network column {network_column!r}; available: {available}")
    assignments = atlas.networks[network_column]
    network_names = sorted({name for name in assignments if name is not None})
    node_indices = {name: np.array([a == name for a in assignments], dtype=bool)
                    for name in network_names}
    result: dict[str, float] = {}
    for i, network_a in enumerate(network_names):
        for j in range(i, len(network_names)):
            network_b = network_names[j]
            block = values[np.ix_(node_indices[network_a], node_indices[network_b])]
            if i == j:
                block = block[~np.eye(block.shape[0], dtype=bool)]
                key = f"{network_a}_within{suffix}"
            else:
                key = f"{network_a}_X_{network_b}{suffix}"
            result[key] = _finite_mean(block)
    name_to_index = {name: i for i, name in enumerate(atlas.names)}
    for roi in roi_names:
        if roi not in name_to_index:
            raise ValueError(f"ROI {roi!r} is not present in the atlas")
        roi_index = name_to_index[roi]
        for network in network_names:
            selected = node_indices[network].copy()
            if assignments[roi_index] == network:
                selected[roi_index] = False
            result[f"{roi}_to_{network}{suffix}"] = _finite_mean(values[roi_index, selected])
    return result


def bilateral_roi_features(
    matrix: np.ndarray, atlas: AtlasMetadata, roi_pairs: Mapping[str, tuple[str, str]],
    network_column: str, suffix: str = "",
) -> dict[str, float]:
    """Compute bilateral seed features as finite means of left/right summaries."""
    output: dict[str, float] = {}
    for label, (left, right) in roi_pairs.items():
        left_features = summarize_connectivity(matrix, atlas, network_column, (left,), suffix)
        right_features = summarize_connectivity(matrix, atlas, network_column, (right,), suffix)
        for key in left_features:
            if "_to_" not in key:
                continue
            network = key.split("_to_", 1)[1]
            right_key = f"{right}_to_{network}"
            vals = np.asarray([left_features[key], right_features[right_key]])
            finite = vals[np.isfinite(vals)]
            output[f"{label}_to_{network}"] = _finite_mean(finite)
    return output


def summarize_connectivity_partitions(
    matrix: np.ndarray,
    atlas: AtlasMetadata,
    network_columns: Mapping[str, tuple[str, str]],
    roi_names: Sequence[str] = (),
    bilateral_pairs: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, float]:
    """Extract multiple network partitions and optional bilateral ROI features.

    ``network_columns`` maps output labels to ``(atlas network column, suffix)``.
    """
    result: dict[str, float] = {}
    for _, (column, suffix) in network_columns.items():
        result.update(summarize_connectivity(
            matrix, atlas, column, roi_names=roi_names, suffix=suffix
        ))
        if bilateral_pairs:
            result.update(bilateral_roi_features(
                matrix, atlas, bilateral_pairs, column, suffix
            ))
    return result
