from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import dask.config
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync

from dask_awkward.layers import AwkwardIOLayer


def basic_optimize(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **kwargs: Any,
) -> Mapping:
    if not isinstance(keys, (list, set)):
        keys = (keys,)  # pragma: no cover
    keys = tuple(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())

    if dask.config.get("awkward.column-projection-optimization"):
        # our column optimization
        dsk = optimize_iolayer_columns(dsk)

    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore

    return dsk


def _attempt_compute_with_columns(dsk: HighLevelGraph, columns: list[str]) -> None:
    layers = dsk.layers.copy()
    deps = dsk.dependencies
    io_layer_names = [k for k, v in dsk.layers.items() if isinstance(v, AwkwardIOLayer)]
    top_io_layer_name = io_layer_names[0]
    layers[top_io_layer_name] = layers[top_io_layer_name].project_and_mock(columns)
    # final necessary key is the 0th partition of the last layer in
    # the graph (hence the toposort to find last layer).
    final_key = (dsk._toposort_layers()[-1], 0)
    new_hlg = HighLevelGraph(layers, deps).cull([final_key])
    get_sync(new_hlg, list(new_hlg.keys()))


def _necessary_columns(dsk: HighLevelGraph) -> list[str]:
    # staring fields should be those belonging to the AwkwardIOLayer's
    # metadata (typetracer) array.
    out_meta = list(dsk.layers.values())[-1]._meta
    columns: list[str] = []
    for k, v in dsk.layers.items():
        if isinstance(v, AwkwardIOLayer):
            columns = v._meta.layout.form.columns()
            break

    # can only select output columns that exist in the input
    # (other names may have come from aliases)
    keep = [c for c in out_meta.layout.form.columns() if c in columns]

    for c in columns:
        if c in keep:
            continue
        holdout = c
        allcolumns = set(columns)
        remaining = list(allcolumns - {holdout})
        try:
            _attempt_compute_with_columns(dsk, columns=remaining)
        except IndexError:
            keep.append(holdout)
    if keep == columns:
        keep = []
    return keep


def _has_projectable_awkward_io_layer(dsk: HighLevelGraph) -> bool:
    for _, v in dsk.layers.items():
        if isinstance(v, AwkwardIOLayer) and hasattr(v.io_func, "project_columns"):
            return True
    return False


def optimize_iolayer_columns(dsk: HighLevelGraph) -> HighLevelGraph:
    # if the task graph doesn't contain a column-projectable
    # AwkwardIOLayer then bail on this optimization (just return the
    # existing task graph).
    if not _has_projectable_awkward_io_layer(dsk):
        return dsk

    # determine the necessary columns to complete the executation of
    # the metadata (typetracer) based task graph.
    necessary_cols = _necessary_columns(dsk)

    # if necessary cols is empty just return the input graph
    if not necessary_cols:
        return dsk

    layers = dsk.layers.copy()
    deps = dsk.dependencies.copy()
    for k, v in dsk.layers.items():
        if isinstance(v, AwkwardIOLayer):
            new_layer = v.project_columns(necessary_cols)
            io_layer_name = k
            layers[io_layer_name] = new_layer
            break

    return HighLevelGraph(layers, deps)
