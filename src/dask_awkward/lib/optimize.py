from __future__ import annotations

import operator
from collections.abc import Hashable, Mapping
from typing import Any

import dask.config
from dask.blockwise import Blockwise, Layer, fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync

from dask_awkward.layers import AwkwardIOLayer


def optimize(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **_: Any,
) -> Mapping:
    if not isinstance(keys, (list, set)):
        keys = (keys,)  # pragma: no cover
    keys = tuple(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())

    confopt = dask.config.get("awkward.column-projection-optimization")
    if confopt == "simple-getitem":
        dsk = optimize_iolayer_columns_getitem(dsk)  # type: ignore
    elif confopt == "brute-force":
        dsk = optimize_iolayer_columns_brute(dsk)  # type: ignore
    elif confopt == "chained":
        raise NotImplementedError(
            '"chained" is not supported (yet), use "simple-getitem" or "brute-force".'
        )
    else:
        pass

    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore

    return dsk


def _is_getitem(layer: Layer) -> bool:
    """Determine if a layer is an ``operator.getitem`` call."""
    if not isinstance(layer, Blockwise):
        return False
    return layer.dsk[layer.output][0] == operator.getitem


def _requested_columns(layer):
    """Determine the columns requested in an ``operator.getitem`` call."""
    fn_arg = layer.indices[1][0]
    if isinstance(fn_arg, tuple):
        fn_arg = fn_arg[0]
        if isinstance(fn_arg, slice):
            return set()
    if isinstance(fn_arg, list):
        if all(isinstance(x, str) for x in fn_arg):
            return set(fn_arg)
    return {fn_arg}


def optimize_iolayer_columns_getitem(dsk: HighLevelGraph) -> HighLevelGraph:
    # find layers that are AwkwardIOLayer with a project_columns io_func method.
    # projectable-I/O --> "pio"
    pio_layer_names = [
        n
        for n, v in dsk.layers.items()
        if isinstance(v, AwkwardIOLayer) and hasattr(v.io_func, "project_columns")
    ]

    # if the task graph doesn't contain a column-projectable
    # AwkwardIOLayer then bail on this optimization (just return the
    # existing task graph).
    if not pio_layer_names:
        return dsk

    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore

    for pio_layer_name in pio_layer_names:
        cols_used_in_getitem = set()
        # dependencies of the current IOLayer
        pio_layer_deps = dsk.dependents[pio_layer_name]
        # which of those dependencies are operator.getitem layers
        deps_that_are_getitem = [
            k for k in pio_layer_deps if _is_getitem(dsk.layers[k])
        ]
        # of the getitem dependencies, determine the columns that were requested.
        for dep_that_is_getitem in deps_that_are_getitem:
            layer_of_interest = dsk.layers[dep_that_is_getitem]
            cols_used_in_getitem |= _requested_columns(layer_of_interest)
        # project columns using the discovered getitem columns.
        if cols_used_in_getitem:
            new_layer = layers[pio_layer_name].project_columns(
                list(cols_used_in_getitem)
            )
            layers[pio_layer_name] = new_layer

    return HighLevelGraph(layers, deps)


def _attempt_compute_with_columns(dsk: HighLevelGraph, columns: list[str]) -> None:
    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore
    io_layer_names = [k for k, v in dsk.layers.items() if isinstance(v, AwkwardIOLayer)]
    top_io_layer_name = io_layer_names[0]
    layers[top_io_layer_name] = layers[top_io_layer_name].project_and_mock(columns)
    # final necessary key is the 0th partition of the last layer in
    # the graph (hence the toposort to find last layer).
    final_key = (dsk._toposort_layers()[-1], 0)
    new_hlg = HighLevelGraph(layers, deps).cull([final_key])
    get_sync(new_hlg, list(new_hlg.keys()))


def _necessary_columns(dsk: HighLevelGraph) -> list[str] | None:
    # staring fields should be those belonging to the AwkwardIOLayer's
    # metadata (typetracer) array.
    out_meta = list(dsk.layers.values())[-1]._meta  # type: ignore
    keep = out_meta.layout.form.columns()
    columns: list[str] = []
    for _, v in dsk.layers.items():
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
        keep = None
    return keep


def _has_projectable_awkward_io_layer(dsk: HighLevelGraph) -> bool:
    for k, v in dsk.layers.items():
        if isinstance(v, AwkwardIOLayer) and hasattr(v.io_func, "project_columns"):
            return True
    return False


def optimize_iolayer_columns_brute(dsk: HighLevelGraph) -> HighLevelGraph:
    # if the task graph doesn't contain a column-projectable
    # AwkwardIOLayer then bail on this optimization (just return the
    # existing task graph).
    if not _has_projectable_awkward_io_layer(dsk):
        return dsk
    # determine the necessary columns to complete the executation of
    # the metadata (typetracer) based task graph.
    necessary_cols = _necessary_columns(dsk)
    if necessary_cols is None:
        return dsk
    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore
    for k, v in dsk.layers.items():
        if isinstance(v, AwkwardIOLayer):
            new_layer = v.project_columns(necessary_cols)
            io_layer_name = k
            break

    layers[io_layer_name] = new_layer

    return HighLevelGraph(layers, deps)
