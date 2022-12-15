from __future__ import annotations

import operator
import warnings
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
        dsk = _optimize_iolayer_columns_getitem(dsk)  # type: ignore
    elif confopt == "brute-force":
        dsk = optimize_iolayer_columns_brute(dsk)  # type: ignore
    elif confopt == "chained":
        raise NotImplementedError(
            '"chained" is not supported (yet), use "simple-getitem" or "brute-force".'
        )
    elif confopt in ("none", False, None):
        pass
    else:
        warnings.warn(
            f"column-projection-optimization option {confopt!r} is unknown; "
            "no column projection optimization will be executed."
        )

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


def _requested_columns_getitem(layer):
    """Determine the columns requested in an ``operator.getitem`` call."""
    fn_arg = layer.indices[1][0]
    if isinstance(fn_arg, tuple):
        fn_arg = fn_arg[0]
        if isinstance(fn_arg, slice):
            return set()
    if isinstance(fn_arg, list):
        if all(isinstance(x, str) for x in fn_arg):
            return set(fn_arg)
    if isinstance(fn_arg, str):
        return {fn_arg}
    return set()


def _projectable_io_layer_names(dsk: HighLevelGraph) -> list[str]:
    """Get list of column-projectable AwkwardIOLayer names.

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph of interest

    Returns
    -------
    list[str]
        Names of the AwkwardIOLayers in the graph that are
        column-projectable.

    """
    return [
        n
        for n, v in dsk.layers.items()
        if isinstance(v, AwkwardIOLayer) and hasattr(v.io_func, "project_columns")
    ]


def _all_getitem_call_columns(dsk: HighLevelGraph) -> set[str]:
    result = set()
    for _, v in dsk.layers.items():
        if _is_getitem(v):
            result |= _requested_columns_getitem(v)
    return result


def _layers_and_columns_getitem(dsk: HighLevelGraph) -> dict[str, list[str] | None]:
    # find layers that are AwkwardIOLayer with a project_columns io_func method.
    # projectable-I/O --> "pio"
    pio_layer_names = _projectable_io_layer_names(dsk)

    # if no projectable AwkwardIOLayers bail and return empty dict
    if not pio_layer_names:
        return {}

    all_getitem_call_columns = _all_getitem_call_columns(dsk)

    last_layer = list(dsk.layers.values())[-1]
    if hasattr(last_layer, "_meta"):
        out_meta = last_layer._meta
        out_meta_columns = out_meta.layout.form.columns()
        if out_meta_columns == [""]:
            out_meta_columns = []
    else:
        out_meta_columns = []
    # can only select output columns that exist in the input
    # (other names may have come from aliases)

    result: dict[str, list[str] | None] = {}
    for pio_layer_name in pio_layer_names:
        columns = dsk.layers[pio_layer_name]._meta.fields  # type: ignore
        starting_columns = set(columns)
        keep = {c for c in out_meta_columns if c in columns}

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
            requested_and_starting = (
                _requested_columns_getitem(layer_of_interest) & starting_columns
            )
            requested_or_starting_and_all_calls = requested_and_starting | (
                starting_columns & all_getitem_call_columns
            )
            cols_used_in_getitem |= requested_or_starting_and_all_calls
        # project columns using the discovered getitem columns.
        if cols_used_in_getitem:
            keep = cols_used_in_getitem | set(keep) | set(out_meta_columns)
            if keep == set(out_meta_columns) or keep == set(starting_columns):
                result[pio_layer_name] = None
            else:
                result[pio_layer_name] = list(keep)

    return result


def _optimize_iolayer_columns_getitem(dsk: HighLevelGraph) -> HighLevelGraph:

    layers_and_cols = _layers_and_columns_getitem(dsk)

    # if the task graph doesn't contain a column-projectable
    # AwkwardIOLayer then bail on this optimization (just return the
    # existing task graph).
    if not layers_and_cols:
        return dsk

    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore

    for pio_layer_name, cols in layers_and_cols.items():
        new_layer = layers[pio_layer_name].project_columns(cols)
        layers[pio_layer_name] = new_layer

    return HighLevelGraph(layers, deps)


def _attempt_compute_with_columns_brute(
    dsk: HighLevelGraph,
    columns: list[str],
) -> None:
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


def _necessary_columns_brute(dsk: HighLevelGraph) -> dict:
    # staring fields should be those belonging to the AwkwardIOLayer's
    # metadata (typetracer) array.
    out_meta = list(dsk.layers.values())[-1]._meta  # type: ignore
    keep = out_meta.layout.form.columns()
    columns: list[str] = []

    pio_layer_names = _projectable_io_layer_names(dsk)
    if len(pio_layer_names) > 1:
        raise RuntimeError(
            "'brute' method of optimization currently only graphs with a single IO layer."
        )
    pio_layer = pio_layer_names[0]

    columns = dsk.layers[pio_layer]._meta.layout.form.columns()  # type: ignore

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
            _attempt_compute_with_columns_brute(dsk, columns=remaining)
        except IndexError:
            keep.append(holdout)
    if keep == columns:
        keep = None
    return {pio_layer: keep}


def _has_projectable_awkward_io_layer(dsk: HighLevelGraph) -> bool:
    for _, v in dsk.layers.items():
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
    necessary_cols = _necessary_columns_brute(dsk)
    layer_name, necessary_cols = list(_necessary_columns_brute(dsk).items())[0]

    if necessary_cols is None:
        return dsk  # type: ignore
    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore
    new_layer: Layer | None = None

    new_layer = dsk.layers[layer_name].project_columns(necessary_cols)  # type: ignore
    layers[layer_name] = new_layer

    return HighLevelGraph(layers, deps)
