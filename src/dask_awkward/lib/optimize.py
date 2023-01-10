from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, Dict, Iterable

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

    # defunct
    # confopt = dask.config.get("awkward.column-projection-optimization")
    dsk = optimize_columns(dsk, keys)

    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore

    return dsk


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


def optimize_columns(dsk: HighLevelGraph, outputs: Iterable[str]):
    reports = _get_column_report(dsk, outputs)
    return _apply_column_optimization(dsk, reports)


def _get_column_report(dsk: HighLevelGraph, outputs: Iterable[str]) -> Dict[str, Any]:

    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore
    reports = {}
    for name in _projectable_io_layer_names(dsk):
        layers[name], report = layers[name].mock()
        reports[name] = report
    hlg = HighLevelGraph(layers, deps)
    outlayer = list(hlg.layers)[-1]
    out = get_sync(hlg, (outlayer, 0))
    out.layout._touch_data(recursive=True)
    return reports


def _apply_column_optimization(
    dsk: HighLevelGraph, reports: Dict[str, Any]
) -> HighLevelGraph:
    # now apply
    layers = dsk.layers.copy()  # type: ignore
    for name, ttr in reports.items():
        cols = set(ttr.data_touched)
        cols.remove(None)
        layers[name] = layers[name].project_columns(list(cols))
    return HighLevelGraph(layers, dsk.dependencies)
