from __future__ import annotations

import copy
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


def _output_io_layer_names(dsk: HighLevelGraph) -> list[str]:
    return [
        n
        for n, v in dsk.layers.items()
        if (v.annotations or {}).get("ak_output")
    ]

def _mock_io_func(*args, **kwargs):
    import awkward as ak
    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, ak.Array):
            arg.layout._touch_data(recursive=True)


def _mock_output(layer: Layer):
    assert len(layer.dsk) == 1
    layer2 = copy.deepcopy(layer)
    mp = layer2.mapping.copy()  # why is this always a MetrializedLayer?
    key = iter(mp).__next__()
    mp[key] = (_mock_io_func, ) + mp[key][1:]
    layer2.mapping = mp
    return layer2


def optimize_columns(dsk: HighLevelGraph, outputs: Iterable[str]):
    reports = _get_column_report(dsk, outputs)
    return _apply_column_optimization(dsk, reports)


def _get_column_report(dsk: HighLevelGraph, outputs: Iterable[str]) -> Dict[str, Any]:
    import awkward as ak
    layers = dsk.layers.copy()
    deps = dsk.dependencies.copy()
    reports = {}
    for name in _projectable_io_layer_names(dsk):
        # these are the places we can select columns
        layers[name], report = layers[name].mock()
        reports[name] = report

    for name in _output_io_layer_names(dsk):
        # these are the places files would get written if we let them
        layers[name] = _mock_output(layers[name])

    hlg = HighLevelGraph(layers, deps)
    outlayer = list(hlg.layers.values())[-1]  # or take from `outputs`?
    out = get_sync(hlg, list(outlayer.keys())[0])
    if isinstance(out, ak.Array):
        # if output is still an array, all columns count as touched
        out.layout._touch_data(recursive=True)
    return reports


def _apply_column_optimization(
    dsk: HighLevelGraph, reports: Dict[str, Any]
) -> HighLevelGraph:
    # now apply
    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()
    for name, ttr in reports.items():
        cols = set(ttr.data_touched)
        select = []
        for col in cols:
            if col is None:
                continue
            n, c = col.split(".", 1)
            if n == name:
                select.append(c)
        layers[name] = layers[name].project_columns(select)
    return HighLevelGraph(layers, deps)
