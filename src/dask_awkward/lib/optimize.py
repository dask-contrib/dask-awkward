from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph

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

    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore

    return dsk


def _attempt_compute_with_columns(dsk: HighLevelGraph, columns: list[str]) -> None:
    layers = dsk.layers.copy()
    deps = dsk.dependencies.copy()
    io_layer_names = [k for k, v in dsk.layers.items() if isinstance(v, AwkwardIOLayer)]
    top_io_layer_name = io_layer_names[0]

    layers[top_io_layer_name] = layers[top_io_layer_name].project_and_mock(columns)

    import dask.local

    new_hlg = HighLevelGraph(layers, deps)
    dask.local.get_sync(new_hlg, list(new_hlg.keys()))


def _necessary_columns(dsk: HighLevelGraph) -> list[str]:
    # staring fields should be those belonging to the AwkwardIOLayer's
    # metadata (typetracer) array.
    for k, v in dsk.layers.items():
        if isinstance(v, AwkwardIOLayer):
            fields = v._meta.fields
            break

    keep = []
    for f in fields:
        holdout = f
        allfields = set(fields)
        remaining = list(allfields - {f})
        try:
            _attempt_compute_with_columns(dsk, columns=remaining)
        except IndexError:
            keep.append(holdout)

    return keep
