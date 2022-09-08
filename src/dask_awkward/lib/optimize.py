from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING

from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers import AwkwardIOLayer

if TYPE_CHECKING:
    from dask.typing import HLGDaskCollection


def basic_optimize(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
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


def fun(
    collection: HLGDaskCollection,
    columns: list[str] | None = None,
) -> HighLevelGraph:
    from dask_awkward.lib.core import new_array_object

    hlg = collection.__dask_graph__()
    layers = hlg.layers.copy()
    deps = hlg.dependencies.copy()
    io_layer_names = [k for k, v in hlg.layers.items() if isinstance(v, AwkwardIOLayer)]
    top_io_layer_name = io_layer_names[0]
    if columns is not None:
        layers[top_io_layer_name] = layers[top_io_layer_name].project_and_mock(columns)
    else:
        layers[top_io_layer_name] = layers[top_io_layer_name].mock()

    return new_array_object(
        HighLevelGraph(layers, deps),
        collection.name,
        meta=collection._meta,
        divisions=collection.divisions,
    )


def column_projection(collection):

    hlg = collection.__dask_graph__()
    layers = hlg.layers.copy()

    for k, v in layers.items():
        if isinstance(v, AwkwardIOLayer):
            fields = v._meta.fields
            break

    keep = []

    for f in fields:

        holdout = f
        allfields = set(fields)
        remaining = list(allfields - {f})

        try:
            fun(collection, columns=remaining).compute()
        except IndexError:
            print(f"I think we should keep {holdout}")
            keep.append(holdout)

    print(keep)
