from __future__ import annotations

from collections.abc import Hashable, Mapping

from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph


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
