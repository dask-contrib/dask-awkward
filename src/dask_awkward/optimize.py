from __future__ import annotations

from typing import Any, Hashable, Mapping

from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph


def optimize(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **kwargs: Any,
) -> Mapping:
    if not isinstance(keys, (list, set)):
        keys = [keys]  # pragma: no cover
    keys = list(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())

    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore

    return dsk
