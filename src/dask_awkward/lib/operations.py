from __future__ import annotations

import awkward as ak
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.lib.core import (
    Array,
    compatible_divisions,
    map_partitions,
    new_array_object,
)
from dask_awkward.utils import DaskAwkwardNotImplemented


class _ConcatenateFnAxisGT0:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args):
        return ak.concatenate(list(args), **self.kwargs)


def concatenate(
    arrays: list[Array],
    axis: int = 0,
    mergebool: bool = True,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    label = "concatenate"
    token = tokenize(arrays, axis, mergebool, highlevel, behavior)
    name = f"{label}-{token}"

    if axis == 0:
        npartitions = sum([a.npartitions for a in arrays])
        g = {}
        i = 0
        metas = []
        for collection in arrays:
            metas.append(collection._meta)
            for k in collection.__dask_keys__():
                g[(name, i)] = k
                i += 1

        meta = ak.concatenate(metas)

        hlg = HighLevelGraph.from_collections(name, g, dependencies=arrays)
        return new_array_object(hlg, name, meta=meta, npartitions=npartitions)

    if axis > 0:
        if not compatible_divisions(*arrays):
            raise ValueError("All arrays must have identical divisions")

        fn = _ConcatenateFnAxisGT0(axis=axis)
        return map_partitions(fn, *arrays)

    else:
        raise DaskAwkwardNotImplemented("TODO")
