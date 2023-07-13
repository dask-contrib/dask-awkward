from __future__ import annotations

import awkward as ak
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers import AwkwardMaterializedLayer
from dask_awkward.lib.core import (
    Array,
    PartitionCompatibility,
    map_partitions,
    new_array_object,
    partition_compatibility,
)
from dask_awkward.utils import DaskAwkwardNotImplemented, IncompatiblePartitions


class _ConcatenateFnAxisGT0:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args):
        return ak.concatenate(list(args), **self.kwargs)


def _concatenate_axis0_multiarg(*args):
    return ak.concatenate(list(args), axis=0)


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

        prev_names = [iarr.name for iarr in arrays]
        g = AwkwardMaterializedLayer(
            g,
            previous_layer_names=prev_names,
            fn=_concatenate_axis0_multiarg,
        )
        hlg = HighLevelGraph.from_collections(name, g, dependencies=arrays)
        return new_array_object(hlg, name, meta=meta, npartitions=npartitions)

    if axis > 0:
        if partition_compatibility(*arrays) == PartitionCompatibility.NO:
            raise IncompatiblePartitions("concatenate", *arrays)

        fn = _ConcatenateFnAxisGT0(axis=axis)
        return map_partitions(fn, *arrays)

    else:
        raise DaskAwkwardNotImplemented("TODO")
