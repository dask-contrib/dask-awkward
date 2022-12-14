from __future__ import annotations

import awkward as ak
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.lib.core import Array, new_array_object


def concatenate(arrays: list[Array]) -> Array:
    label = "concatenate"
    token = tokenize(arrays)
    name = f"{label}-{token}"
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
