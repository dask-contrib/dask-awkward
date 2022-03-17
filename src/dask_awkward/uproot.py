from __future__ import annotations

from math import ceil

import awkward._v2 as ak
import uproot
from awkward._v2.tmp_for_testing import v1_to_v2
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.core import Array, new_array_object


class UprootReadWrapper:
    def __init__(
        self,
        source: str,
        tree_name: str,
        branches: list[str] | None,
    ) -> None:
        self.source = source
        self.tree_name = tree_name
        self.branches = branches

    def __call__(
        self,
        start: int | None,
        stop: int | None,
    ) -> ak.Array:
        t = uproot.open(self.source)[self.tree_name]
        arr = t.arrays(self.branches, entry_start=start, entry_stop=stop)
        # uproot returns an awkward v1 array; we convert to v2
        return ak.Array(v1_to_v2(arr.layout))


def from_uproot(
    source: str,
    tree_name: str,
    npartitions: int,
    branches: list[str] | None = None,
) -> Array:
    token = tokenize(source, tree_name, npartitions, branches)
    name = f"from-uproot-{token}"

    # determine partitioning based on npartitions argument and
    # determine meta from the first 5 entries in the tree.
    tree = uproot.open(source)[tree_name]
    nrows = tree.num_entries
    chunksize = int(ceil(nrows / npartitions))
    locs = list(range(0, nrows, chunksize)) + [nrows]
    start_stop_pairs = [(start, stop) for start, stop in zip(locs[:-1], locs[1:])]
    first5 = tree.arrays(branches, entry_start=0, entry_stop=5)
    v2first5 = ak.Array(v1_to_v2(first5.layout))
    meta = ak.Array(v2first5.layout.typetracer.forget_length())

    # low level graph; one partition per (start, stop) pair.
    llg = {
        (name, i): (
            UprootReadWrapper(
                source,
                tree_name,
                branches,
            ),
            start,
            stop,
        )
        for i, (start, stop) in enumerate(tuple(start_stop_pairs))
    }

    # instantiate the collection
    hlg = HighLevelGraph.from_collections(name, llg, dependencies=set())
    return new_array_object(hlg, name, divisions=tuple(locs), meta=meta)


def from_uproot_files(
    files: list[str],
    tree_name: str,
    branches: list[str] | None = None,
) -> Array:
    token = tokenize(files, tree_name, branches)
    name = f"from-uproot-{token}"

    # use first 5 entries in the first file to derive meta
    tree1 = uproot.open(files[0])[tree_name]
    first5 = tree1.arrays(branches, entry_start=0, entry_stop=5)
    v2first5 = ak.Array(v1_to_v2(first5.layout))
    meta = ak.Array(v2first5.layout.typetracer.forget_length())

    # low level graph; one partition per file.
    llg = {
        (name, i): (
            UprootReadWrapper(
                fname,
                tree_name,
                branches,
            ),
            None,
            None,
        )
        for i, fname in enumerate(files)
    }

    # instantiate the collection
    hlg = HighLevelGraph.from_collections(name, llg, dependencies=set())
    return new_array_object(hlg, name, npartitions=len(files), meta=meta)
