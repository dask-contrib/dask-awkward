from __future__ import annotations

from math import ceil

import awkward._v2 as ak
import fsspec
import uproot
from awkward._v2.tmp_for_testing import v1_to_v2
from dask.base import tokenize

from dask_awkward.core import Array
from dask_awkward.io.io import from_map


class _UprootReadStartStopFn:
    def __init__(self, source: str, tree_name: str, branches: list[str] | None) -> None:
        self.branches = branches
        self.tree = uproot.open(source)[tree_name]

    def __call__(self, start_stop: tuple[int | None, ...]) -> ak.Array:
        start, stop = start_stop
        arr = self.tree.arrays(self.branches, entry_start=start, entry_stop=stop)
        # uproot returns an awkward v1 array; we convert to v2
        return ak.Array(v1_to_v2(arr.layout))


class _UprootReadIndivFileFn:
    def __init__(self, tree_name: str, branches: list[str] | None):
        self.tree_name = tree_name
        self.branches = branches

    def __call__(self, file_name: str) -> ak.Array:
        tree = uproot.open(file_name)[self.tree_name]
        return ak.Array(v1_to_v2(tree.arrays(self.branches).layout))


def from_root(
    source: str,
    tree_name: str,
    npartitions: int,
    branches: list[str] | None = None,
) -> Array:
    token = tokenize(source, tree_name, npartitions, branches)

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

    return from_map(
        _UprootReadStartStopFn(source, tree_name, branches),
        start_stop_pairs,
        label="from-root",
        token=token,
        meta=meta,
    )


def from_root_files(
    files: list[str] | str,
    tree_name: str,
    branches: list[str] | None = None,
) -> Array:
    _, fstoken, paths = fsspec.get_fs_token_paths(files)

    token = tokenize(fstoken, tree_name, branches)

    # use first 5 entries in the first file to derive meta
    tree1 = uproot.open(paths[0])[tree_name]
    first5 = tree1.arrays(branches, entry_start=0, entry_stop=5)
    v2first5 = ak.Array(v1_to_v2(first5.layout))
    meta = ak.Array(v2first5.layout.typetracer.forget_length())

    return from_map(
        _UprootReadIndivFileFn(tree_name, branches),
        paths,
        label="from-root",
        token=token,
        meta=meta,
    )
