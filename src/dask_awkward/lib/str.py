from __future__ import annotations

import awkward as ak

from dask_awkward.lib.core import Array, map_partitions


def split_whitespace(
    array: Array,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    highlevel: bool = True,
    behavior: dict | None = None,
):
    return map_partitions(
        ak.str.split_whitespace,
        array,
        max_splits=max_splits,
        reverse=reverse,
        behavior=behavior,
    )
