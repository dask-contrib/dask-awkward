from __future__ import annotations

from typing import Literal

import dask_awkward.lib.optimize as opt
from dask_awkward.lib.core import Array, Record, Scalar


def necessary_columns(
    collection: Array | Record | Scalar,
    strategy: Literal["brute"] | Literal["getitem"],
) -> dict:
    if strategy == "brute":
        necessary = opt._necessary_columns_brute(collection.dask)
        return {None: necessary}
    elif strategy == "getitem":
        return opt._layers_and_columns_getitem(collection.dask)
    raise ValueError(  # pragma: no cover
        "strategy argument should be 'brute' or 'getitem'"
    )
