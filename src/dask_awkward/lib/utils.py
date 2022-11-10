from __future__ import annotations

from typing import Literal

import dask_awkward.lib.optimize as opt
from dask_awkward.lib.core import Array, Record, Scalar


def necessary_columns(
    collection: Array | Record | Scalar,
    method: Literal["brute"] | Literal["getitem"],
) -> dict:
    if method == "brute":
        necessary = opt._necessary_columns_brute(collection.dask)
        return {None: necessary}
    elif method == "getitem":
        return opt._layers_and_columns_getitem(collection.dask)
    raise ValueError(  # pragma: no cover
        "method argument should be 'brute' or 'getitem'"
    )
