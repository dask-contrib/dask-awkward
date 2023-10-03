from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from awkward.highlevel import Array as AwkArray

    from dask_awkward.lib.core import Array


def _random_boolean_like(array_like: AwkArray, probability: float) -> AwkArray:
    import awkward as ak

    backend = ak.backend(array_like)
    layout = ak.to_layout(array_like)

    if ak.backend(array_like) == "typetracer":
        return ak.Array(
            ak.to_layout(np.empty(0, dtype=np.bool_)).to_typetracer(forget_length=True),
            behavior=array_like.behavior,
        )
    else:
        return ak.Array(
            np.random.random(layout.length) < probability,
            behavior=array_like.behavior,
            backend=backend,
        )


def sample(
    arr: Array,
    factor: int | None = None,
    probability: float | None = None,
) -> Array:
    """Decimate the data to a smaller number of rows.

    Must give either `factor` or `probability`.

    Parameters
    ----------
    arr : dask_awkward.Array
        Array collection to sample
    factor : int, optional
        if given, every Nth row will be kept. The counting restarts for each
        partition, so reducing the row count by an exact factor is not guaranteed
    probability : float, optional
        a number between 0 and 1, giving the chance of any particular
        row surviving. For instance, for probability=0.1, roughly 1-in-10
        rows will remain.

    """
    if not (factor is None) ^ (probability is None):
        raise ValueError("Give exactly one of factor or probability")
    if factor:
        return arr.map_partitions(lambda x: x[::factor], meta=arr._meta)
    else:
        return arr.map_partitions(
            lambda x: x[_random_boolean_like(x, probability)], meta=arr._meta
        )
