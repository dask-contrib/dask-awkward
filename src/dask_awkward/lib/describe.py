from __future__ import annotations

import awkward as ak

from dask_awkward.lib.core import Array, Record, Scalar


def fields(collection: Array | Record) -> list[str] | None:
    """Get the fields of a Array collection.

    Parameters
    ----------
    collection : dask_awkward.Array or dask_awkward.Record
        Array or Record collection

    Returns
    -------
    list[str] or None
        The fields of the collection; if the collection does not
        contain metadata ``None`` is returned.

    """
    return collection.fields


def backend(*arrays: Array | Record) -> str:
    """Get the name of the backend used by `arrays`.

    Parameters
    ----------
        arrays : dask_awkward.Array or dask_awkward.Record
            Array or Record collection

    Returns
    -------
    str
        The backend name, which is always `"typetracer"` for
        dask-awkward arrays.
    """
    return ak.backend(
        *[x._meta if isinstance(x, (Array, Record, Scalar)) else x for x in arrays]
    )
