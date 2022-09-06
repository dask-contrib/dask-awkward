from __future__ import annotations

from dask_awkward.lib.core import Array, Record


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
