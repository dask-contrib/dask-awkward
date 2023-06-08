from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from dask.base import unpack_collections
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers import AwkwardInputLayer

if TYPE_CHECKING:
    from dask_awkward.lib.core import Array


def necessary_columns(*args: Any, traverse: bool = True) -> dict[str, list[str]]:
    r"""Determine the columns necessary to compute a collection.

    Parameters
    ----------
    *args : Dask collections or HighLevelGraphs
        The collection (or collection graph) of interest. These can be
        individual objects, lists, sets, or dictionaries.
    traverse : bool, optional
        If True (default), builtin Python collections are traversed
        looking for any Dask collections they might contain.

    Returns
    -------
    dict[str, list[str]]
        Mapping that pairs the input layers in the graph to the
        columns that have been determined necessary from that layer.
        These are not necessarily in the same order as the original
        input.

    Examples
    --------
    If we have a hypothetical parquet dataset (``ds``) with the fields

    - "foo"
    - "bar"
    - "baz"

    And the "baz" field has fields

    - "x"
    - "y"

    The calculation of ``ds.bar + ds.baz.x`` will only require the
    ``bar`` and ``baz.x`` columns from the parquet file.

    >>> import dask_awkward as dak
    >>> ds = dak.from_parquet("some-dataset")
    >>> ds.fields
    ["foo", "bar", "baz"]
    >>> ds.baz.fields
    ["x", "y"]
    >>> x = ds.bar + ds.baz.x
    >>> dak.necessary_columns(x)
    {"from-parquet-abc123": ["bar", "baz.x"]}

    Notice that ``foo`` and ``baz.y`` are not determined to be
    necessary.

    """
    import dask_awkward.lib.optimize as o

    collections, _ = unpack_collections(*args, traverse=traverse)
    if not collections:
        return {}

    out: dict[str, list[str]] = {}
    for obj in collections:
        dsk = obj if isinstance(obj, HighLevelGraph) else obj.dask
        cols_this_dsk = o._necessary_columns(dsk)

        for name in cols_this_dsk:
            neccols = cols_this_dsk[name]
            if not isinstance(dsk.layers[name], AwkwardInputLayer):
                raise TypeError(f"Layer {name} should be an AwkwardInputLayer.")
            cols_this_dsk[name] = o._prune_wildcards(neccols, dsk.layers[name]._meta)

        for key, cols in cols_this_dsk.items():
            prev = out.get(key, [])
            update = list(set(prev + cols))
            out[key] = update

    return out


def sample(arr, factor: int | None = None, probability: float | None = None) -> Array:
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
            lambda x: x[np.random.random(len(x)) < probability], meta=arr._meta
        )
