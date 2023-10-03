from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from dask.base import unpack_collections
from dask.highlevelgraph import HighLevelGraph

if TYPE_CHECKING:
    from awkward.highlevel import Array as AwkArray

    from dask_awkward.lib.core import Array


class NecessaryBuffers(NamedTuple):
    data_and_shape: frozenset[str]
    shape_only: frozenset[str]


def report_necessary_buffers(
    *args: Any, traverse: bool = True
) -> dict[str, NecessaryBuffers | None]:
    r"""Determine the buffer keys necessary to compute a collection.
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
    dict[str, NecessaryBuffers | None]
        Mapping that pairs the input layers in the graph to the
        typetracer report objects that have been populated by column
        optimisation of the given layer.
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
    >>> dak.report_necessary_buffers(x)
    {
        "from-parquet-abc123": NecessaryBuffers(
            data_and_shape=frozenset(...), shape_only=frozenset(...)
        )
    }

    """
    import dask_awkward.lib.optimize as o

    collections, _ = unpack_collections(*args, traverse=traverse)
    if not collections:
        return {}

    seen_names = set()

    name_to_necessary_buffers: dict[str, NecessaryBuffers | None] = {}
    for obj in collections:
        dsk = obj if isinstance(obj, HighLevelGraph) else obj.dask
        projection_data = o._prepare_buffer_projection(dsk)
        if projection_data is None:
            # Ensure that we have a record of this layer
            seen_names.update(dsk.keys())
            continue

        # Unpack projection information
        layer_to_reports, _ = projection_data
        for name, report in layer_to_reports.items():
            existing_buffers = name_to_necessary_buffers.setdefault(
                name, NecessaryBuffers(frozenset(), frozenset())
            )
            data_and_shape = frozenset(report.data_touched)
            shape_only = frozenset(report.shape_touched) - data_and_shape

            # Update set of touched keys
            name_to_necessary_buffers[name] = NecessaryBuffers(
                data_and_shape=existing_buffers.data_and_shape | data_and_shape,
                shape_only=existing_buffers.shape_only | shape_only,
            )

    # Populate result with names of seen layers
    for k in seen_names:
        name_to_necessary_buffers.setdefault(k, None)
    return name_to_necessary_buffers


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
