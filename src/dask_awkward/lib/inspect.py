from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from dask.base import unpack_collections
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers import AwkwardInputLayer

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
        Mapping that pairs the input layers in the graph to objects
        describing the data and shape buffers that have been tagged
        as required by column optimisation of the given layer.

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

        # If the projection failed, or there are no input layers
        if projection_data is None:
            # Ensure that we have a record of the seen layers, if they're inputs
            for name, layer in dsk.items():
                if isinstance(layer, AwkwardInputLayer):
                    seen_names.add(name)
            continue

        # Unpack projection information
        layer_to_reports, _ = projection_data
        for name, report in layer_to_reports.items():
            existing_buffers = name_to_necessary_buffers.setdefault(
                name, NecessaryBuffers(frozenset(), frozenset())
            )
            # Compute the shape-only keys in addition to the data and shape
            data_and_shape = frozenset(report.data_touched)
            shape_only = frozenset(report.shape_touched) - data_and_shape

            # Update set of touched keys
            assert existing_buffers is not None
            name_to_necessary_buffers[name] = NecessaryBuffers(
                data_and_shape=existing_buffers.data_and_shape | data_and_shape,
                shape_only=existing_buffers.shape_only | shape_only,
            )

    # Populate result with names of seen layers
    for k in seen_names:
        name_to_necessary_buffers.setdefault(k, None)
    return name_to_necessary_buffers


def report_necessary_columns(
    *args: Any, traverse: bool = True
) -> dict[str, frozenset[str] | None]:
    r"""Get columns necessary to compute a collection

    This function is specific to sources that are columnar (e.g. Parquet).

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
    dict[str, frozenset[str] | None]
        Mapping that pairs the input layers in the graph to the
        set of necessary IO columns that have been identified by column
        optimisation of the given layer. If the layer is not backed by a
        columnar source, then None is returned instead of a set.

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
    >>> dak.report_necessary_columns(x)
    {
        "from-parquet-abc123": frozenset({"bar", "baz.x"})
    }

    """
    import dask_awkward.lib.optimize as o

    collections, _ = unpack_collections(*args, traverse=traverse)
    if not collections:
        return {}

    seen_names = set()

    name_to_necessary_columns: dict[str, frozenset | None] = {}
    for obj in collections:
        dsk = obj if isinstance(obj, HighLevelGraph) else obj.dask
        projection_data = o._prepare_buffer_projection(dsk)

        # If the projection failed, or there are no input layers
        if projection_data is None:
            # Ensure that we have a record of the seen layers, if they're inputs
            for name, layer in dsk.items():
                if isinstance(layer, AwkwardInputLayer):
                    seen_names.add(name)
            continue

        # Unpack projection information
        layer_to_reports, layer_to_projection_state = projection_data
        for name, report in layer_to_reports.items():
            layer = dsk.layers[name]
            if not (isinstance(layer, AwkwardInputLayer) and layer.is_columnar):
                continue

            existing_columns = name_to_necessary_columns.setdefault(name, frozenset())

            assert existing_columns is not None
            # Update set of touched keys
            name_to_necessary_columns[
                name
            ] = existing_columns | layer.necessary_columns(
                report=report, state=layer_to_projection_state[name]
            )

    # Populate result with names of seen layers
    for k in seen_names:
        name_to_necessary_columns.setdefault(k, None)
    return name_to_necessary_columns


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
    if (factor is None and probability is None) or (
        factor is not None and probability is not None
    ):
        raise ValueError("Give exactly one of factor or probability")
    if factor:
        return arr.map_partitions(lambda x: x[::factor], meta=arr._meta)
    assert probability is not None
    proba = float(probability)
    return arr.map_partitions(
        lambda x: x[_random_boolean_like(x, proba)], meta=arr._meta
    )
