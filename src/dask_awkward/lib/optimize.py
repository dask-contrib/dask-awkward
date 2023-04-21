from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any

import dask.config
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync

from dask_awkward.layers import AwkwardInputLayer

log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from awkward import Array as AwkwardArray


def all_optimizations(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **_: Any,
) -> Mapping:
    """Run all optimizations that benefit dask-awkward computations.

    This function will run both dask-awkward specific and upstream
    general optimizations from core dask.

    """
    if not isinstance(keys, (list, set)):
        keys = (keys,)  # pragma: no cover
    keys = tuple(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())

    else:
        # Perform dask-awkward specific optimizations.
        dsk = optimize(dsk, keys=keys)
        # Perform Blockwise optimizations for HLG input
        dsk = optimize_blockwise(dsk, keys=keys)
        # fuse nearby layers
        dsk = fuse_roots(dsk, keys=keys)  # type: ignore

    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore

    return dsk


def optimize(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **_: Any,
) -> Mapping:
    """Run optimizations specific to dask-awkward.

    This is currently limited to determining the necessary columns for
    input layers.

    """
    if dask.config.get("awkward.optimization.enabled", default=False):
        dsk = optimize_columns(dsk)  # type: ignore
    return dsk


def optimize_columns(dsk: HighLevelGraph) -> HighLevelGraph:
    """Run column projection optimization.

    This optimization determines which columns from an
    ``AwkwardInputLayer`` are necessary for a complete computation.

    For example, if a parquet dataset is loaded with fields:
    ``["foo", "bar", "baz.x", "baz.y"]``

    And the following task graph is made:

    >>> ds = dak.from_parquet("/path/to/dataset")
    >>> z = ds["foo"] - ds["baz"]["y"]

    Upon calling z.compute() the AwkwardInputLayer created in the
    from_parquet call will only read the parquet columns ``foo`` and
    ``baz.y``.

    Parameters
    ----------
    dsk : HighLevelGraph
        Original high level dask graph

    Returns
    -------
    HighLevelGraph
        New dask graph with a modified ``AwkwardInputLayer``.

    """
    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore

    layer_to_necessary_columns = _necessary_columns(dsk)

    for name, neccols in layer_to_necessary_columns.items():
        meta = layers[name]._meta
        neccols = _prune_wildcards(neccols, meta)
        layers[name] = layers[name].project_columns(neccols)

    return HighLevelGraph(layers, deps)


def _projectable_input_layer_names(dsk: HighLevelGraph) -> list[str]:
    """Get list of column-projectable AwkwardInputLayer names.

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph of interest

    Returns
    -------
    list[str]
        Names of the AwkwardInputLayers in the graph that are
        column-projectable.

    """
    return [
        n
        for n, v in dsk.layers.items()
        if isinstance(v, AwkwardInputLayer) and hasattr(v.io_func, "project_columns")
        # following condition means dep/pickled layers cannot be optimised
        and hasattr(v, "_meta")
    ]


def _layers_with_annotation(dsk: HighLevelGraph, key: str) -> list[str]:
    return [n for n, v in dsk.layers.items() if (v.annotations or {}).get(key)]


def _ak_output_layer_names(dsk: HighLevelGraph) -> list[str]:
    """Get a list output layer names.

    Output layer names are annotated with 'ak_output'.

    Parameters
    ----------
    dsk : HighLevelGraph
        Graph of interest.

    Returns
    -------
    list[str]
        Names of the output layers.

    """
    return _layers_with_annotation(dsk, "ak_output")


def _opt_touch_all_layer_names(dsk: HighLevelGraph) -> list[str]:
    return [n for n, v in dsk.layers.items() if hasattr(v, "_opt_touch_all")]
    # return _layers_with_annotation(dsk, "ak_touch_all")


def _has_projectable_awkward_io_layer(dsk: HighLevelGraph) -> bool:
    """Check if a graph at least one AwkwardInputLayer that is project-able."""
    for _, v in dsk.layers.items():
        if isinstance(v, AwkwardInputLayer) and hasattr(v.io_func, "project_columns"):
            return True
    return False


def _touch_all_data(*args, **kwargs):
    """Mock writing an ak.Array to disk by touching data buffers."""
    import awkward as ak

    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, ak.Array):
            arg.layout._touch_data(recursive=True)


def _mock_output(layer):
    """Update a layer to run the _touch_all_data."""
    assert len(layer.dsk) == 1

    new_layer = copy.deepcopy(layer)
    mp = new_layer.dsk.copy()
    for k in iter(mp.keys()):
        mp[k] = (_touch_all_data,) + mp[k][1:]
    new_layer.dsk = mp
    return new_layer


def _touch_and_call_fn(fn, *args, **kwargs):
    _touch_all_data(*args, **kwargs)
    return fn(*args, **kwargs)


def _touch_and_call(layer):
    assert len(layer.dsk) == 1

    new_layer = copy.deepcopy(layer)
    mp = new_layer.dsk.copy()
    for k in iter(mp.keys()):
        mp[k] = (_touch_and_call_fn,) + mp[k]
    new_layer.dsk = mp
    return new_layer


def _get_column_reports(dsk: HighLevelGraph) -> dict[str, Any]:
    """Get the TypeTracerReport for each input layer in a task graph."""
    if not _has_projectable_awkward_io_layer(dsk):
        return {}

    import awkward as ak

    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore
    reports = {}

    # make labelled report
    projectable = _projectable_input_layer_names(dsk)
    for name, lay in dsk.layers.copy().items():
        if name in projectable:
            layers[name], report = lay.mock()
            reports[name] = report
        elif hasattr(lay, "mock"):
            layers[name] = lay.mock()

    for name in _projectable_input_layer_names(dsk):
        layers[name], report = layers[name].mock()
        reports[name] = report

    for name in _ak_output_layer_names(dsk):
        layers[name] = _mock_output(layers[name])

    for name in _opt_touch_all_layer_names(dsk):
        layers[name] = _touch_and_call(layers[name])

    hlg = HighLevelGraph(layers, deps)
    outlayer = hlg.layers[hlg._toposort_layers()[-1]]

    try:
        for layer in hlg.layers.values():
            layer.__dict__.pop("_cached_dict", None)
        out = get_sync(hlg, list(outlayer.keys())[0])
    except Exception as err:
        on_fail = dask.config.get("awkward.optimization.on-fail")
        # this is the default, throw a warning but skip the optimization.
        if on_fail == "warn":
            warnings.warn(f"Column projection optimization failed: {type(err)}, {err}")
            return {}
        # option "pass" means do not throw warning but skip the optimization.
        elif on_fail == "pass":
            log.debug("Column projection optimization failed; optimization skipped.")
            return {}
        # option "raise" to raise the exception here
        elif on_fail == "raise":
            raise
        else:
            raise ValueError(
                f"Invalid awkward.optimization.on-fail option: {on_fail}.\n"
                "Valid options are 'warn', 'pass', or 'raise'."
            )

    if isinstance(out, (ak.Array, ak.Record)):
        out.layout._touch_data(recursive=True)
    return reports


def _necessary_columns(dsk: HighLevelGraph) -> dict[str, list[str]]:
    """Pair layer names with lists of necessary columns."""
    kv = {}
    for name, report in _get_column_reports(dsk).items():
        cols = {_ for _ in report.data_touched if _ is not None}
        select = []
        for col in sorted(cols):
            if col == name:
                continue
            n, c = col.split(".", 1)
            if n == name:
                if c.endswith("__list__"):
                    cnew = c[:-9].rstrip(".")
                    if cnew not in select:
                        select.append(f"{cnew}.*")
                else:
                    select.append(c)
        kv[name] = select
    return kv


def _prune_wildcards(columns: list[str], meta: AwkwardArray) -> list[str]:
    """Prune wildcard '.*' suffix from necessary columns results.

    The _necessary_columns logic will provide some results of the
    form:

    "foo.bar.*"

    This function will eliminate the wildcard in one of two ways
    (continuing to use "foo.bar.*" as an example):

    1. If "foo.bar" has leaves (subfields) "x", "y" and "z", and _any_
       of those (so "foo.bar.x", for example) also appears in the
       columns list, then essentially nothing will happen (except we
       drop the wildcard string), because we can be sure that a leaf
       of "foo.bar" will be read (in this case it's "foo.bar.x").

    2. If "foo.bar" has multiple leaves but none of them appear in the
       columns list, we will just pick the first one that we find
       (that is, foo.bar.fields[0]).

    Parameters
    ----------
    columns : list[str]
        The "raw" columns deemed necessary by the necessary columns
        logic; can still contain the wildcard syntax we've adopted.
    meta : ak.Array
        The metadata (typetracer array) from the AwkwardInputLayer
        that is getting optimized.

    Returns
    -------
    list[str]
        Columns with the wildcard syntax pruned and (also augmented
        with a leaf node if necessary).

    """

    good_columns: list[str] = []
    wildcard_columns: list[str] = []
    for col in columns:
        if ".*" in col:
            wildcard_columns.append(col)
        else:
            good_columns.append(col)

    for col in wildcard_columns:
        # each time we meet a wildcard column we need to start back
        # with the original meta array.
        imeta = meta
        colsplit = col.split(".")[:-1]
        parts = list(reversed(colsplit))
        while parts:
            part = parts.pop()
            # for unnamed roots part may be an empty string, so we
            # need this if statement.
            if part:
                imeta = imeta[part]

        for field in imeta.fields:
            wholecol = f"{col[:-2]}.{field}"
            if wholecol in good_columns:
                break
        else:
            if imeta.fields:
                good_columns.append(f"{col[:-2]}.{imeta.fields[0]}")

    return good_columns
