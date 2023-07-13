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

from dask_awkward.layers import AwkwardBlockwiseLayer, AwkwardInputLayer

log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from awkward import Array as AwkwardArray


COLUMN_OPT_FAILED_WARNING_MSG = """The necessary columns optimization failed; exception raised:

{exception} with message {message}.

Please see the FAQ section of the docs for more information:
https://dask-awkward.readthedocs.io/en/stable/me-faq.html

"""


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

        # blockwise layer chaining optimization.
        dsk = rewrite_layer_chains(dsk)

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


def rewrite_layer_chains(dsk: HighLevelGraph) -> HighLevelGraph:
    # dask.optimization.fuse_liner for blockwise layers
    import copy

    chains = []
    deps = dsk.dependencies.copy()

    layers = {}
    # find chains; each chain list is at least two keys long
    dependents = dsk.dependents
    all_layers = set(dsk.layers)
    while all_layers:
        lay = all_layers.pop()
        val = dsk.layers[lay]
        if not isinstance(val, AwkwardBlockwiseLayer):
            # shortcut to avoid making comparisons
            layers[lay] = val  # passthrough unchanged
            continue
        children = dependents[lay]
        chain = [lay]
        lay0 = lay
        while (
            len(children) == 1
            and dsk.dependencies[list(children)[0]] == {lay}
            and isinstance(dsk.layers[list(children)[0]], AwkwardBlockwiseLayer)
            and len(dsk.layers[lay]) == len(dsk.layers[list(children)[0]])
        ):
            # walk forwards
            lay = list(children)[0]
            chain.append(lay)
            all_layers.remove(lay)
            children = dependents[lay]
        lay = lay0
        parents = dsk.dependencies[lay]
        while (
            len(parents) == 1
            and dependents[list(parents)[0]] == {lay}
            and isinstance(dsk.layers[list(parents)[0]], AwkwardBlockwiseLayer)
            and len(dsk.layers[lay]) == len(dsk.layers[list(parents)[0]])
        ):
            # walk backwards
            lay = list(parents)[0]
            chain.insert(0, lay)
            all_layers.remove(lay)
            parents = dsk.dependencies[lay]
        if len(chain) > 1:
            chains.append(chain)
            layers[chain[-1]] = copy.copy(
                dsk.layers[chain[-1]]
            )  # shallow copy to be mutated
        else:
            layers[lay] = val  # passthrough unchanged

    # do rewrite
    for chain in chains:
        # inputs are the inputs of chain[0]
        # outputs are the outputs of chain[-1]
        # .dsk is composed from the .dsk of each layer
        outkey = chain[-1]
        layer0 = dsk.layers[chain[0]]
        outlayer = layers[outkey]
        numblocks = [nb[0] for nb in layer0.numblocks.values() if nb[0] is not None][0]
        deps[outkey] = deps[chain[0]]
        [deps.pop(ch) for ch in chain[:-1]]

        subgraph = layer0.dsk.copy()
        indices = list(layer0.indices)
        parent = chain[0]

        outlayer.io_deps = layer0.io_deps
        for chain_member in chain[1:]:
            layer = dsk.layers[chain_member]
            for k in layer.io_deps:
                outlayer.io_deps[k] = layer.io_deps[k]
            func, *args = layer.dsk[chain_member]
            args2 = _recursive_replace(args, layer, parent, indices)
            subgraph[chain_member] = (func,) + tuple(args2)
            parent = chain_member
        outlayer.numblocks = {i[0]: (numblocks,) for i in indices if i[1] is not None}
        outlayer.dsk = subgraph
        if hasattr(outlayer, "_dims"):
            del outlayer._dims
        outlayer.indices = tuple(
            (i[0], (".0",) if i[1] is not None else None) for i in indices
        )
        outlayer.output_indices = (".0",)
        outlayer.inputs = getattr(layer0, "inputs", set())
        if hasattr(outlayer, "_cached_dict"):
            del outlayer._cached_dict  # reset, since original can be mutated
    return HighLevelGraph(layers, deps)


def _recursive_replace(args, layer, parent, indices):
    args2 = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith("__dask_blockwise__"):
            ind = int(arg[18:])
            if layer.indices[ind][1] is None:
                # this is a simple arg
                args2.append(layer.indices[ind][0])
            elif layer.indices[ind][0] == parent:
                # arg refers to output of previous layer
                args2.append(parent)
            else:
                # arg refers to things defined in io_deps
                indices.append(layer.indices[ind])
                args2.append(f"__dask_blockwise__{len(indices) - 1}")
        elif isinstance(arg, list):
            args2.append(_recursive_replace(arg, layer, parent, indices))
        elif isinstance(arg, tuple):
            args2.append(tuple(_recursive_replace(arg, layer, parent, indices)))
        # elif isinstance(arg, dict):
        else:
            args2.append(arg)
    return args2


def _get_column_reports(dsk: HighLevelGraph) -> dict[str, Any]:
    """Get the TypeTracerReport for each input layer in a task graph."""
    if not _has_projectable_awkward_io_layer(dsk):
        return {}

    import awkward as ak

    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore
    dependents = dsk.dependents

    reports = {}

    # make labelled report
    projectable = _projectable_input_layer_names(dsk)
    for name, lay in dsk.layers.copy().items():
        if name in projectable:
            layers[name], report = lay.mock()
            reports[name] = report
        elif hasattr(lay, "mock"):
            layers[name], _ = lay.mock()

    for name in _ak_output_layer_names(dsk):
        layers[name] = _mock_output(layers[name])

    for name in _opt_touch_all_layer_names(dsk):
        layers[name] = _touch_and_call(layers[name])

    hlg = HighLevelGraph(layers, deps)

    # this loop builds up what are the possible final leaf nodes by
    # inspecting the dependents dictionary. If something does not have
    # a dependent, it must be the end of a graph. These are the things
    # we need to compute for; we only use a single partition (the
    # first). for a single collection `.compute()` this list will just
    # be length 1; but if we are using `dask.compute` to pass in
    # multiple collections to be computed simultaneously, this list
    # will increase in length.
    leaf_layers_keys = [
        (k, 0) for k, v in dependents.items() if isinstance(v, set) and len(v) == 0
    ]

    # now we try to compute for each possible output layer key (leaf
    # node on partition 0); this will cause the typetacer reports to
    # get correct fields/columns touched. If the result is a record or
    # an array we of course want to touch all of the data/fields.
    try:
        for layer in hlg.layers.values():
            layer.__dict__.pop("_cached_dict", None)
        results = get_sync(hlg, leaf_layers_keys)
        for out in results:
            if isinstance(out, (ak.Array, ak.Record)):
                out.layout._touch_data(recursive=True)
    except Exception as err:
        on_fail = dask.config.get("awkward.optimization.on-fail")
        # this is the default, throw a warning but skip the optimization.
        if on_fail == "warn":
            warnings.warn(
                COLUMN_OPT_FAILED_WARNING_MSG.format(exception=type(err), message=err)
            )
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
            else:
                good_columns.append(col[:-2])

    return good_columns
