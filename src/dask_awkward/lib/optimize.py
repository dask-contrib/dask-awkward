from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import dask.config
from awkward.typetracer import touch_data
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync

from dask_awkward.layers import AwkwardBlockwiseLayer, AwkwardInputLayer

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport

log = logging.getLogger(__name__)

COLUMN_OPT_FAILED_WARNING_MSG = """The necessary columns optimization failed; exception raised:

{exception} with message {message}.

Please see the FAQ section of the docs for more information:
https://dask-awkward.readthedocs.io/en/stable/more/faq.html

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
        dsk = HighLevelGraph.from_collections(str(id(dsk)), dsk, dependencies=())

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
    dsk: HighLevelGraph,
    keys: Hashable | list[Hashable] | set[Hashable],
    **_: Any,
) -> Mapping:
    """Run optimizations specific to dask-awkward.

    This is currently limited to determining the necessary columns for
    input layers.

    """
    if dask.config.get("awkward.optimization.enabled"):
        which = dask.config.get("awkward.optimization.which")
        if "columns" in which:
            dsk = optimize_columns(dsk)
        if "layer-chains" in which:
            dsk = rewrite_layer_chains(dsk, keys)

    return dsk


def _prepare_buffer_projection(
    dsk: HighLevelGraph,
) -> tuple[dict[str, TypeTracerReport], dict[str, Any]] | None:
    """Pair layer names with lists of necessary columns."""
    import awkward as ak

    if not _has_projectable_awkward_io_layer(dsk):
        return None

    layer_to_projection_state: dict[str, Any] = {}
    layer_to_reports: dict[str, TypeTracerReport] = {}
    projection_layers = dict(dsk.layers)

    for name, lay in dsk.layers.items():
        if isinstance(lay, AwkwardInputLayer):
            if lay.is_projectable:
                # Insert mocked array into layers, replacing generation func
                # Keep track of mocked state
                (
                    projection_layers[name],
                    layer_to_reports[name],
                    layer_to_projection_state[name],
                ) = lay.prepare_for_projection()
            elif lay.is_mockable:
                projection_layers[name] = lay.mock()
        elif hasattr(lay, "mock"):
            projection_layers[name] = lay.mock()

    for name in _ak_output_layer_names(dsk):
        projection_layers[name] = _mock_output(projection_layers[name])

    hlg = HighLevelGraph(projection_layers, dsk.dependencies)

    # this loop builds up what are the possible final leaf nodes by
    # inspecting the dependents dictionary. If something does not have
    # a dependent, it must be the end of a graph. These are the things
    # we need to compute for; we only use a single partition (the
    # first). for a single collection `.compute()` this list will just
    # be length 1; but if we are using `dask.compute` to pass in
    # multiple collections to be computed simultaneously, this list
    # will increase in length.
    leaf_layers_keys = [
        (k, 0) for k, v in dsk.dependents.items() if isinstance(v, set) and len(v) == 0
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
                touch_data(out)
    except Exception as err:
        on_fail = dask.config.get("awkward.optimization.on-fail")
        # this is the default, throw a warning but skip the optimization.
        if on_fail == "warn":
            warnings.warn(
                COLUMN_OPT_FAILED_WARNING_MSG.format(exception=type(err), message=err)
            )
        # option "pass" means do not throw warning but skip the optimization.
        elif on_fail == "pass":
            log.debug("Column projection optimization failed; optimization skipped.")
        # option "raise" to raise the exception here
        elif on_fail == "raise":
            raise
        else:
            raise ValueError(
                f"Invalid awkward.optimization.on-fail option: {on_fail}.\n"
                "Valid options are 'warn', 'pass', or 'raise'."
            )
        return None
    else:
        return layer_to_reports, layer_to_projection_state


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
        Task graph to optimize.

    Returns
    -------
    HighLevelGraph
        New, optimized task graph with column-projected ``AwkwardInputLayer``.

    """
    projection_data = _prepare_buffer_projection(dsk)
    if projection_data is None:
        return dsk

    # Unpack result
    layer_to_reports, layer_to_projection_state = projection_data

    # Project layers using projection state
    layers = dict(dsk.layers)
    for name, state in layer_to_projection_state.items():
        layers[name] = cast(AwkwardInputLayer, layers[name]).project(
            report=layer_to_reports[name], state=state
        )

    return HighLevelGraph(layers, dsk.dependencies)


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


def _has_projectable_awkward_io_layer(dsk: HighLevelGraph) -> bool:
    """Check if a graph at least one AwkwardInputLayer that is project-able."""
    for _, v in dsk.layers.items():
        if isinstance(v, AwkwardInputLayer) and v.is_projectable:
            return True
    return False


def _touch_all_data(*args, **kwargs):
    """Mock writing an ak.Array to disk by touching data buffers."""
    for arg in args + tuple(kwargs.values()):
        touch_data(arg)


def _mock_output(layer):
    """Update a layer to run the _touch_all_data."""
    assert len(layer.dsk) == 1

    new_layer = copy.deepcopy(layer)
    mp = new_layer.dsk.copy()
    for k in iter(mp.keys()):
        mp[k] = (_touch_all_data,) + mp[k][1:]
    new_layer.dsk = mp
    return new_layer


def rewrite_layer_chains(dsk: HighLevelGraph, keys: Any) -> HighLevelGraph:
    """Smush chains of blockwise layers into a single layer.

    The logic here identifies chains by popping layers (in arbitrary
    order) from a set of all layers in the task graph and walking
    through the dependencies (parent layers) and dependents (child
    layers). If a multi layer chain is discovered we compress it into
    a single layer with the second loop below (for chain in chains;
    that step rewrites the graph). In the chain building logic, if a
    layer exists in the `keys` argument (the keys necessary for the
    compute that we are optimizing for), we shortcircuit the logic to
    ensure we do not chain layers that contain a necessary key inside
    (these layers are called `required_layers` below).

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph to optimize.
    keys : Any
        Keys that are requested by the compute that is being
        optimized.

    Returns
    -------
    HighLevelGraph
        New, optimized task graph.

    """
    # dask.optimization.fuse_liner for blockwise layers
    import copy

    chains = []
    deps = copy.copy(dsk.dependencies)

    required_layers = {k[0] for k in keys}
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
            and lay not in required_layers
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
            and list(parents)[0] not in required_layers
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
        numblocks = [nb[0] for nb in layer0.numblocks.values() if nb[0] is not None][0]  # type: ignore
        deps[outkey] = deps[chain[0]]  # type: ignore
        [deps.pop(ch) for ch in chain[:-1]]  # type: ignore

        subgraph = layer0.dsk.copy()  # type: ignore
        indices = list(layer0.indices)  # type: ignore
        parent = chain[0]

        outlayer.io_deps = layer0.io_deps  # type: ignore
        for chain_member in chain[1:]:
            layer = dsk.layers[chain_member]
            for k in layer.io_deps:  # type: ignore
                outlayer.io_deps[k] = layer.io_deps[k]  # type: ignore
            func, *args = layer.dsk[chain_member]  # type: ignore
            args2 = _recursive_replace(args, layer, parent, indices)
            subgraph[chain_member] = (func,) + tuple(args2)
            parent = chain_member
        outlayer.numblocks = {i[0]: (numblocks,) for i in indices if i[1] is not None}  # type: ignore
        outlayer.dsk = subgraph  # type: ignore
        if hasattr(outlayer, "_dims"):
            del outlayer._dims
        outlayer.indices = tuple(  # type: ignore
            (i[0], (".0",) if i[1] is not None else None) for i in indices
        )
        outlayer.output_indices = (".0",)  # type: ignore
        outlayer.inputs = getattr(layer0, "inputs", set())  # type: ignore
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


def _buffer_keys_for_layer(
    buffer_keys: Iterable[str], known_buffer_keys: frozenset[str]
) -> set[str]:
    return {k for k in buffer_keys if k in known_buffer_keys}
