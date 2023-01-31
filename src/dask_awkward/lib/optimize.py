from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Hashable, Mapping
from typing import Any

import dask.config
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync

from dask_awkward.layers import AwkwardInputLayer

log = logging.getLogger(__name__)


def all_optimizations(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **_: Any,
) -> Mapping:
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
    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore

    layer_to_necessary_columns = _necessary_columns(dsk)

    for name, neccols in layer_to_necessary_columns.items():
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
    ]


def _output_layer_names(dsk: HighLevelGraph) -> list[str]:
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
    return [n for n, v in dsk.layers.items() if (v.annotations or {}).get("ak_output")]


def _has_projectable_awkward_io_layer(dsk: HighLevelGraph) -> bool:
    for _, v in dsk.layers.items():
        if isinstance(v, AwkwardInputLayer) and hasattr(v.io_func, "project_columns"):
            return True
    return False


def _mock_output_func(*args, **kwargs):
    import awkward as ak

    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, ak.Array):
            arg.layout._touch_data(recursive=True)


def _mock_output(layer):
    assert len(layer.dsk) == 1

    new_layer = copy.deepcopy(layer)
    mp = new_layer.mapping.copy()
    for k in iter(mp.keys()):
        mp[k] = (_mock_output_func,) + mp[k][1:]
    new_layer.mapping = mp
    return new_layer


def _get_column_reports(dsk: HighLevelGraph) -> dict[str, Any]:
    if not _has_projectable_awkward_io_layer(dsk):
        return {}

    import awkward as ak

    layers = dsk.layers.copy()  # type: ignore
    deps = dsk.dependencies.copy()  # type: ignore
    reports = {}

    for name in _projectable_input_layer_names(dsk):
        layers[name], report = layers[name].mock()
        reports[name] = report

    for name in _output_layer_names(dsk):
        layers[name] = _mock_output(layers[name])

    hlg = HighLevelGraph(layers, deps)
    outlayer = list(hlg.layers.values())[-1]

    try:
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

    if isinstance(out, ak.Array):
        out.layout._touch_data(recursive=True)
    return reports


def _necessary_columns(dsk: HighLevelGraph) -> dict[str, list[str]]:
    kv = {}
    for name, report in _get_column_reports(dsk).items():
        cols = set(report.data_touched)
        select = []
        for col in cols:
            if col is None or col == name:
                continue
            n, c = col.split(".", 1)
            if n == name:
                select.append(c)
        kv[name] = select
    return kv
