from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import TYPE_CHECKING, Any, Callable, Hashable, Mapping

import awkward._v2 as ak
import numpy as np
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import HighLevelGraph
from dask.utils import funcname

from dask_awkward.core import (
    DaskAwkwardNotImplemented,
    map_partitions,
    new_array_object,
    typetracer_array,
)
from dask_awkward.utils import LazyInputsDict

if TYPE_CHECKING:
    from dask.array.core import Array as DaskArray
    from dask.delayed import Delayed

    from dask_awkward.core import Array


class FromAwkwardWrapper:
    def __init__(self, arr: ak.Array) -> None:
        self.arr = arr

    def __call__(self, source: tuple[int, int]) -> ak.Array:
        start, stop = source
        return self.arr[start:stop]


def from_awkward(source: ak.Array, npartitions: int, name: str | None = None) -> Array:
    if name is None:
        name = f"from-awkward-{tokenize(source, npartitions)}"
    nrows = len(source)
    chunksize = int(ceil(nrows / npartitions))
    locs = list(range(0, nrows, chunksize)) + [nrows]
    inputs = list(zip(locs[:-1], locs[1:]))
    meta = typetracer_array(source)
    return from_map(
        FromAwkwardWrapper(source),
        inputs,
        label="from-awkward",
        token=tokenize(source, npartitions),
        divisions=tuple(locs),
        meta=meta,
    )


def from_delayed(
    arrays: list[Delayed] | Delayed,
    meta: ak.Array | None = None,
    divisions: tuple[int | None, ...] | None = None,
    prefix: str = "from-delayed",
) -> Array:
    """Create a Dask Awkward Array from Dask Delayed objects.

    Parameters
    ----------
    arrays : list[Delayed] | Delayed
        Iterable of ``dask.delayed.Delayed`` objects (or a single
        object). Each Delayed object represents a single partition in
        the resulting awkward array.
    meta : ak.Array, optional
        Metadata (typetracer array) if known, if ``None`` the first
        partition (first element of the list of ``Delayed`` objects)
        will be computed to determine the metadata.
    divisions : tuple[int | None, ...], optional
        Partition boundaries (if known).
    prefix : str
        Prefix for the keys in the task graph.

    Returns
    -------
    Array
        Resulting Array collection.

    """
    from dask.delayed import Delayed

    parts = [arrays] if isinstance(arrays, Delayed) else arrays
    name = f"{prefix}-{tokenize(arrays)}"
    dsk = {(name, i): part.key for i, part in enumerate(parts)}
    if divisions is None:
        divs: tuple[int | None, ...] = (None,) * (len(arrays) + 1)
    else:
        divs = tuple(divisions)
        if len(divs) != len(arrays) + 1:
            raise ValueError("divisions must be a tuple of length len(arrays) + 1")
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=arrays)
    return new_array_object(hlg, name=name, meta=meta, divisions=divs)


def to_delayed(array: Array, optimize_graph: bool = True) -> list[Delayed]:
    """Convert the collection to a list of delayed objects.

    One dask.delayed.Delayed object per partition.

    Parameters
    ----------
    optimize_graph : bool
        If True the task graph associated with the collection will
        be optimized before conversion to the list of Delayed
        objects.

    Returns
    -------
    list[Delayed]
        List of delayed objects (one per partition).

    """
    from dask.delayed import Delayed

    keys = array.__dask_keys__()
    graph = array.__dask_graph__()
    layer = array.__dask_layers__()[0]
    if optimize_graph:
        graph = array.__dask_optimize__(graph, keys)
        layer = f"delayed-{array.name}"
        graph = HighLevelGraph.from_collections(layer, graph, dependencies=())
    return [Delayed(k, graph, layer=layer) for k in keys]


def to_dask_array(array: Array) -> DaskArray:
    from dask.array.core import new_da_object

    new = map_partitions(ak.to_numpy, array)
    graph = new.dask
    dtype = new._meta.dtype if new._meta is not None else None

    # TODO: define chunks if we can.
    #
    # if array.known_divisions:
    #     divs = np.array(array.divisions)
    #     chunks = (tuple(divs[1:] - divs[:-1]),)

    chunks = ((np.nan,) * array.npartitions,)
    if new._meta is not None:
        if new._meta.ndim > 1:
            raise DaskAwkwardNotImplemented(
                "only one dimensional arrays are supported."
            )
    return new_da_object(
        graph,
        new.name,
        meta=None,
        chunks=chunks,
        dtype=dtype,
    )


def from_dask_array(array: DaskArray) -> Array:
    """Convert a Dask Array collection to a Dask Awkard Array collection.

    Parameters
    ----------
    array : dask.array.Array
        Array to convert.

    Returns
    -------
    Array
        The Awkward Array Dask collection.

    Examples
    --------
    >>> import dask.array as da
    >>> import dask_awkward as dak
    >>> x = da.ones(1000, chunks=250)
    >>> y = dak.from_dask_array(x)
    >>> y
    dask.awkward<from-dask-array, npartitions=4>

    """

    from dask.blockwise import blockwise as dask_blockwise

    token = tokenize(array)
    name = f"from-dask-array-{token}"
    meta = typetracer_array(ak.from_numpy(array._meta))
    pairs = [array.name, "i"]
    numblocks = {array.name: array.numblocks}
    layer = dask_blockwise(
        ak.from_numpy,
        name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
    )
    hlg = HighLevelGraph.from_collections(name, layer, dependencies=[array])
    if np.any(np.isnan(array.chunks)):
        return new_array_object(hlg, name, npartitions=array.npartitions, meta=meta)
    else:
        divs = (0, *np.cumsum(array.chunks))
        return new_array_object(hlg, name, divisions=divs, meta=meta)


class AwkwardIOLayer(Blockwise):
    def __init__(
        self,
        name: str,
        inputs: Any,
        io_func: Callable,
        label: str | None = None,
        produces_tasks: bool = False,
        creation_info: dict | None = None,
        annotations: dict | None = None,
    ):
        self.name = name
        self.inputs = inputs
        self.io_func = io_func
        self.label = label
        self.produces_tasks = produces_tasks
        self.annotations = annotations
        self.creation_info = creation_info

        io_arg_map = BlockwiseDepDict(
            mapping=LazyInputsDict(self.inputs),  # type: ignore
            produces_tasks=self.produces_tasks,
        )

        dsk = {self.name: (io_func, blockwise_token(0))}
        super().__init__(
            output=self.name,
            output_indices="i",
            dsk=dsk,
            indices=[(io_arg_map, "i")],
            numblocks={},
            annotations=annotations,
        )


def from_map(
    func: Callable,
    inputs: Sequence[Hashable],
    label: str | None = None,
    token: str | None = None,
    divisions: tuple[int, ...] | None = None,
    meta: ak.Array | None = None,
    **kwargs: Any,
) -> Array:

    # Define collection name
    label = label or funcname(func)
    token = token or tokenize(func, inputs, meta, **kwargs)
    name = f"{label}-{token}"

    # Check for `produces_tasks` and `creation_info`
    produces_tasks = kwargs.pop("produces_tasks", False)
    creation_info = kwargs.pop("creation_info", None)

    deps: set[Any] | list[Any] = set()
    dsk: Mapping = AwkwardIOLayer(
        name,
        inputs,
        func,
        produces_tasks=produces_tasks,
        creation_info=creation_info,
    )

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    if divisions is not None:
        return new_array_object(hlg, name, meta=meta, divisions=divisions)
    else:
        return new_array_object(hlg, name, meta=meta, npartitions=len(inputs))
