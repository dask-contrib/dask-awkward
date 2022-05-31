from __future__ import annotations

from collections.abc import Callable, Iterable
from math import ceil
from typing import TYPE_CHECKING, Any

import awkward._v2 as ak
import numpy as np
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import HighLevelGraph
from dask.utils import funcname

from dask_awkward.core import map_partitions, new_array_object, typetracer_array
from dask_awkward.utils import DaskAwkwardNotImplemented, LazyInputsDict

if TYPE_CHECKING:
    from dask.array.core import Array as DaskArray
    from dask.delayed import Delayed

    from dask_awkward.core import Array


class FromAwkwardWrapper:
    def __init__(self, arr: ak.Array) -> None:
        self.arr = arr

    def __call__(self, start: int, stop: int) -> ak.Array:
        return self.arr[start:stop]


def from_awkward(source: ak.Array, npartitions: int, label: str | None = None) -> Array:
    """Create a Dask collection from a concrete awkward array.

    Parameters
    ----------
    source : ak.Array
        The concrete awkward array.
    npartitions : int
        The total number of partitions for the collection.
    label : str, optional
        Label for the task.

    Returns
    -------
    Array
        Resulting awkward array collection.

    Examples
    --------
    >>> import dask_awkward as dak
    >>> import awkward._v2 as ak
    >>> a = ak.Array([[1, 2, 3], [4], [5, 6, 7, 8]])
    >>> c = dak.from_awkward(a, npartitions=3)
    >>> c.partitions[[0, 1]].compute()
    <Array [[1, 2, 3], [4]] type='2 * var * int64'>

    """
    nrows = len(source)
    chunksize = int(ceil(nrows / npartitions))
    locs = list(range(0, nrows, chunksize)) + [nrows]
    starts = locs[:-1]
    stops = locs[1:]
    meta = typetracer_array(source)
    return from_map(
        FromAwkwardWrapper(source),
        starts,
        stops,
        label=label or "from-awkward",
        token=tokenize(source, npartitions),
        divisions=tuple(locs),
        meta=meta,
    )


def from_lists(source: list[list[Any]]) -> Array:
    """Create a Dask collection from a list of lists.

    Parameters
    ----------
    source : list[list[Any]]
        List of lists, each outer list will become a partition in the
        collection.

    Returns
    -------
    Array
        Resulting Array collection.

    Examples
    --------
    >>> import dask_awkward as dak
    >>> a = [[1, 2, 3], [4]]
    >>> b = [[5], [6, 7, 8]]
    >>> c = dak.from_lists([a, b])
    >>> c
    dask.awkward<from-lists, npartitions=2>
    >>> c.compute()
    <Array [[1, 2, 3], [4], [5], [6, 7, 8]] type='4 * var * int64'>

    """
    lists = list(source)
    divs = (0, *np.cumsum(list(map(len, lists))))
    return from_map(
        lambda x: ak.Array(x),
        lists,
        meta=typetracer_array(ak.Array(lists[0])),
        divisions=divs,
        label="from-lists",
    )


def from_delayed(
    source: list[Delayed] | Delayed,
    meta: ak.Array | None = None,
    divisions: tuple[int | None, ...] | None = None,
    prefix: str = "from-delayed",
) -> Array:
    """Create a Dask Awkward Array from Dask Delayed objects.

    Parameters
    ----------
    source : list[dask.delayed.Delayed] | dask.delayed.Delayed
        List of :py:class:`~dask.delayed.Delayed` objects (or a single
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

    parts = [source] if isinstance(source, Delayed) else source
    name = f"{prefix}-{tokenize(source)}"
    dsk = {(name, i): part.key for i, part in enumerate(parts)}
    if divisions is None:
        divs: tuple[int | None, ...] = (None,) * (len(source) + 1)
    else:
        divs = tuple(divisions)
        if len(divs) != len(source) + 1:
            raise ValueError("divisions must be a tuple of length len(source) + 1")
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=source)
    return new_array_object(hlg, name=name, meta=meta, divisions=divs)


def to_delayed(array: Array, optimize_graph: bool = True) -> list[Delayed]:
    """Convert the collection to a list of delayed objects.

    One dask.delayed.Delayed object per partition.

    Parameters
    ----------
    optimize_graph : bool
        If ``True`` the task graph associated with the collection will
        be optimized before conversion to the list of Delayed objects.

    Returns
    -------
    list[dask.delayed.Delayed]
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
    """Convert awkward array collection to a Dask array collection.

    This conversion requires the awkward array to have a rectilinear
    shape (that is, no lists of variable length lists).

    Parameters
    ----------
    array : Array
        The dask awkward array collection.

    Returns
    -------
    dask.array.Array
        The new :py:class:`dask.array.Array` collection.

    """
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
    pairs = (array.name, "i")
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


# class AwkwardIOLayer(Blockwise):
#     def __init__(
#         self,
#         name: str,
#         inputs: Any,
#         io_func: Callable,
#         label: str | None = None,
#         produces_tasks: bool = False,
#         creation_info: dict | None = None,
#         annotations: dict | None = None,
#     ):
#         self.name = name
#         self.inputs = inputs
#         self.io_func = io_func
#         self.label = label
#         self.produces_tasks = produces_tasks
#         self.annotations = annotations
#         self.creation_info = creation_info

#         io_arg_map = BlockwiseDepDict(
#             mapping=LazyInputsDict(self.inputs),  # type: ignore
#             produces_tasks=self.produces_tasks,
#         )

#         dsk = {self.name: (io_func, blockwise_token(0))}
#         super().__init__(
#             output=self.name,
#             output_indices="i",
#             dsk=dsk,
#             indices=[(io_arg_map, "i")],
#             numblocks={},
#             annotations=annotations,
#         )


class _PackedArgCallable:
    """Wrap a callable such that packed arguments can be unrolled.

    Inspired by dask.dataframe.io.io._PackedArgCallable.

    """

    def __init__(
        self,
        func: Callable,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        packed: bool = False,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.packed = packed

    def __call__(self, packed_arg: Any):
        if not self.packed:
            packed_arg = (packed_arg,)
        return self.func(
            *packed_arg,
            *(self.args or []),
            **(self.kwargs or {}),
        )


def from_map(
    func: Callable,
    *iterables: Iterable,
    args: tuple[Any, ...] | None = None,
    label: str | None = None,
    token: str | None = None,
    divisions: tuple[int, ...] | None = None,
    meta: ak.Array | None = None,
    **kwargs: Any,
) -> Array:
    """Create an Array collection from a custom mapping.

    Parameters
    ----------
    func : Callable
        Function used to create each partition.
    *iterables : Iterable
        Iterable objects to map to each output partition. All
        iterables must be the same length. This length determines the
        number of partitions in the output collection (only one
        element of each iterable will be passed to `func` for each
        partition).
    label : str, optional
        String to use as the function-name label in the output
        collection-key names.
    token : str, optional
        String to use as the "token" in the output collection-key names.
    divisions : tuple[int | None, ...], optional
        Partition boundaries (if known).
    meta : Array, optional
        Collection metadata array, if known (the awkward-array type
        tracer)
    **kwargs : Any
        Keyword arguments passed to `func`.

    Returns
    -------
    Array
        Array collection.

    """

    if not callable(func):
        raise ValueError("`func` argument must be `callable`")
    lengths = set()
    iters: list[Iterable] = list(iterables)
    for i, iterable in enumerate(iters):
        if not isinstance(iterable, Iterable):
            raise ValueError(
                f"All elements of `iterables` must be Iterable, got {type(iterable)}"
            )
        try:
            lengths.add(len(iterable))  # type: ignore
        except (AttributeError, TypeError):
            iters[i] = list(iterable)
            lengths.add(len(iters[i]))  # type: ignore
    if len(lengths) == 0:
        raise ValueError("`from_map` requires at least one Iterable input")
    elif len(lengths) > 1:
        raise ValueError("All `iterables` must have the same length")
    if lengths == {0}:
        raise ValueError("All `iterables` must have a non-zero length")

    # Check for `produces_tasks` and `creation_info`
    produces_tasks = kwargs.pop("produces_tasks", False)
    # creation_info = kwargs.pop("creation_info", None)

    if produces_tasks or len(iters) == 1:
        if len(iters) > 1:
            # Tasks are not detected correctly when they are "packed"
            # within an outer list/tuple
            raise ValueError(
                "Multiple iterables not supported when produces_tasks=True"
            )
        inputs = list(iters[0])
        packed = False
    else:
        # Structure inputs such that the tuple of arguments pair each 0th,
        # 1st, 2nd, ... elements together; for example:
        # from_map(f, [1, 2, 3], [4, 5, 6]) --> [f(1, 4), f(2, 5), f(3, 6)]
        inputs = list(zip(*iters))
        packed = True

    # Define collection name
    label = label or funcname(func)
    token = token or tokenize(func, iters, meta, **kwargs)
    name = f"{label}-{token}"

    # Define io_func
    if packed or args or kwargs:
        io_func: Callable = _PackedArgCallable(
            func,
            args=args,
            kwargs=kwargs,
            packed=packed,
        )
    else:
        io_func = func

    io_arg_map = BlockwiseDepDict(
        mapping=LazyInputsDict(inputs),  # type: ignore
        produces_tasks=produces_tasks,
    )

    dsk = Blockwise(
        output=name,
        output_indices="i",
        dsk={name: (io_func, blockwise_token(0))},
        indices=[(io_arg_map, "i")],
        numblocks={},
        annotations=None,
    )

    hlg = HighLevelGraph.from_collections(name, dsk)
    if divisions is not None:
        result = new_array_object(hlg, name, meta=meta, divisions=divisions)
    else:
        result = new_array_object(hlg, name, meta=meta, npartitions=len(inputs))

    return result
