from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

import awkward as ak
import numpy as np
from awkward.types.numpytype import primitive_to_dtype
from dask.base import flatten, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import funcname

from dask_awkward.layers import AwkwardBlockwiseLayer, AwkwardInputLayer
from dask_awkward.layers.layers import AwkwardMaterializedLayer
from dask_awkward.lib.core import (
    empty_typetracer,
    map_partitions,
    new_array_object,
    typetracer_array,
)

if TYPE_CHECKING:
    from dask.array.core import Array as DaskArray
    from dask.bag.core import Bag as DaskBag
    from dask.dataframe.core import DataFrame as DaskDataFrame
    from dask.delayed import Delayed

    from dask_awkward.lib.core import Array


class ImplementsFormTransformation(Protocol):
    behavior: dict | None

    def __call__(self, form: ak.form.Form, docstr: str | None) -> ak.form.Form:
        raise NotImplementedError

    def extract_form_keys_base_columns(self, form_keys: Iterable[str]) -> Iterable[str]:
        raise NotImplementedError

    def create_column_mapping_and_key(
        self,
        column_source: Any,
        start: int,
        stop: int,
        **kwargs: Any,
    ) -> tuple[Mapping[str, ak.Array], Callable[[str, ak.forms.Form, str], str] | str]:
        raise NotImplementedError


class _FromAwkwardFn:
    def __init__(self, arr: ak.Array) -> None:
        self.arr = arr

    def __call__(self, start: int, stop: int, **kwargs: Any) -> ak.Array:
        return self.arr[start:stop]


def from_awkward(
    source: ak.Array,
    npartitions: int,
    behavior: dict | None = None,
    label: str | None = None,
) -> Array:
    """Create an Array collection from a concrete :class:`awkward.Array` object.

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
    >>> import awkward as ak
    >>> a = ak.Array([[1, 2, 3], [4], [5, 6, 7, 8]])
    >>> c = dak.from_awkward(a, npartitions=3)
    >>> c.partitions[[0, 1]].compute()
    <Array [[1, 2, 3], [4]] type='2 * var * int64'>

    """
    nrows = len(source)
    chunksize = int(math.ceil(nrows / npartitions))
    locs = list(range(0, nrows, chunksize)) + [nrows]
    starts = locs[:-1]
    stops = locs[1:]
    meta = typetracer_array(source)
    return from_map(
        _FromAwkwardFn(source),
        starts,
        stops,
        label=label or "from-awkward",
        token=tokenize(source, npartitions),
        divisions=tuple(locs),
        meta=meta,
        behavior=behavior,
    )


class _FromListsFn:
    def __init__(self, behavior: dict | None = None):
        self.behavior = behavior

    def __call__(self, x, **kwargs):
        return ak.Array(x, behavior=self.behavior)


def from_lists(source: list[list[Any]], behavior: dict | None = None) -> Array:
    """Create an Array collection from a list of lists.

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
        _FromListsFn(),
        lists,
        meta=typetracer_array(ak.Array(lists[0])),
        divisions=divs,
        label="from-lists",
    )


def from_delayed(
    source: list[Delayed] | Delayed,
    meta: ak.Array | None = None,
    behavior: dict | None = None,
    divisions: tuple[int | None, ...] | None = None,
    prefix: str = "from-delayed",
) -> Array:
    """Create an Array collection from a set of :class:`~dask.delayed.Delayed` objects.

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
    name = f"{prefix}-{tokenize(parts)}"
    dsk = AwkwardMaterializedLayer(
        {(name, i): part.key for i, part in enumerate(parts)},
        previous_layer_names=[parts[0].name],
    )
    if divisions is None:
        divs: tuple[int | None, ...] = (None,) * (len(parts) + 1)
    else:
        divs = tuple(divisions)
        if len(divs) != len(parts) + 1:
            raise ValueError("divisions must be a tuple of length len(source) + 1")
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=parts)
    return new_array_object(
        hlg, name=name, meta=meta, behavior=behavior, divisions=divs
    )


def to_delayed(array: Array, optimize_graph: bool = True) -> list[Delayed]:
    """Convert Arrray the collection to a list of :class:`~dask.delayed.Delayed` objects.

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


def to_dask_bag(array: Array) -> DaskBag:
    """Convert Array collection to a :class:`dask.bag.Bag` collection."""
    from dask.bag.core import Bag

    return Bag(array.dask, array.name, array.npartitions)


def to_dask_array(
    array: Array,
    *,
    dtype: Any = None,
    optimize_graph: bool = True,
) -> DaskArray:
    """Convert Array collection to a :class:`dask.array.Array` collection.

    This conversion requires the awkward array to have a rectilinear
    shape (that is, no lists of variable length lists).

    Parameters
    ----------
    array : Array
        The dask awkward array collection.
    dtype : DType
        NumPy dtype for the resulting array.
    optimize_graph : bool
        Optimize the graph associated with `array` (the
        ``dask_awkward.Array``) before converting to
        ``dask.array.Array``.

    Returns
    -------
    dask.array.Array
        The new :py:class:`dask.array.Array` collection.

    """
    from dask.array.core import new_da_object

    if array._meta is None:
        raise ValueError("Array metadata required for determining dtype")

    ndim = array.ndim

    if optimize_graph:
        keys = array.__dask_keys__()
        graph = array.__dask_graph__()
        layer = array.__dask_layers__()[0]
        graph = array.__dask_optimize__(graph, keys)
        hlg = HighLevelGraph.from_collections(layer, graph, dependencies=())
        array = new_array_object(
            hlg,
            name=layer,
            divisions=array.divisions,
            meta=array._meta,
        )

    if ndim == 1:
        new = map_partitions(ak.to_numpy, array, meta=empty_typetracer())
        graph = new.dask
        dtype = dtype or primitive_to_dtype(array._meta.layout.form.type.primitive)
        if array.known_divisions:
            divs = np.array(array.divisions)
            chunks: tuple[tuple[float, ...], ...] = (tuple(divs[1:] - divs[:-1]),)
        else:
            chunks = ((np.nan,) * array.npartitions,)
        return new_da_object(
            graph,
            new.name,
            meta=None,
            chunks=chunks,
            dtype=dtype,
        )

    else:
        # assert ndim > 1
        content = array._meta.layout.form.type.content
        no_primitive = not hasattr(content, "primitive")
        while no_primitive:
            content = content.content
            no_primitive = not hasattr(content, "primitive")
        dtype = dtype or primitive_to_dtype(content.primitive)

        name = f"to-dask-array-{tokenize(array)}"
        nan_tuples_innerdims = ((np.nan,),) * (ndim - 1)
        chunks = ((np.nan,) * array.npartitions, *nan_tuples_innerdims)
        zeros = (0,) * (ndim - 1)

        # eventually convert to HLG (if possible)
        llg = {
            (name, i, *zeros): (ak.to_numpy, k)
            for i, k in enumerate(flatten(array.__dask_keys__()))
        }

        graph = HighLevelGraph.from_collections(name, llg, dependencies=[array])
        return new_da_object(graph, name, meta=None, chunks=chunks, dtype=dtype)


def from_dask_array(array: DaskArray, behavior: dict | None = None) -> Array:
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
    layer = AwkwardBlockwiseLayer.from_blockwise(layer)
    hlg = HighLevelGraph.from_collections(name, layer, dependencies=[array])
    if np.any(np.isnan(array.chunks)):
        return new_array_object(
            hlg, name, npartitions=array.npartitions, meta=meta, behavior=behavior
        )
    else:
        divs = (0, *np.cumsum(array.chunks))
        return new_array_object(hlg, name, divisions=divs, meta=meta, behavior=behavior)


def to_dataframe(
    array,
    optimize_graph: bool = True,
    **kwargs: Any,
) -> DaskDataFrame:
    """Convert :class:`dask_awkward.Array` collection to :class:`~dask.dataframe.DataFrame`.

    Parameters
    ----------
    array : dask_awkward.Array
        Array collection to be converted.
    optimize_graph : bool
        If ``True``, optimize the Array collection task graph before
        converting to DataFrame collection.
    **kwargs : Any
        Additional arguments passed to :func:`ak.to_dataframe`.

    Returns
    -------
    dask.dataframe.DataFrame
        Resulting DataFrame collection.

    """
    import dask
    from dask.dataframe.core import new_dd_object

    if optimize_graph:
        (array,) = dask.optimize(array)
    intermediate = map_partitions(
        ak.to_dataframe,
        array,
        meta=empty_typetracer(),
        label="to-dataframe",
        **kwargs,
    )
    meta = ak.to_dataframe(
        array._meta.layout.form.length_zero_array(highlevel=False), **kwargs
    )
    return new_dd_object(
        intermediate.dask,
        intermediate.name,
        meta,
        intermediate.divisions,
    )


class PackedArgCallable:
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

    def __call__(self, packed_arg):
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
    behavior: dict | None = None,
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
        func = PackedArgCallable(
            func,
            args=args,
            kwargs=kwargs,
            packed=packed,
        )

    dsk = AwkwardInputLayer(
        name=name,
        columns=None,
        inputs=inputs,
        io_func=func,
        meta=meta,
        behavior=behavior,
    )

    hlg = HighLevelGraph.from_collections(name, dsk)
    if divisions is not None:
        result = new_array_object(
            hlg, name, meta=meta, behavior=dsk._behavior, divisions=divisions
        )
    else:
        result = new_array_object(
            hlg, name, meta=meta, behavior=dsk._behavior, npartitions=len(inputs)
        )

    return result
