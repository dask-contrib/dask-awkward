from __future__ import annotations

import operator
import warnings
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable

import awkward as ak
import numpy as np
from awkward._v2._connect.numpy import NDArrayOperatorsMixin
from awkward._v2.highlevel import Array as Array_v2
from awkward._v2.tmp_for_testing import v1_to_v2
from awkward.highlevel import Array as Array_v1
from dask.base import (
    DaskMethodsMixin,
    is_dask_collection,
    replace_name_in_key,
    tokenize,
)
from dask.blockwise import blockwise as upstream_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import IndexCallable, cached_property, funcname, key_split

from .utils import normalize_single_outer_inner_index

if TYPE_CHECKING:
    from awkward._v2.contents import Content
    from awkward._v2.forms.form import Form
    from awkward._v2.types.type import Type
    from dask.blockwise import Blockwise


def _finalize_daskawkwardarray(results: Any) -> Any:
    if len(results) == 1:
        return results[0]
    elif all(isinstance(r, Array_v1) for r in results):
        return ak.concatenate(results)
    elif all(isinstance(r, Array_v2) for r in results):
        warnings.warn(
            "v2 Record Arrays cannot be concatenated (yet); "
            "returning the list of results on each node."
        )
        return results
    else:
        return ak.from_iter(results)


def _finalize_scalar(results: Any) -> Any:
    return results[0]


class Scalar(DaskMethodsMixin):
    def __init__(self, dsk: HighLevelGraph, key: str) -> None:
        self._dask: HighLevelGraph = dsk
        self._key: str = key

    def __dask_graph__(self) -> HighLevelGraph:
        return self._dask

    def __dask_keys__(self) -> list[str]:
        return [self._key]

    def __dask_layers__(self) -> tuple[str, ...]:
        if isinstance(self._dask, HighLevelGraph) and len(self._dask.layers) == 1:
            return tuple(self._dask.layers)
        return (self.key,)

    def __dask_tokenize__(self) -> str:
        return self.key

    @staticmethod
    def __dask_optimize__(dsk: Any, keys: Any, **kwargs: Any) -> HighLevelGraph:
        return dsk

    __dask_scheduler__ = staticmethod(threaded_get)

    def __dask_postcompute__(self) -> Any:
        return _finalize_scalar, ()

    def __dask_postpersist__(self) -> Any:
        return self._rebuild, ()

    def _rebuild(self, dsk: Any, *, rename: Any | None = None) -> Any:
        key = replace_name_in_key(self.key, rename) if rename else self.key
        return Scalar(dsk, key)

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self.key

    def __add__(self, other: Scalar) -> Scalar:
        name = f"add-{tokenize(self, other)}"
        deps = [self, other]
        llg = {name: (operator.add, self.key, other.key)}
        g = HighLevelGraph.from_collections(name, llg, dependencies=deps)
        return new_scalar_object(g, name, None)


def new_scalar_object(dsk: HighLevelGraph, name: str, meta: Any) -> Scalar:
    return Scalar(dsk, name)


class DaskAwkwardArray(DaskMethodsMixin, NDArrayOperatorsMixin):
    """Partitioned, lazy, and parallel Awkward Array Dask collection.

    The class constructor is not intended for users. Instead use
    factory functions like :py:func:`dask_awkward.from_parquet`,
    :py:func:`dask_awkward.from_json`, etc.

    Within dask-awkward the ``new_array_object`` factory function is
    used for creating new instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        key: str,
        meta: Any,
        divisions: tuple[int | None, ...],
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._key: str = key
        self._divisions = divisions
        self._meta = meta

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> list[tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return _finalize_daskawkwardarray, ()

    @staticmethod
    def __dask_optimize__(dsk: Any, keys: Any, **kwargs: Any) -> HighLevelGraph:
        return dsk

    __dask_scheduler__ = staticmethod(threaded_get)

    def _rebuild(self, dsk: Any, *, rename: Any | None = None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.meta, divisions=self.divisions)

    def __len__(self) -> int:
        self._divisions = calculate_known_divisions(self)
        return self._divisions[-1] + 1

    def _shorttypestr(self, max: int = 10) -> str:
        return str(_type(self))[0:max]

    def _typestr(self, max: int = 0) -> str:
        tstr = str(_type(self))
        if max and len(tstr) > max:
            tstr = f"{tstr[0:max]} ... }}"
        length = "var" if self.divisions[-1] is None else (self.divisions[-1] + 1)
        return f"{length} * {tstr}"

    def __str__(self) -> str:
        return (
            f"dask.awkward<{key_split(self.name)}, "
            f"npartitions={self.npartitions}, "
            f"type='{self._typestr(max=30)}'"
            ">"
        )

    __repr__ = __str__

    def _ipython_display_(self) -> None:
        if self.meta is None:
            return None

        import json

        from IPython.display import display_json

        display_json(json.loads(self.meta.form.to_json()), raw=True)

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self.key

    @property
    def ndim(self) -> int:
        raise NotImplementedError("TODO")

    @property
    def divisions(self) -> tuple[int | None, ...]:
        return self._divisions

    @property
    def known_divisions(self) -> bool:
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self) -> int:
        return len(self.divisions) - 1

    @property
    def meta(self) -> Any | None:
        return self._meta

    @property
    def typetracer(self) -> Any | None:
        return self.meta

    @property
    def fields(self) -> list[str] | None:
        if self.meta is not None:
            return self.meta.fields
        return None

    @property
    def form(self) -> Form | None:
        if self.meta is not None:
            return self.meta.form
        return None

    @cached_property
    def keys_array(self) -> np.ndarray:
        return np.array(self.__dask_keys__(), dtype=object)

    def _partitions(self, index: Any) -> DaskAwkwardArray:
        if not isinstance(index, tuple):
            index = (index,)
        token = tokenize(self, index)
        from dask.array.slicing import normalize_index

        index = normalize_index(index, (self.npartitions,))
        index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in index)  # type: ignore
        name = f"partitions-{token}"
        new_keys = self.keys_array[index].tolist()
        divisions = [self.divisions[i] for _, i in new_keys] + [
            self.divisions[new_keys[-1][1] + 1]
        ]
        dsk = {(name, i): tuple(key) for i, key in enumerate(new_keys)}
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return new_array_object(graph, name, None, divisions=tuple(divisions))

    @property
    def partitions(self) -> IndexCallable:
        """Get a specific partition or slice of partitions.

        Returns
        -------
        dask.utils.IndexCallable

        Examples
        --------
        >>> import dask_awkward as dak
        >>> import dask_awkward.data as dakd
        >>> a = dak.from_json(dakd.json_data())
        >>> a
        dask.awkward<from-json, npartitions=3, type='var * var * var * int64'>
        >>> a.partitions[0]
        dask.awkward<partitions, npartitions=1, type='var * var * var * int64'>
        >>> a.partitions[0:2]
        dask.awkward<partitions, npartitions=2, type='var * var * var * int64'>

        """
        return IndexCallable(self._partitions)

    def _getitem_inner(self, key: Any) -> DaskAwkwardArray:
        token = tokenize(self, key)
        name = f"getitem-{token}"
        graphlayer = partitionwise_layer(
            lambda x, gikey: operator.getitem(x, gikey), name, self, gikey=key
        )
        hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
        meta = self.meta[key] if self.meta is not None else None
        return new_array_object(hlg, name, meta=meta, divisions=self.divisions)

    def _getitem_singleint(self, key: int) -> DaskAwkwardArray:
        # get divisions
        self._divisions = calculate_known_divisions(self)

        # if only 1 division
        if len(self.divisions) == 2:
            return self._getitem_inner(key=key)

        p, k = normalize_single_outer_inner_index(self._divisions, key)
        return self.partitions[p][k]

    def __getitem__(self, key: Any) -> DaskAwkwardArray:
        if not isinstance(key, tuple):
            key = (key,)
        if isinstance(key[0], list):
            if any(isinstance(k, int) for k in key[0]):
                raise NotImplementedError("Lists containing integers not supported.")
        if (
            isinstance(key[0], (str, list))
            or key[0] is Ellipsis
            or key[0] == slice(None, None, None)
        ):
            return self._getitem_inner(key=key)
        if isinstance(key[0], slice):
            pass
        if isinstance(key[0], int) and len(key) == 1:
            return self._getitem_singleint(key=key[0])
        if isinstance(key[0], int) and len(key) > 1:
            pass
        return key

    def __getattr__(self, attr: str) -> DaskAwkwardArray:
        try:
            return self.__getitem__(attr)
        except (IndexError, KeyError):
            raise AttributeError(f"{attr} not in fields.")

    def map_partitions(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> DaskAwkwardArray:
        """Map a function across all partitions of the collection.

        Parameters
        ----------
        func : Callable
            Function to call on all partitions.
        *args : Collections and function arguments
            Additional arguments passed to `func` after the
            collection, if arguments are DaskAwkwardArray collections
            they must be compatibly partitioned with the object this
            method is being called from.
        **kwargs : Any
            Additional keyword arguments passed to the `func`.

        Returns
        -------
        DaskAwkwardArray
            The new collection.

        See Also
        --------
        dask_awkward.map_partitions

        """
        return map_partitions(func, self, *args, **kwargs)

    def _compute_divisions(self) -> None:
        self._divisions = calculate_known_divisions(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return map_partitions(ufunc, *inputs, **kwargs)


def _first_partition(array: DaskAwkwardArray) -> ak.Array:
    """Compute the first partition of a DaskAwkwardArray collection.

    Parameters
    ----------
    array : DaskAwkwardArray
        Awkward collection.

    Returns
    -------
    ak.Array
        Concrete awkward array for the first partition of the Dask
        awkward array.

    """

    computed = array.__dask_scheduler__(
        array.__dask_graph__(), array.__dask_keys__()[0]
    )
    return computed


def _typetracer_via_v1_to_v2(array: DaskAwkwardArray) -> Content:
    """Obtain the awkward type tracker using the v1 to v2 conversion

    This will compute the first partition of the array collection if
    necessary.

    Parameters
    ----------
    array : DaskAwkwardArray
        The collection.

    Returns
    -------
    Content
        Awkward Content object representing the typetracer (metadata).

    """
    first_part = _first_partition(array)
    if isinstance(first_part, Array_v1):
        return v1_to_v2(_first_partition(array).layout).typetracer
    elif isinstance(first_part, Array_v2):
        return first_part.layout
    else:
        raise TypeError(f"Should have an Array type, got {type(first_part)}")


def new_array_object(
    dsk: HighLevelGraph,
    name: str,
    meta: Any | None = None,
    npartitions: int | None = None,
    divisions: tuple[Any, ...] | None = None,
) -> DaskAwkwardArray:
    """Instantiate a new DaskAwkwardArray collection object.

    Parameters
    ----------
    dsk : dask.highlevelgraph.HighLevelGraph
        Graph backing the collection.
    name : str
        Unique name for the collection.
    meta : awkward-array type tracing information, optional
        Object metadata; this is awkward-array TypeTracer.
    npartitions : int, optional
        Total number of partitions; if used `divisions` will be a
        tuple of length `npartitions` + 1 with all elements``None``.
    divisions : tuple[int or None, ...], optional
        Tuple identifying the locations of the divisions between the
        partitions.

    Returns
    -------
    DaskAwkwardArray
        Resulting collection.

    """
    if divisions is None and npartitions is not None:
        divisions = (None,) * (npartitions + 1)
    elif divisions is not None and npartitions is not None:
        raise ValueError("Only one of either divisions or npartitions must be defined.")
    elif divisions is None and npartitions is None:
        raise ValueError("One of either divisions or npartitions must be defined.")
    array = DaskAwkwardArray(dsk, name, meta, divisions)  # type: ignore
    if meta is None:
        try:
            array._meta = _typetracer_via_v1_to_v2(array)
        except (AttributeError, AssertionError, TypeError):
            array._meta = None
    return array


def partitionwise_layer(
    func: Callable,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Blockwise:
    """Create a partitionwise graph layer.

    Parameters
    ----------
    func : Callable
        Function to apply on all partitions.
    name : str
        Name for the layer.
    *args : Any
        Arguments that will be passed to `func`.
    **kwargs : Any
        Keyword arguments that will be passed to `func`.

    Returns
    -------
    dask.blockwise.Blockwise
        The Dask HighLevelGraph Blockwise layer.

    """
    pairs: list[Any] = []
    numblocks: dict[Any, int | tuple[int, ...]] = {}
    for arg in args:
        if isinstance(arg, DaskAwkwardArray):
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = (arg.npartitions,)
        elif is_dask_collection(arg):
            raise NotImplementedError(
                "Use of DaskAwkwardArray with other Dask "
                "collections is currently unsupported."
            )
        else:
            pairs.extend([arg, None])
    return upstream_blockwise(
        func,
        name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )


def map_partitions(
    func: Callable,
    *args: Any,
    name: str | None = None,
    **kwargs: Any,
) -> DaskAwkwardArray:
    """Map a callable across all partitions of a collection.

    Parameters
    ----------
    func : Callable
        Function to apply on all partitions.
    *args : Collections and function arguments
        Arguments passed to the function, if arguments are
        DaskAwkwardArray collections they must be compatibly
        partitioned.
    name : str, optional
        Name for the Dask graph layer; if left to ``None`` (default),
        the name of the function will be used.
    **kwargs : Any
        Additional keyword arguments passed to the `func`.

    Returns
    -------
    DaskAwkwardArray
        The new collection.

    """
    token = tokenize(func, *args, **kwargs)
    name = name or funcname(func)
    name = f"{name}-{token}"
    lay = partitionwise_layer(func, name, *args, **kwargs)
    deps = [a for a in args if is_dask_collection(a)] + [
        v for _, v in kwargs.items() if is_dask_collection(v)
    ]
    hlg = HighLevelGraph.from_collections(name, lay, dependencies=deps)
    return new_array_object(hlg, name, None, npartitions=args[0].npartitions)


def pw_reduction_with_agg_to_scalar(
    array: DaskAwkwardArray,
    func: Callable,
    agg: Callable,
    *,
    name: str | None = None,
    **kwargs: Any,
) -> Scalar:
    """Partitionwise operation with aggregation to scalar.

    Parameters
    ----------
    array : DaskAwkwardArray
        Awkward array collection.
    func : Callable
        Function to apply on all partitions.
    agg : Callable
        Function to aggregate the result on each partition.
    name : str | None
        Name for the computation, if ``None`` we use the name of
        `func`.
    **kwargs : Any
        Keyword arguments passed to `func`.

    Returns
    -------
    Scalar
        Resulting scalar Dask collection.

    """
    token = tokenize(array)
    name = func.__name__ if name is None else name
    name = f"{name}-{token}"
    func = partial(func, **kwargs)
    dsk = {(name, i): (func, k) for i, k in enumerate(array.__dask_keys__())}
    dsk[name] = (agg, list(dsk.keys()))  # type: ignore
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[array])
    return new_scalar_object(hlg, name, None)


def calculate_known_divisions(array: DaskAwkwardArray) -> tuple[int, ...]:
    """Determine the divisions of a collection.

    This function triggers an immediate computation.

    Parameters
    ----------
    array : DaskAwkwardArray
        Awkard array collection.

    Returns
    -------
    tuple[int, ...]
        Locations (indices) of division boundaries.

    """
    # if divisions are known, quick return
    if array.known_divisions:
        return array.divisions  # type: ignore
    # handle the case where we have 1 partition
    if array.npartitions == 1:
        num = array.map_partitions(ak.num, axis=0).compute()
        if isinstance(num, int):
            return (0, num - 1)
        return (0, num.slot0 - 1)
    # finally handle the more common > 1 partition case
    nums = array.map_partitions(ak.num, axis=0).compute()
    try:
        cs = list(np.cumsum(nums))
    except TypeError:
        cs = list(np.cumsum(nums.slot0))
    cs[-1] -= 1
    return tuple([0, *cs])


class _TrivialPartitionwiseOp:
    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._func = func
        self.name = func.__name__ if name is None else name
        self._kwargs = kwargs

    def __call__(self, collection: DaskAwkwardArray, **kwargs: Any) -> DaskAwkwardArray:
        # overwrite any saved kwargs in self._kwargs
        for k, v in kwargs.items():
            self._kwargs[k] = v
        return map_partitions(self._func, collection, name=self.name, **self._kwargs)


def _type(array: DaskAwkwardArray) -> Type | None:
    """Get the type object associated with an array.

    Parameters
    ----------
    array : DaskAwkwardArray
        The collection.

    Returns
    -------
    Type
        The awkward type object of the array; if the array does not
        contain metadata ``None`` is returned.

    """
    f = array.form
    return f.type if f is not None else None


def fields(array: DaskAwkwardArray) -> list[str] | None:
    """Get the fields of a DaskAwkwardArray collection.

    Parameters
    ----------
    array : DaskAwkwardArray
        The collection.

    Returns
    -------
    list[str] or None
        The fields of the array; if the array does not contain
        metadata ``None`` is returned.

    """
    if array.meta is not None:
        return array.meta.fields
    else:
        return None
