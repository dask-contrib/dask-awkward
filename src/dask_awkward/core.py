from __future__ import annotations

import operator
from functools import partial
from numbers import Number
from typing import Any, Callable

import awkward as ak
import numpy as np
from awkward._v2.tmp_for_testing import v1_to_v2
from dask.base import (
    DaskMethodsMixin,
    is_dask_collection,
    replace_name_in_key,
    tokenize,
)
from dask.blockwise import blockwise as core_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import IndexCallable, cached_property, funcname, key_split


def _finalize_daskawkwardarray(results: Any) -> Any:
    if all(isinstance(r, ak.Array) for r in results):
        return ak.concatenate(results)
    if all(isinstance(r, ak.Record) for r in results):
        raise NotImplementedError("Records not supported yet.")
    else:
        return results


def _finalize_scalar(results: Any) -> Any:
    return results[0]


class Scalar(DaskMethodsMixin):
    def __init__(self, dsk: HighLevelGraph, key: str) -> None:
        self._dask: HighLevelGraph = dsk
        self._key: str = key

    def __dask_graph__(self):
        return self._dask

    def __dask_keys__(self):
        return [self._key]

    def __dask_layers__(self):
        if isinstance(self._dask, HighLevelGraph) and len(self._dask.layers) == 1:
            return tuple(self._dask.layers)
        return (self.key,)

    def __dask_tokenize__(self):
        return self.key

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    __dask_scheduler__ = staticmethod(threaded_get)

    def __dask_postcompute__(self):
        return _finalize_scalar, ()

    def __dask_postpersist__(self):
        return self._rebuild, ()

    def _rebuild(self, dsk, *, rename=None):
        key = replace_name_in_key(self.key, rename) if rename else self.key
        return Scalar(dsk, key)

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


def new_scalar_object(dsk: HighLevelGraph, name: str, meta: Any):
    return Scalar(dsk, name)


class DaskAwkwardArray(DaskMethodsMixin):
    """Partitioned, lazy, and parallel Awkward Array Dask collection.

    The class constructor is not intended for users. Instead use
    factory functions like :py:func:`dask_awkward.from_parquet,
    :py:func:`dask_awkward.from_json`, etc.

    Within dask-awkward the ``new_array_object`` factory function is
    used for creating new instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        key: str,
        divisions: tuple[Any, ...] | None = None,
        npartitions: int | None = None,
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._key: str = key
        if divisions is None and npartitions is not None:
            self._npartitions: int = npartitions
            self._divisions: tuple[Any, ...] = (None,) * (npartitions + 1)
        elif divisions is not None and npartitions is None:
            self._divisions = divisions
            self._npartitions = len(divisions) - 1
        self._fields: list[str] | None = None
        self._typetracer = None

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
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    def _rebuild(self, dsk: Any, *, rename: Any | None = None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, divisions=self.divisions)

    def __str__(self) -> str:
        return (
            f"DaskAwkwardArray<{key_split(self.name)}, npartitions={self.npartitions}>"
        )

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(threaded_get)

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def fields(self) -> list[str] | None:
        return self._fields

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self.key

    @property
    def ndim(self) -> int:
        raise NotImplementedError("Not known without metadata; a current TODO")

    @property
    def divisions(self) -> tuple[Any, ...]:
        return self._divisions

    @property
    def known_divisions(self) -> bool:
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self) -> int:
        return self._npartitions

    @property
    def typetracer(self) -> Any:
        return self._typetracer

    @cached_property
    def keys_array(self) -> np.ndarray:
        return np.array(self.__dask_keys__(), dtype=object)

    def _partitions(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        token = tokenize(self, index)
        from dask.array.slicing import normalize_index

        index = normalize_index(index, (self.npartitions,))
        index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in index)
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
        DaskAwkwardArray<from-json, npartitions=3>
        >>> a.partitions[0]
        DaskAwkwardArray<partitions, npartitions=1>
        >>> a.partitions[0:2]
        DaskAwkwardArray<partitions, npartitions=2>

        """
        return IndexCallable(self._partitions)

    def __getitem__(self, key) -> Any:
        if not isinstance(key, str):
            raise NotImplementedError("__getitem__ supports only string (for now).")
        token = tokenize(self, key)
        name = f"getitem-{token}"
        graphlayer = partitionwise_layer(
            lambda x, gikey: operator.getitem(x, gikey), name, self, gikey=key
        )
        hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
        new_typetracer = self._typetracer[key]
        return new_array_object(
            hlg,
            name,
            meta=new_typetracer,
            npartitions=self.npartitions,
        )

    def __getattr__(self, attr) -> Any:
        return self.__getitem__(attr)

    def map_partitions(self, func, *args, **kwargs):
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


def _first_partition_layout(arr: DaskAwkwardArray):
    return arr.__dask_scheduler__(arr.__dask_graph__(), arr.__dask_keys__()[0]).layout


def new_array_object(
    dsk: HighLevelGraph,
    name: str,
    meta: Any | None = None,
    npartitions: int | None = None,
    divisions: tuple[Any, ...] | None = None,
):
    arr = DaskAwkwardArray(dsk, name, npartitions=npartitions, divisions=divisions)
    if meta is None:
        layout = v1_to_v2(_first_partition_layout(arr))
        arr._typetracer = layout.typetracer
    else:
        arr._typetracer = meta
    return arr


def partitionwise_layer(func, name, *args, **kwargs):
    pairs: list[Any] = []
    numblocks: dict[Any, int] = {}
    for arg in args:
        if isinstance(arg, DaskAwkwardArray):
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = (arg.npartitions,)
    return core_blockwise(
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
        Function to call on all partitions.
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
    deps = []
    for a in args:
        if is_dask_collection(a):
            deps.append(a)
    hlg = HighLevelGraph.from_collections(name, lay, dependencies=deps)
    return new_array_object(hlg, name, None, npartitions=args[0].npartitions)


def pw_reduction_with_agg_to_scalar(
    a: DaskAwkwardArray,
    func: Callable,
    agg: Callable,
    *,
    name: str | None = None,
    **kwargs,
):
    token = tokenize(a)
    name = func.__name__ if name is None else name
    name = f"{name}-{token}"
    func = partial(func, **kwargs)
    dsk = {(name, i): (func, k) for i, k in enumerate(a.__dask_keys__())}
    dsk[name] = (agg, list(dsk.keys()))  # type: ignore
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[a])
    return new_scalar_object(hlg, name, None)


class TrivialPartitionwiseOp:
    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._func = func
        self.__name__ = func.__name__ if name is None else name
        self._kwargs = kwargs

    def __call__(self, collection, **kwargs):
        # overwrite any saved kwargs in self._kwargs
        for k, v in kwargs.items():
            self._kwargs[k] = v
        token = tokenize(collection, kwargs)
        name = f"{self.__name__}-{token}"
        layer = partitionwise_layer(self._func, name, collection, **self._kwargs)
        hlg = HighLevelGraph.from_collections(name, layer, dependencies=[collection])
        return new_array_object(hlg, name, None, collection.npartitions)
