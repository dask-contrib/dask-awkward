from __future__ import annotations

import functools
import operator
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import awkward as ak
import numpy as np
from dask.base import DaskMethodsMixin, replace_name_in_key, tokenize
from dask.blockwise import blockwise as core_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import key_split


def _finalize_daskawkwardarray(results: Any) -> Any:
    if all(isinstance(r, ak.Array) for r in results):
        return ak.concatenate(results)
    if all(isinstance(r, ak.Record) for r in results):
        return ak.concatenate(results)
    else:
        return results


def _finalize_scalar(results: Any) -> Any:
    return results[0]


class Scalar(DaskMethodsMixin):
    def __init__(self, dsk: HighLevelGraph, key: str) -> None:
        self._dsk: HighLevelGraph = dsk
        self._key: str = key

    def __dask_graph__(self):
        return self._dsk

    def __dask_keys__(self):
        return [self._key]

    def __dask_layers__(self):
        if isinstance(self._dsk, HighLevelGraph) and len(self._dsk.layers) == 1:
            return tuple(self._dsk.layers)
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
        name = "add-{}".format(tokenize(self, other))
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

    def __init__(self, dsk: HighLevelGraph, key: str, npartitions: int) -> None:
        self._dsk: HighLevelGraph = dsk
        self._key: str = key
        self._npartitions: int = npartitions
        self._fields: List[str] = None

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> List[Tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> Tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return _finalize_daskawkwardarray, ()

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    def _rebuild(self, dsk: Any, *, rename: Any = None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.npartitions)

    def __str__(self) -> str:
        return (
            f"DaskAwkwardArray<{key_split(self.name)}, npartitions={self.npartitions}>"
        )

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(threaded_get)

    @property
    def dask(self) -> HighLevelGraph:
        return self._dsk

    @property
    def fields(self) -> Iterable[str]:
        return self._fields

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self.key

    @property
    def npartitions(self) -> int:
        return self._npartitions

    def __getitem__(self, key) -> Any:
        if not isinstance(key, (int, str)):
            raise NotImplementedError(
                "getitem supports only string and integer for now."
            )
        token = tokenize(self, key)
        name = f"getitem-{token}"
        graphlayer = pw_layer(
            lambda x, gikey: operator.getitem(x, gikey), name, self, gikey=key
        )
        hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
        return new_array_object(hlg, name, None, self.npartitions)

    def __getattr__(self, attr) -> Any:
        return self.__getitem__(attr)


def new_array_object(dsk: HighLevelGraph, name: str, meta: Any, npartitions: int):
    return DaskAwkwardArray(dsk, name, npartitions)


def pw_layer(func, name, *args, **kwargs):
    pairs: List[Any] = []
    numblocks: Dict[Any, int] = {}
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


def pw_reduction_with_agg(
    a: DaskAwkwardArray,
    func: Callable,
    agg: Callable,
    *,
    name: str = None,
    **kwargs,
):
    token = tokenize(a)
    name = func.__name__ if name is None else name
    name = f"{name}-{token}"
    func = functools.partial(func, **kwargs)
    dsk = {(name, i): (func, k) for i, k in enumerate(a.__dask_keys__())}
    dsk[name] = (agg, list(dsk.keys()))
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[a])
    return new_scalar_object(hlg, name, None)


class TrivialPartitionwiseOp:
    def __init__(self, func: Callable, name: str = None) -> None:
        self._func = func
        self.__name__ = func.__name__ if name is None else name

    def __call__(self, collection, **kwargs):
        token = tokenize(collection)
        name = f"{self.__name__}-{token}"
        layer = pw_layer(self._func, name, collection, **kwargs)
        hlg = HighLevelGraph.from_collections(name, layer, dependencies=[collection])
        return new_array_object(hlg, name, None, collection.npartitions)


_count_trivial = TrivialPartitionwiseOp(ak.count)
_flatten_trivial = TrivialPartitionwiseOp(ak.flatten)
_max_trivial = TrivialPartitionwiseOp(ak.max)
_min_trivial = TrivialPartitionwiseOp(ak.min)
_num_trivial = TrivialPartitionwiseOp(ak.num)
_sum_trivial = TrivialPartitionwiseOp(ak.sum)


def count(a, axis: Optional[int] = None, **kwargs):
    if axis is not None and axis > 0:
        return _count_trivial(a, axis=axis, **kwargs)
    elif axis is None:
        trivial_result = _count_trivial(a, axis=1, **kwargs)
        return pw_reduction_with_agg(trivial_result, ak.sum, ak.sum)
    elif axis == 0 or axis == -1 * a.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


def flatten(a: DaskAwkwardArray, axis: int = 1, **kwargs):
    if axis == 1:
        return _flatten_trivial(a, axis=axis, **kwargs)


def max(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    pass


def min(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    pass


def num(a: DaskAwkwardArray, axis: int = 1, **kwargs):
    pass


def sum(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    pass
