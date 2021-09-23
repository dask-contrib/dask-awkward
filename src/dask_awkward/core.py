from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Iterable, List, Tuple

import awkward as ak
from dask.base import DaskMethodsMixin, tokenize
from dask.blockwise import blockwise as core_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import key_split


def _finalize_daskawkwardarray(results: Any) -> Any:
    return ak.concatenate(results)


class AwkwardDaskArray(DaskMethodsMixin):
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
            f"AwkwardDaskArray<{key_split(self.name)}, npartitions={self.npartitions}>"
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
        graphlayer = partitionwise(_getitem, name, self, gikey=key)
        hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
        return AwkwardDaskArray(hlg, name, self.npartitions)

    def __getattr__(self, attr) -> Any:
        return self.__getitem__(attr)


def _getitem(coll, *, gikey):
    return operator.getitem(coll, gikey)


def partitionwise(func, name, *args, **kwargs):
    pairs: List[Any] = []
    numblocks: Dict[Any, int] = {}
    for arg in args:
        if isinstance(arg, AwkwardDaskArray):
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


class PartitionwiseOp:
    def __init__(self, func: Callable, name: str = None) -> None:
        self._func = func
        self.__name__ = func.__name__ if name is None else name
        self.__doc__ = func.__doc__

    def __call__(self, collection, **kwargs):
        token = tokenize(collection)
        name = f"{self.__name__}-{token}"
        layer = partitionwise(self._func, name, collection, **kwargs)
        hlg = HighLevelGraph.from_collections(name, layer, dependencies=[collection])
        return AwkwardDaskArray(hlg, name, collection.npartitions)


flatten = PartitionwiseOp(ak.flatten)
num = PartitionwiseOp(ak.num)
count = PartitionwiseOp(ak.count)
sum = PartitionwiseOp(ak.sum)


# def flatten(
#     collection: AwkwardDaskArray,
#     axis: int = 1,
#     highlevel: bool = True,
#     behavior: Any = None,
# ) -> AwkwardDaskArray:
#     token = tokenize(collection)
#     name = f"flatten-{token}"
#     graphlayer = partitionwise(
#         ak.flatten,
#         name,
#         collection,
#         axis=axis,
#         highlevel=highlevel,
#         behavior=behavior,
#     )
#     hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[collection])
#     return AwkwardDaskArray(hlg, name, collection.npartitions)

# def num(
#     collection: AwkwardDaskArray,
#     axis: int = 1,
#     highlevel: bool = True,
#     behavior: Any = None,
# ) -> AwkwardDaskArray:
#     token = tokenize(collection)
#     name = f"num-{token}"
#     graphlayer = partitionwise(
#         ak.num,
#         name,
#         collection,
#         axis=axis,
#         highlevel=highlevel,
#         behavior=behavior,
#     )
#     hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[collection])
#     return AwkwardDaskArray(hlg, name, collection.npartitions)
