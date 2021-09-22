from __future__ import annotations

from typing import Any, List, Tuple

import awkward as ak
from dask.base import DaskMethodsMixin, tokenize
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
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self.key

    @property
    def npartitions(self) -> int:
        return self._npartitions

    @property
    def dask(self) -> HighLevelGraph:
        return self._dsk

    def __getattr__(self, attr) -> Any:
        token = tokenize(self, attr)
        name = f"{attr}-{token}"
        dsk = {
            (name, i): (getattr, k, attr) for i, k in enumerate(self.__dask_keys__())
        }
        hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return AwkwardDaskArray(hlg, name, self.npartitions)


# def partitionwise(func, name, *args, **kwargs):
#     pairs = []
#     numblocks = {}
#     pass
