from __future__ import annotations

from typing import Any, List, Tuple
import functools

import awkward as ak
from dask.base import DaskMethodsMixin
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import key_split


def _finalize_daskawkwardarray(results: Any) -> Any:
    return results


class AwkwardDaskArray(DaskMethodsMixin):
    npartitions: int

    def __init__(self, dsk: HighLevelGraph, key: str, npartitions: int) -> None:
        self._dsk = dsk
        self._key = key
        self._npartitions = npartitions

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

    __dask_scheduler__ = staticmethod(threaded_get)

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

    def __str__(self) -> str:
        return f"AwkwardDaskArray<{key_split(self.name)}>"
