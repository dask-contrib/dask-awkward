from __future__ import annotations

import abc
from typing import Any, Protocol, runtime_checkable

try:
    from dask.typing import HLGDaskCollection
except ImportError:
    raise ImportError(
        "Using dask-awkward's typing module requires a version "
        "of Dask with support for the dask.typing module."
    )


@runtime_checkable
class AwkwardDaskCollection(HLGDaskCollection, Protocol):
    _meta: Any

    @property
    @abc.abstractmethod
    def fields(self) -> list[str]:
        """ """

    @property
    @abc.abstractmethod
    def layout(self) -> Any:
        """ """

    @property
    @abc.abstractmethod
    def npartitions(self) -> int:
        """ """

    @abc.abstractmethod
    def __getitem__(self, where: Any) -> AwkwardDaskCollection:
        """ """

    @abc.abstractmethod
    def __getattr__(self, attr: str) -> Any:
        """ """
