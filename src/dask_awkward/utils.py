from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dask.base import is_dask_collection

from .data import json_data
from .io import from_json

if TYPE_CHECKING:
    from .core import DaskAwkwardArray


def load_nested() -> DaskAwkwardArray:
    return from_json(json_data(kind="records"))


def load_array() -> DaskAwkwardArray:
    return from_json(json_data(kind="numbers"))


def assert_eq(a: Any, b: Any) -> None:
    if is_dask_collection(a) and not is_dask_collection(b):
        assert a.compute().to_list() == b.to_list()
    elif is_dask_collection(b) and not is_dask_collection(a):
        assert a.to_list() == b.compute().to_list()
    else:
        assert a.compute().to_list() == b.compute().to_list()
