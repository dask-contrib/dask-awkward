from __future__ import annotations

from typing import TYPE_CHECKING

from .data import json_data
from .io import from_json

if TYPE_CHECKING:
    from .core import DaskAwkwardArray


def load_nested() -> DaskAwkwardArray:
    return from_json(json_data(kind="records"))


def load_array() -> DaskAwkwardArray:
    return from_json(json_data(kind="numbers"))
