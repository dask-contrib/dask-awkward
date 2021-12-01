from __future__ import annotations

try:
    import ujson as json
except ImportError:
    import json

import pathlib
from typing import TYPE_CHECKING

import fsspec
from awkward._v2.highlevel import Array
from awkward._v2.operations.convert import from_iter

from dask_awkward.io import from_json

if TYPE_CHECKING:
    from dask_awkward.core import DaskAwkwardArray


_DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"


def resolved_data_file(name: str) -> str:
    return str(_DATA_DIR / name)


def load_records_lazy(
    blocksize: int | str = 1024,
    by_file: bool = False,
    ntimes: int = 1,
) -> DaskAwkwardArray:
    records_file = resolved_data_file("records.json")
    if by_file:
        return from_json([records_file] * ntimes)
    return from_json(records_file, blocksize=blocksize)


def load_records_eager(ntimes: int = 1) -> Array:
    records_file = resolved_data_file("records.json")
    with fsspec.open(records_file) as f:
        return from_iter(json.loads(line) for line in f)


def load_single_record_lazy() -> DaskAwkwardArray:
    record_file = resolved_data_file("single-record.json")
    return from_json(
        record_file,
        delimiter=None,
        blocksize=None,
        one_obj_per_file=True,
    )
