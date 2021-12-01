from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import awkward as ak

from dask_awkward.io import from_json

if TYPE_CHECKING:
    from dask_awkward.core import DaskAwkwardArray


_DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"


def resolved_data_file(name: str) -> str:
    return str(_DATA_DIR / name)


def load_records_lazy(
    blocksize: int | str = 1024,
    by_file: bool = False,
) -> DaskAwkwardArray:
    records_file = resolved_data_file("records.json")
    if by_file:
        return from_json([records_file, records_file, records_file])
    return from_json(records_file, blocksize=blocksize)


def load_records_eager() -> ak.Array:
    records_file = resolved_data_file("records.json")
    return ak.from_json(records_file)


def load_single_record_lazy() -> DaskAwkwardArray:
    record_file = resolved_data_file("single-record.json")
    return from_json(
        record_file,
        delimiter=None,
        blocksize=None,
        one_obj_per_file=True,
    )
