from __future__ import annotations

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import pathlib
from typing import TYPE_CHECKING

import fsspec
from awkward._v2.highlevel import Array
from awkward._v2.operations.convert import from_iter

from dask_awkward.io import from_json

if TYPE_CHECKING:
    from dask_awkward.core import DaskAwkwardArray

import pytest

_DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"


# fmt: off
MANY_RECORDS = \
    """{"analysis":{"x1":[1,2,3],"y1":[2,3,4],"z1":[2,6,6],"t1":[7,8,9],"x2":[],"y2":[],"z2":[],"t2":[]}}
{"analysis":{"x1":[1,2],"y1":[2,3],"z1":[3,4],"t1":[4,5],"x2":[2,9],"y2":[2,8],"z2":[2,7],"t2":[0,6]}}
{"analysis":{"x1":[],"y1":[],"z1":[],"t1":[],"x2":[3,2,1],"y2":[4,3,2],"z2":[5,4,3],"t2":[6,5,4]}}
{"analysis":{"x1":[],"y1":[],"z1":[],"t1":[],"x2":[1,2,3],"y2":[2,3,4],"z2":[2,6,6],"t2":[7,8,9]}}
{"analysis":{"x1":[3,2,1],"y1":[4,3,2],"z1":[5,4,3],"t1":[6,5,4],"x2":[],"y2":[],"z2":[],"t2":[]}}
{"analysis":{"x1":[],"y1":[],"z1":[],"t1":[],"x2":[1,2],"y2":[2,3],"z2":[2,6],"t2":[7,8]}}
{"analysis":{"x1":[3,2,1,4],"y1":[4,3,2,5],"z1":[5,4,3,6],"t1":[6,5,4,7],"x2":[1,2],"y2":[3,4],"z2":[5,6],"t2":[7,8]}}
{"analysis":{"x1":[1,2,3],"y1":[2,3,4],"z1":[2,6,6],"t1":[7,8,9],"x2":[],"y2":[],"z2":[],"t2":[]}}
{"analysis":{"x1":[1,2],"y1":[2,3],"z1":[3,4],"t1":[4,5],"x2":[2,9],"y2":[2,8],"z2":[2,7],"t2":[0,6]}}
{"analysis":{"x1":[],"y1":[],"z1":[],"t1":[],"x2":[3,2,1],"y2":[4,3,2],"z2":[5,4,3],"t2":[6,5,4]}}
{"analysis":{"x1":[],"y1":[],"z1":[],"t1":[],"x2":[1,2,3],"y2":[2,3,4],"z2":[2,6,6],"t2":[7,8,9]}}
{"analysis":{"x1":[3,2,1],"y1":[4,3,2],"z1":[5,4,3],"t1":[6,5,4],"x2":[],"y2":[],"z2":[],"t2":[]}}
{"analysis":{"x1":[2,9],"y1":[2,8],"z1":[2,7],"t1":[0,6],"x2":[1,2],"y2":[2,3],"z2":[3,4],"t2":[4,5]}}
{"analysis":{"x1":[],"y1":[],"z1":[],"t1":[],"x2":[3,2,1],"y2":[4,3,2],"z2":[5,4,3],"t2":[6,5,4]}}
{"analysis":{"x1":[3,2,1],"y1":[4,3,2],"z1":[5,4,3],"t1":[6,5,4],"x2":[],"y2":[],"z2":[],"t2":[]}}
{"analysis":{"x1":[2,9],"y1":[2,8],"z1":[2,7],"t1":[0,6],"x2":[1,2],"y2":[2,3],"z2":[3,4],"t2":[4,5]}}
{"analysis":{"x1":[1,9,1],"y1":[1,8,2],"z1":[1,7,3],"t1":[1,6,4],"x2":[3,2,5],"y2":[3,3,6],"z2":[3,4,7],"t2":[3,5,8]}}
{"analysis":{"x1":[],"y1":[],"z1":[],"t1":[],"x2":[1,2],"y2":[2,3],"z2":[2,6],"t2":[7,8]}}
{"analysis":{"x1":[3,2,1,4],"y1":[4,3,2,5],"z1":[5,4,3,6],"t1":[6,5,4,7],"x2":[1,2],"y2":[3,4],"z2":[5,6],"t2":[7,8]}}
{"analysis":{"x1":[1,2,3],"y1":[2,3,4],"z1":[2,6,6],"t1":[7,8,9],"x2":[],"y2":[],"z2":[],"t2":[]}}"""


SINGLE_RECORD = """{"a":[1,2,3]}"""

# fmt: on


@pytest.fixture(scope="session")
def line_delim_records_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("records.json")
    with open(fn, "w") as f:
        f.write(MANY_RECORDS)
    return str(fn)


@pytest.fixture(scope="session")
def single_record_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("single-record.json")
    with open(fn, "w") as f:
        f.write(SINGLE_RECORD)
    return str(fn)


def load_records_lazy(
    fn,
    blocksize: int | str = 700,
    by_file: bool = False,
    ntimes: int = 1,
) -> DaskAwkwardArray:
    if by_file:
        return from_json([fn] * ntimes)
    return from_json(fn, blocksize=blocksize)


def load_records_eager(fn, ntimes: int = 1) -> Array:
    with fsspec.open(fn) as f:
        return from_iter(json.loads(line) for line in f)


def load_single_record_lazy(fn) -> DaskAwkwardArray:
    return from_json(
        fn,
        delimiter=None,
        blocksize=None,
        one_obj_per_file=True,
    )
