from __future__ import annotations

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import os
import tempfile
from typing import Any

import awkward._v2 as ak
import fsspec
import pytest
from dask.base import is_dask_collection

import dask_awkward as dak

from ..core import Array, Record, from_awkward
from ..io import from_json
from ..utils import idempotent_concatenate

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


def aeq(a, b):
    a_is_coll = is_dask_collection(a)
    b_is_coll = is_dask_collection(b)
    a_comp = a.compute() if a_is_coll else a
    b_comp = b.compute() if b_is_coll else b
    a_tt = dak.typetracer_array(a)
    b_tt = dak.typetracer_array(b)

    # first check high level values
    assert a_comp.tolist() == b_comp.tolist()

    # then check forms
    a_concated_form = idempotent_concatenate(a_tt).layout.form
    b_concated_form = idempotent_concatenate(b_tt).layout.form
    assert a_concated_form == b_concated_form
    if a_is_coll:
        assert a_comp.layout.form == a_concated_form
    if b_is_coll:
        assert b_comp.layout.form == b_concated_form

    # check the unconcatenated versions as well
    if a_is_coll and not b_is_coll:
        assert b_tt.layout.form == a.partitions[0].compute().layout.form
    if not a_is_coll and b_is_coll:
        assert a_tt.layout.form == b.partitions[0].compute().layout.form


def assert_eq(a: Any, b: Any) -> None:
    if isinstance(a, (Array, ak.Array)):
        aeq(a, b)
    elif isinstance(a, (Record, ak.Record)):
        aeq(a, b)
    else:
        assert_eq_other(a, b)


def assert_eq_arrays(a: Array | ak.Array, b: Array | ak.Array) -> None:
    ares = a.compute() if is_dask_collection(a) else a
    bres = b.compute() if is_dask_collection(b) else b
    assert ares.tolist() == bres.tolist()


def assert_eq_records(a: Record | ak.Record, b: Record | ak.Record) -> None:
    ares = a.compute() if is_dask_collection(a) else a
    bres = b.compute() if is_dask_collection(b) else b
    assert ares.tolist() == bres.tolist()


def assert_eq_other(a: Any, b: Any) -> None:
    if is_dask_collection(a) and not is_dask_collection(b):
        assert a.compute() == b
    if is_dask_collection(b) and not is_dask_collection(a):
        assert a == b.compute()
    if not is_dask_collection(a) and not is_dask_collection(b):
        assert a == b


@pytest.fixture(scope="session")
def line_delim_records_file(tmpdir_factory):
    """Fixture providing a file name pointing to line deliminted JSON records."""
    fn = tmpdir_factory.mktemp("data").join("records.json")
    with open(fn, "w") as f:
        f.write(MANY_RECORDS)
    return str(fn)


@pytest.fixture(scope="session")
def single_record_file(tmpdir_factory):
    """Fixture providing file name pointing to a single JSON record."""
    fn = tmpdir_factory.mktemp("data").join("single-record.json")
    with open(fn, "w") as f:
        f.write(SINGLE_RECORD)
    return str(fn)


@pytest.fixture(scope="session")
def daa(tmpdir_factory):
    """Fixture providing a file name pointing to line deliminted JSON records."""
    fn = tmpdir_factory.mktemp("data").join("records.json")
    with open(fn, "w") as f:
        f.write(MANY_RECORDS)
    return load_records_lazy(fn)


@pytest.fixture(scope="session")
def caa(tmpdir_factory):
    """Fixture providing a file name pointing to line deliminted JSON records."""
    fn = tmpdir_factory.mktemp("data").join("records.json")
    with open(fn, "w") as f:
        f.write(MANY_RECORDS)
    return load_records_eager(fn)


def records_from_temp_file(n_times: int = 1) -> ak.Array:
    """Get a concrete Array of records from a temporary file.

    Parameters
    ----------
    n_times : int
        Number of times to parse the file file of records.

    Returns
    -------
    Array
        Resulting concrete Awkward Array.

    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(MANY_RECORDS)
        name = f.name
    x = load_records_eager(name, n_times=n_times)
    os.remove(name)
    return x


def single_record_from_temp_file() -> ak.Array:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(SINGLE_RECORD)
        name = f.name
    x = load_single_record_eager(name)
    os.remove(name)
    return x


def load_records_lazy(
    fn: str,
    blocksize: int | str = 700,
    by_file: bool = False,
    n_times: int = 1,
) -> Array:
    """Load a record array Dask Awkward Array collection.

    Parameters
    ----------
    fn : str
        File name.
    blocksize : int | str
        Blocksize in bytes for lazy reading.
    by_file : bool
        Read by file instead of by bytes.
    n_times : int
        Number of times to read the file.

    Returns
    -------
    Array
        Resulting Dask Awkward Array collection.

    """
    if by_file:
        return from_json([fn] * n_times)
    return from_json(fn, blocksize=blocksize)


def load_records_eager(fn: str, n_times: int = 1) -> ak.Array:
    """Load a concrete Awkward record array.

    Parameters
    ----------
    fn : str
        File name.
    n_times : int
        Number of times to read the file.

    Returns
    -------
    Array
        Resulting concrete Awkward Array.

    """
    files = [fn] * n_times
    loaded = []
    for ff in files:
        with fsspec.open(ff) as f:
            loaded += list(json.loads(line) for line in f)
    return ak.from_iter(loaded)


def load_single_record_lazy(fn: str) -> Array:
    return from_json(
        fn,
        delimiter=None,
        blocksize=None,
        one_obj_per_file=True,
    )


def load_single_record_eager(fn: str) -> ak.Array:
    with fsspec.open(fn) as f:
        d = json.load(f)
    return ak.Array([d])


def _lazyrecords() -> Array:
    return from_awkward(records_from_temp_file(), npartitions=5)


def _lazyrecord() -> Array:
    return from_awkward(single_record_from_temp_file(), npartitions=1)


def _lazyjsonrecords() -> Array:
    with tempfile.NamedTemporaryFile("w", delete=False) as fp:
        name = fp.name
        fp.write(MANY_RECORDS)
    arr = from_json([name, name])
    return arr
