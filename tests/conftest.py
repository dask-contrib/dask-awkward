from __future__ import annotations

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import awkward._v2 as ak
import awkward_datasets as akds
import fsspec
import pytest

import dask_awkward as dak
import dask_awkward.testutils as daktu
from dask_awkward import from_json
from dask_awkward.core import Array


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


@pytest.fixture(scope="session")
def line_delim_records_file() -> str:
    """Fixture providing a file name pointing to line deliminted JSON records."""
    return str(akds.line_delimited_records())


@pytest.fixture(scope="session")
def concrete_from_line_delim(line_delim_records_file) -> ak.Array:
    """Fixture returning a concrete array from the line delim records file."""
    with fsspec.open(line_delim_records_file, "rt") as f:
        return ak.from_json(f.read())


@pytest.fixture(scope="session")
def single_record_file() -> str:
    """Fixture providing file name pointing to a single JSON record."""
    return str(akds.single_record())


@pytest.fixture(scope="session")
def daa() -> Array:
    """Fixture providing a Dask Awkward Array collection."""
    return load_records_lazy(str(akds.line_delimited_records()))


@pytest.fixture(scope="session")
def caa() -> ak.Array:
    """Fixture providing a concrete Awkward Array."""
    return load_records_eager(str(akds.line_delimited_records()))


@pytest.fixture(scope="session")
def points_ndjson_file1(tmpdir_factory) -> str:
    array = daktu.awkward_xy_points()
    fn = tmpdir_factory.mktemp("data").join("points_ndjson1.json")
    with fsspec.open(fn, "w") as f:
        for entry in array.tolist():
            print(json.dumps({"points": entry}), file=f)
    return fn


@pytest.fixture(scope="session")
def points_ndjson_file2(tmpdir_factory) -> str:
    array = daktu.awkward_xy_points()
    fn = tmpdir_factory.mktemp("data").join("points_ndjson2.json")
    with fsspec.open(fn, "w") as f:
        for entry in array.tolist():
            print(json.dumps({"points": entry}), file=f)
    return fn


@pytest.fixture(scope="session")
def daa_p1(points_ndjson_file1: str) -> dak.Array:
    return dak.from_json([points_ndjson_file1] * 3)


@pytest.fixture(scope="session")
def daa_p2(points_ndjson_file2: str) -> dak.Array:
    return dak.from_json([points_ndjson_file2] * 3)


@pytest.fixture(scope="session")
def caa_p1(points_ndjson_file1: str) -> ak.Array:
    with open(points_ndjson_file1) as f:
        lines = [json.loads(line) for line in f]
    return ak.Array(lines * 3)


@pytest.fixture(scope="session")
def caa_p2(points_ndjson_file2: str) -> ak.Array:
    with open(points_ndjson_file2) as f:
        lines = [json.loads(line) for line in f]
    return ak.Array(lines * 3)


@pytest.fixture(scope="session")
def L1() -> list:
    return [
        [{"x": 1.0, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4.0, "y": 4.4}, {"x": 5, "y": 5.5}],
        [{"x": 6.0, "y": 6.6}],
        [{"x": 7.0, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}],
    ]


@pytest.fixture(scope="session")
def L2() -> list:
    return [
        [{"x": 0.9, "y": 1.0}, {"x": 2, "y": 2.2}, {"x": 2.9, "y": 3}],
        [],
        [{"x": 3.9, "y": 4.0}, {"x": 5, "y": 5.5}],
        [{"x": 5.9, "y": 6.0}],
        [{"x": 6.9, "y": 7.0}, {"x": 8, "y": 8.8}, {"x": 8.9, "y": 9}],
    ]


@pytest.fixture(scope="session")
def L3() -> list:
    return [
        [{"x": 1.9, "y": 9.0}, {"x": 2, "y": 8.2}, {"x": 9.9, "y": 9}],
        [],
        [{"x": 1.9, "y": 8.0}, {"x": 4, "y": 6.5}],
        [{"x": 1.9, "y": 7.0}],
        [{"x": 1.9, "y": 6.0}, {"x": 6, "y": 4.8}, {"x": 9.9, "y": 9}],
    ]
