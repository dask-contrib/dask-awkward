from __future__ import annotations

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import awkward._v2 as ak
import awkward_datasets as akds
import fsspec
import pytest

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
