import pytest

from dask_awkward.testutils import (
    MANY_RECORDS,
    SINGLE_RECORD,
    load_records_eager,
    load_records_lazy,
)


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
    """Fixture providing a Dask Awkward Array collection."""
    fn = tmpdir_factory.mktemp("data").join("records.json")
    with open(fn, "w") as f:
        f.write(MANY_RECORDS)
    return load_records_lazy(fn)


@pytest.fixture(scope="session")
def caa(tmpdir_factory):
    """Fixture providing a concrete Awkward Array."""
    fn = tmpdir_factory.mktemp("data").join("records.json")
    with open(fn, "w") as f:
        f.write(MANY_RECORDS)
    return load_records_eager(fn)
