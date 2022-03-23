import awkward_datasets as akds
import pytest

from dask_awkward.testutils import load_records_eager, load_records_lazy


@pytest.fixture(scope="session")
def line_delim_records_file():
    """Fixture providing a file name pointing to line deliminted JSON records."""
    return str(akds.line_delimited_records())


@pytest.fixture(scope="session")
def single_record_file():
    """Fixture providing file name pointing to a single JSON record."""
    return str(akds.single_record())


@pytest.fixture(scope="session")
def daa():
    """Fixture providing a Dask Awkward Array collection."""
    return load_records_lazy(str(akds.line_delimited_records()))


@pytest.fixture(scope="session")
def caa():
    """Fixture providing a concrete Awkward Array."""
    return load_records_eager(str(akds.line_delimited_records()))
