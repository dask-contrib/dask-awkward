import os
import shutil
import tempfile

import pytest

import dask_awkward as dak


@pytest.fixture(scope="module")
def local_copy():
    # could have done this by chaining "simplecache::" into the github URLs
    tmpdir = str(tempfile.mkdtemp())
    import fsspec

    fs = fsspec.filesystem("github", org="scikit-hep", repo="awkward-1.0")
    fs.get("tests/samples/*.parquet", tmpdir)
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_remote_single():
    arr = dak.read_parquet(
        "github://scikit-hep:awkward-1.0@/tests/samples/"
        "nullable-record-primitives-simple.parquet"
    )
    assert arr.compute().to_list() == [
        {"u4": None, "u8": 1},
        {"u4": None, "u8": 2},
        {"u4": None, "u8": 3},
        {"u4": None, "u8": 4},
        {"u4": None, "u8": 5},
    ]


def test_remote_double():
    arr = dak.read_parquet(
        [
            "github://scikit-hep:awkward-1.0@/tests/samples/"
            "nullable-record-primitives-simple.parquet",
            "github://scikit-hep:awkward-1.0@/tests/samples/"
            "nullable-record-primitives-simple.parquet",
        ]
    )
    assert arr.npartitions == 2
    assert (
        arr.compute().to_list()
        == [
            {"u4": None, "u8": 1},
            {"u4": None, "u8": 2},
            {"u4": None, "u8": 3},
            {"u4": None, "u8": 4},
            {"u4": None, "u8": 5},
        ]
        * 2
    )


def test_fixture(local_copy):
    files = os.listdir(local_copy)
    assert files
    assert all([s.endswith("parquet") for s in files])


def test_dir_of_one_file(local_copy, tmpdir):
    pass


def test_dir_of_one_file_metadata(local_copy, tmpdir):
    pass


def test_dir_of_two_files(local_copy, tmpdir):
    pass


@pytest.mark.parametrize("ignore_metadata", [True, False])
def test_dir_of_two_files_metadata(local_copy, tmpdir, ignore_metadata):
    pass
