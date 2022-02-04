import fsspec
import pyarrow as pa
import pyarrow.dataset as pad
import pytest

import dask_awkward as dak
from dask_awkward.parquet import _write__metadata

data = [[1, 2, 3], [4, None], None]
arr = pa.array(data)
ds = pa.Table.from_arrays([arr], names=["arr"])
fs = fsspec.filesystem("file")


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


def test_dir_of_one_file(tmpdir):
    pad.write_dataset(ds, tmpdir, format="parquet")
    arr = dak.read_parquet(tmpdir)
    assert arr["arr"].compute().to_list() == data


def test_dir_of_one_file_metadata(tmpdir):
    tmpdir = str(tmpdir)

    pad.write_dataset(ds, tmpdir, format="parquet")
    _write__metadata(["/".join([tmpdir, "part-0.parquet"])], fs, tmpdir)

    arr = dak.read_parquet(tmpdir)
    assert arr["arr"].compute().to_list() == data


def test_dir_of_two_files(tmpdir):
    tmpdir = str(tmpdir)
    paths = ["/".join([tmpdir, _]) for _ in ["part-0.parquet", "part-1.parquet"]]
    pad.write_dataset(ds, tmpdir, format="parquet")
    fs.cp(paths[0], paths[1])
    arr = dak.read_parquet(tmpdir)
    assert arr["arr"].compute().to_list() == data * 2


@pytest.mark.parametrize("ignore_metadata", [True, False])
def test_dir_of_two_files_metadata(tmpdir, ignore_metadata):
    tmpdir = str(tmpdir)
    paths = ["/".join([tmpdir, _]) for _ in ["part-0.parquet", "part-1.parquet"]]
    pad.write_dataset(ds, tmpdir, format="parquet")
    fs.cp(paths[0], paths[1])
    _write__metadata(paths, fs, tmpdir)

    arr = dak.read_parquet(tmpdir, ignore_metadata=ignore_metadata)
    assert arr["arr"].compute().to_list() == data * 2
