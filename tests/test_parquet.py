import os

import pytest

import dask_awkward.parquet

# from dask_awkward.utils import assert_eq

pyarrow = pytest.importorskip("pyarrow")


def test_simple(tmpdir):
    import pyarrow.parquet

    fn = os.path.join(tmpdir, "test.parq")
    data = {"a": [0, 1], "b": [b"hi", b"oi"], "c": [[], [1, 2, 3]]}
    table = pyarrow.Table.from_pydict(data)
    pyarrow.parquet.write_table(table, fn)

    out = dask_awkward.parquet.ak_read_parquet(fn)
    expected = [{"a": 0, "b": b"hi", "c": []}, {"a": 1, "b": b"oi", "c": [1, 2, 3]}]
    assert out.compute().tolist() == expected
