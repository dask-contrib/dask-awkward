from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
import dask_awkward.lib.testutils
from dask_awkward.lib.inspect import necessary_columns


def test_necessary_columns(
    daa: dak.Array,
    tmpdir_factory: pytest.TempdirFactory,
) -> None:
    dname = tmpdir_factory.mktemp("pq")
    dak.to_parquet(daa, str(dname), compute=True)
    ds = dak.from_parquet(str(dname))
    aioname = list(ds.dask.layers.items())[0][0]
    assert necessary_columns(ds.points.x, "brute") == {aioname: ["points.x"]}
    assert necessary_columns(ds.points.x, "getitem") == {aioname: ["points"]}
    with pytest.raises(ValueError, match="strategy argument should"):
        assert necessary_columns(ds.points, "okokok")  # type: ignore


def test_necessary_columns_gh126(tmpdir_factory):
    d = tmpdir_factory.mktemp("pq126")
    dname1 = d / "f1.parquet"
    dname2 = d / "f2.parquet"
    lists = dask_awkward.lib.testutils.lists().compute()
    ak.to_parquet(lists, str(dname1))
    ak.to_parquet(lists, str(dname2))
    ds = dak.from_parquet([str(dname1), str(dname2)])
    selection = ds[ds.x > 1.0]
    nc = necessary_columns(selection, "getitem")
    assert list(nc.values())[0] == ["x"]
    assert selection.compute()
