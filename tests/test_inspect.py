from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
import dask_awkward.lib.testutils as daktu
from dask_awkward.lib.inspect import necessary_columns
from dask_awkward.lib.testutils import assert_eq


def test_necessary_columns(
    daa: dak.Array,
    tmpdir_factory: pytest.TempdirFactory,
) -> None:
    dname = tmpdir_factory.mktemp("pq")
    dak.to_parquet(daa, str(dname), compute=True)
    ds = dak.from_parquet(str(dname))
    aioname = list(ds.dask.layers.items())[0][0]
    assert necessary_columns(ds.points.x, "brute") == {aioname: ["points.x"]}
    assert necessary_columns(ds.points.x, "getitem") == {aioname: None}
    with pytest.raises(ValueError, match="strategy argument should"):
        assert necessary_columns(ds.points, "okokok")  # type: ignore


def test_necessary_columns_gh126(tmpdir_factory):
    d = tmpdir_factory.mktemp("pq126")
    dname1 = d / "f1.parquet"
    dname2 = d / "f2.parquet"
    lists = daktu.lists().compute()
    ak.to_parquet(lists, str(dname1), extensionarray=False)
    ak.to_parquet(lists, str(dname2), extensionarray=False)
    daa = dak.from_parquet([str(dname1), str(dname2)])
    caa = ak.concatenate([lists, lists])

    case0 = daa[daa.x > 1.0].x
    nc = necessary_columns(case0, "getitem")
    assert list(nc.values())[0] == ["x"]
    assert_eq(case0, caa[caa.x > 1.0].x, check_forms=False)

    case1 = daa[daa.x > 1.0].y
    nc = necessary_columns(case1, "getitem")
    assert list(nc.values())[0] is None
    assert_eq(case1, caa[caa.x > 1.0].y)

    case2 = daa[daa.x > 1.0]
    nc = necessary_columns(case2, "getitem")
    assert list(nc.values())[0] is None  # none because we never select anything
    assert_eq(caa[caa.x > 1.0], case2)

    case3 = daa[daa.y > 1.0].y
    nc2 = necessary_columns(case3, "brute")
    assert list(nc2.values())[0] == ["y"]
    assert_eq(caa[caa.y > 1.0].y, case3, check_forms=False)

    case4 = daa[daa.y > 1.0].x
    nc2 = necessary_columns(case4, "brute")
    assert list(nc2.values())[0] is None
    assert_eq(caa[caa.y > 1.0].x, case4, check_forms=False)
