from __future__ import annotations

import operator
from collections.abc import Callable

import awkward._v2 as ak
import pytest

import dask_awkward as dak
import dask_awkward.core as dakc
from dask_awkward.core import DaskAwkwardNotImplemented, IncompatiblePartitions
from dask_awkward.testutils import assert_eq


def test_getattr_raise(daa: dak.Array) -> None:
    dar = daa[0]
    assert type(dar) is dakc.Record
    with pytest.raises(AttributeError, match="not in fields"):
        assert daa.abcdefg
    with pytest.raises(AttributeError, match="not in fields"):
        assert dar.x3


def test_multi_string(daa: dak.Array, caa: ak.Array) -> None:
    assert_eq(
        daa["points"][["x", "y"]],
        caa["points"][["x", "y"]],
    )


def test_single_string(daa: dak.Array, caa: ak.Array) -> None:
    assert_eq(daa["points"], caa["points"])


def test_layered_string(daa: dak.Array, caa: ak.Array) -> None:
    assert_eq(daa["points", "x"], caa["points", "x"])
    assert_eq(daa["points", "x"], caa["points"]["x"])
    assert_eq(caa["points", "x"], daa["points"]["x"])
    assert_eq(daa[["points"], ["x", "y"]], caa[["points"], ["x", "y"]])
    assert_eq(daa["points", ["x", "y"]], caa["points", ["x", "y"]])


def test_list_with_ints_raise(daa: dak.Array) -> None:
    with pytest.raises(RuntimeError, match="Lists containing integers"):
        assert daa[[1, 2]]


def test_single_int(daa: dak.Array, caa: ak.Array) -> None:
    total = len(daa)
    assert daa.known_divisions
    for i in range(total):
        a = daa["points"]["x"]
        c = caa["points"]["x"]
        assert a.known_divisions
        assert_eq(a[i], c[i])
        assert_eq(a[-i], c[-i])
        a = daa[i, "points", "x"]
        c = caa[i, "points", "x"]
        assert not a.known_divisions
        assert_eq(a, c)
    for i in range(total):
        assert caa[i].tolist() == daa[i].compute().tolist()
        assert caa["points"][i].tolist() == daa["points"][i].compute().tolist()


def test_single_ellipsis(daa: dak.Array, caa: ak.Array) -> None:
    assert_eq(daa[...], caa[...])


def test_empty_slice(daa: dak.Array, caa: ak.Array) -> None:
    assert_eq(daa[:], caa[:])
    assert_eq(daa[:, "points"], caa[:, "points"])


def test_record_getitem(daa: dak.Array, caa: ak.Array) -> None:
    assert daa[0].compute().to_list() == caa[0].to_list()
    assert daa["points"]["x"][0][0].compute() == caa["points"]["x"][0][0]
    assert daa[0]["points"].compute().to_list() == caa[0]["points"].to_list()
    assert daa["points"][0].compute().to_list() == caa["points"][0].to_list()
    assert daa["points"][0].x.compute().to_list() == caa["points"][0].x.to_list()
    assert daa[0]["points"]["x"][0].compute() == caa[0]["points", "x"][0]


@pytest.mark.parametrize("op", [operator.gt, operator.ge, operator.le, operator.lt])
def test_boolean_array(daa: dak.Array, op: Callable) -> None:
    caa = daa.compute()
    dx = daa.points.x
    cx = caa.points.x
    dxs = op(dx, 2)
    cxs = op(cx, 2)
    dx_p = dx[dxs]
    cx_p = cx[cxs]
    assert_eq(dx_p, cx_p)


def test_boolean_array_from_awkward(daa: dak.Array) -> None:
    cx_2 = daa.points.x.compute()
    dx_2 = dak.from_awkward(cx_2, npartitions=6)
    dx_2 = dx_2[dx_2 > 2]
    assert_eq(dx_2, cx_2[cx_2 > 2])


def test_tuple_boolean_array_raise(daa: dak.Array) -> None:
    sel = dak.num(daa.points.x, axis=1) >= 2
    with pytest.raises(
        DaskAwkwardNotImplemented,
        match="tuple style input boolean selection is not supported",
    ):
        daa[sel, "points"]


def test_bad_partition_boolean_array(ndjson_points_file: str) -> None:
    daa1 = dak.from_json([ndjson_points_file] * 2)
    daa2 = dak.from_json([ndjson_points_file] * 3)
    sel = dak.num(daa1.points.x > 2, axis=1) >= 2
    with pytest.raises(IncompatiblePartitions):
        daa2[sel]
