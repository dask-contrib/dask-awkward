from __future__ import annotations

import awkward._v2 as ak
import pytest

import dask_awkward as dak
from dask_awkward.testutils import assert_eq


@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("mask_identity", [True, False])
@pytest.mark.parametrize("testval", [-1, 3, 100])
def test_all(daa, caa, axis, keepdims, mask_identity, testval) -> None:
    xd = daa.points.x
    xc = caa.points.x
    maskd = xd > testval
    maskc = xc > testval
    ar = ak.all(xc[maskc], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    dr = dak.all(xd[maskd], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("mask_identity", [True, False])
@pytest.mark.parametrize("testval", [-1, 3, 100])
def test_any(daa, caa, axis, keepdims, mask_identity, testval) -> None:
    xd = daa.points.x
    xc = caa.points.x
    maskd = xd > testval
    maskc = xc > testval
    ar = ak.any(xc[maskc], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    dr = dak.any(xd[maskd], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [1])
def test_argmax(daa, caa, axis) -> None:
    xd = daa.points.x
    xc = caa.points.x
    dr = dak.argmax(xd, axis=axis)
    ar = ak.argmax(xc, axis=axis)
    assert_eq(dr, ar)


@pytest.mark.parametrize("axis", [1])
def test_argmin(daa, caa, axis) -> None:
    xd = daa.points.x
    xc = caa.points.x
    dr = dak.argmin(xd, axis=axis)
    ar = ak.argmin(xc, axis=axis)
    assert_eq(dr, ar)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
def test_count(daa, caa, axis) -> None:
    ar = ak.count(caa["points"]["x"], axis=axis)
    dr = dak.count(daa["points"]["x"], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
def test_count_nonzero(daa, caa, axis) -> None:
    ar = ak.count_nonzero(caa["points", "x"], axis=axis)
    dr = dak.count_nonzero(daa["points"]["x"], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x", "y"])
def test_max(daa, caa, axis, attr) -> None:
    ar = ak.max(caa.points[attr], axis=axis)
    dr = dak.max(daa.points[attr], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("attr", ["y", "x"])
def test_mean(daa, caa, axis, attr) -> None:
    ar = ak.mean(caa.points[attr], axis=axis)
    dr = dak.mean(daa.points[attr], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x", "y"])
def test_min(daa, caa, axis, attr) -> None:
    ar = ak.min(caa.points[attr], axis=axis)
    dr = dak.min(daa["points", attr], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x", "y"])
def test_sum(daa, caa, axis, attr) -> None:
    ar = ak.sum(caa.points[attr], axis=axis)
    dr = dak.sum(daa.points[attr], axis=axis)
    assert_eq(ar, dr)
