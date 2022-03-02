from __future__ import annotations

import awkward._v2 as ak
import pytest

import dask_awkward as dak
from dask_awkward.testutils import assert_eq, caa, daa  # noqa: F401


@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("mask_identity", [True, False])
@pytest.mark.parametrize("testval", [-1, 3, 100])
def test_all(daa, caa, axis, keepdims, mask_identity, testval) -> None:  # noqa: F811
    x1d = daa.analysis.x1
    x1c = caa.analysis.x1
    maskd = x1d > testval
    maskc = x1c > testval
    ar = ak.all(x1c[maskc], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    dr = dak.all(x1d[maskd], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("mask_identity", [True, False])
@pytest.mark.parametrize("testval", [-1, 3, 100])
def test_any(daa, caa, axis, keepdims, mask_identity, testval) -> None:  # noqa: F811
    x1d = daa.analysis.x1
    x1c = caa.analysis.x1
    maskd = x1d > testval
    maskc = x1c > testval
    ar = ak.any(x1c[maskc], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    dr = dak.any(x1d[maskd], axis=axis, keepdims=keepdims, mask_identity=mask_identity)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_min(daa, caa, axis, attr) -> None:  # noqa: F811
    ar = ak.min(caa.analysis[attr], axis=axis)
    dr = dak.min(daa["analysis", attr], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_max(daa, caa, axis, attr) -> None:  # noqa: F811
    ar = ak.max(caa.analysis[attr], axis=axis)
    dr = dak.max(daa.analysis[attr], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_sum(daa, caa, axis, attr) -> None:  # noqa: F811
    ar = ak.sum(caa.analysis[attr], axis=axis)
    dr = dak.sum(daa.analysis[attr], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
def test_count(daa, caa, axis) -> None:  # noqa: F811
    ar = ak.count(caa["analysis"]["x1"], axis=axis)
    dr = dak.count(daa["analysis"]["x1"], axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
def test_count_nonzero(daa, caa, axis) -> None:  # noqa: F811
    ar = ak.count_nonzero(caa["analysis", "x1"], axis=axis)
    dr = dak.count_nonzero(daa["analysis"]["x1"], axis=axis)
    assert_eq(ar, dr)
