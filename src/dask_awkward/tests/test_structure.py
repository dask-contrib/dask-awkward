from __future__ import annotations

import awkward._v2 as ak
import pytest

import dask_awkward as dak
from dask_awkward.testutils import assert_eq


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_flatten(caa, daa, axis) -> None:
    cr = ak.flatten(caa.analysis.x2, axis=axis)
    dr = dak.flatten(daa.analysis.x2, axis=axis)
    assert_eq(cr, dr)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(None, marks=pytest.mark.xfail),
        0,
        1,
    ],
)
def test_num(daa, caa, axis) -> None:
    da = daa["analysis"]
    ca = caa["analysis"]

    if axis == 0:
        assert_eq(dak.num(da.x1, axis=axis), ak.num(ca.x1, axis=axis))
        da.eager_compute_divisions()

    assert_eq(dak.num(da.x1, axis=axis), ak.num(ca.x1, axis=axis))

    if axis == 1:
        c1 = dak.num(da.x1, axis=axis) > 2
        c2 = ak.num(ca.x1, axis=axis) > 2
        assert_eq(da[c1], ca[c2])


def test_zip(daa, caa) -> None:
    da1 = daa["analysis"]["x1"]
    da2 = daa["analysis"]["x1"]
    ca1 = caa["analysis"]["x1"]
    ca2 = caa["analysis"]["x1"]

    da_z = dak.zip({"a": da1, "b": da2})
    ca_z = ak.zip({"a": ca1, "b": ca2})
    assert_eq(da_z, ca_z)
