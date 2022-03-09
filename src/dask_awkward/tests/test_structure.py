from __future__ import annotations

import awkward._v2 as ak
import pytest

import dask_awkward as dak
from dask_awkward.testutils import assert_eq, load_records_eager, load_records_lazy


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
def test_num(line_delim_records_file, axis) -> None:
    da = load_records_lazy(line_delim_records_file)["analysis"]
    ca = load_records_eager(line_delim_records_file)["analysis"]

    if axis == 0:
        assert_eq(dak.num(da.x1, axis=axis), ak.num(ca.x1, axis=axis))
        da.eager_compute_divisions()

    assert_eq(dak.num(da.x1, axis=axis), ak.num(ca.x1, axis=axis))

    if axis == 1:
        c1 = dak.num(da.x1, axis=axis) > 2
        c2 = ak.num(ca.x1, axis=axis) > 2
        assert_eq(da[c1], ca[c2])
