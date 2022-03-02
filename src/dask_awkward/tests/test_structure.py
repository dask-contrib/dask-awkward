from __future__ import annotations

import awkward._v2 as ak
import pytest

import dask_awkward as dak
from dask_awkward.testutils import (  # noqa: F401
    assert_eq,
    line_delim_records_file,
    load_records_eager,
    load_records_lazy,
)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(None, marks=pytest.mark.xfail),
        0,
        1,
    ],
)
def test_num(line_delim_records_file, axis) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]
    caa = load_records_eager(line_delim_records_file)["analysis"]

    # TODO: also test before this forced computation.
    if axis == 0:
        daa.eager_compute_divisions()

    assert_eq(dak.num(daa.x1, axis=axis), ak.num(caa.x1, axis=axis))

    if axis == 1:
        c1 = dak.num(daa.x1, axis=axis) > 2
        c2 = ak.num(caa.x1, axis=axis) > 2
        assert_eq(daa[c1], caa[c2])
