from __future__ import annotations

import awkward._v2 as ak
import pytest

import dask_awkward as dak
from dask_awkward.testutils import assert_eq

from .helpers import (  # noqa: F401
    line_delim_records_file,
    load_records_eager,
    load_records_lazy,
)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(None, marks=pytest.mark.xfail),
        pytest.param(0, marks=pytest.mark.xfail),
        1,
    ],
)
def test_num(line_delim_records_file, axis) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]
    caa = load_records_eager(line_delim_records_file)["analysis"]
    assert_eq(dak.num(daa.x1, axis=axis), ak.num(caa.x1, axis=axis))
