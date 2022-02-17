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


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
def test_min(line_delim_records_file, axis) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file).analysis.x1
    caa = load_records_eager(line_delim_records_file).analysis.x1
    ar = ak.min(caa, axis=axis)
    dr = dak.min(daa, axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_max(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file).analysis[attr]
    caa = load_records_eager(line_delim_records_file).analysis[attr]
    ar = ak.max(caa, axis=axis)
    dr = dak.max(daa, axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_sum(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file).analysis[attr]
    caa = load_records_eager(line_delim_records_file).analysis[attr]
    ar = ak.sum(caa, axis=axis)
    dr = dak.sum(daa, axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_count(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]["x1"]
    caa = load_records_eager(line_delim_records_file)["analysis"]["x1"]
    ar = ak.count(caa, axis=axis)
    dr = dak.count(daa, axis=axis)
    assert_eq(ar, dr)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_count_nonzero(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]["x1"]
    caa = load_records_eager(line_delim_records_file)["analysis"]["x1"]
    ar = ak.count_nonzero(caa, axis=axis)
    dr = dak.count_nonzero(daa, axis=axis)
    assert_eq(ar, dr)
