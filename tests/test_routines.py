from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from helpers import (  # noqa: F401
    line_delim_records_file,
    load_records_eager,
    load_records_lazy,
)


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.xfail
def test_min(line_delim_records_file, axis) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file).analysis.x1
    caa = load_records_eager(line_delim_records_file).analysis.x1
    ar = ak.min(caa, axis=axis)
    dr = dak.min(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert list(ar) == list(dr)
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
@pytest.mark.xfail
def test_max(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file).analysis[attr]
    caa = load_records_eager(line_delim_records_file).analysis[attr]
    ar = ak.max(caa, axis=axis)
    dr = dak.max(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
@pytest.mark.xfail
def test_sum(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file).analysis[attr]
    caa = load_records_eager(line_delim_records_file).analysis[attr]
    ar = ak.sum(caa, axis=axis)
    dr = dak.sum(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize(
    "axis",
    [
        None,
        0,
        1,
        pytest.param(2, marks=pytest.mark.xfail),
        -1,
        -2,
        pytest.param(3, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.xfail
def test_flatten(line_delim_records_file, axis) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]["x1"]
    caa = load_records_eager(line_delim_records_file)["analysis"]["x1"]
    ar = ak.flatten(caa, axis=axis)
    dr = dak.flatten(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(None, marks=pytest.mark.xfail),
        pytest.param(0, marks=pytest.mark.xfail),
        1,
        pytest.param(2, marks=pytest.mark.xfail),
        -1,
        pytest.param(-2, marks=pytest.mark.xfail),
        pytest.param(3, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.xfail
def test_num(line_delim_records_file, axis) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]["x1"]
    caa = load_records_eager(line_delim_records_file)["analysis"]["x1"]
    ar = ak.num(caa, axis=axis)
    dr = dak.num(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
@pytest.mark.xfail
def test_count(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]["x1"]
    caa = load_records_eager(line_delim_records_file)["analysis"]["x1"]
    ar = ak.count(caa, axis=axis)
    dr = dak.count(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
@pytest.mark.xfail
def test_count_nonzero(line_delim_records_file, axis, attr) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)["analysis"]["x1"]
    caa = load_records_eager(line_delim_records_file)["analysis"]["x1"]
    ar = ak.count_nonzero(caa, axis=axis)
    dr = dak.count_nonzero(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr
