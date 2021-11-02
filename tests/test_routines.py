from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from dask_awkward.utils import load_array, load_nested


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
def test_min(axis) -> None:
    daa = load_nested().analysis.x1
    aa = daa.compute()
    ar = ak.min(aa, axis=axis)
    dr = dak.min(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert list(ar) == list(dr)
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_max(axis, attr) -> None:
    daa = load_nested().analysis[attr]
    aa = daa.compute()
    ar = ak.max(aa, axis=axis)
    dr = dak.max(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_sum(axis, attr) -> None:
    daa = load_nested().analysis[attr]
    aa = daa.compute()
    ar = ak.sum(aa, axis=axis)
    dr = dak.sum(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize(
    "axis", [None, 0, 1, 2, -1, -2, pytest.param(3, marks=pytest.mark.xfail)]
)
def test_flatten(axis) -> None:
    daa = load_array()
    aa = daa.compute()
    ar = ak.flatten(aa, axis=axis)
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
        2,
        -1,
        -2,
        pytest.param(3, marks=pytest.mark.xfail),
    ],
)
def test_num(axis) -> None:
    daa = load_array()
    aa = daa.compute()
    ar = ak.num(aa, axis=axis)
    dr = dak.num(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_count(axis, attr) -> None:
    daa = load_nested().analysis[attr]
    aa = daa.compute()
    ar = ak.count(aa, axis=axis)
    dr = dak.count(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("attr", ["x1", "z2"])
def test_count_nonzero(axis, attr) -> None:
    daa = load_nested().analysis[attr]
    aa = daa.compute()
    ar = ak.count_nonzero(aa, axis=axis)
    dr = dak.count_nonzero(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr
