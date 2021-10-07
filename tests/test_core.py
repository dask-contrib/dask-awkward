from __future__ import annotations

from typing import TYPE_CHECKING

import awkward as ak
import pytest

import dask_awkward as dak
import dask_awkward.data as dakd

if TYPE_CHECKING:
    from dask_awkward.core import DaskAwkwardArray


def load_nested() -> DaskAwkwardArray:
    return dak.from_json(dakd.json_data(kind="records"))


def load_array() -> DaskAwkwardArray:
    return dak.from_json(dakd.json_data(kind="numbers"))


@pytest.mark.parametrize("axis", [None, 1, pytest.param(-1, marks=pytest.mark.xfail)])
def test_min(axis):
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
def test_max(axis, attr):
    daa = load_nested().analysis[attr]
    aa = daa.compute()
    ar = ak.max(aa, axis=axis)
    dr = dak.max(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr


@pytest.mark.parametrize(
    "axis", [None, 0, 1, 2, -1, -2, pytest.param(3, marks=pytest.mark.xfail)]
)
def test_flatten(axis):
    daa = load_array()
    aa = daa.compute()
    ar = ak.flatten(aa, axis=axis)
    dr = dak.flatten(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr
