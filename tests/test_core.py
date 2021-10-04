from __future__ import annotations

from typing import TYPE_CHECKING

import awkward as ak
import pytest

import dask_awkward as dak

if TYPE_CHECKING:
    from dask_awkward.core import DaskAwkwardArray


def load_json() -> DaskAwkwardArray:
    files = [f"scratch/data/simple{i}.json" for i in range(1, 4)]
    a = dak.from_json(files)
    return a


@pytest.mark.parametrize("axis", [None, 1])
def test_min(axis):
    daa = load_json().analysis.x
    aa: ak.Array = daa.compute()
    ar = ak.min(aa, axis=axis)
    dr = dak.min(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert list(ar) == list(dr)
    else:
        assert ar == dr


@pytest.mark.parametrize("axis", [None, 1])
@pytest.mark.parametrize("attr", ["x", "y", "z"])
def test_max(axis, attr):
    daa = load_json().analysis[attr]
    aa: ak.Array = daa.compute()
    ar = ak.max(aa, axis=axis)
    dr = dak.max(daa, axis=axis).compute()
    if isinstance(ar, ak.Array):
        assert ar.to_list() == dr.to_list()
    else:
        assert ar == dr
