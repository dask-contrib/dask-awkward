from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
import dask_awkward.core as dakc
from dask_awkward.utils import load_nested


def test_meta_exists() -> None:
    daa = load_nested()
    assert daa.meta is not None
    assert daa["analysis"]["x1"].meta is not None


@pytest.mark.xfail
def test_fields() -> None:
    daa = load_nested()
    assert dak.fields(daa) == ak.fields(daa.compute())
    assert dak.fields(daa["analysis"]) == ak.fields(daa["analysis"].compute())


def test_calculate_known_divisions() -> None:
    daa = load_nested()
    target = (3, 6, 9)
    assert dakc.calculate_known_divisions(daa) == target
    assert dakc.calculate_known_divisions(daa.analysis) == target
    assert dakc.calculate_known_divisions(daa.analysis.x1) == target
    assert dakc.calculate_known_divisions(daa["analysis"][["x1", "x2"]]) == target
