from __future__ import annotations

import awkward as ak

import dask_awkward as dak
from dask_awkward.utils import load_nested


def test_meta_exists() -> None:
    daa = load_nested()
    assert daa.meta is not None
    assert daa["analysis"]["x1"].meta is not None


def test_fields() -> None:
    daa = load_nested()
    assert dak.fields(daa) == ak.fields(daa.compute())
    assert dak.fields(daa["analysis"]) == ak.fields(daa["analysis"].compute())
