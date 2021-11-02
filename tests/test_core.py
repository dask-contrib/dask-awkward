from __future__ import annotations

import pytest

from dask_awkward.utils import load_nested


def test_meta_exists() -> None:
    daa = load_nested()
    assert daa.meta is not None
    assert daa.analysis.x1.meta is not None


@pytest.mark.xfail
def test_fields() -> None:
    daa = load_nested()
    assert daa.fields == daa.compute().fields
    assert daa.analysis.fields == daa.analysis.compute().fields
