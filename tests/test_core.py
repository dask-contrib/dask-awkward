from __future__ import annotations

import dask_awkward.core as dakc
from helpers import load_records_lazy


def test_meta_exists() -> None:
    daa = load_records_lazy()
    assert daa.meta is not None
    assert daa["analysis"]["x1"].meta is not None


def test_calculate_known_divisions() -> None:
    daa = load_records_lazy()
    target = (0, 20, 40, 60, 62)
    assert dakc.calculate_known_divisions(daa) == target
    assert dakc.calculate_known_divisions(daa.analysis) == target
    assert dakc.calculate_known_divisions(daa.analysis.x1) == target
    assert dakc.calculate_known_divisions(daa["analysis"][["x1", "x2"]]) == target


def test_len() -> None:
    daa = load_records_lazy()
    assert len(daa) == 63
