from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq


def test_concatenate_axis0(daa, caa):
    assert_eq(
        ak.concatenate([caa.points.x, caa.points.y], axis=0),
        dak.concatenate([daa.points.x, daa.points.y], axis=0),
    )


def test_concatenate_axis1():
    a = [[1, 2, 3], [], [100, 101], [12, 13]]
    b = [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]]
    one = dak.from_lists([a, a])
    two = dak.from_lists([b, b])
    c = dak.concatenate([one, two], axis=1)
    aa = ak.concatenate([a, a])
    bb = ak.concatenate([b, b])
    cc = ak.concatenate([aa, bb], axis=1)
    assert_eq(c, cc)

    # add an additional entry to a to trigger bad divisions
    a = [[1, 2, 3], [], [100, 101], [12, 13], [6]]
    b = [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]]
    one = dak.from_lists([a, a])
    two = dak.from_lists([b, b])
    with pytest.raises(ValueError, match="arrays must have identical divisions"):
        c = dak.concatenate([one, two], axis=1)
