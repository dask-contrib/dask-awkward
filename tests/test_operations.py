from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq
from dask_awkward.utils import IncompatiblePartitions


def test_concatenate_simple(daa, caa):
    assert_eq(
        ak.concatenate([caa.points.x, caa.points.y], axis=0),
        dak.concatenate([daa.points.x, daa.points.y], axis=0),
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_concatenate_more_axes(axis):
    a = [[[1, 2, 3], [], [100, 101], [12, 13]], [[1, 2, 3], [], [100, 101], [12, 13]]]
    b = [
        [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
        [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
    ]
    one = dak.from_lists([a, a])
    two = dak.from_lists([b, b])
    c = dak.concatenate([one, two], axis=axis)
    aa = ak.concatenate([a, a])
    bb = ak.concatenate([b, b])
    cc = ak.concatenate([aa, bb], axis=axis)
    assert_eq(c, cc)

    if axis > 0:
        # add an additional entry to a to trigger bad divisions
        a = [
            [[1, 2, 3], [], [100, 101], [12, 13]],
            [[1, 2, 3], [], [100, 101], [12, 13]],
            [],
        ]
        b = [
            [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
            [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
        ]
        b = [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]]
        one = dak.from_lists([a, a])
        two = dak.from_lists([b, b])
        with pytest.raises(
            IncompatiblePartitions,
            match="The inputs to concatenate are incompatibly partitioned",
        ):
            c = dak.concatenate([one, two], axis=axis)
