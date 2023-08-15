from __future__ import annotations

import awkward as ak

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq


def test_split_whitespace():
    a = ak.Array(
        [
            ["abc 123", "fooo   ooo", "123"],
            ["hij\tj"],
            ["lmn op", ""],
            ["123 456 789", "98765 43210"],
        ]
    )
    b = dak.from_awkward(a, npartitions=2)
    a2 = ak.str.split_whitespace(a)
    b2 = dak.str.split_whitespace(b)
    assert_eq(a2, b2)
    assert_eq(ak.num(a2, axis=2), ak.num(b2, axis=2))
