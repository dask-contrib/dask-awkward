from __future__ import annotations

import awkward._v2 as ak
import numpy as np
import pytest

import dask_awkward as dak
from dask_awkward.testutils import assert_eq


class Point(ak.Record):
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class PointArray(ak.Array):
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


ak.behavior["point"] = Point
ak.behavior[".", "point"] = PointArray
ak.behavior["*", "point"] = PointArray

one = ak.Array(
    [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        [{"x": 6, "y": 6.6}],
        [{"x": 7, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}],
    ],
)

two = ak.Array(
    [
        [{"x": 0.9, "y": 1}, {"x": 2, "y": 2.2}, {"x": 2.9, "y": 3}],
        [],
        [{"x": 3.9, "y": 4}, {"x": 5, "y": 5.5}],
        [{"x": 5.9, "y": 6}],
        [{"x": 6.9, "y": 7}, {"x": 8, "y": 8.8}, {"x": 8.9, "y": 9}],
    ],
)


def test_distance_behavior() -> None:
    onedak = dak.with_name(dak.from_awkward(one, npartitions=2), "point")
    twodak = dak.with_name(dak.from_awkward(two, npartitions=2), "point")

    assert_eq(
        onedak.distance(twodak),
        ak.Array(one, with_name="point").distance(ak.Array(two, with_name="point")),
    )


def test_nonexistent_behavior() -> None:
    onea = ak.Array(one, with_name="point")
    twoa = ak.Array(two)
    onedak = dak.from_awkward(onea, npartitions=2)
    twodak = dak.from_awkward(twoa, npartitions=2)

    with pytest.raises(
        AttributeError,
        match="Method doesnotexist is not available to this collection",
    ):
        onedak._call_behavior("doesnotexist", twodak)

    # in this case the field check is where we raise
    with pytest.raises(AttributeError, match="distance not in fields"):
        twodak.distance(onedak)
