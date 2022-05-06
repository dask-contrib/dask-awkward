from __future__ import annotations

import awkward._v2 as ak
import numpy as np
import pytest
from awkward._v2.behaviors.mixins import mixin_class as ak_mixin_class
from awkward._v2.behaviors.mixins import mixin_class_method as ak_mixin_class_method

import dask_awkward as dak
from dask_awkward.testutils import assert_eq

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


behaviors = {}


@ak_mixin_class(behaviors)
class Point:
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @property
    def x2(self):
        return self.x * self.x

    @ak_mixin_class_method(np.abs)
    def point_abs(self):
        return np.sqrt(self.x**2 + self.y**2)


def test_distance_behavior() -> None:
    onedak = dak.with_name(
        dak.from_awkward(one, npartitions=2),
        name="Point",
        behavior=behaviors,
    )
    twodak = dak.with_name(
        dak.from_awkward(two, npartitions=2),
        name="Point",
        behavior=behaviors,
    )

    onec = ak.Array(one, with_name="Point", behavior=behaviors)
    twoc = ak.Array(two)

    assert_eq(onedak.distance(twodak), onec.distance(twoc))
    assert_eq(np.abs(onedak), np.abs(onec))


def test_property_behavior() -> None:
    onedak = dak.with_name(
        dak.from_awkward(one, npartitions=2),
        name="Point",
        behavior=behaviors,
    )
    onec = ak.Array(one, with_name="Point", behavior=behaviors)
    assert_eq(onedak.x2, onec.x2)


def test_nonexistent_behavior() -> None:
    onec = ak.Array(one, with_name="Point")
    twoc = ak.Array(two)
    onedak = dak.from_awkward(onec, npartitions=2)
    twodak = dak.from_awkward(twoc, npartitions=2)

    with pytest.raises(
        AttributeError,
        match="Method doesnotexist is not available to this collection",
    ):
        onedak._call_behavior_method("doesnotexist", twodak)

    with pytest.raises(
        AttributeError,
        match="Property doesnotexist is not available to this collection",
    ):
        onedak._call_behavior_property("doesnotexist")

    # in this case the field check is where we raise
    with pytest.raises(AttributeError, match="distance not in fields"):
        twodak.distance(onedak)
