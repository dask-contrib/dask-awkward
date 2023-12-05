from __future__ import annotations

from typing import no_type_check

import awkward as ak
import numpy as np
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import BAD_NP_AK_MIXIN_VERSIONING, assert_eq

behaviors: dict = {}


@ak.mixin_class(behaviors)
class Point:
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @property
    def x2(self):
        return self.x * self.x

    @ak.mixin_class_method(np.abs)
    def point_abs(self):
        return np.sqrt(self.x**2 + self.y**2)

    @dak.dask_property
    def some_property(self):
        return "this is a non-dask property"

    @some_property.dask
    def some_property_dask(self, array):
        return f"this is a dask property ({type(array).__name__})"

    @dak.dask_property(no_dispatch=True)
    def some_property_both(self):
        return "this is a dask AND non-dask property"

    @dak.dask_method
    def some_method(self):
        return None

    @no_type_check
    @some_method.dask
    def some_method_dask(self, array):
        return array

    @dak.dask_method(no_dispatch=True)
    def some_method_both(self):
        return "NO DISPATCH!"


@pytest.mark.xfail(
    BAD_NP_AK_MIXIN_VERSIONING,
    reason="NumPy 1.25 mixin __slots__ change",
)
def test_distance_behavior(
    daa_p1: dak.Array,
    daa_p2: dak.Array,
    caa_p1: ak.Array,
    caa_p2: ak.Array,
) -> None:
    daa1 = dak.with_name(daa_p1.points, name="Point", behavior=behaviors)
    daa2 = dak.with_name(daa_p2.points, name="Point", behavior=behaviors)
    caa1 = ak.Array(caa_p1.points, with_name="Point", behavior=behaviors)
    caa2 = ak.Array(caa_p2.points)
    assert_eq(daa1.distance(daa2), caa1.distance(caa2))
    assert_eq(np.abs(daa1), np.abs(caa1))


@pytest.mark.xfail(
    BAD_NP_AK_MIXIN_VERSIONING,
    reason="NumPy 1.25 mixin __slots__ change",
)
def test_property_method_behavior(daa_p1: dak.Array, caa_p1: ak.Array) -> None:
    daa = dak.with_name(daa_p1.points, name="Point", behavior=behaviors)
    caa = ak.Array(caa_p1.points, with_name="Point", behavior=behaviors)
    assert_eq(daa.x2, caa.x2)

    assert daa.behavior == caa.behavior

    assert caa.some_property == "this is a non-dask property"
    assert daa.some_property == "this is a dask property (Array)"

    assert repr(daa.some_method()) == repr(daa)
    assert repr(caa.some_method()) == repr(None)

    assert (
        daa.some_property_both
        == caa.some_property_both
        == "this is a dask AND non-dask property"
    )
    assert daa.some_method_both() == caa.some_method_both() == "NO DISPATCH!"


@pytest.mark.xfail(
    BAD_NP_AK_MIXIN_VERSIONING,
    reason="NumPy 1.25 mixin __slots__ change",
)
def test_nonexistent_behavior(daa_p1: dak.Array, daa_p2: dak.Array) -> None:
    daa1 = dak.with_name(daa_p1["points"], "Point", behavior=behaviors)
    daa2 = daa_p2

    # in this case the field check is where we raise
    with pytest.raises(AttributeError, match="distance not in fields"):
        daa2.distance(daa1)
