from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq
from dask_awkward.utils import DaskAwkwardNotImplemented


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_flatten(caa: ak.Array, daa: dak.Array, axis: int | None) -> None:
    cr = ak.flatten(caa.points.x, axis=axis)
    dr = dak.flatten(daa.points.x, axis=axis)
    assert_eq(cr, dr)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_num(caa: ak.Array, daa: dak.Array, axis: int | None) -> None:
    da = daa["points"]
    ca = caa["points"]

    if axis == 0:
        assert_eq(dak.num(da.x, axis=axis), ak.num(ca.x, axis=axis))
        da.eager_compute_divisions()

    assert_eq(dak.num(da.x, axis=axis), ak.num(ca.x, axis=axis))

    if axis == 1:
        c1 = dak.num(da.x, axis=axis) > 2
        c2 = ak.num(ca.x, axis=axis) > 2
        assert_eq(da[c1], ca[c2])


def test_zip_dict_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa["points"]["x"]
    da2 = daa["points"]["x"]
    ca1 = caa["points"]["x"]
    ca2 = caa["points"]["x"]

    da_z = dak.zip({"a": da1, "b": da2})
    ca_z = ak.zip({"a": ca1, "b": ca2})
    assert_eq(da_z, ca_z)


def test_zip_list_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa.points.x
    ca1 = caa.points.x
    dz1 = dak.zip([da1, da1])
    cz1 = ak.zip([ca1, ca1])
    assert_eq(dz1, cz1)
    dz2 = dak.zip([da1, da1, da1])
    cz2 = ak.zip([ca1, ca1, ca1])
    assert_eq(dz2, cz2)


def test_zip_tuple_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa.points.x
    ca1 = caa.points.x
    dz1 = dak.zip((da1, da1))
    cz1 = ak.zip((ca1, ca1))
    assert_eq(dz1, cz1)
    dz2 = dak.zip((da1, da1, da1))
    cz2 = ak.zip((ca1, ca1, ca1))
    assert_eq(dz2, cz2)


def test_zip_bad_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa.points.x
    gd = (x for x in (da1, da1))
    with pytest.raises(DaskAwkwardNotImplemented, match="only sized iterables"):
        dak.zip(gd)


def test_cartesian(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa["points", "x"]
    da2 = daa["points", "y"]
    ca1 = caa["points", "x"]
    ca2 = caa["points", "y"]

    dz = dak.cartesian([da1, da2], axis=1)
    cz = ak.cartesian([ca1, ca2], axis=1)
    assert_eq(dz, cz)


def test_ones_like(caa: ak.Array, daa: dak.Array) -> None:
    da1 = dak.ones_like(daa.points.x)
    ca1 = ak.ones_like(caa["points", "x"])
    assert_eq(da1, ca1)


def test_zeros_like(caa: ak.Array, daa: dak.Array) -> None:
    da1 = dak.zeros_like(daa["points", "x"])
    ca1 = ak.zeros_like(caa.points.x)
    assert_eq(da1, ca1)
