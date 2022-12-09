from __future__ import annotations

from typing import Any

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


def test_zip_bad_input(daa: dak.Array) -> None:
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


@pytest.mark.parametrize("vf", [9, 99.9])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_fill_none(vf: int | float | str, axis: int | None) -> None:
    a = [[1, 2, None], [], [None], [5, 6, 7, None], [1, 2], None]
    b = [[None, 2, 1], [None], [], None, [7, 6, None, 5], [None, None]]
    c = dak.from_lists([a, b])
    d = dak.fill_none(c, vf, axis=axis)
    e = ak.fill_none(ak.from_iter(a + b), vf, axis=axis)
    assert_eq(d, e, check_forms=(not isinstance(vf, str)))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_is_none(axis: int) -> None:
    a: list[Any] = [[1, 2, None], None, None, [], [None], [5, 6, 7, None], [1, 2], None]
    b: list[Any] = [[None, 2, 1], [None], [], None, [7, 6, None, 5], [None, None]]
    c = dak.from_lists([a, b])
    d = dak.is_none(c, axis=axis)
    e = ak.is_none(ak.from_iter(a + b), axis=axis)
    assert_eq(d, e)


@pytest.mark.parametrize("axis", [1, -1, 2, -2])
@pytest.mark.parametrize("target", [5, 10, 1])
def test_pad_none(axis: int, target: int) -> None:
    a = [[1, 2, 3], [4], None]
    b = [[7], [], None, [6, 7, 8]]
    c = dak.from_lists([[a, b], [b, a]])
    d = ak.from_iter([a, b] + [b, a])
    assert_eq(
        dak.pad_none(c, target=target, axis=axis),
        ak.pad_none(d, target=target, axis=axis),
    )


def test_with_parameter() -> None:
    a = [[1, 2, 3], [], [4]]
    b = [[], [3], []]
    c = dak.from_lists([a, b])
    d = dak.with_parameter(c, "something", {})
    x = ak.from_iter(a + b)
    y = ak.with_parameter(x, "something", {})
    assert_eq(d, y)

    assert d.compute().layout.parameters == y.layout.parameters
    assert d._meta.layout.parameters == y.layout.parameters

    d2 = dak.without_parameters(d)
    y2 = ak.without_parameters(y)
    assert_eq(d2, y2)
    assert not d2.compute().layout.parameters
    assert d2.compute().layout.parameters == y2.layout.parameters
    assert d2._meta.layout.parameters == y2.layout.parameters


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("fields", [None, ["a", "b"]])
def test_combinations(caa, daa, axis, fields):
    assert_eq(
        dak.combinations(daa, 2, axis=axis),
        ak.combinations(caa, 2, axis=axis),
    )


def test_combinations_raise(daa):
    with pytest.raises(ValueError, match="if provided, the length"):
        dak.combinations(daa, 2, fields=["a", "b", "c"])
