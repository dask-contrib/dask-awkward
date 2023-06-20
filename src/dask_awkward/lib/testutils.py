from __future__ import annotations

import random
from typing import Any

import awkward as ak
import numpy as np
from dask.base import is_dask_collection
from packaging.version import Version

from dask_awkward.lib.core import Array, Record, typetracer_array
from dask_awkward.lib.io.io import from_lists

_RG = random.Random(414)


DEFAULT_SCHEDULER: Any = "sync"


NP_LTE_1_25_0 = Version(np.__version__) >= Version("1.25.0")
AK_GTE_2_2_3 = Version(ak.__version__) <= Version("2.2.3")
BAD_NP_AK_MIXIN_VERSIONING = NP_LTE_1_25_0 and AK_GTE_2_2_3


def assert_eq(
    a: Any,
    b: Any,
    check_forms: bool = False,
    check_divisions: bool = True,
    scheduler: Any | None = None,
    **kwargs: Any,
) -> None:
    scheduler = scheduler or DEFAULT_SCHEDULER
    if isinstance(a, (Array, ak.Array)):
        assert_eq_arrays(
            a,
            b,
            check_forms=check_forms,
            check_divisions=check_divisions,
            scheduler=scheduler,
            **kwargs,
        )
    elif isinstance(a, (Record, ak.Record)):  # type: ignore
        assert_eq_records(a, b, scheduler=scheduler, **kwargs)
    else:
        assert_eq_other(a, b, scheduler=scheduler, **kwargs)


def assert_eq_arrays(
    a: Array | ak.Array,
    b: Array | ak.Array,
    isclose_equal_nan: bool = False,
    check_forms: bool = False,
    check_divisions: bool = True,
    scheduler: Any | None = None,
) -> None:
    scheduler = scheduler or DEFAULT_SCHEDULER
    a_is_coll = is_dask_collection(a)
    b_is_coll = is_dask_collection(b)
    a_comp = a.compute(scheduler=scheduler) if a_is_coll else a
    b_comp = b.compute(scheduler=scheduler) if b_is_coll else b

    a_tt = typetracer_array(a)
    b_tt = typetracer_array(b)
    assert a_tt is not None
    assert b_tt is not None

    if check_forms:
        a_form = ak.concatenate([a.layout, a.layout[0:0]], highlevel=False).form
        b_form = ak.concatenate([b.layout, b.layout[0:0]], highlevel=False).form
        # a_form = a.layout.form
        # b_form = b.layout.form
        a_comp_form = a_comp.layout.form
        b_comp_form = b_comp.layout.form
        assert a_comp_form == a_form
        assert a_comp_form == b_form
        assert b_comp_form == a_comp_form

    if check_divisions:
        # check divisions if both collections
        if a_is_coll and b_is_coll:
            if a.known_divisions and b.known_divisions:
                assert a.divisions == b.divisions
            else:
                assert a.npartitions == b.npartitions

    # finally check the values
    if isclose_equal_nan:
        assert ak.all(
            ak.isclose(
                ak.from_iter(a_comp.tolist()),
                ak.from_iter(b_comp.tolist()),
                equal_nan=True,
            )
        )
    else:
        assert a_comp.tolist() == b_comp.tolist()


def assert_eq_records(
    a: Record | ak.Record,
    b: Record | ak.Record,
    scheduler: Any | None = None,
) -> None:
    scheduler = scheduler or DEFAULT_SCHEDULER
    ares = a.compute(scheduler=scheduler) if is_dask_collection(a) else a
    bres = b.compute(scheduler=scheduler) if is_dask_collection(b) else b

    assert ares.tolist() == bres.tolist()


def assert_eq_other(
    a: Any,
    b: Any,
    scheduler: Any | None = None,
) -> None:
    scheduler = scheduler or DEFAULT_SCHEDULER
    ares = a.compute(scheduler=scheduler) if is_dask_collection(a) else a
    bres = b.compute(scheduler=scheduler) if is_dask_collection(b) else b
    assert ares == bres


def make_xy_point() -> dict[str, int]:
    return {"x": _RG.randint(0, 10), "y": _RG.randint(0, 10)}


def make_xy_point_str() -> dict[str, str]:
    return {"x": str(_RG.randint(0, 10)), "y": str(_RG.randint(0, 10))}


def list_of_xy_points(n: int) -> list[dict[str, int]]:
    return [make_xy_point() for _ in range(n)]


def list_of_xy_points_str(n: int) -> list[dict[str, str]]:
    return [make_xy_point_str() for _ in range(n)]


def awkward_xy_points(lengths: tuple[int, ...] | None = None) -> ak.Array:
    if lengths is None:
        lengths = (3, 0, 2, 1, 3)
    return ak.Array([list_of_xy_points(n) for n in lengths])


def awkward_xy_points_str(lengths: tuple[int, ...] | None = None) -> ak.Array:
    if lengths is None:
        lengths = (3, 0, 2, 1, 3)
    return ak.Array([list_of_xy_points_str(n) for n in lengths])


def list1() -> list:
    return [
        [{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4.0, "y": 4.4}, {"x": 5.0, "y": 5.5}],
        [{"x": 6.0, "y": 6.6}],
        [{"x": 7.0, "y": 7.7}, {"x": 8.0, "y": 8.8}, {"x": 9, "y": 9.9}],
    ]  # pragma: no cover


def list2() -> list:
    return [
        [{"x": 0.9, "y": 1.0}, {"x": 2.0, "y": 2.2}, {"x": 2.9, "y": 3.0}],
        [],
        [{"x": 3.9, "y": 4.0}, {"x": 5.0, "y": 5.5}],
        [{"x": 5.9, "y": 6.0}],
        [{"x": 6.9, "y": 7.0}, {"x": 8.0, "y": 8.8}, {"x": 8.9, "y": 9.0}],
    ]  # pragma: no cover


def list3() -> list:
    return [
        [{"x": 1.9, "y": 9.0}, {"x": 2.0, "y": 8.2}, {"x": 9.9, "y": 9.0}],
        [],
        [{"x": 1.9, "y": 8.0}, {"x": 4.0, "y": 6.5}],
        [{"x": 1.9, "y": 7.0}],
        [{"x": 1.9, "y": 6.0}, {"x": 6.0, "y": 4.8}, {"x": 9.9, "y": 9.0}],
    ]  # pragma: no cover


def lists() -> Array:
    return from_lists([list1(), list2(), list3()])  # pragma: no cover


def unnamed_root_ds() -> Array:
    ds = [
        [
            {
                "minutes": 33,
                "passes": {"to": [2, 5, 6], "success": [True, True, False]},
                "assists": [
                    {"distance": 3.3, "scorer": 6},
                    {"distance": 4.4, "scorer": 6},
                ],
            },
            {
                "minutes": 34,
                "passes": {"to": [5, 6, 7, 8], "success": [False, False, True, True]},
                "assists": [],
            },
        ],
        [
            {
                "minutes": 24,
                "passes": {"to": [0, 3, 4, 5], "success": [True, True, True, False]},
                "assists": [
                    {"distance": 10.3, "scorer": 0},
                    {"distance": 14.0, "scorer": 0},
                ],
            },
            {
                "minutes": 3,
                "passes": {"to": [], "success": []},
                "assists": [
                    {"distance": 3.3, "scorer": 0},
                    {"distance": 2.3, "scorer": 4},
                    {"distance": 1.0, "scorer": 5},
                ],
            },
            {
                "minutes": 18,
                "passes": {"to": [0, 3, 4, 5], "success": [False, True, True, False]},
                "assists": [{"distance": 4.3, "scorer": 0}],
            },
        ],
    ]
    return ak.Array(ds * 3)
