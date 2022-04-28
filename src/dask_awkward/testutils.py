from __future__ import annotations

from typing import Any

import awkward._v2 as ak
from dask.base import is_dask_collection

from dask_awkward.core import Array, Record, typetracer_array


def assert_eq(
    a: Any,
    b: Any,
    check_forms: bool = True,
    check_divisions: bool = True,
) -> None:
    if isinstance(a, (Array, ak.Array)):
        assert_eq_arrays(
            a,
            b,
            check_forms=check_forms,
            check_divisions=check_divisions,
        )
    elif isinstance(a, (Record, ak.Record)):
        assert_eq_records(a, b)
    else:
        assert_eq_other(a, b)


def idempotent_concatenate(x: ak.Array) -> ak.Array:
    return ak.concatenate([x, x[0:0]])


def assert_eq_arrays(
    a: Any,
    b: Any,
    check_forms: bool = True,
    check_divisions: bool = True,
) -> None:
    a_is_coll = is_dask_collection(a)
    b_is_coll = is_dask_collection(b)
    a_comp = a.compute() if a_is_coll else a
    b_comp = b.compute() if b_is_coll else b

    if check_forms:
        a_tt = typetracer_array(a)
        b_tt = typetracer_array(b)
        assert a_tt is not None
        assert b_tt is not None

        # that that the idempotent concatation of the typetracers for
        # both a and b yield the same form.
        a_concated_form = idempotent_concatenate(a_tt).layout.form
        b_concated_form = idempotent_concatenate(b_tt).layout.form
        assert a_concated_form == b_concated_form

        # if a is a collection with multiple partitions its computed
        # from should be the same the concated version
        if a_is_coll and a.npartitions > 1:
            assert a_comp.layout.form == a_concated_form

        # if a is a collection with a _single_ partition then we don't
        # have to use the concated typetracer.
        elif a_is_coll and a.npartitions == 1:
            assert a_comp.layout.form == a_tt.layout.form

        # if b is a collection with multiple partitions its computed
        # from should be the same the concated version
        if b_is_coll and b.npartitions > 1:
            assert b_comp.layout.form == b_concated_form

        # if b is a collection with a _single_ partition then we don't
        # have to use the concated typetracer.
        elif b_is_coll and b.npartitions == 1:
            assert b_comp.layout.form == b_tt.layout.form

        # check the unconcatenated versions as well; a single
        # partition does not have the concatenation effect.
        if a_is_coll and not b_is_coll:
            assert b_tt.layout.form == a.partitions[0].compute().layout.form
        if not a_is_coll and b_is_coll:
            assert a_tt.layout.form == b.partitions[0].compute().layout.form

    if check_divisions:
        # check divisions if both collections
        if a_is_coll and b_is_coll:
            if a.known_divisions and b.known_divisions:
                assert a.divisions == b.divisions
            else:
                assert a.npartitions == b.npartitions

    # finally check the values
    assert a_comp.tolist() == b_comp.tolist()


def assert_eq_records(a: Record | ak.Record, b: Record | ak.Record) -> None:
    ares = a.compute() if is_dask_collection(a) else a
    bres = b.compute() if is_dask_collection(b) else b
    assert ares.tolist() == bres.tolist()


def assert_eq_other(a: Any, b: Any) -> None:
    ares = a.compute() if is_dask_collection(a) else a
    bres = b.compute() if is_dask_collection(b) else b
    assert ares == bres
