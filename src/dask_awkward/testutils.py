from typing import Any

import awkward._v2 as ak
from dask.base import is_dask_collection

from .core import Array, Record, typetracer_array


def idempotent_concatenate(x: ak.Array) -> ak.Array:
    return ak.concatenate([x, x[0:0]])


def aeq(a, b):
    a_is_coll = is_dask_collection(a)
    b_is_coll = is_dask_collection(b)
    a_comp = a.compute() if a_is_coll else a
    b_comp = b.compute() if b_is_coll else b
    a_tt = typetracer_array(a)
    b_tt = typetracer_array(b)

    assert a_tt is not None
    assert b_tt is not None

    # checking forms
    a_concated_form = idempotent_concatenate(a_tt).layout.form
    b_concated_form = idempotent_concatenate(b_tt).layout.form
    assert a_concated_form == b_concated_form
    # if a is a collection its computed from should be the same the
    # concated version
    if a_is_coll:
        assert a_comp.layout.form == a_concated_form
    # if b is a collection its computed from should be the same the
    # concated version
    if b_is_coll:
        assert b_comp.layout.form == b_concated_form

    # check the unconcatenated versions as well
    if a_is_coll and not b_is_coll:
        assert b_tt.layout.form == a.partitions[0].compute().layout.form
    if not a_is_coll and b_is_coll:
        assert a_tt.layout.form == b.partitions[0].compute().layout.form

    # check divisions if both collections
    if a_is_coll and b_is_coll:
        if a.known_divisions and b.known_divisions:
            assert a.divisions == b.divisions
        else:
            assert a.npartitions == b.npartitions

    # finally check the values
    assert a_comp.tolist() == b_comp.tolist()


def assert_eq(a: Any, b: Any) -> None:
    if isinstance(a, (Array, ak.Array)):
        aeq(a, b)
    elif isinstance(a, (Record, ak.Record)):
        aeq(a, b)
    else:
        assert_eq_other(a, b)


def assert_eq_arrays(a: Array | ak.Array, b: Array | ak.Array) -> None:
    ares = a.compute() if is_dask_collection(a) else a
    bres = b.compute() if is_dask_collection(b) else b
    assert ares.tolist() == bres.tolist()


def assert_eq_records(a: Record | ak.Record, b: Record | ak.Record) -> None:
    ares = a.compute() if is_dask_collection(a) else a
    bres = b.compute() if is_dask_collection(b) else b
    assert ares.tolist() == bres.tolist()


def assert_eq_other(a: Any, b: Any) -> None:
    if is_dask_collection(a) and not is_dask_collection(b):
        assert a.compute() == b
    if is_dask_collection(b) and not is_dask_collection(a):
        assert a == b.compute()
    if not is_dask_collection(a) and not is_dask_collection(b):
        assert a == b
