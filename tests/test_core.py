from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Any

import awkward as ak
import fsspec
import numpy as np
import pytest

try:
    import ujson as json
except ImportError:
    import json  # type: ignore[no-redef]

import sys

import dask_awkward as dak
from dask_awkward.lib.core import (
    Record,
    Scalar,
    calculate_known_divisions,
    compatible_partitions,
    compute_typetracer,
    is_typetracer,
    meta_or_identity,
    new_array_object,
    new_known_scalar,
    new_record_object,
    new_scalar_object,
    normalize_single_outer_inner_index,
    to_meta,
    typetracer_array,
)
from dask_awkward.lib.testutils import assert_eq

if TYPE_CHECKING:
    from dask_awkward.lib.core import Array


def test_clear_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    assert daa.known_divisions
    daa.clear_divisions()
    assert len(daa.divisions) == daa.npartitions + 1
    assert not daa.known_divisions


def test_dunder_str(daa: Array) -> None:
    assert str(daa) == "dask.awkward<from-json, npartitions=3>"


def test_calculate_known_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    target = (0, 5, 10, 15)
    assert calculate_known_divisions(daa) == target
    assert calculate_known_divisions(daa.points) == target
    assert calculate_known_divisions(daa.points.x) == target
    assert calculate_known_divisions(daa["points"]["y"]) == target
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    assert daa.known_divisions
    assert calculate_known_divisions(daa) == target


def test_fields(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    # records fields same as array of records fields
    assert daa[0].points.fields == daa.points.fields
    aa = daa.compute()
    assert daa.fields == aa.fields
    daa.reset_meta()
    assert daa.fields == []


def test_form(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    assert daa.form
    daa.reset_meta()

    from awkward.forms.emptyform import EmptyForm

    assert daa.form == EmptyForm()


@pytest.mark.parametrize("nparts", [2, 3, 4])
def test_from_awkward(caa: ak.Array, nparts: int) -> None:
    daa = dak.from_awkward(caa, npartitions=nparts)
    assert_eq(caa, daa, check_forms=False)
    assert_eq(daa, daa, check_forms=False)


def test_compute_typetracer(daa: Array) -> None:
    tt = compute_typetracer(daa.dask, daa.name)
    daa2 = new_array_object(daa.dask, daa.name, meta=tt, divisions=daa.divisions)
    assert_eq(daa, daa2)


def test_len(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert len(daa) == 10


def test_meta_exists(daa: Array) -> None:
    assert daa._meta is not None
    assert daa["points"]._meta is not None


def test_meta_raise(ndjson_points_file: str) -> None:
    with pytest.raises(
        TypeError, match="meta must be an instance of an Awkward Array."
    ):
        dak.from_json([ndjson_points_file], meta=5)


def test_ndim(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert daa.ndim == daa.compute().ndim


def test_new_array_object_raises(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    name = daa.name
    hlg = daa.dask
    with pytest.raises(
        ValueError, match="One of either divisions or npartitions must be defined."
    ):
        new_array_object(hlg, name, meta=None, npartitions=None, divisions=None)
    with pytest.raises(
        ValueError, match="Only one of either divisions or npartitions can be defined."
    ):
        new_array_object(hlg, name, meta=None, npartitions=3, divisions=(0, 2, 4, 7))


def test_partitions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 4)
    for i in range(daa.npartitions):
        part = daa.partitions[i]
        assert part.npartitions == 1


def test_partitions_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    divs = daa.divisions
    t1 = daa.partitions[1:3]
    assert not t1.known_divisions
    t2 = daa.partitions[1]
    assert t2.known_divisions
    assert t2.divisions == (0, divs[2] - divs[1])


def test_array_rebuild(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    x = daa.compute()
    daa = daa._rebuild(daa.dask)
    y = daa.compute()
    assert x.tolist() == y.tolist()

    with pytest.raises(ValueError, match="rename= unsupported"):
        daa._rebuild(daa.dask, rename={"x": "y"})


def test_type(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert dak.type(daa) is not None
    daa._meta = None
    assert dak.type(daa) is None


def test_short_typestr(daa: Array) -> None:
    ts = daa._shorttypestr(max=12)
    assert len(ts) == 12


def test_typestr(daa: Array) -> None:
    aa = daa.compute()
    assert str(aa.layout.form.type) in daa._typestr()
    extras = len("var *  ... }")
    assert len(daa._typestr(max=20)) == 20 + extras


def test_record_collection(daa: Array) -> None:
    assert type(daa[0]) is Record
    aa = daa.compute()
    assert_eq(daa[0], aa[0])
    # assert daa[0].compute().tolist() == aa[0].tolist()


def test_scalar_collection(daa: Array) -> None:
    assert type(daa["points", "x"][0][0]) is Scalar


def test_scalar_getitem_getattr() -> None:
    d = {"a": 5}
    s = new_known_scalar(d)
    assert s["a"].compute() == d["a"]
    Thing = namedtuple("Thing", "a b c")
    t = Thing(c=3, b=2, a=1)
    s = new_known_scalar(t)
    assert s.c.compute() == t.c


def test_is_typetracer(daa: Array) -> None:
    assert not is_typetracer(daa)
    assert not is_typetracer(daa[0])
    assert not is_typetracer(daa["points"])
    assert not is_typetracer(daa.compute())
    assert is_typetracer(daa._meta)
    assert is_typetracer(daa[0]._meta)
    assert is_typetracer(daa["points"]._meta)
    assert is_typetracer(daa["points"][0]["x"][0]._meta)


def test_meta_or_identity(daa: Array) -> None:
    assert is_typetracer(meta_or_identity(daa))
    assert meta_or_identity(daa) is daa._meta
    assert meta_or_identity(5) == 5


def test_to_meta(daa: Array) -> None:
    x1 = daa["points"]["x"]
    x1_0 = x1[0]
    metad = to_meta([x1, 5, "ok", x1_0])
    assert isinstance(metad, tuple)
    for a, b in zip(metad, (x1._meta, 5, "ok", x1_0._meta)):
        if is_typetracer(a):
            assert a is b
        else:
            assert a == b


def test_record_str(daa: Array) -> None:
    r = daa[0]
    assert type(r) == dak.Record
    assert str(r) == "dask.awkward<getitem, type=Record>"


def test_record_to_delayed(daa: Array) -> None:
    r = daa[0]
    assert type(r) == dak.Record
    d = r.to_delayed()
    assert r.compute().tolist() == d.compute().tolist()


def test_record_fields(daa: Array) -> None:
    r = daa[0]
    assert type(r) == dak.Record
    r._meta = None
    with pytest.raises(TypeError, match="metadata is missing"):
        assert not r.fields


def test_record_dir(daa: Array) -> None:
    r = daa["points"][0][0]
    assert type(r) == dak.Record
    d = dir(r)
    for f in r.fields:
        assert f in d


# @pytest.mark.xfail(reason="ak.Record typetracer fails to pickle")
# def test_record_pickle(daa: Array) -> None:
#     import pickle

#     r = daa[0]
#     assert type(r) == dak.Record
#     assert isinstance(r._meta, ak.Record)

#     dumped = pickle.dumps(r)
#     new = pickle.loads(dumped)
#     assert_eq(dumped, new)


def test_array_dir(daa: Array) -> None:
    a = daa["points"]
    d = dir(a)
    for f in a.fields:
        assert f in d


def test_typetracer_function(daa: Array) -> None:
    aa = daa.compute()
    assert typetracer_array(daa) is not None
    assert typetracer_array(daa) is daa._meta
    tta = typetracer_array(aa)
    assert tta is not None
    assert tta.layout.form == aa.layout.form
    with pytest.raises(TypeError, match="Got type <class 'int'>"):
        typetracer_array(3)


def test_single_partition(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    with fsspec.open(ndjson_points_file, "r") as f:
        caa = ak.from_iter([json.loads(line) for line in f])

    assert daa.npartitions == 1
    assert_eq(daa, caa)
    assert_eq(caa, daa)


def test_new_known_scalar() -> None:
    s1 = 5
    c = new_known_scalar(s1)
    assert c.compute() == s1
    assert c._meta is not None
    s2 = 5.5
    c = new_known_scalar(s2)
    assert c.compute() == 5.5
    assert c._meta is not None

    c = dak.Scalar.from_known(s1)
    assert c.known_value == s1
    assert c.compute() == s1


def test_scalar_dtype() -> None:
    s = 2
    c = new_known_scalar(s)
    assert c.dtype == np.dtype(type(s))
    c._meta = None
    assert c.dtype is None


# def test_scalar_pickle(daa: Array) -> None:
#    import pickle
#
#    s = 2
#    c1 = new_known_scalar(s)
#    s_dumped = pickle.dumps(c1)
#    c2 = pickle.loads(s_dumped)
#    assert_eq(c1, c2)
#    s1 = dak.sum(daa["points"]["y"], axis=None)
#    s_dumped = pickle.dumps(s1)
#    s2 = pickle.loads(s_dumped)
#    assert_eq(s1.compute(), s2.compute())
#
#    assert s1.known_value is None


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_scalar_to_delayed(daa: Array, optimize_graph: bool) -> None:
    s1 = dak.sum(daa["points", "x"], axis=None)
    d1 = s1.to_delayed(optimize_graph=optimize_graph)
    s1c = s1.compute()
    assert d1.compute() == s1c


def test_compatible_partitions(ndjson_points_file: str) -> None:
    daa1 = dak.from_json([ndjson_points_file] * 5)
    daa2 = dak.from_awkward(daa1.compute(), npartitions=4)
    assert compatible_partitions(daa1, daa1)
    assert compatible_partitions(daa1, daa1, daa1)
    assert not compatible_partitions(daa1, daa2)
    daa1.eager_compute_divisions()
    assert compatible_partitions(daa1, daa1)
    x = ak.Array([[1, 2, 3], [1, 2, 3], [3, 4, 5]])
    y = ak.Array([[1, 2, 3], [3, 4, 5]])
    x = dak.from_awkward(x, npartitions=2)
    y = dak.from_awkward(y, npartitions=2)
    assert not compatible_partitions(x, y)
    assert not compatible_partitions(x, x, y)
    assert compatible_partitions(y, y)


@pytest.mark.parametrize("meta", [5, False, [1, 2, 3]])
def test_bad_meta_type(ndjson_points_file: str, meta: Any) -> None:
    with pytest.raises(TypeError, match="meta must be an instance of an Awkward Array"):
        dak.from_json([ndjson_points_file] * 3, meta=meta)


def test_bad_meta_backend_array(daa):
    with pytest.raises(TypeError, match="meta Array must have a typetracer backend"):
        daa.points.x.map_partitions(lambda x: x**2, meta=ak.Array([]))


def test_bad_meta_backend_record(daa):
    with pytest.raises(TypeError, match="meta Record must have a typetracer backend"):
        a = daa.points[0]
        new_record_object(a.dask, a.name, meta=ak.Record({"x": 1}))


def test_bad_meta_backend_scalar(daa):
    with pytest.raises(TypeError, match="meta Scalar must have a typetracer backend"):
        a = daa.points.x[0][0]
        new_scalar_object(a.dask, a.name, meta=5)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="skip if windows")
def test_scalar_repr(daa: Array) -> None:
    s = dak.max(daa.points.y)
    sstr = str(s)
    assert "type=Scalar" in sstr
    s = new_known_scalar(5)
    sstr = str(s)
    assert (
        sstr == r"dask.awkward<known-scalar, type=Scalar, dtype=int64, known_value=5>"
    )


def test_scalar_divisions(daa: Array) -> None:
    s = dak.max(daa.points.x, axis=None)
    assert s.divisions == (None, None)


def test_array_persist(daa: Array) -> None:
    daa2 = daa["points"]["x"].persist()
    assert_eq(daa["points"]["x"], daa2)
    daa2 = daa.persist()
    assert_eq(daa2, daa)


def test_scalar_persist_and_rebuild(daa: Array) -> None:
    coll = daa["points"][0]["x"][0]
    coll2 = coll.persist()
    assert_eq(coll, coll2)

    m = dak.max(daa.points.x, axis=None)
    with pytest.raises(ValueError, match="rename= unsupported"):
        m._rebuild(m.dask, rename={m._name: "max2"})


def test_output_divisions(daa: Array) -> None:
    assert dak.max(daa.points.y, axis=1).divisions == daa.divisions
    assert dak.num(daa.points.y, axis=1).divisions == (None,) * (daa.npartitions + 1)
    assert daa["points"][["x", "y"]].divisions == daa.divisions
    assert daa["points"].divisions == daa.divisions


def test_record_npartitions(daa: Array) -> None:
    analysis0 = daa[0]
    assert analysis0.npartitions == 1


def test_iter(daa: Array) -> None:
    with pytest.raises(
        NotImplementedError,
        match="Iteration over a Dask Awkward collection is not supported",
    ):
        for a in daa:
            pass


def test_normalize_single_outer_inner_index() -> None:
    divisions = (0, 12, 14, 20, 23, 24)
    indices = [0, 1, 2, 8, 12, 13, 14, 15, 17, 20, 21, 22]
    results = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 8),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    for i, r in zip(indices, results):
        res = normalize_single_outer_inner_index(divisions, i)
        assert r == res

    divisions = (0, 12)  # type: ignore
    indices = [0, 2, 3, 6, 8, 11]
    results = [
        (0, 0),
        (0, 2),
        (0, 3),
        (0, 6),
        (0, 8),
        (0, 11),
    ]
    for i, r in zip(indices, results):
        res = normalize_single_outer_inner_index(divisions, i)
        assert r == res
