from __future__ import annotations

import awkward._v2 as ak
import dask.config
import fsspec
import numpy as np
import pytest

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import dask_awkward as dak
import dask_awkward.core as dakc
from dask_awkward.testutils import assert_eq


def test_clear_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    assert daa.known_divisions
    daa.clear_divisions()
    assert not daa.known_divisions
    assert len(daa.divisions) == daa.npartitions + 1
    assert_eq(daa, daa)


def test_dunder_str(daa) -> None:
    assert str(daa) == "dask.awkward<from-json, npartitions=3>"


def test_calculate_known_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    target = (0, 5, 10, 15)
    assert dakc.calculate_known_divisions(daa) == target
    assert dakc.calculate_known_divisions(daa.points) == target
    assert dakc.calculate_known_divisions(daa.points.x) == target
    assert dakc.calculate_known_divisions(daa["points"]["y"]) == target
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    assert daa.known_divisions
    assert dakc.calculate_known_divisions(daa) == target


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

    from awkward._v2.forms.emptyform import EmptyForm

    assert daa.form == EmptyForm()


def test_from_awkward(caa: ak.Array) -> None:
    daa = dak.from_awkward(caa, npartitions=4)
    assert_eq(caa, daa)
    assert_eq(daa, daa)


def test_get_typetracer(daa) -> None:
    assert dakc._get_typetracer(daa) is daa._meta


def test_len(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert len(daa) == 10


def test_meta_exists(daa: dak.Array) -> None:
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
    daa._meta = None
    assert daa.ndim is None


def test_new_array_object_raises(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    name = daa.name
    hlg = daa.dask
    with pytest.raises(
        ValueError, match="One of either divisions or npartitions must be defined."
    ):
        dakc.new_array_object(hlg, name, meta=None, npartitions=None, divisions=None)
    with pytest.raises(
        ValueError, match="Only one of either divisions or npartitions must be defined."
    ):
        dakc.new_array_object(
            hlg, name, meta=None, npartitions=3, divisions=(0, 2, 4, 7)
        )


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
    assert t2.divisions == (0, divs[2] - divs[1])  # type: ignore


def test_raise_in_finalize(daa: dak.Array) -> None:
    with dask.config.set({"awkward.compute-unknown-meta": False}):
        res = daa.map_partitions(str)
    with pytest.raises(RuntimeError, match="type of first result: <class 'str'>"):
        res.compute()


def test_rebuild(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    x = daa.compute()
    daa = daa._rebuild(daa.dask)
    y = daa.compute()
    assert x.tolist() == y.tolist()


def test_type(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert dak.type(daa) is not None
    daa._meta = None
    assert dak.type(daa) is None


def test_short_typestr(daa: dak.Array) -> None:
    ts = daa._shorttypestr(max=12)
    assert len(ts) == 12


def test_typestr(daa: dak.Array) -> None:
    aa = daa.compute()
    assert str(aa.layout.form.type) in daa._typestr()
    extras = len("var *  ... }")
    assert len(daa._typestr(max=20)) == 20 + extras


def test_record_collection(daa: dak.Array) -> None:
    assert type(daa[0]) is dakc.Record
    aa = daa.compute()
    assert_eq(daa[0], aa[0])
    # assert daa[0].compute().tolist() == aa[0].tolist()


def test_scalar_collection(daa: dak.Array) -> None:
    assert type(daa["points", "x"][0][0]) is dakc.Scalar


def test_is_typetracer(daa: dak.Array) -> None:
    assert not dakc.is_typetracer(daa)
    assert not dakc.is_typetracer(daa[0])
    assert not dakc.is_typetracer(daa["points"])
    assert not dakc.is_typetracer(daa.compute())
    assert dakc.is_typetracer(daa._meta)
    assert dakc.is_typetracer(daa[0]._meta)
    assert dakc.is_typetracer(daa["points"]._meta)
    assert dakc.is_typetracer(daa["points"][0]["x"][0]._meta)


def test_meta_or_identity(daa: dak.Array) -> None:
    assert dakc.is_typetracer(dakc.meta_or_identity(daa))
    assert dakc.meta_or_identity(daa) is daa._meta
    assert dakc.meta_or_identity(5) == 5


def test_to_meta(daa: dak.Array) -> None:
    x1 = daa["points"]["x"]
    x1_0 = x1[0]
    metad = dakc.to_meta([x1, 5, "ok", x1_0])
    assert isinstance(metad, tuple)
    for a, b in zip(metad, (x1._meta, 5, "ok", x1_0._meta)):
        if dakc.is_typetracer(a):
            assert a is b
        else:
            assert a == b


def test_record_str(daa: dak.Array) -> None:
    r = daa[0]
    assert str(r) == "dask.awkward<getitem, type=Record>"


def test_record_to_delayed(daa: dak.Array) -> None:
    r = daa[0]
    d = r.to_delayed()
    assert r.compute().tolist() == d.compute().tolist()


def test_record_fields(daa: dak.Array) -> None:
    r = daa[0]
    r._meta = None
    assert r.fields is None


def test_record_dir(daa: dak.Array) -> None:
    r = daa["points"][0]
    d = dir(r)
    for f in r.fields:
        assert f in d


def test_array_dir(daa: dak.Array) -> None:
    a = daa["points"]
    d = dir(a)
    for f in a.fields:
        assert f in d


def test_typetracer_function(daa: dak.Array) -> None:
    aa = daa.compute()
    assert dakc.typetracer_array(daa) is not None
    assert dakc.typetracer_array(daa) is daa._meta
    tta = dakc.typetracer_array(aa)
    assert tta is not None
    assert tta.layout.form == aa.layout.form
    with pytest.raises(TypeError, match="Got type <class 'int'>"):
        dakc.typetracer_array(3)


def test_single_partition(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    with fsspec.open(ndjson_points_file, "r") as f:
        caa = ak.from_iter([json.loads(line) for line in f])

    assert daa.npartitions == 1
    assert_eq(daa, caa)
    assert_eq(caa, daa)


def test_new_known_scalar() -> None:
    s1 = 5
    c = dakc.new_known_scalar(s1)
    assert c.compute() == s1
    assert c._meta is not None
    s2 = 5.5
    c = dakc.new_known_scalar(s2)
    assert c.compute() == 5.5
    assert c._meta is not None

    c = dak.Scalar.from_known(s1)
    assert c.known_value == s1
    assert c.compute() == s1


def test_scalar_dtype() -> None:
    s = 2
    c = dakc.new_known_scalar(s)
    assert c.dtype == np.dtype(type(s))
    c._meta = None
    assert c.dtype is None


def test_scalar_pickle(daa: dak.Array) -> None:
    import pickle

    s = 2
    c1 = dakc.new_known_scalar(s)
    s_dumped = pickle.dumps(c1)
    c2 = pickle.loads(s_dumped)
    assert_eq(c1, c2)
    s1 = dak.sum(daa["points"]["y"], axis=None)
    s_dumped = pickle.dumps(s1)
    s2 = pickle.loads(s_dumped)
    assert_eq(s1, s2)

    assert s1.known_value is None


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_scalar_to_delayed(daa, optimize_graph) -> None:
    s1 = dak.sum(daa["points", "x"], axis=None)
    d1 = s1.to_delayed(optimize_graph=optimize_graph)
    s1c = s1.compute()
    assert d1.compute() == s1c  # type: ignore


def test_compatible_partitions(ndjson_points_file: str) -> None:
    daa1 = dak.from_json([ndjson_points_file] * 5)
    daa2 = dak.from_awkward(daa1.compute(), npartitions=4)
    assert dakc.compatible_partitions(daa1, daa1)
    assert dakc.compatible_partitions(daa1, daa1, daa1)
    assert not dakc.compatible_partitions(daa1, daa2)
    daa1.eager_compute_divisions()
    assert dakc.compatible_partitions(daa1, daa1)
    x = ak.Array([[1, 2, 3], [1, 2, 3], [3, 4, 5]])
    y = ak.Array([[1, 2, 3], [3, 4, 5]])
    x = dak.from_awkward(x, npartitions=2)
    y = dak.from_awkward(y, npartitions=2)
    assert not dakc.compatible_partitions(x, y)
    assert not dakc.compatible_partitions(x, x, y)
    assert dakc.compatible_partitions(y, y)


@pytest.mark.parametrize("meta", [5, False, [1, 2, 3]])
def test_bad_meta_type(ndjson_points_file: str, meta) -> None:
    with pytest.raises(TypeError, match="meta must be an instance of an Awkward Array"):
        dak.from_json([ndjson_points_file] * 3, meta=meta)


def test_scalar_repr(daa: dakc.Array) -> None:
    s = dak.max(daa.points.y)
    sstr = str(s)
    assert "type=Scalar" in sstr


def test_array_persist(daa: dakc.Array) -> None:
    daa2 = daa["points"]["x"].persist()
    assert_eq(daa["points"]["x"], daa2)
    daa2 = daa.persist()
    assert_eq(daa2, daa)


def test_scalar_persist(daa: dakc.Array) -> None:
    coll = daa["points"][0]["x"][0]
    coll2 = coll.persist()
    assert_eq(coll, coll2)


def test_output_divisions(daa: dakc.Array) -> None:
    assert dak.max(daa.points.y, axis=1).divisions == daa.divisions
    assert dak.num(daa.points.y, axis=1).divisions == (None,) * (daa.npartitions + 1)
    assert daa["points"][["x", "y"]].divisions == daa.divisions
    assert daa["points"].divisions == daa.divisions


def test_record_npartitions(daa: dakc.Array) -> None:
    analysis0 = daa[0]
    assert analysis0.npartitions == 1


def test_iter(daa: dakc.Array) -> None:
    with pytest.raises(
        NotImplementedError,
        match="Iteration over a Dask Awkward collection is not supported",
    ):
        for a in daa:
            pass
