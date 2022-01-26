from __future__ import annotations

import pytest

import dask_awkward as dak
import dask_awkward.core as dakc

from .helpers import (  # noqa: F401
    _lazyrecord,
    _lazyrecords,
    assert_eq,
    line_delim_records_file,
    load_records_eager,
)


def test_clear_divisions() -> None:
    daa = dak.from_awkward(_lazyrecords().compute(), npartitions=3)
    assert daa.known_divisions
    daa.clear_divisions()
    assert not daa.known_divisions
    assert len(daa.divisions) == daa.npartitions + 1


def test_dunder_str(line_delim_records_file) -> None:  # noqa: F811
    aa = load_records_eager(line_delim_records_file)
    daa = dak.from_awkward(aa, npartitions=6)
    assert str(daa) == "dask.awkward<from-awkward, npartitions=5>"


def test_calculate_known_divisions(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json([line_delim_records_file] * 3)
    target = (0, 20, 40, 60)
    assert dakc.calculate_known_divisions(daa) == target
    assert dakc.calculate_known_divisions(daa.analysis) == target
    assert dakc.calculate_known_divisions(daa.analysis.x1) == target
    assert dakc.calculate_known_divisions(daa["analysis"][["x1", "x2"]]) == target
    daa = dak.from_json([line_delim_records_file] * 3)
    daa._compute_divisions()
    assert daa.known_divisions
    assert dakc.calculate_known_divisions(daa) == target


def test_fields(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file, blocksize=340)
    # records fields same as array of records fields
    assert daa[0].analysis.fields == daa.analysis.fields
    aa = daa.compute()
    assert daa.fields == aa.fields
    assert dak.fields(daa) == aa.fields
    daa.meta = None
    assert daa.fields is None
    assert dak.fields(daa) is None


def test_form(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file)
    assert daa.form
    daa.meta = None
    assert daa.form is None


@pytest.mark.xfail
def test_form_equality(line_delim_records_file) -> None:  # noqa: F811
    # NOTE: forms come from meta which currently depends on partitioning
    daa = dak.from_json(line_delim_records_file)
    assert daa.form == daa.compute().layout.form
    daa = _lazyrecords()
    assert daa.form == daa.compute().layout.form


def test_from_awkward(line_delim_records_file) -> None:  # noqa: F811
    aa = load_records_eager(line_delim_records_file)
    daa = dak.from_awkward(aa, npartitions=4)
    assert_eq(aa, daa)
    assert_eq(daa, daa)


def test_get_typetracer() -> None:
    daa = _lazyrecords()
    assert dakc._get_typetracer(daa) is daa.meta


def test_len(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file)
    assert len(daa) == 20


def test_meta_and_typetracer_exist(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file, blocksize=700)
    assert daa.meta is not None
    assert daa["analysis"]["x1"].meta is not None
    assert daa.typetracer is daa.meta


def test_meta_raise(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file)
    with pytest.raises(
        TypeError, match="meta must be an instance of an Awkward Array."
    ):
        daa.meta = "hello"


def test_ndim(line_delim_records_file) -> None:  # noqa
    daa = dak.from_json(line_delim_records_file, blocksize=700)
    assert daa.ndim == daa.compute().ndim
    daa.meta = None
    assert daa.ndim is None


def test_new_array_object_raises(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file)
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


def test_partitions() -> None:
    daa = _lazyrecords()
    lop = list(daa.partitions)
    for part in lop:
        assert part.npartitions == 1
    assert len(lop) == daa.npartitions


def test_partitions_divisions() -> None:
    daa = _lazyrecords()
    divs = daa.divisions
    t1 = daa.partitions[1:3]
    assert not t1.known_divisions
    t2 = daa.partitions[1]
    assert t2.known_divisions
    assert t2.divisions == (0, divs[2] - divs[1])  # type: ignore


def test_raise_in_finalize() -> None:
    daa = _lazyrecords()
    res = daa.map_partitions(str)
    with pytest.raises(RuntimeError, match="type of first result: <class 'str'>"):
        res.compute()


def test_rebuild(line_delim_records_file):  # noqa: F811
    daa = dak.from_json(line_delim_records_file)
    x = daa.compute()
    daa = daa._rebuild(daa.dask)
    y = daa.compute()
    assert x.tolist() == y.tolist()


def test_type(line_delim_records_file) -> None:  # noqa: F811
    daa = _lazyrecords()
    assert dak.type(daa) is not None
    daa = dak.from_json(line_delim_records_file)
    daa.meta = None
    assert dak.type(daa) is None


def test_short_typestr() -> None:
    daa = _lazyrecords()
    ts = daa._shorttypestr(max=12)
    assert len(ts) == 12


def test_typestr() -> None:
    daa = _lazyrecord()
    aa = daa.compute()
    assert str(aa.layout.form.type) in daa._typestr()
    daa = _lazyrecords()
    aa = daa.compute()
    extras = len("var *  ... }")
    assert len(daa._typestr(max=20)) == 20 + extras


def test_record_collection() -> None:
    daa = _lazyrecords()
    assert type(daa[0]) is dakc.Record
    aa = daa.compute()
    assert daa[0].compute().tolist() == aa[0].tolist()


def test_scalar_collection() -> None:
    daa = _lazyrecords()
    assert type(daa["analysis"]["x1"][0][0]) is dakc.Scalar


def test_is_typetracer() -> None:
    daa = _lazyrecords()
    assert not dakc.is_typetracer(daa)
    assert not dakc.is_typetracer(daa[0])
    assert not dakc.is_typetracer(daa["analysis"])
    assert not dakc.is_typetracer(daa.compute())
    assert dakc.is_typetracer(daa.meta)
    assert dakc.is_typetracer(daa[0].meta)
    assert dakc.is_typetracer(daa["analysis"].meta)
    assert dakc.is_typetracer(daa["analysis"][0]["x1"][0].meta)


def test_meta_or_identity() -> None:
    daa = _lazyrecords()
    assert dakc.is_typetracer(dakc.meta_or_identity(daa))
    assert dakc.meta_or_identity(daa) is daa.meta
    assert dakc.meta_or_identity(5) == 5


def test_to_meta() -> None:
    daa = _lazyrecords()
    x1 = daa["analysis"]["x1"]
    x1_0 = x1[0]
    metad = dakc.to_meta([x1, 5, "ok", x1_0])
    assert isinstance(metad, tuple)
    for a, b in zip(metad, (x1.meta, 5, "ok", x1_0.meta)):
        if dakc.is_typetracer(a):
            assert a is b
        else:
            assert a == b
