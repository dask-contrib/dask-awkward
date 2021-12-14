from __future__ import annotations

import pytest

import dask_awkward as dak
import dask_awkward.core as dakc
from dask_awkward.utils import assert_eq

from .helpers import (  # noqa: F401
    LAZY_RECORD,
    LAZY_RECORDS,
    line_delim_records_file,
    load_records_eager,
)


def test_clear_divisions() -> None:
    daa = LAZY_RECORDS
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
    daa = LAZY_RECORDS
    assert daa.form == daa.compute().layout.form


def test_from_awkward(line_delim_records_file) -> None:  # noqa: F811
    aa = load_records_eager(line_delim_records_file)
    daa = dak.from_awkward(aa, npartitions=4)
    assert_eq(aa, daa)
    assert_eq(daa, daa)


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
    daa = LAZY_RECORDS
    lop = list(daa.partitions)
    for part in lop:
        assert part.npartitions == 1
    assert len(lop) == daa.npartitions


def test_partitions_divisions() -> None:
    daa = LAZY_RECORDS
    divs = daa.divisions
    t1 = daa.partitions[1:3]
    assert not t1.known_divisions
    t2 = daa.partitions[1]
    assert t2.known_divisions
    assert daa.partitions[1].divisions == (0, divs[2] - divs[1])  # type: ignore


def test_raise_in_finalize() -> None:
    daa = LAZY_RECORDS
    res = daa.map_partitions(str)
    with pytest.raises(RuntimeError, match="type of first result: <class 'str'>"):
        res.compute()


def test_type(line_delim_records_file) -> None:  # noqa: F811
    daa = LAZY_RECORDS
    assert dak.type(daa) is not None
    daa = dak.from_json(line_delim_records_file)
    daa.meta = None
    assert dak.type(daa) is None


def test_short_typestr() -> None:
    daa = LAZY_RECORDS
    ts = daa._shorttypestr(max=12)
    assert len(ts) == 12


def test_typestr() -> None:
    daa = LAZY_RECORD
    aa = daa.compute()
    assert str(aa.layout.form.type) in daa._typestr()
    daa = LAZY_RECORDS
    aa = daa.compute()
    extras = len("var *  ... }")
    assert len(daa._typestr(max=20)) == 20 + extras
