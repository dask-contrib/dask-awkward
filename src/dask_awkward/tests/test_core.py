from __future__ import annotations

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


def test_fields(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file, blocksize=340)
    aa = daa.compute()
    assert daa.fields == aa.fields
    assert dak.fields(daa) == aa.fields


def test_from_awkward(line_delim_records_file) -> None:  # noqa: F811
    aa = load_records_eager(line_delim_records_file)
    daa = dak.from_awkward(aa, npartitions=4)
    assert_eq(daa, aa)


def test_len(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file)
    assert len(daa) == 20


def test_meta_and_typetracer_exist(line_delim_records_file) -> None:  # noqa: F811
    daa = dak.from_json(line_delim_records_file, blocksize=700)
    assert daa.meta is not None
    assert daa["analysis"]["x1"].meta is not None
    assert daa.typetracer is daa.meta


def test_partitions() -> None:  # noqa: F811
    daa = LAZY_RECORDS
    lop = list(daa.partitions)
    for part in lop:
        assert part.npartitions == 1
    assert len(lop) == daa.npartitions


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
