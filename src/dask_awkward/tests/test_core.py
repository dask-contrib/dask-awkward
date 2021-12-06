from __future__ import annotations

import dask_awkward.core as dakc
from dask_awkward.io import from_json
from dask_awkward.utils import assert_eq

from .helpers import (  # noqa: F401
    line_delim_records_file,
    load_records_eager,
    load_records_lazy,
    single_record_file,
)


def test_meta_exists(line_delim_records_file) -> None:  # noqa: F811
    daa = from_json(line_delim_records_file, blocksize=1024)
    assert daa.meta is not None
    assert daa["analysis"]["x1"].meta is not None


def test_calculate_known_divisions(line_delim_records_file) -> None:  # noqa: F811
    print(line_delim_records_file)
    daa = from_json([line_delim_records_file] * 3)
    target = (0, 20, 40, 60)
    assert dakc.calculate_known_divisions(daa) == target
    assert dakc.calculate_known_divisions(daa.analysis) == target
    assert dakc.calculate_known_divisions(daa.analysis.x1) == target
    assert dakc.calculate_known_divisions(daa["analysis"][["x1", "x2"]]) == target


def test_len(line_delim_records_file) -> None:  # noqa: F811
    daa = from_json(line_delim_records_file)
    assert len(daa) == 20


def test_from_awkward(line_delim_records_file) -> None:  # noqa: F811
    aa = load_records_eager(line_delim_records_file)
    daa = dakc.from_awkward(aa, npartitions=3)
    assert_eq(daa, aa)
