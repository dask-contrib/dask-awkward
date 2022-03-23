from __future__ import annotations

import awkward._v2 as ak
import fsspec
import pytest
from dask.array.utils import assert_eq as da_assert_eq

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import dask_awkward as dak
from dask_awkward.testutils import assert_eq


def test_force_by_lines_meta(line_delim_records_file: str) -> None:
    daa1 = dak.from_json(
        [line_delim_records_file] * 5,
        derive_meta_kwargs={"force_by_lines": True},
    )
    daa2 = dak.from_json([line_delim_records_file, line_delim_records_file])
    assert daa1._meta is not None
    assert daa2._meta is not None
    f1 = daa1._meta.layout.form
    f2 = daa2._meta.layout.form
    assert f1 == f2


def test_derive_json_meta_trigger_warning(line_delim_records_file: str) -> None:
    with pytest.warns(UserWarning):
        dak.from_json(line_delim_records_file, derive_meta_kwargs={"bytechunks": 64})


def test_json_one_obj_per_file(single_record_file: str) -> None:
    daa = dak.from_json(
        [single_record_file] * 5,
        one_obj_per_file=True,
    )
    with fsspec.open(single_record_file, "r") as f:
        content = json.load(f)
    caa = ak.from_iter([content] * 5)
    assert_eq(daa, caa)


def test_json_delim_defined(line_delim_records_file: str) -> None:
    source = [line_delim_records_file] * 6
    daa = dak.from_json(source, delimiter=b"\n")

    concretes = []
    for s in source:
        with open(s) as f:
            for line in f:
                concretes.append(json.loads(line))
    caa = ak.from_iter(concretes)

    assert_eq(
        daa["analysis"][["x1", "z2"]],
        caa["analysis"][["x1", "z2"]],
    )


def test_to_and_from_dask_array(line_delim_records_file) -> None:
    daa = dak.from_json([line_delim_records_file] * 3)
    computed = ak.flatten(daa.analysis.x1.compute())
    x1 = dak.flatten(daa.analysis.x1)
    daskarr = dak.to_dask_array(x1)
    da_assert_eq(daskarr, computed.to_numpy())

    back_to_dak = dak.from_dask_array(daskarr)
    assert_eq(back_to_dak, computed)


def test_from_dask_array() -> None:
    from dask.array.wrap import ones

    darr = ones(100, chunks=25)
    daa = dak.from_dask_array(darr)
    assert daa.known_divisions
    assert_eq(daa, ak.from_numpy(darr.compute()))
