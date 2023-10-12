from __future__ import annotations

import awkward as ak
import awkward.operations.str as akstr
import fsspec
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq


def test_from_text() -> None:
    pytest.importorskip("pyarrow")
    f1 = "https://raw.githubusercontent.com/dask-contrib/dask-awkward/main/README.md"
    f2 = "https://raw.githubusercontent.com/dask-contrib/dask-awkward/main/LICENSE"

    daa = dak.from_text([f1, f2])
    assert daa.npartitions == 2

    with fsspec.open(f1, "rt") as f:
        caa1 = ak.Array([line.rstrip() for line in f.readlines()])
    with fsspec.open(f2, "rt") as f:
        caa2 = ak.Array([line.rstrip() for line in f.readlines()])

    caa = ak.concatenate([caa1, caa2])

    assert_eq(daa, caa)

    daa1 = daa.map_partitions(akstr.split_whitespace).map_partitions(akstr.is_upper)
    caa1 = akstr.is_upper(akstr.split_whitespace(caa))
    assert_eq(daa1, caa1)
