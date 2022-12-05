from __future__ import annotations

import pytest

import dask_awkward as dak
from dask_awkward.lib.inspect import necessary_columns


def test_necessary_columns(
    daa: dak.Array,
    tmpdir_factory: pytest.TempdirFactory,
) -> None:
    dname = tmpdir_factory.mktemp("pq")
    dak.to_parquet(daa, str(dname), compute=True)
    ds = dak.from_parquet(str(dname))
    aioname = list(ds.dask.layers.items())[0][0]
    assert necessary_columns(ds.points.x, "brute") == {aioname: ["points.x"]}
    assert necessary_columns(ds.points.x, "getitem") == {aioname: ["points"]}
    with pytest.raises(ValueError, match="strategy argument should"):
        assert necessary_columns(ds.points, "okokok")  # type: ignore
