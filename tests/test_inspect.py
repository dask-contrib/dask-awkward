from __future__ import annotations

from pathlib import Path

import pytest

import dask_awkward as dak


def test_necessary_columns(
    daa: dak.Array, tmpdir_factory: pytest.TempdirFactory
) -> None:
    z = daa.points.x + daa.points.y
    for k, v in dak.necessary_columns(z).items():
        assert set(v) == {"points.x", "points.y"}

    w = dak.to_parquet(
        daa.points.x, str(Path(tmpdir_factory.mktemp("pq")) / "out"), compute=False
    )
    for k, v in dak.necessary_columns(w).items():
        assert set(v) == {"points.x"}
