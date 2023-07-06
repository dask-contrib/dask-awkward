from __future__ import annotations

import dask

import dask_awkward as dak


def test_multiple_computes(pq_points_dir) -> None:
    ds1 = dak.from_parquet(pq_points_dir)
    # add a columns= argument to force a new tokenize result in
    # from_parquet so we get two unique collections.
    ds2 = dak.from_parquet(pq_points_dir, columns=["points"])

    assert ds1.name != ds2.name
    assert dask.compute(ds1.points.x, ds2.points.y)
