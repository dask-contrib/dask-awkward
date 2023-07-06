from __future__ import annotations

import awkward as ak
import dask

import dask_awkward as dak


def test_multiple_computes(pq_points_dir) -> None:
    ds1 = dak.from_parquet(pq_points_dir)
    # add a columns= argument to force a new tokenize result in
    # from_parquet so we get two unique collections.
    ds2 = dak.from_parquet(pq_points_dir, columns=["points"])

    lists = [[[1, 2, 3], [4, 5]], [[], [0, 0, 0]]]
    ds3 = dak.from_lists([[[1, 2, 3], [4, 5]], [[], [0, 0, 0]]])

    assert ds1.name != ds2.name
    things1 = dask.compute(ds1.points.x, ds2.points.y)
    things2 = dask.compute(ds1.points)
    assert things2[0].x.tolist() == things1[0].tolist()

    things3 = dask.compute(ds2.points.y, ds1.points.partitions[0])
    assert things3[0].tolist() == things1[1].tolist()

    assert len(things3[1]) < len(things3[0])

    things = dask.compute(ds1.points, ds2.points.x, ds2.points.y, ds1.points.y, ds3)
    assert things[-1].tolist() == ak.Array(lists[0] + lists[1]).tolist()
