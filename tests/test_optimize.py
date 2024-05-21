from __future__ import annotations

import awkward as ak
import dask
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq


def test_multiple_computes(ndjson_points_file: str) -> None:
    ds1 = dak.from_json([ndjson_points_file] * 2)
    # add a kwarg argument to force a new tokenize result in
    # from_json so we get two unique collections.
    ds2 = dak.from_json([ndjson_points_file] * 2, buffersize=65536 // 2)

    lists = [[[1, 2, 3], [4, 5]], [[], [0, 0, 0]]]
    ds3 = dak.from_lists(lists)

    assert ds1.name != ds2.name
    things1 = dask.compute(ds1.points.x, ds2.points.y)
    things2 = dask.compute(ds1.points)
    assert things2[0].x.tolist() == things1[0].tolist()

    things3 = dask.compute(ds2.points.y, ds1.points.partitions[0])
    assert things3[0].tolist() == things1[1].tolist()

    assert len(things3[1]) < len(things3[0])

    things = dask.compute(ds1.points, ds2.points.x, ds2.points.y, ds1.points.y, ds3)
    assert things[-1].tolist() == ak.Array(lists[0] + lists[1]).tolist()  # type: ignore


def identity(x):
    return x


def test_multiple_compute_incapsulated():
    array = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])[[0, 2]]
    darray = dak.from_awkward(array, 1)
    darray_result = darray.map_partitions(identity)

    first, second = dask.compute(darray, darray_result)

    assert ak.almost_equal(first, second)
    assert first.layout.form == second.layout.form


def test_multiple_computes_multiple_incapsulated(daa, caa):
    dstep1 = daa.points.x
    dstep2 = dstep1**2
    dstep3 = dstep2 + 2
    dstep4 = dstep3 - 1
    dstep5 = dstep4 - dstep2

    cstep1 = caa.points.x
    cstep2 = cstep1**2
    cstep3 = cstep2 + 2
    cstep4 = cstep3 - 1
    cstep5 = cstep4 - cstep2

    # multiple computes all work and evaluate to the expected result
    c5, c4, c2 = dask.compute(dstep5, dstep4, dstep2)
    assert_eq(c5, cstep5)
    assert_eq(c2, cstep2)
    assert_eq(c4, cstep4)

    # if optimized together we still have 2 layers
    opt4, opt3 = dask.optimize(dstep4, dstep3)
    assert len(opt4.dask.layers) == 2
    assert len(opt3.dask.layers) == 2
    assert_eq(opt4, cstep4)
    assert_eq(opt3, cstep3)

    # if optimized alone we get optimized to 1 entire chain smushed
    # down to 1 layer
    (opt4_alone,) = dask.optimize(dstep4)
    assert len(opt4_alone.dask.layers) == 1
    assert_eq(opt4_alone, opt4)


def test_optimization_runs_on_multiple_collections_gh430(tmp_path_factory):
    pytest.importorskip("pyarrow")
    d = tmp_path_factory.mktemp("opt")
    array1 = ak.Array(
        [
            {
                "points": [
                    {"x": 3.0, "y": 4.0, "z": 1.0},
                    {"x": 2.0, "y": 5.0, "z": 2.0},
                ],
            },
            {
                "points": [
                    {"x": 2.0, "y": 5.0, "z": 2.0},
                    {"x": 3.0, "y": 4.0, "z": 1.0},
                ],
            },
            {
                "points": [],
            },
            {
                "points": [
                    {"x": 2.0, "y": 6.0, "z": 2.0},
                ],
            },
        ]
    )
    ak.to_parquet(array1, d / "p0.parquet", extensionarray=False)

    array2 = ak.Array(
        [
            {
                "points": [
                    {"x": 1.0, "y": 4.0, "z": 1.0},
                ],
            },
            {
                "points": [
                    {"x": 7.0, "y": 5.0, "z": 2.0},
                    {"x": 3.0, "y": 4.0, "z": 1.0},
                    {"x": 5.0, "y": 5.0, "z": 2.0},
                ],
            },
            {
                "points": [
                    {"x": 2.0, "y": 6.0, "z": 2.0},
                ],
            },
            {
                "points": [],
            },
        ]
    )
    ak.to_parquet(array2, d / "p1.parquet", extensionarray=False)

    ds = dak.from_parquet(d)
    a1 = ds.partitions[0].points
    a2 = ds.partitions[1].points
    a, b = ak.unzip(ak.cartesian([a1, a2], axis=1, nested=True))

    def something(j, k):
        return j.x + k.x

    a_compute = something(a, b)
    print()
    nc1 = dak.necessary_columns(a_compute)
    assert list(nc1.values())[0] == ["points.x"]

    nc2 = dak.necessary_columns(a_compute, a, b)
    assert list(nc2.items())[0][1] == ["points.x", "points.y", "points.z"]

    x, (y, z) = dask.compute(a_compute, (a, b))
    assert str(x)
    assert str(y)
    assert str(z)
