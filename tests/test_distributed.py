from __future__ import annotations

import pytest

distributed = pytest.importorskip("distributed")

from pathlib import Path

import awkward as ak
import numpy as np
from dask import persist
from dask.delayed import delayed
from distributed import Client
from distributed.client import _wait
from distributed.utils_test import cleanup  # noqa
from distributed.utils_test import loop  # noqa
from distributed.utils_test import loop_in_thread  # noqa
from distributed.utils_test import cluster, gen_cluster

import dask_awkward as dak
from dask_awkward.lib.testutils import BAD_NP_AK_MIXIN_VERSIONING, assert_eq

# @pytest.fixture(scope="session")
# def small_cluster():
#     with LocalCluster(n_workers=2, threads_per_worker=1) as cluster:
#         yield cluster


# @pytest.fixture
# def simple_client(small_cluster):
#     with Client(small_cluster) as client:
#         yield client


def test_compute(loop, ndjson_points_file):  # noqa
    caa = ak.from_json(Path(ndjson_points_file).read_text(), line_delimited=True)
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as client:
            daa = dak.from_json([ndjson_points_file])
            assert_eq(daa.points.x, caa.points.x, scheduler=client)


@gen_cluster(client=True)
async def test_persist(c, s, a, b, ndjson_points_file):
    daa = dak.from_json([ndjson_points_file])
    (x1,) = persist(daa, scheduler=c)
    await _wait(x1)
    assert x1.__dask_keys__()[0] in daa.__dask_keys__()


@pytest.mark.parametrize("optimize_graph", [True, False])
@gen_cluster(client=True)
async def test_compute_gen_cluster(
    c,
    s,
    a,
    b,
    ndjson_points_file,
    optimize_graph,
):
    files = [ndjson_points_file] * 3
    daa = dak.from_json(files)
    caa = ak.concatenate(
        [ak.from_json(Path(f).read_text(), line_delimited=True) for f in files]
    )
    res = await c.compute(
        dak.num(daa.points.x, axis=1),
        optimize_graph=optimize_graph,
    )
    assert res.tolist() == ak.num(caa.points.x, axis=1).tolist()


def make_a_concrete(file: str) -> ak.Array:
    with open(file) as f:
        return ak.from_json(f.read(), line_delimited=True)


def test_from_delayed(loop, ndjson_points_file):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as client:
            make_a_delayed = delayed(make_a_concrete, pure=True)
            x = dak.from_delayed([make_a_delayed(f) for f in [ndjson_points_file] * 3])
            y = ak.concatenate([make_a_concrete(f) for f in [ndjson_points_file] * 3])
            assert_eq(x, y, scheduler=client)


behaviors: dict = {}


@ak.mixin_class(behaviors)
class Point:
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @property
    def x2(self):
        return self.x * self.x

    @ak.mixin_class_method(np.abs)
    def point_abs(self):
        return np.sqrt(self.x**2 + self.y**2)


@pytest.mark.xfail(
    BAD_NP_AK_MIXIN_VERSIONING,
    reason="NumPy 1.25 mixin __slots__ change",
)
def test_from_list_behaviorized(loop, L1, L2):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as client:
            daa = dak.from_lists([L1, L2])
            daa = dak.with_name(daa, name="Point", behavior=behaviors)
            caa = ak.Array(L1 + L2, with_name="Point", behavior=behaviors)

            assert_eq(daa.x2, caa.x2, scheduler=client)
            assert_eq(daa.distance(daa), caa.distance(caa), scheduler=client)


# def test_from_list_behaviorized2(simple_client, L1, L2) -> None:
#     client = simple_client
#     daa = dak.from_lists([L1, L2])
#     daa = dak.with_name(daa, name="Point", behavior=behaviors)
#     caa = ak.Array(L1 + L2, with_name="Point", behavior=behaviors)

#     assert_eq(daa.x2, caa.x2, scheduler=client)
#     assert_eq(daa.distance(daa), caa.distance(caa), scheduler=client)
