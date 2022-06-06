from __future__ import annotations

import pytest

distributed = pytest.importorskip("distributed")

from pathlib import Path
from typing import TYPE_CHECKING

import awkward._v2 as ak
import numpy as np
from dask import persist
from dask.delayed import delayed
from distributed.client import _wait
from distributed.utils_test import cluster_fixture  # noqa
from distributed.utils_test import loop  # noqa
from distributed.utils_test import a, b, cleanup  # noqa
from distributed.utils_test import client as c  # noqa
from distributed.utils_test import cluster, gen_cluster, s, varying  # noqa

import dask_awkward as dak
from dask_awkward.testutils import assert_eq

if TYPE_CHECKING:
    from distributed import Client


def test_simple_compute(c, daa_p1, caa_p1) -> None:  # noqa
    assert_eq(daa_p1.points.x, caa_p1.points.x, scheduler=c)


@gen_cluster(client=True)
async def test_persist(c: Client, s, a, b, daa_p1) -> None:  # noqa
    (x1,) = persist(daa_p1, scheduler=c)
    await _wait(x1)
    assert x1.__dask_keys__()[0] in daa_p1.__dask_keys__()


@pytest.mark.parametrize("optimize_graph", [True, False])
@gen_cluster(client=True)
async def test_compute(
    c,  # noqa
    s,  # noqa
    a,  # noqa
    b,  # noqa
    ndjson_points_file,
    optimize_graph,
) -> None:
    files = [ndjson_points_file] * 3
    daa = dak.from_json(files)
    caa = ak.concatenate([ak.from_json(Path(f).read_text()) for f in files])
    res = await c.compute(
        dak.num(daa.points.x, axis=1),
        optimize_graph=optimize_graph,
    )
    assert res.tolist() == ak.num(caa.points.x, axis=1).tolist()


def test_from_delayed(c: Client, ndjson_points_file: str) -> None:  # noqa
    def make_a_concrete(file: str) -> ak.Array:
        with open(file) as f:
            return ak.from_json(f.read())

    make_a_delayed = delayed(make_a_concrete, pure=True)

    x = dak.from_delayed([make_a_delayed(f) for f in [ndjson_points_file] * 3])
    y = ak.concatenate([make_a_concrete(f) for f in [ndjson_points_file] * 3])
    assert_eq(x, y, scheduler=c, check_unconcat_form=False)


from awkward._v2.behaviors.mixins import mixin_class as ak_mixin_class
from awkward._v2.behaviors.mixins import mixin_class_method as ak_mixin_class_method

behaviors = {}


@ak_mixin_class(behaviors)
class Point:
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @property
    def x2(self):
        return self.x * self.x

    @ak_mixin_class_method(np.abs)
    def point_abs(self):
        return np.sqrt(self.x**2 + self.y**2)


def test_from_list_behaviorized(c: Client, L1: list, L2: list) -> None:  # noqa
    daa = dak.from_lists([L1, L2])
    daa = dak.with_name(daa, name="Point", behavior=behaviors)
    caa = ak.Array(L1 + L2, with_name="Point", behavior=behaviors)

    assert_eq(daa.x2, caa.x2, scheduler=c)
    assert_eq(daa.distance(daa), caa.distance(caa), scheduler=c)
