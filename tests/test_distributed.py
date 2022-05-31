from __future__ import annotations

import pytest

distributed = pytest.importorskip("distributed")

import copy
from typing import TYPE_CHECKING

import awkward._v2 as ak
import numpy as np
from dask import persist
from dask.delayed import delayed
from distributed.client import _wait
from distributed.utils_test import cluster_fixture  # noqa
from distributed.utils_test import loop  # noqa
from distributed.utils_test import a, b  # noqa
from distributed.utils_test import client as c  # noqa
from distributed.utils_test import cluster, gen_cluster, inc, s, varying  # noqa

import dask_awkward as dak
import dask_awkward.testutils as daktu
from dask_awkward.testutils import assert_eq

if TYPE_CHECKING:
    from distributed import Client

X = ak.from_iter([[1, 2, 3], [4], [5, 6, 7]])


def test_simple_compute(c) -> None:  # noqa
    x1 = copy.copy(X)
    x2 = dak.from_awkward(x1, npartitions=3)
    assert_eq(x1, x2, scheduler=c)


@gen_cluster(client=True)
async def test_persist(c: Client, s, a, b) -> None:  # noqa
    x1 = copy.copy(X)
    x2 = dak.from_awkward(x1, npartitions=3)
    (x3,) = persist(x2, scheduler=c)
    await _wait(x3)
    assert x3.__dask_keys__()[0] in x2.__dask_keys__()


@pytest.mark.parametrize("optimize_graph", [True, False])
@gen_cluster(client=True)
async def test_compute(
    c,  # noqa
    s,  # noqa
    a,  # noqa
    b,  # noqa
    line_delim_records_file,
    concrete_from_line_delim,
    optimize_graph,
) -> None:
    files = [line_delim_records_file] * 3
    daa = dak.from_json(files)
    caa = ak.concatenate([concrete_from_line_delim] * 3)
    res = await c.compute(
        dak.num(daa.analysis.x1, axis=1),
        optimize_graph=optimize_graph,
    )
    assert res.tolist() == ak.num(caa.analysis.x1, axis=1).tolist()


def test_from_delayed(c: Client, line_delim_records_file: str) -> None:  # noqa
    def make_a_concrete(file: str) -> ak.Array:
        with open(file) as f:
            return ak.from_json(f.read())

    make_a_delayed = delayed(make_a_concrete, pure=True)

    x = dak.from_delayed([make_a_delayed(f) for f in [line_delim_records_file] * 3])
    y = ak.concatenate([make_a_concrete(f) for f in [line_delim_records_file] * 3])
    assert_eq(x, y, scheduler=c, check_unconcat_form=False)


def test_from_lists(c: Client) -> None:  # noqa

    daa = dak.from_lists([daktu.A1, daktu.A2])
    caa = ak.Array(daktu.A1 + daktu.A2)
    assert_eq(daa, caa, scheduler=c)
    assert_eq(daa.x, caa.x, scheduler=c)


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


def test_from_list_behaviorized(c: Client) -> None:  # noqa
    daa = dak.from_lists([daktu.A1, daktu.A2])
    daa = dak.with_name(daa, name="Point", behavior=behaviors)
    caa = ak.Array(daktu.A1 + daktu.A2, with_name="Point", behavior=behaviors)

    assert_eq(daa.x2, caa.x2, scheduler=c)
    assert_eq(daa.distance(daa), caa.distance(caa), scheduler=c)
