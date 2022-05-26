from __future__ import annotations

import pytest

distributed = pytest.importorskip("distributed")

import copy
from typing import TYPE_CHECKING

import awkward._v2 as ak
from dask import persist
from dask.delayed import delayed
from distributed import wait
from distributed.utils_test import cluster_fixture  # noqa
from distributed.utils_test import loop  # noqa
from distributed.utils_test import a, b  # noqa
from distributed.utils_test import client as c  # noqa
from distributed.utils_test import cluster, gen_cluster, inc, s, varying  # noqa

import dask_awkward as dak
from dask_awkward.testutils import assert_eq

if TYPE_CHECKING:
    from distributed import Client

X = ak.from_iter([[1, 2, 3], [4], [5, 6, 7]])


def test_simple_compute(c) -> None:  # noqa
    x1 = copy.copy(X)
    x2 = dak.from_awkward(x1, npartitions=3)
    assert_eq(x1, x2, scheduler=c)


@gen_cluster(client=True)
async def test_persist(c, s, a, b) -> None:  # noqa
    x1 = copy.copy(X)
    x2 = dak.from_awkward(x1, npartitions=3)

    (x3,) = persist(x2)

    await wait(x3)

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
