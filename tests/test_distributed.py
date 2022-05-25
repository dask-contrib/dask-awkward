from __future__ import annotations

import pytest

distributed = pytest.importorskip("distributed")

import awkward._v2 as ak
from distributed.utils_test import client as c  # noqa
from distributed.utils_test import cluster_fixture  # noqa
from distributed.utils_test import loop  # noqa
from distributed.utils_test import cluster, gen_cluster, inc, varying  # noqa

import dask_awkward as dak
from dask_awkward.testutils import assert_eq


def test_compute(c):  # noqa
    x1 = ak.from_iter([[1, 2, 3], [4], [5, 6, 7]])
    x2 = dak.from_awkward(x1, npartitions=3)
    assert_eq(x1, x2, scheduler=c)
