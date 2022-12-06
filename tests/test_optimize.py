import pytest

pytest.importorskip("pyarrow")

import dask

import dask_awkward as dak
import dask_awkward.lib.optimize as o
from dask_awkward.layers import AwkwardIOLayer
from dask_awkward.lib.testutils import assert_eq


def test_is_getitem(caa_parquet):
    a = dak.from_parquet([caa_parquet] * 2)
    for _, v in a.points[["x", "y"]].dask.layers.items():
        if isinstance(v, AwkwardIOLayer):
            continue
        else:
            assert o._is_getitem(v)


def test_requested_columns(caa_parquet):
    tg = dak.from_parquet([caa_parquet] * 2).points["x"].dask
    assert any(o._is_getitem(v) for _, v in tg.layers.items())
    for k, v in tg.layers.items():
        if isinstance(v, AwkwardIOLayer):
            continue
        if k.startswith("points"):
            assert o._requested_columns_getitem(v) == {"points"}
        if k.startswith("x"):
            assert o._requested_columns_getitem(v) == {"x"}


@pytest.mark.parametrize("method", ["brute-force", "simple-getitem"])
@pytest.mark.parametrize("nullmethod", [False, None, "none"])
def test_config_adjust(caa_parquet, method, nullmethod):
    with dask.config.set({"awkward.column-projection-optimization": nullmethod}):
        a = dak.from_parquet([caa_parquet] * 3)
        ref = a.points.x.compute()

    with dask.config.set({"awkward.column-projection-optimization": method}):
        a = dak.from_parquet([caa_parquet] * 3)
        opted = a.points.x.compute()

    assert_eq(ref, opted, check_forms=False)
