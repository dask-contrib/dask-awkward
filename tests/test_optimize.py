import pytest

pytest.importorskip("pyarrow")

import dask

import dask_awkward as dak
import dask_awkward.lib.optimize as o
from dask_awkward.layers import AwkwardIOLayer
from dask_awkward.lib.testutils import assert_eq


def test_requested_columns(caa_parquet):
    tg = dak.from_parquet([caa_parquet] * 2).points["x"].dask
    assert o._projectable_io_layer_names(tg)
    tg2 = o.optimize_columns(tg, [])
    layer_name = [_ for _ in tg.layers if _.startswith("read-parquet")][0]
    assert tg.layers[layer_name].columns is None
    assert tg2.layers[layer_name].columns == ["points.x"]


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
