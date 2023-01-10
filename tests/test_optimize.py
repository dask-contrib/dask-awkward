import pytest

pytest.importorskip("pyarrow")

import dask_awkward as dak
import dask_awkward.lib.optimize as o


def test_requested_columns(caa_parquet):
    tg = dak.from_parquet([caa_parquet] * 2).points["x"].dask
    assert o._projectable_io_layer_names(tg)
    tg2 = o.optimize_columns(tg, [])
    layer_name = [_ for _ in tg.layers if _.startswith("read-parquet")][0]
    assert tg.layers[layer_name].columns is None
    assert tg2.layers[layer_name].columns == ["points.x"]
