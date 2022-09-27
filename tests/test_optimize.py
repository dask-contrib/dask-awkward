import dask_awkward as dak
import dask_awkward.lib.optimize as o
from dask_awkward.layers import AwkwardIOLayer


def test_is_getitem(caa_parquet):
    a = dak.from_parquet([caa_parquet] * 2)
    for k, v in a.points[["x", "y"]].dask.layers.items():
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
            assert o._requested_columns(v) == {"points"}
        if k.startswith("x"):
            assert o._requested_columns(v) == {"x"}
