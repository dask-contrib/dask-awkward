import dask_awkward as dak
from dask_awkward.layers import AwkwardInputLayer


def test_idempotent_layer_column_project(caa):
    daa = dak.from_awkward(caa, npartitions=2)
    n = 0
    for _, v in daa.dask.layers.items():
        if isinstance(v, AwkwardInputLayer):
            n += 1
            assert v is v.project_columns(["abc"])
    assert n > 0
