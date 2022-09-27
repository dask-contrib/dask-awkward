from dask_awkward.layers import AwkwardIOLayer


def test_idempotent_layer_column_project(daa):
    # daa is a dask_awkward Array created via a dak.from_json call; so
    # it is _not_ column projectable. If we call project_columns on
    # the AwkwardIOLayer it just returns itself no matter what the
    # arguments are.
    for k, v in daa.dask.layers.items():
        if isinstance(v, AwkwardIOLayer):
            assert v is v.project_columns(["abc"])
