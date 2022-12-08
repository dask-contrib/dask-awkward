import dask_awkward
import dask_awkward.config
import dask_awkward.layers
import dask_awkward.typing
import dask_awkward.utils  # noqa


def test_import() -> None:
    try:
        import awkward as ak  # noqa
    except ImportError:
        pass
    assert True
