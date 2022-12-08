import dask_awkward
import dask_awkward.config
import dask_awkward.layers
import dask_awkward.typing
import dask_awkward.utils  # noqa


def test_import():
    try:
        pass
    except ImportError:
        assert True
