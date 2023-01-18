import pytest

pytest.importorskip("pyarrow")

import dask

import dask_awkward as dak
import dask_awkward.lib.optimize as o
from dask_awkward.layers import AwkwardIOLayer
from dask_awkward.lib.testutils import assert_eq


def test_foo():
    assert True
