from __future__ import annotations

from collections.abc import Callable

import awkward._v2 as ak
import numpy as np
import pytest

import dask_awkward as dak
from dask_awkward.testutils import assert_eq


def test_ufunc_add(daa: dak.Array, caa: ak.Array) -> None:
    a1 = daa.points.x + 2
    a2 = caa.points.x + 2
    assert_eq(a1, a2)


def test_ufunc_sin(daa: dak.Array, caa: ak.Array) -> None:
    daa = daa.points.x
    caa = caa.points.x
    a1 = np.sin(daa)
    a2 = np.sin(caa)
    assert_eq(a1, a2)


@pytest.mark.parametrize("f", [np.add.accumulate, np.add.reduce])
def test_ufunc_method_raise(daa: dak.Array, f: Callable) -> None:
    daa = daa.points.x
    with pytest.raises(RuntimeError, match="Array ufunc supports only method"):
        f(daa, daa)
