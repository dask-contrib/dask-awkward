from __future__ import annotations

import numpy as np

from dask_awkward.data import load_array
from dask_awkward.utils import assert_eq


def test_ufunc_sin():
    daa = load_array()
    a1 = np.sin(daa)
    a2 = np.sin(daa.compute())
    assert_eq(a1, a2)


def test_ufunc_add():
    daa = load_array()
    a1 = daa
    a2 = daa + 2
    a3 = a1 + 2
    a4 = a2.compute()
    assert_eq(a3, a4)
