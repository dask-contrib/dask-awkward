from __future__ import annotations

import numpy as np
import pytest

from dask_awkward.testutils import assert_eq


def test_ufunc_add(daa, caa) -> None:
    a1 = daa.points.x + 2
    a2 = caa.points.x + 2
    assert_eq(a1, a2)


def test_ufunc_sin(daa, caa) -> None:
    daa = daa.points.x
    caa = caa.points.x
    a1 = np.sin(daa)
    a2 = np.sin(caa)
    assert_eq(a1, a2)


@pytest.mark.parametrize("f", [np.add.accumulate, np.add.reduce])
def test_ufunc_method_raise(daa, f) -> None:
    daa = daa.points.x
    with pytest.raises(RuntimeError, match="Array ufunc supports only method"):
        f(daa, daa)
