from __future__ import annotations

import numpy as np
import pytest

from dask_awkward.testutils import assert_eq


def test_ufunc_add(daa, caa) -> None:
    a1 = daa.analysis.x1 + 2
    a2 = caa.analysis.x1 + 2
    assert_eq(a1, a2)


def test_ufunc_sin(daa, caa) -> None:
    daa = daa.analysis.x1
    caa = caa.analysis.x1
    a1 = np.sin(daa)
    a2 = np.sin(caa)
    assert_eq(a1, a2)


@pytest.mark.parametrize("f", [np.add.accumulate, np.add.reduce])
def test_ufunc_method_raise(daa, f) -> None:
    daa = daa.analysis.x1
    with pytest.raises(RuntimeError, match="Array ufunc supports only method"):
        f(daa, daa)
