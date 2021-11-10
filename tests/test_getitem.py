from dask_awkward.data import load_array, load_nested
from dask_awkward.utils import assert_eq


def test_single_string() -> None:
    a = load_nested()
    b = a.compute()
    assert_eq(a["analysis"], b["analysis"])


def test_multi_string() -> None:
    a = load_nested()
    b = a.compute()
    assert_eq(
        a["analysis"][["x1", "y2"]],
        b["analysis"][["x1", "y2"]],
    )


def test_single_int() -> None:
    a = load_array()
    for i in range(len(a)):
        assert_eq(a[i], a.compute()[i])


def test_test() -> None:
    a = load_nested()
    b = a.compute()
    assert_eq(a["analysis", "x1"][:, ::2], b["analysis", "x1"][:, ::2])
