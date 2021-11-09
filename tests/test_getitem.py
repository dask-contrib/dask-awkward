from dask_awkward.utils import assert_eq, load_nested


def test_single_string():
    a = load_nested()
    b = a.compute()
    assert_eq(a["analysis"], b["analysis"])


def test_multi_string():
    a = load_nested()
    b = a.compute()
    assert_eq(
        a["analysis"][["x1", "y2"]],
        b["analysis"][["x1", "y2"]],
    )


def test_test():
    a = load_nested()
    b = a.compute()
    assert_eq(a["analysis", "x1"][:, ::2], b["analysis", "x1"][:, ::2])
