from dask_awkward.utils import assert_eq
from helpers import load_records_eager, load_records_lazy


def test_single_string() -> None:
    daa = load_records_lazy()
    caa = load_records_eager()
    assert_eq(daa["analysis"], caa["analysis"])


def test_multi_string() -> None:
    daa = load_records_lazy()
    caa = load_records_eager()
    assert_eq(
        daa["analysis"][["x1", "y2"]],
        caa["analysis"][["x1", "y2"]],
    )


def test_single_int() -> None:
    daa = load_records_lazy()["analysis"]["y1"]
    caa = load_records_eager()["analysis"]["y1"]
    for i in range(len(daa)):
        assert_eq(daa[i], caa[i])


def test_test() -> None:
    daa = load_records_lazy()
    caa = load_records_eager()
    assert_eq(daa["analysis", "x1"][:, ::2], caa["analysis", "x1"][:, ::2])
