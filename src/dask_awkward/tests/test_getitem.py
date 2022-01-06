from __future__ import annotations

import pytest

import dask_awkward.core as dakc

from .helpers import assert_eq, caa, daa  # noqa: F401


def test_getattr_raise(daa) -> None:  # noqa: F811
    dar = daa[0]
    assert type(dar) is dakc.Record
    with pytest.raises(AttributeError, match="not in fields"):
        assert daa.abcdefg
    with pytest.raises(AttributeError, match="not in fields"):
        assert dar.x3


def test_multi_string(daa, caa) -> None:  # noqa: F811
    assert_eq(
        daa["analysis"][["x1", "y2"]],
        caa["analysis"][["x1", "y2"]],
    )


def test_single_string(daa, caa) -> None:  # noqa: F811
    assert_eq(daa["analysis"], caa["analysis"])


def test_list_with_ints_raise(daa) -> None:  # noqa: F811
    with pytest.raises(NotImplementedError, match="Lists containing integers"):
        assert daa[[1, 2]]


def test_single_int(daa, caa) -> None:  # noqa: F811
    total = len(daa)
    for i in range(total):
        assert_eq(daa["analysis"]["x1"][i], caa["analysis"]["x1"][i])
        assert_eq(daa["analysis"]["y2"][-i], caa["analysis"]["y2"][-i])
    for i in range(total):
        caa[i].tolist() == daa[i].compute().tolist()
        caa["analysis"][i].tolist() == daa["analysis"][i].compute().tolist()


def test_single_ellipsis(daa, caa) -> None:  # noqa: F811
    assert_eq(daa[...], caa[...])


def test_empty_slice(daa, caa) -> None:  # noqa: F811
    assert_eq(daa[:], caa[:])


def test_record_getitem(daa, caa) -> None:  # noqa: F811
    assert daa[0].compute().to_list() == caa[0].to_list()
    assert daa["analysis"]["x1"][0][0].compute() == caa["analysis"]["x1"][0][0]
    assert daa[0]["analysis"].compute().to_list() == caa[0]["analysis"].to_list()
    assert daa["analysis"][0].compute().to_list() == caa["analysis"][0].to_list()
    assert daa["analysis"][0].x1.compute().to_list() == caa["analysis"][0].x1.to_list()
    assert daa[0]["analysis"]["x1"][0].compute() == caa[0]["analysis", "x1"][0]
