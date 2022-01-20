from __future__ import annotations

import operator

import pytest

import dask_awkward as dak
import dask_awkward.core as dakc

from .helpers import assert_eq, caa, daa, line_delim_records_file  # noqa: F401


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


def test_layered_string(daa, caa) -> None:  # noqa: F811
    assert_eq(daa["analysis", "x1"], caa["analysis", "x1"])
    assert_eq(daa["analysis", "x1"], caa["analysis"]["x1"])
    assert_eq(caa["analysis", "x1"], daa["analysis"]["x1"])
    assert_eq(daa["analysis", ["x1", "t1"]], caa["analysis", ["x1", "t1"]])


def test_list_with_ints_raise(daa) -> None:  # noqa: F811
    with pytest.raises(NotImplementedError, match="Lists containing integers"):
        assert daa[[1, 2]]


def test_single_int(daa, caa) -> None:  # noqa: F811
    total = len(daa)
    for i in range(total):
        assert_eq(daa["analysis"]["x1"][i], caa["analysis"]["x1"][i])
        assert_eq(daa["analysis"]["y2"][-i], caa["analysis"]["y2"][-i])
        assert_eq(daa[0, "analysis", "x1"], caa[0, "analysis", "x1"])
    for i in range(total):
        caa[i].tolist() == daa[i].compute().tolist()
        caa["analysis"][i].tolist() == daa["analysis"][i].compute().tolist()


def test_single_ellipsis(daa, caa) -> None:  # noqa: F811
    assert_eq(daa[...], caa[...])


def test_empty_slice(daa, caa) -> None:  # noqa: F811
    assert_eq(daa[:], caa[:])
    assert_eq(daa[:, "analysis"], caa[:, "analysis"])


def test_record_getitem(daa, caa) -> None:  # noqa: F811
    assert daa[0].compute().to_list() == caa[0].to_list()
    assert daa["analysis"]["x1"][0][0].compute() == caa["analysis"]["x1"][0][0]
    assert daa[0]["analysis"].compute().to_list() == caa[0]["analysis"].to_list()
    assert daa["analysis"][0].compute().to_list() == caa["analysis"][0].to_list()
    assert daa["analysis"][0].x1.compute().to_list() == caa["analysis"][0].x1.to_list()
    assert daa[0]["analysis"]["x1"][0].compute() == caa[0]["analysis", "x1"][0]


@pytest.mark.parametrize("op", [operator.gt, operator.ge, operator.le, operator.lt])
def test_boolean_array(line_delim_records_file, op) -> None:  # noqa: F811
    daa = dak.from_json([line_delim_records_file] * 3)  # noqa
    caa = daa.compute()  # noqa
    dx1 = daa.analysis.x1
    cx1 = caa.analysis.x1
    dx1s = op(dx1, 2)
    cx1s = op(cx1, 2)
    dx1_p = dx1[dx1s]
    cx1_p = cx1[cx1s]
    assert_eq(dx1_p, cx1_p)
