from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq
from dask_awkward.utils import IncompatiblePartitions


@pytest.mark.parametrize("axis", [0, 1])
def test_concatenate_simple(daa, caa, axis):
    assert_eq(
        ak.concatenate([caa.points.x, caa.points.y], axis=axis),
        dak.concatenate([daa.points.x, daa.points.y], axis=axis),
    )


def test_concatenate_axis_0_logical_same(daa):
    result = dak.concatenate([daa, daa], axis=0)
    print(daa.form)
    buffers_report = dak.report_necessary_buffers(result.points.x)
    assert len(buffers_report) == 1

    buffers = next(iter(buffers_report.values()))

    assert buffers.data_and_shape == frozenset(
        ["@.points.content.x-data", "@.points-offsets"]
    )
    assert buffers.shape_only == frozenset()


def test_concatenate_axis_0_logical_different(daa):
    import dask.config

    with dask.config.set(
        {"awkward.optimization.on-fail": "raise", "awkward.raise-failed-meta": True}
    ):
        empty_form = ak.forms.from_dict(
            {
                "class": "RecordArray",
                "fields": ["points"],
                "contents": [
                    {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": {
                            "class": "RecordArray",
                            "fields": ["x", "y"],
                            "contents": ["int64", "float64"],
                        },
                    }
                ],
            }
        )
        empty_array = ak.Array(empty_form.length_zero_array(highlevel=False))
        empty_dak_array = dak.from_awkward(empty_array, npartitions=1)
        result = dak.concatenate([daa, empty_dak_array], axis=0)

        buffers_report = dak.report_necessary_buffers(result.points.x)
        assert len(buffers_report) == 1

        buffers = next(iter(buffers_report.values()))
        assert buffers.data_and_shape == frozenset(
            ["@.points.content.x-data", "@.points.content.y-data", "@.points-offsets"]
        )
        assert buffers.shape_only == frozenset()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_concatenate_more_axes(axis):
    a = [[[1, 2, 3], [], [100, 101], [12, 13]], [[1, 2, 3], [], [100, 101], [12, 13]]]
    b = [
        [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
        [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
    ]
    one = dak.from_lists([a, a])
    two = dak.from_lists([b, b])
    c = dak.concatenate([one, two], axis=axis)
    aa = ak.concatenate([a, a])
    bb = ak.concatenate([b, b])
    cc = ak.concatenate([aa, bb], axis=axis)
    assert_eq(c, cc)

    if axis > 0:
        # add an additional entry to a to trigger bad divisions
        a = [
            [[1, 2, 3], [], [100, 101], [12, 13]],
            [[1, 2, 3], [], [100, 101], [12, 13]],
            [],
        ]
        b = [
            [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
            [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]],
        ]
        b = [[4, 5], [10, 11, 12, 13], [102], [9, 9, 9]]
        one = dak.from_lists([a, a])
        two = dak.from_lists([b, b])
        with pytest.raises(
            IncompatiblePartitions,
            match="The inputs to concatenate are incompatibly partitioned",
        ):
            c = dak.concatenate([one, two], axis=axis)
