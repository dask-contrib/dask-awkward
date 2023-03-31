import awkward as ak

from dask_awkward.lib.unproject_layout import unproject_layout


def _compare_values(index, projected, x, unprojected):
    if isinstance(x, list):
        for i, xi in enumerate(x):
            _compare_values(index + (i,), projected, xi, unprojected)

    elif isinstance(x, dict):
        for f, xf in x.items():
            _compare_values(index + (f,), projected, xf, unprojected)

    else:
        assert x == unprojected[index], f"{projected}\n\nat {index}"


def compare_values(projected, unprojected):
    p = ak.to_list(projected)
    _compare_values((), p, p, unprojected)


def test_EmptyArray():
    form = ak.contents.RecordArray(
        [ak.from_iter([], highlevel=False), ak.from_iter([], highlevel=False)],
        ["x", "y"],
        0,
    ).form
    projected = ak.contents.RecordArray(
        [ak.from_iter([], highlevel=False)],
        ["x"],
        0,
    )
    unprojected = unproject_layout(form, projected)
    compare_values(projected, unprojected)


def test_NumpyArray():
    form = ak.from_iter(
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], highlevel=False
    ).form
    projected = ak.from_iter([{"x": 1}, {"x": 2}, {"x": 3}], highlevel=False)
    unprojected = unproject_layout(form, projected)
    compare_values(projected, unprojected)


def test_ListOffsetArray():
    form = ak.from_iter(
        [{"x": 1, "y": []}, {"x": 2, "y": [1]}, {"x": 3, "y": [1, 2]}], highlevel=False
    ).form
    projected = ak.from_iter([{"x": 1}, {"x": 2}, {"x": 3}], highlevel=False)
    unprojected = unproject_layout(form, projected)
    compare_values(projected, unprojected)


def test_string():
    form = ak.from_iter(
        [{"x": 1, "y": "one"}, {"x": 2, "y": "two"}, {"x": 3, "y": "three"}],
        highlevel=False,
    ).form
    projected = ak.from_iter([{"x": 1}, {"x": 2}, {"x": 3}], highlevel=False)
    unprojected = unproject_layout(form, projected)
    compare_values(projected, unprojected)


def test_RecordArray():
    form = ak.from_iter(
        [
            {"in": {"x": 1, "y": []}},
            {"in": {"x": 2, "y": [1]}},
            {"in": {"x": 3, "y": [1, 2]}},
        ],
        highlevel=False,
    ).form
    projected = ak.from_iter(
        [{"in": {"x": 1}}, {"in": {"x": 2}}, {"in": {"x": 3}}], highlevel=False
    )
    unprojected = unproject_layout(form, projected)
    compare_values(projected, unprojected)


def test_UnionArray():
    form = ak.from_iter(
        [{"x": 1, "y": 1}, {"x": 2, "y": "two"}, {"x": 3, "y": 3}], highlevel=False
    ).form
    projected = ak.from_iter([{"x": 1}, {"x": 2}, {"x": 3}], highlevel=False)
    unprojected = unproject_layout(form, projected)
    compare_values(projected, unprojected)
