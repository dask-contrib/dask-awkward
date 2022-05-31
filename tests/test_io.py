from __future__ import annotations

from pathlib import Path

import awkward._v2 as ak
import fsspec
import pytest
from dask.array.utils import assert_eq as da_assert_eq

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import dask_awkward as dak
import dask_awkward.testutils as daktu
from dask_awkward.testutils import assert_eq


def test_force_by_lines_meta(line_delim_records_file: str) -> None:
    daa1 = dak.from_json(
        [line_delim_records_file] * 5,
        derive_meta_kwargs={"force_by_lines": True},
    )
    daa2 = dak.from_json([line_delim_records_file, line_delim_records_file])
    assert daa1._meta is not None
    assert daa2._meta is not None
    f1 = daa1._meta.layout.form
    f2 = daa2._meta.layout.form
    assert f1 == f2


def test_derive_json_meta_trigger_warning(line_delim_records_file: str) -> None:
    with pytest.warns(UserWarning):
        dak.from_json(line_delim_records_file, derive_meta_kwargs={"bytechunks": 64})


def test_json_one_obj_per_file(single_record_file: str) -> None:
    daa = dak.from_json(
        [single_record_file] * 5,
        one_obj_per_file=True,
    )
    with fsspec.open(single_record_file, "r") as f:
        content = json.load(f)
    caa = ak.from_iter([content] * 5)
    assert_eq(daa, caa)


def test_json_delim_defined(line_delim_records_file: str) -> None:
    source = [line_delim_records_file] * 6
    daa = dak.from_json(source, delimiter=b"\n")

    concretes = []
    for s in source:
        with open(s) as f:
            for line in f:
                concretes.append(json.loads(line))
    caa = ak.from_iter(concretes)

    assert_eq(
        daa["analysis"][["x1", "z2"]],
        caa["analysis"][["x1", "z2"]],
    )


def test_to_and_from_dask_array(line_delim_records_file: str) -> None:
    daa = dak.from_json([line_delim_records_file] * 3)
    computed = ak.flatten(daa.analysis.x1.compute())
    x1 = dak.flatten(daa.analysis.x1)
    daskarr = dak.to_dask_array(x1)
    da_assert_eq(daskarr, computed.to_numpy())

    back_to_dak = dak.from_dask_array(daskarr)
    assert_eq(back_to_dak, computed)


def test_from_dask_array() -> None:
    import dask.array as da

    darr = da.ones(100, chunks=25)
    daa = dak.from_dask_array(darr)
    assert daa.known_divisions
    assert_eq(daa, ak.from_numpy(darr.compute()))


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_to_and_from_delayed(
    line_delim_records_file: str,
    optimize_graph: bool,
) -> None:
    daa = dak.from_json([line_delim_records_file] * 3)
    daa = daa[dak.num(daa.analysis.x1, axis=1) > 2]
    delayeds = daa.to_delayed(optimize_graph=optimize_graph)
    daa2 = dak.from_delayed(delayeds)
    assert_eq(daa, daa2)
    for i in range(daa.npartitions):
        assert_eq(daa.partitions[i], delayeds[i].compute())

    daa2 = dak.from_delayed(delayeds, divisions=daa.divisions)
    assert_eq(daa, daa2)

    with pytest.raises(ValueError, match="divisions must be a tuple of length"):
        dak.from_delayed(delayeds, divisions=(1, 5, 7, 9, 11))


def test_from_map_with_args_kwargs() -> None:
    import dask.core

    def f(a, b, c, n, pad_zero=False):
        if pad_zero:
            return ak.Array([a * n, b * n, c * n, 0])
        else:
            return ak.Array([a * n, b * n, c * n])

    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    n = 3

    # dask version
    x = dak.from_map(f, a, b, c, args=(n,))

    # concrete version
    y = list(zip(a, b, c))
    y = dask.core.flatten(list(map(list, y)))
    y = map(lambda x: x * n, y)
    y = ak.from_iter(y)

    assert_eq(x, y)

    # dask version
    x = dak.from_map(f, a, b, c, args=(n,), pad_zero=True)

    # concrete version
    y = list(zip(a, b, c, [0, 0, 0]))
    y = dask.core.flatten(list(map(list, y)))
    y = map(lambda x: x * n, y)
    y = ak.from_iter(y)

    assert_eq(x, y)


def test_from_map_pack_single_iterable(line_delim_records_file) -> None:
    def g(fname, c=1):
        return ak.from_json(Path(fname).read_text()).analysis.x1 * c

    n = 3
    c = 2

    fmt = "{t}\n" * n
    jsontext = fmt.format(t=Path(line_delim_records_file).read_text())
    x = dak.from_map(g, [line_delim_records_file] * n, c=c)
    y = ak.from_json(jsontext).analysis.x1 * c
    assert_eq(x, y)


def test_from_map_enumerate() -> None:
    def f(t):
        i = t[0]
        x = t[1]
        return ak.Array([{"x": (i + 1) * x}])

    x = [[1, 2, 3], [4, 5, 6]]

    a1 = dak.from_map(f, enumerate(x))
    a2 = ak.Array([{"x": [1, 2, 3]}, {"x": [4, 5, 6, 4, 5, 6]}])
    assert_eq(a1, a2)


def test_from_map_exceptions() -> None:
    def f(a, b):
        return ak.Array([a, b])

    with pytest.raises(ValueError, match="same length"):
        dak.from_map(f, [1, 2], [3, 4, 5])

    with pytest.raises(ValueError, match="must be `callable`"):
        dak.from_map(5, [1], [2])  # type: ignore

    with pytest.raises(ValueError, match="must be Iterable"):
        dak.from_map(f, 1, [1, 2])  # type: ignore

    with pytest.raises(ValueError, match="non-zero length"):
        dak.from_map(f, [], [], [])

    with pytest.raises(ValueError, match="at least one Iterable input"):
        dak.from_map(f, args=(5,))


def test_from_lists() -> None:
    daa = dak.from_lists([daktu.A1, daktu.A2])
    caa = ak.Array(daktu.A1 + daktu.A2)
    assert_eq(daa, caa)
