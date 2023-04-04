from __future__ import annotations

import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from dask.array.utils import assert_eq as da_assert_eq
from dask.delayed import delayed
from numpy.typing import DTypeLike

try:
    import ujson as json
except ImportError:
    import json

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq


def test_force_by_lines_meta(ndjson_points_file: str) -> None:
    daa1 = dak.from_json(
        [ndjson_points_file] * 5,
        derive_meta_kwargs={"force_by_lines": True},
    )
    daa2 = dak.from_json([ndjson_points_file] * 3)
    assert daa1._meta is not None
    assert daa2._meta is not None
    f1 = daa1._meta.layout.form
    f2 = daa2._meta.layout.form
    assert f1 == f2


def test_derive_json_meta_trigger_warning(ndjson_points_file: str) -> None:
    with pytest.warns(UserWarning):
        dak.from_json([ndjson_points_file], derive_meta_kwargs={"bytechunks": 64})


def test_json_one_obj_per_file(single_record_file: str) -> None:
    daa = dak.from_json(
        [single_record_file] * 5,
        one_obj_per_file=True,
    )
    caa = ak.concatenate([ak.from_json(Path(single_record_file))] * 5)
    assert_eq(daa, caa)


def test_json_delim_defined(ndjson_points_file: str) -> None:
    source = [ndjson_points_file] * 6
    daa = dak.from_json(source, delimiter=b"\n")

    concretes = []
    for s in source:
        with open(s) as f:
            for line in f:
                concretes.append(json.loads(line))
    caa = ak.from_iter(concretes)
    assert_eq(
        daa["points"][["x", "y"]],
        caa["points"][["x", "y"]],
    )


def test_json_sample_rows_true(ndjson_points_file: str) -> None:
    source = [ndjson_points_file] * 5

    daa = dak.from_json(
        source,
        derive_meta_kwargs={"force_by_lines": True, "sample_rows": 2},
    )

    concretes = []
    for s in source:
        with open(s) as f:
            for line in f:
                concretes.append(json.loads(line))
    caa = ak.from_iter(concretes)

    assert_eq(daa, caa)


def test_json_bytes_no_delim_defined(ndjson_points_file: str) -> None:
    source = [ndjson_points_file] * 7
    daa = dak.from_json(source, blocksize=650, delimiter=None)

    concretes = []
    for s in source:
        with open(s) as f:
            for line in f:
                concretes.append(json.loads(line))

    caa = ak.from_iter(concretes)
    assert_eq(daa, caa)


def test_to_and_from_dask_array(daa: dak.Array) -> None:
    computed = ak.flatten(daa.points.x.compute())
    x = dak.flatten(daa.points.x)
    daskarr = dak.to_dask_array(x)
    da_assert_eq(daskarr, computed.to_numpy())

    back_to_dak = dak.from_dask_array(daskarr)
    assert_eq(back_to_dak, computed)

    a = dak.from_lists([[1, 2, 3], [4, 5, 6]])
    a._meta = None
    with pytest.raises(ValueError, match="metadata required"):
        dak.to_dask_array(a)


def test_from_dask_array() -> None:
    import dask.array as da

    darr = da.ones(100, chunks=25)
    daa = dak.from_dask_array(darr)
    assert daa.known_divisions
    assert_eq(daa, ak.from_numpy(darr.compute()))


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_to_and_from_delayed(daa: dak.Array, optimize_graph: bool) -> None:
    daa1 = daa[dak.num(daa.points.x, axis=1) > 2]
    delayeds = daa1.to_delayed(optimize_graph=optimize_graph)
    daa2 = dak.from_delayed(delayeds)
    assert_eq(daa1, daa2)
    for i in range(daa1.npartitions):
        assert_eq(daa1.partitions[i], delayeds[i].compute())

    daa2 = dak.from_delayed(delayeds, divisions=daa.divisions)
    assert_eq(daa1, daa2)

    with pytest.raises(ValueError, match="divisions must be a tuple of length"):
        dak.from_delayed(delayeds, divisions=(1, 5, 7, 9, 11))


def test_delayed_single_node():
    a = ak.Array([1, 2, 3])
    b = delayed(a)
    c = dak.from_delayed(b)
    assert c.npartitions == 1
    assert c.divisions == (None, None)
    assert_eq(c, a)


def test_column_ordering(tmpdir):
    fn = f"{tmpdir}/temp.json"
    j = """{"a": [1, 2, 3], "b": [4, 5, 6]}"""
    with open(fn, "w") as f:
        f.write(j)
    b = dak.from_json(fn)

    def assert_1(arr):
        # after loading
        assert arr.fields == ["a", "b"]
        return arr

    def assert_2(arr):
        # after reorder
        assert arr.fields == ["b", "a"]
        return arr

    c = b.map_partitions(assert_1)[["b", "a"]].map_partitions(assert_2)

    # arbitrary order here
    assert set(list(dak.lib.necessary_columns(c).values())[0]) == {"b", "a"}

    arr = c.compute()
    assert arr.fields == ["b", "a"]  # output has required order


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
    y = dask.core.flatten(list(map(list, y)))  # type: ignore
    y = map(lambda x: x * n, y)  # type: ignore
    y = ak.from_iter(y)

    assert_eq(x, y)

    # dask version
    x = dak.from_map(f, a, b, c, args=(n,), pad_zero=True)

    # concrete version
    y = list(zip(a, b, c, [0, 0, 0]))  # type: ignore
    y = dask.core.flatten(list(map(list, y)))  # type: ignore
    y = map(lambda x: x * n, y)  # type: ignore
    y = ak.from_iter(y)

    assert_eq(x, y)


def test_from_map_pack_single_iterable(ndjson_points_file: str) -> None:
    def g(fname, c=1):
        return ak.from_json(Path(fname).read_text(), line_delimited=True).points.x * c

    n = 3
    c = 2

    fmt = "{t}\n" * n
    jsontext = fmt.format(t=Path(ndjson_points_file).read_text())
    x = dak.from_map(g, [ndjson_points_file] * n, c=c)
    y = ak.from_json(jsontext, line_delimited=True).points.x * c
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
        dak.from_map(5, [1], [2])

    with pytest.raises(ValueError, match="must be Iterable"):
        dak.from_map(f, 1, [1, 2])

    with pytest.raises(ValueError, match="non-zero length"):
        dak.from_map(f, [], [], [])

    with pytest.raises(ValueError, match="at least one Iterable input"):
        dak.from_map(f, args=(5,))


def test_from_map_raise_produces_tasks() -> None:
    def f(a, b):
        return ak.Array([a, b])

    with pytest.raises(ValueError, match="Multiple iterables not supported"):
        dak.from_map(f, [1, 2, 3], [4, 5, 6], produces_tasks=True)


def test_from_lists(caa_p1: ak.Array) -> None:
    listed = caa_p1.tolist()
    one = listed[:5]
    two = listed[5:]
    daa = dak.from_lists([one, two])
    caa = ak.Array(one + two)
    assert_eq(daa, caa)


def test_to_dask_array(daa: dak.Array, caa: dak.Array) -> None:
    from dask.array.utils import assert_eq as da_assert_eq

    da = dak.to_dask_array(dak.flatten(daa.points.x))
    ca = ak.to_numpy(ak.flatten(caa.points.x))
    da_assert_eq(da, ca)
    da = dak.flatten(daa.points.x).to_dask_array()
    da_assert_eq(da, ca)


def test_to_dask_array_multidim() -> None:
    c = ak.Array([[[1, 2, 3]], [[4, 5, 6]]])
    a = dak.from_awkward(c, npartitions=2)
    d = dak.to_dask_array(a)
    da_assert_eq(d, ak.to_numpy(c))


@pytest.mark.parametrize("clear_divs", [True, False])
@pytest.mark.parametrize("optimize_graph", [True, False])
@pytest.mark.parametrize("dtype", [np.uint32, np.float64])
def test_to_dask_array_divs(
    clear_divs: bool,
    optimize_graph: bool,
    dtype: DTypeLike,
) -> None:
    a = ak.Array(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype))
    b = dak.from_awkward(a, npartitions=3)
    if clear_divs:
        b.clear_divisions()
    c = b.to_dask_array(optimize_graph=optimize_graph)
    d = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype)
    da_assert_eq(c, d)


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_to_delayed(daa, caa, optimize_graph):
    delayeds = dak.to_delayed(daa.points, optimize_graph=optimize_graph)
    comped = ak.concatenate([d.compute() for d in delayeds])
    assert caa.points.tolist() == comped.tolist()
    delayeds = daa.points.to_delayed(optimize_graph=optimize_graph)
    comped = ak.concatenate([d.compute() for d in delayeds])
    assert caa.points.tolist() == comped.tolist()


def test_to_bag(daa, caa):
    a = daa.to_dask_bag()
    for comprec, entry in zip(a.compute(), caa):
        assert comprec.tolist() == entry.tolist()


def test_to_json(daa, tmpdir_factory):
    tdir = str(tmpdir_factory.mktemp("json_temp"))

    p1 = os.path.join(tdir, "z", "z")

    dak.to_json(daa, p1, compute=True, line_delimited=True)
    paths = list((Path(tdir) / "z" / "z").glob("part*.json"))
    assert len(paths) == daa.npartitions
    arrays = ak.concatenate([ak.from_json(p, line_delimited=True) for p in paths])
    assert_eq(daa, arrays)

    x = dak.from_json(os.path.join(p1, "*.json"))
    assert_eq(arrays, x)

    s = dak.to_json(
        daa,
        os.path.join(tdir, "file-*.json.gz"),
        compute=False,
        line_delimited=True,
    )
    s.compute()
    r = dak.from_json(os.path.join(tdir, "*.json.gz"))
    assert_eq(x, r)


def test_to_json_raise_filenotfound(
    daa: dak.Array,
    tmpdir_factory: pytest.TempdirFactory,
) -> None:
    p = tmpdir_factory.mktemp("onelevel")
    p2 = os.path.join(str(p), "two")
    with pytest.raises(FileNotFoundError, match="Parent directory for output file"):
        dak.to_json(
            daa,
            os.path.join(p2, "three", "four", "*.json"),
            compute=True,
            line_delimited=True,
        )


def test_to_dask_dataframe(daa: dak.Array, caa: ak.Array) -> None:
    pytest.importorskip("pandas")

    from dask.dataframe.utils import assert_eq

    dd = dak.to_dask_dataframe(daa)
    df = ak.to_dataframe(caa)

    assert_eq(dd, df, check_index=False)
