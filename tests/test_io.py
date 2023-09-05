from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from dask.array.utils import assert_eq as da_assert_eq
from dask.delayed import delayed
from fsspec.core import get_fs_token_paths
from numpy.typing import DTypeLike

import dask_awkward as dak
from dask_awkward.lib.io.io import _bytes_with_sample
from dask_awkward.lib.testutils import assert_eq


def test_to_and_from_dask_array(daa: dak.Array) -> None:
    daa = dak.from_awkward(daa.compute(), npartitions=3)

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
    assert set(list(dak.necessary_columns(c).values())[0]) == {"b", "a"}

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

    daa = dak.from_awkward(daa.compute(), npartitions=4)

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


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_to_dataframe(daa: dak.Array, caa: ak.Array, optimize_graph: bool) -> None:
    pytest.importorskip("pandas")

    from dask.dataframe.utils import assert_eq

    daa = daa["points", ["x", "y"]]
    caa = caa["points", ["x", "y"]]

    dd = dak.to_dataframe(daa, optimize_graph=optimize_graph)
    df = ak.to_dataframe(caa)

    assert_eq(dd, df, check_index=False)


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_to_dataframe_str(
    daa_str: dak.Array, caa_str: ak.Array, optimize_graph: bool
) -> None:
    pytest.importorskip("pandas")

    from dask.dataframe.utils import assert_eq

    daa = daa_str["points", ["x", "y"]]
    caa = caa_str["points", ["x", "y"]]

    dd = dak.to_dataframe(daa, optimize_graph=optimize_graph)
    df = ak.to_dataframe(caa)

    assert_eq(dd, df, check_index=False)


def test_from_awkward_empty_array(daa) -> None:
    # no form
    c1 = ak.Array([])
    assert len(c1) == 0
    a1 = dak.from_awkward(c1, npartitions=1)
    assert_eq(a1, c1)
    assert len(a1) == 0

    # with a form
    c2 = ak.Array(daa.layout.form.length_zero_array(highlevel=False))
    assert len(c2) == 0
    a2 = dak.from_awkward(c2, npartitions=1)
    assert len(a2) == 0
    daa.layout.form == a2.layout.form


@pytest.mark.parametrize("sample", [False, 28])
@pytest.mark.parametrize("not_zero", [True, False])
def test_bytes_with_sample(
    sample: int | str | bool,
    not_zero: bool,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    tmppath = tmp_path_factory.mktemp("bytes_with_sample")

    lines1 = b"\n".join([b"a" * 127, b"b" * 127, b"c" * 127, b"d" * 128])
    lines2 = b"\n".join([b"e" * 127, b"f" * 127, b"g" * 127, b"h" * 128])

    with open(tmppath / "file1.txt", "wb") as f:
        f.write(lines1)
    with open(tmppath / "file2.txt", "wb") as f:
        f.write(lines2)

    fs, _, paths = get_fs_token_paths(str(tmppath / "*.txt"))

    bytes_instructions, sample_bytes = _bytes_with_sample(
        fs=fs,
        paths=paths,
        compression="infer",
        delimiter=b"\n",
        not_zero=not_zero,
        blocksize=256,
        sample=sample,
    )

    assert len(bytes_instructions) == 2

    assert len(bytes_instructions[0]) == 2
    assert len(bytes_instructions[1]) == 2

    # seek until next delimieter from 1 byte forward if not_zero is True
    assert bytes_instructions[0][0].offset == 0 if not not_zero else 1
    assert bytes_instructions[0][0].length == 256 if not not_zero else 255

    assert bytes_instructions[1][1].offset == 256
    assert bytes_instructions[1][1].length == 256

    if not sample:
        assert sample_bytes == b""
    else:
        assert len(sample_bytes) == 127
