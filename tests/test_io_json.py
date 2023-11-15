from __future__ import annotations

import json
import os
from pathlib import Path

import awkward as ak
import dask
import fsspec
import pytest

import dask_awkward as dak
from dask_awkward.lib.core import Array
from dask_awkward.lib.optimize import optimize as dak_optimize
from dask_awkward.lib.testutils import assert_eq

data1 = r"""{"name":"Bob","team":"tigers","goals":[0,0,0,1,2,0,1]}
{"name":"Alice","team":"bears","goals":[null]}
{"name":"Jack","team":"bears","goals":[0,0,0,0,0,0,0,0,1]}
{"name":"Jill","team":"bears","goals":[3,0,2]}
{"name":"Ted","team":"tigers","goals":null}
"""

data2 = r"""{"name":"Ellen","team":"tigers","goals":[1,0,0,0,2,0,1]}
{"name":"Dan","team":"bears","goals":[0,0,3,1,0,2,0,0]}
{"name":"Brad","team":"bears","goals":[null]}
{"name":"Nancy","team":"tigers","goals":[0,0,1,1,1,1,0]}
{"name":"Lance","team":"bears","goals":[1,1,1,1,1]}
"""

data3 = r"""{"name":"Sara","team":"tigers","goals":[0,1,0,2,0,3]}
{"name":"Ryan","team":"tigers","goals":[1,2,3,0,0,0,0]}
"""


@pytest.fixture(scope="session")
def json_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    toplevel = tmp_path_factory.mktemp("json_data")
    for i, datum in enumerate((data1, data2, data3)):
        with open(toplevel / f"file{i}.json", "w") as f:
            print(datum, file=f)
    return toplevel


@pytest.fixture(scope="session")
def concrete_data(json_data_dir: Path) -> ak.Array:
    array = ak.concatenate(
        [
            ak.from_json(json_data_dir / "file0.json", line_delimited=True),
            ak.from_json(json_data_dir / "file1.json", line_delimited=True),
            ak.from_json(json_data_dir / "file2.json", line_delimited=True),
        ],
    )

    return array


def test_json_sanity(json_data_dir: Path, concrete_data: ak.Array) -> None:
    source = os.path.join(str(json_data_dir))
    ds = dak.from_json(source)
    assert not ds.known_divisions
    with pytest.raises(
        NotImplementedError,
        match=(
            "Cannot determine length of collection with unknown partition sizes without executing the graph.\\n"
            "Use `dask_awkward.num\\(\\.\\.\\., axis=0\\)` if you want a lazy Scalar of the length.\\n"
            "If you want to eagerly compute the partition sizes to have the ability to call `len` on the collection"
            ", use `\\.eager_compute_divisions\\(\\)` on the collection."
        ),
    ):
        assert ds
    ds.eager_compute_divisions()
    assert ds

    assert_eq(ds, concrete_data)


def input_layer_array_partition0(collection: Array) -> ak.Array:
    """Get first partition concrete array after the input layer.

    Parameteters
    ------------
    collection : dask_awkward.Array
        dask-awkward Array collection of interest

    Returns
    -------
    ak.Array
        Concrete awkward array representing the first partition
        immediately after the input layer.

    """
    with dask.config.set({"awkward.optimization.which": ["columns"]}):
        optimized_hlg = dak_optimize(collection.dask, [])
        layers = list(optimized_hlg.layers)  # type: ignore
        layer_name = [name for name in layers if name.startswith("from-json")][0]
        sgc, arg = optimized_hlg[(layer_name, 0)]
        array = sgc.dsk[layer_name][0](arg)
    return array


def test_json_column_projection_off(json_data_dir: Path) -> None:
    source = os.path.join(str(json_data_dir), "*.json")
    ds = dak.from_json(source)
    fields_to_keep = ["name", "goals"]

    ds2 = ds[fields_to_keep]
    with dask.config.set({"awkward.optimization.columns-opt-formats": []}):
        array = input_layer_array_partition0(ds2)

    assert array.fields == ["name", "team", "goals"]


def test_json_column_projection1(json_data_dir: Path) -> None:
    source = os.path.join(str(json_data_dir), "*.json")
    ds = dak.from_json(source)
    fields_to_keep = ["name", "goals"]
    ds2 = ds[fields_to_keep]
    with dask.config.set({"awkward.optimization.columns-opt-formats": ["json"]}):
        array = input_layer_array_partition0(ds2)

    assert array.fields == fields_to_keep


def test_json_column_projection2(json_data_dir: Path) -> None:
    source = os.path.join(str(json_data_dir), "*.json")
    ds = dak.from_json(source)
    # grab name and goals but then only use goals!
    ds2 = dak.max(ds[["name", "goals"]].goals, axis=1)
    with dask.config.set({"awkward.optimization.columns-opt-formats": ["json"]}):
        array = input_layer_array_partition0(ds2)

    assert array.fields == ["goals"]


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


@pytest.mark.parametrize("compression", [None, "xz"])
def test_json_one_obj_per_file(
    single_record_file: str,
    tmp_path_factory: pytest.TempPathFactory,
    compression: str | None,
) -> None:
    if compression:
        d = tmp_path_factory.mktemp("sopf")
        p = str(d / f"file.json.{compression}")
        with fsspec.open(single_record_file, "rt") as f:
            text = f.read()
        with fsspec.open(p, mode="w", compression=compression) as f:
            print(text, file=f)
        file_of_interest = p
    else:
        file_of_interest = single_record_file

    daa = dak.from_json([file_of_interest] * 5, line_delimited=False)
    single_record = ak.from_json(Path(single_record_file), line_delimited=False)
    single_entry_array = ak.Array([single_record])
    caa = ak.concatenate([single_entry_array] * 5)
    assert_eq(daa, caa)


@pytest.mark.parametrize("compression", ["xz", "gzip", "zip"])
def test_to_and_from_json(
    daa: Array,
    tmp_path_factory: pytest.TempPathFactory,
    compression: str,
) -> None:
    tdir = str(tmp_path_factory.mktemp("json_temp"))

    p1 = os.path.join(tdir, "z", "z")

    dak.to_json(daa, p1)
    paths = list((Path(tdir) / "z" / "z").glob("part*.json"))
    assert len(paths) == daa.npartitions
    arrays = ak.concatenate([ak.from_json(p, line_delimited=True) for p in paths])
    assert_eq(daa, arrays)

    x = dak.from_json(os.path.join(p1, "*.json"))
    assert_eq(arrays, x)

    s = dak.to_json(
        array=daa,
        path=tdir,
        compression=compression,
        compute=False,
    )
    assert isinstance(s, dak.Scalar)
    s.compute()
    suffix = "gz" if compression == "gzip" else compression
    r = dak.from_json(os.path.join(tdir, f"*.json.{suffix}"))
    assert_eq(x, r)
