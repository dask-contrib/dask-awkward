from __future__ import annotations

try:
    import ujson as json
except ImportError:
    import json

from pathlib import Path

import awkward as ak
import fsspec
import pytest

import dask_awkward as dak
import dask_awkward.lib.testutils as daktu


@pytest.fixture(scope="session")
def single_record_file(tmpdir_factory: pytest.TempdirFactory) -> str:
    fname = Path(tmpdir_factory.mktemp("data")) / "single_record.json"
    record = [{"record": [1, 2, 3]}]
    with fsspec.open(fname, "w") as f:
        print(json.dumps(record), file=f)
    return str(fname)


@pytest.fixture(scope="session")
def ndjson_points1(tmpdir_factory: pytest.TempdirFactory) -> str:
    array = daktu.awkward_xy_points()
    fname = Path(tmpdir_factory.mktemp("data")) / "points_ndjson1.json"
    with fsspec.open(fname, "w") as f:
        for entry in array.tolist():
            print(json.dumps({"points": entry}), file=f)
    return str(fname)


@pytest.fixture(scope="session")
def ndjson_points1_str(tmpdir_factory: pytest.TempdirFactory) -> str:
    array = daktu.awkward_xy_points_str()
    fname = Path(tmpdir_factory.mktemp("data")) / "points_ndjson1.json"
    with fsspec.open(fname, "w") as f:
        for entry in array.tolist():
            print(json.dumps({"points": entry}), file=f)
    return str(fname)


@pytest.fixture(scope="session")
def ndjson_points2(tmpdir_factory: pytest.TempdirFactory) -> str:
    array = daktu.awkward_xy_points()
    fname = Path(tmpdir_factory.mktemp("data")) / "points_ndjson2.json"
    with fsspec.open(fname, "w") as f:
        for entry in array.tolist():
            print(json.dumps({"points": entry}), file=f)
    return str(fname)


@pytest.fixture(scope="session")
def ndjson_points_file(ndjson_points1: str) -> str:
    return ndjson_points1


@pytest.fixture(scope="session")
def ndjson_points_file_str(ndjson_points1_str: str) -> str:
    return ndjson_points1_str


@pytest.fixture(scope="session")
def daa(ndjson_points1: str) -> dak.Array:
    return dak.from_json([ndjson_points1] * 3)


@pytest.fixture(scope="session")
def daa_str(ndjson_points1_str: str) -> dak.Array:
    return dak.from_json([ndjson_points1_str] * 3)


@pytest.fixture(scope="session")
def caa(ndjson_points1: str) -> ak.Array:
    with open(ndjson_points1, "rb") as f:
        a = ak.from_json(f, line_delimited=True)
    return ak.concatenate([a, a, a])


@pytest.fixture(scope="session")
def caa_str(ndjson_points1_str: str) -> ak.Array:
    with open(ndjson_points1_str, "rb") as f:
        a = ak.from_json(f, line_delimited=True)
    return ak.concatenate([a, a, a])


@pytest.fixture(scope="session")
def daa_p1(ndjson_points1: str) -> dak.Array:
    return dak.from_json([ndjson_points1] * 3)


@pytest.fixture(scope="session")
def daa_p2(ndjson_points2: str) -> dak.Array:
    return dak.from_json([ndjson_points2] * 3)


@pytest.fixture(scope="session")
def caa_p1(ndjson_points1: str) -> ak.Array:
    with open(ndjson_points1) as f:
        lines = [json.loads(line) for line in f]
    return ak.Array(lines * 3)


@pytest.fixture(scope="session")
def caa_p2(ndjson_points2: str) -> ak.Array:
    with open(ndjson_points2) as f:
        lines = [json.loads(line) for line in f]
    return ak.Array(lines * 3)


@pytest.fixture(scope="session")
def L1() -> list[list[dict[str, float]]]:
    return [
        [{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4.0, "y": 4.4}, {"x": 5.0, "y": 5.5}],
        [{"x": 6.0, "y": 6.6}],
        [{"x": 7.0, "y": 7.7}, {"x": 8.0, "y": 8.8}, {"x": 9, "y": 9.9}],
    ]


@pytest.fixture(scope="session")
def L2() -> list[list[dict[str, float]]]:
    return [
        [{"x": 0.9, "y": 1.0}, {"x": 2.0, "y": 2.2}, {"x": 2.9, "y": 3.0}],
        [],
        [{"x": 3.9, "y": 4.0}, {"x": 5.0, "y": 5.5}],
        [{"x": 5.9, "y": 6.0}],
        [{"x": 6.9, "y": 7.0}, {"x": 8.0, "y": 8.8}, {"x": 8.9, "y": 9.0}],
    ]


@pytest.fixture(scope="session")
def L3() -> list[list[dict[str, float]]]:
    return [
        [{"x": 1.9, "y": 9.0}, {"x": 2.0, "y": 8.2}, {"x": 9.9, "y": 9.0}],
        [],
        [{"x": 1.9, "y": 8.0}, {"x": 4.0, "y": 6.5}],
        [{"x": 1.9, "y": 7.0}],
        [{"x": 1.9, "y": 6.0}, {"x": 6.0, "y": 4.8}, {"x": 9.9, "y": 9.0}],
    ]


@pytest.fixture(scope="session")
def L4() -> list[list[dict[str, float]] | None]:
    return [
        [{"x": 1.9, "y": 9.0}, {"x": 2.0, "y": 8.2}, {"x": 9.9, "y": 9.0}],
        None,
        [{"x": 1.9, "y": 8.0}, {"x": 4.0, "y": 6.5}],
        [{"x": 1.9, "y": 7.0}],
        [{"x": 1.9, "y": 6.0}, {"x": 6.0, "y": 4.8}, {"x": 9.9, "y": 9.0}],
    ]


@pytest.fixture(scope="session")
def caa_parquet(caa: ak.Array, tmpdir_factory: pytest.TempdirFactory) -> str:
    fname = tmpdir_factory.mktemp("parquet_data") / "caa.parquet"  # type: ignore
    ak.to_parquet(caa, str(fname), extensionarray=False)
    return str(fname)


@pytest.fixture(scope="session")
def unnamed_root_parquet_file(tmpdir_factory: pytest.TempdirFactory) -> str:
    from dask_awkward.lib.testutils import unnamed_root_ds

    fname = Path(tmpdir_factory.mktemp("unnamed_parquet_data")) / "file.parquet"
    ak.to_parquet(unnamed_root_ds(), str(fname), extensionarray=False, row_group_size=3)
    return str(fname)
