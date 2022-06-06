from __future__ import annotations

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import awkward._v2 as ak
import fsspec
import pytest

import dask_awkward as dak
import dask_awkward.testutils as daktu


@pytest.fixture(scope="session")
def single_record_file(tmpdir_factory) -> str:
    fn = tmpdir_factory.mktemp("data").join("single_record.json")
    record = {"record": [1, 2, 3]}
    with fsspec.open(fn, "w") as f:
        print(json.dumps(record), file=f)
    return fn


@pytest.fixture(scope="session")
def ndjson_points1(tmpdir_factory) -> str:
    array = daktu.awkward_xy_points()
    fn = tmpdir_factory.mktemp("data").join("points_ndjson1.json")
    with fsspec.open(fn, "w") as f:
        for entry in array.tolist():
            print(json.dumps({"points": entry}), file=f)
    return fn


@pytest.fixture(scope="session")
def ndjson_points2(tmpdir_factory) -> str:
    array = daktu.awkward_xy_points()
    fn = tmpdir_factory.mktemp("data").join("points_ndjson2.json")
    with fsspec.open(fn, "w") as f:
        for entry in array.tolist():
            print(json.dumps({"points": entry}), file=f)
    return fn


@pytest.fixture(scope="session")
def ndjson_points_file(ndjson_points1: str) -> str:
    return ndjson_points1


@pytest.fixture(scope="session")
def daa(ndjson_points1: str) -> dak.Array:
    return dak.from_json([ndjson_points1] * 3)


@pytest.fixture(scope="session")
def caa(ndjson_points1: str) -> ak.Array:
    with open(ndjson_points1) as f:
        lines = [json.loads(line) for line in f]
    return ak.Array(lines * 3)


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
def L1() -> list:
    return [
        [{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4.0, "y": 4.4}, {"x": 5.0, "y": 5.5}],
        [{"x": 6.0, "y": 6.6}],
        [{"x": 7.0, "y": 7.7}, {"x": 8.0, "y": 8.8}, {"x": 9, "y": 9.9}],
    ]


@pytest.fixture(scope="session")
def L2() -> list:
    return [
        [{"x": 0.9, "y": 1.0}, {"x": 2.0, "y": 2.2}, {"x": 2.9, "y": 3.0}],
        [],
        [{"x": 3.9, "y": 4.0}, {"x": 5.0, "y": 5.5}],
        [{"x": 5.9, "y": 6.0}],
        [{"x": 6.9, "y": 7.0}, {"x": 8.0, "y": 8.8}, {"x": 8.9, "y": 9.0}],
    ]


@pytest.fixture(scope="session")
def L3() -> list:
    return [
        [{"x": 1.9, "y": 9.0}, {"x": 2.0, "y": 8.2}, {"x": 9.9, "y": 9.0}],
        [],
        [{"x": 1.9, "y": 8.0}, {"x": 4.0, "y": 6.5}],
        [{"x": 1.9, "y": 7.0}],
        [{"x": 1.9, "y": 6.0}, {"x": 6.0, "y": 4.8}, {"x": 9.9, "y": 9.0}],
    ]
