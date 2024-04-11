from typing import Any

import awkward as ak
import pytest

import dask_awkward as dak


@pytest.mark.parametrize("quak", [ak, dak])
def test_fields(ndjson_points_file: str, quak: Any) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    # records fields same as array of records fields
    assert quak.fields(daa[0].points) == quak.fields(daa.points)
    # computed is same as collection
    assert quak.fields(daa) == ak.fields(daa.compute())
    daa.reset_meta()
    # removed meta gives None fields
    assert quak.fields(daa) == []


@pytest.mark.parametrize("quak", [ak, dak])
def test_backend(ndjson_points_file: str, quak: Any) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert quak.backend(daa) == "typetracer"
