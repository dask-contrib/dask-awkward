import awkward._v2 as ak

import dask_awkward as dak


def test_fields(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    # records fields same as array of records fields
    assert dak.fields(daa[0].points) == dak.fields(daa.points)
    # computed is same as collection
    assert dak.fields(daa) == ak.fields(daa.compute())
    daa.reset_meta()
    # removed meta gives None fields
    assert dak.fields(daa) == []
