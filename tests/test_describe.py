import awkward._v2 as ak

import dask_awkward as dak


def test_fields(line_delim_records_file) -> None:
    daa = dak.from_json(line_delim_records_file, blocksize=340)
    # records fields same as array of records fields
    assert dak.fields(daa[0].analysis) == dak.fields(daa.analysis)
    # computed is same as collection
    assert dak.fields(daa) == ak.fields(daa.compute())
    daa.reset_meta()
    # removed meta gives None fields
    assert dak.fields(daa) == []
