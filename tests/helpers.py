import pathlib

import awkward as ak

from dask_awkward.io import from_json

this_path = pathlib.Path(__file__).parent.resolve()
records_file = str(this_path / "data" / "records.json")


def load_records_lazy(blocksize=2048):
    return from_json(records_file, blocksize=blocksize)


def load_records_eager():
    return ak.from_json(records_file)
