from __future__ import annotations

import pickle

import awkward as ak
import dask.config

import dask_awkward as dak


def test_pickle_ak_array():
    buffers = []
    array = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])[[0, 2]]
    next_array = pickle.loads(
        pickle.dumps(array, protocol=5, buffer_callback=buffers.append), buffers=buffers
    )
    assert ak.almost_equal(array, next_array)
    assert array.layout.form == next_array.layout.form


def identity(x):
    return x


def test_worker_transfer():
    array = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])[[0, 2]]
    darray = dak.from_awkward(array, 1)
    darray_result = darray.map_partitions(identity)

    first, second = dask.compute(darray, darray_result)

    assert ak.almost_equal(first, second)
    assert first.layout.form == second.layout.form
