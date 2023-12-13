from __future__ import annotations

import pickle

import awkward as ak
import numpy as np


def test_pickle_ak_array():
    buffers = []
    attrs = {"foo": "keep", "@foo": "drop"}
    behavior = {("some", "behavior"): "key"}
    array = ak.Array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], attrs=attrs, behavior=behavior
    )[[0, 2]]
    next_array = pickle.loads(
        pickle.dumps(array, protocol=5, buffer_callback=buffers.append), buffers=buffers
    )
    assert ak.almost_equal(array, next_array)
    assert array.layout.form == next_array.layout.form
    assert buffers
    assert np.shares_memory(
        array.layout.content.data,
        next_array.layout.content.data,
    )
    assert np.shares_memory(
        array.layout.starts.data,
        next_array.layout.starts.data,
    )
    assert np.shares_memory(
        array.layout.stops.data,
        next_array.layout.stops.data,
    )
    assert next_array.behavior == behavior
    assert next_array.attrs == {"foo": "keep"}


def test_pickle_ak_record():
    buffers = []
    attrs = {"foo": "keep", "@foo": "drop"}
    behavior = {("some", "behavior"): "key"}
    record = ak.zip(
        {"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]},
        depth_limit=1,
        behavior=behavior,
        attrs=attrs,
    )[2]
    next_record = pickle.loads(
        pickle.dumps(record, protocol=5, buffer_callback=buffers.append),
        buffers=buffers,
    )
    assert record.layout.at == next_record.layout.at
    assert next_record.behavior == behavior
    assert next_record.attrs == {"foo": "keep"}

    array = ak.Array(record.layout.array)
    next_array = ak.Array(next_record.layout.array)

    assert buffers
    assert np.shares_memory(
        array.layout.content("x").content.data,
        next_array.layout.content("x").content.data,
    )
    assert np.shares_memory(
        array.layout.content("x").starts.data,
        next_array.layout.content("x").starts.data,
    )
    assert np.shares_memory(
        array.layout.content("x").stops.data,
        next_array.layout.content("x").stops.data,
    )
