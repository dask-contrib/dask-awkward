from functools import partial

import awkward as ak
import numpy as np
import pytest

import dask_awkward as dak


def test_mapfilter_single_return():
    ak_array = ak.zip({"foo": [1, 2, 3, 4], "bar": [1, 1, 1, 1]})
    dak_array = dak.from_awkward(ak_array, 2)

    @dak.mapfilter
    def fun(x):
        y = x.foo + 1
        return y

    assert ak.all(
        fun(dak_array).compute()
        == dak.map_partitions(fun.wrapped_fn, dak_array).compute()
    )


def test_mapfilter_multiple_return():
    ak_array = ak.zip({"foo": [1, 2, 3, 4], "bar": [1, 1, 1, 1]})
    dak_array = dak.from_awkward(ak_array, 2)

    class some: ...

    @dak.mapfilter
    def fun(x):
        y = x.foo + 1
        return y, (np.sum(y),), some(), ak.Array(np.ones(4))

    y, y_sum, something, arr = fun(dak_array)

    assert ak.all(y.compute() == ak_array.foo + 1)
    assert np.all(y_sum.compute() == [np.array(5), np.array(9)])
    something = something.compute()
    assert len(something) == 2
    assert all(isinstance(s, some) for s in something)
    array = arr.compute()
    assert len(array) == 8
    assert array.ndim == 1
    assert ak.all(array == ak.Array(np.ones(8)))


def test_mapfilter_needs_outlike():
    ak_array = ak.zip({"pt": [10, 20, 30, 40], "eta": [1, 1, 1, 1]})
    dak_array = dak.from_awkward(ak_array, 2)

    def untraceable_fun(muons):
        # a non-traceable computation for ak.typetracer
        # which needs "pt" column from muons and returns a 1-element array
        pt = ak.to_numpy(muons.pt)
        return ak.Array([np.sum(pt)])

    # first check that the function is not traceable
    with pytest.raises(TypeError):
        dak.map_partitions(untraceable_fun, dak_array)

    # now check that the necessary columns are reported correctly
    wrap = partial(dak.mapfilter, needs={"muons": ["pt"]}, out_like=ak.Array([0.0]))
    out = wrap(untraceable_fun)(dak_array)
    cols = next(iter(dak.report_necessary_columns(out).values()))
    assert cols == {"pt"}
