import awkward as ak
import dask.sizeof


def test_sizeof():
    x = ak.Array([[1, 2, 3], [4], [5, 6]])

    assert dask.sizeof.sizeof(x) == x.nbytes
