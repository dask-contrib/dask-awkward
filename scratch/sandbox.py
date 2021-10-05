import awkward as ak  # noqa
import dask_awkward.core as dakc  # noqa
import dask_awkward as dak


daa = dak.from_json(["data/arr1.json", "data/arr2.json", "data/arr3.json"])
aa = daa.compute()
a1 = ak.from_json("data/arr1.json")
a2 = ak.from_json("data/arr2.json")
a3 = ak.from_json("data/arr3.json")
