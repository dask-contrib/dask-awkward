import awkward as ak  # noqa
import dask_awkward.core as dakc  # noqa
import dask_awkward as dak


daa = dak.from_json(["data/simple1.json", "data/simple2.json", "data/simple3.json"])
aa = daa.compute()
a1 = ak.from_json("data/simple1.json")
a2 = ak.from_json("data/simple2.json")
a3 = ak.from_json("data/simple3.json")
