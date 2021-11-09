import awkward as ak  # noqa
import dask.array as da  # noqa
import dask.dataframe as dd  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa

import dask_awkward as dak  # noqa
import dask_awkward.core as dakc  # noqa
import dask_awkward.data as dakd


class Recs:
    def __init__(self):
        self.daa = dak.from_json(dakd.json_data(kind="records"))
        self.aa = self.daa.compute()
        self.a0 = self.daa.partitions[0].compute()
        self.a1 = self.daa.partitions[1].compute()
        self.a2 = self.daa.partitions[2].compute()


class Nums:
    def __init__(self):
        self.daa = dak.from_json(dakd.json_data(kind="numbers"))
        self.aa = self.daa.compute()
        self.a0 = self.daa.partitions[0].compute()
        self.a1 = self.daa.partitions[1].compute()
        self.a2 = self.daa.partitions[2].compute()


r = Recs()
n = Nums()
