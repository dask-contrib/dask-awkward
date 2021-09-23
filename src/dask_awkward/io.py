import awkward as ak
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from .core import DaskAwkwardArray


def _from_json(source, kwargs):
    return ak.from_json(source, **kwargs)


def from_json(source, **kwargs) -> DaskAwkwardArray:
    token = tokenize(source)
    name = f"from-json-{token}"
    dsk = {(name, i): (_from_json, f, kwargs) for i, f in enumerate(source)}
    hlg = HighLevelGraph.from_collections(name, dsk)
    return DaskAwkwardArray(hlg, name, len(source))


def _from_parquet_single(source, kwargs):
    return ak.from_parquet(source, **kwargs)


def _from_parquet_rowgroups(source, row_groups, kwargs):
    return ak.from_parquet(source, row_groups=row_groups, **kwargs)


def from_parquet(source, **kwargs) -> DaskAwkwardArray:
    token = tokenize(source)
    name = f"from-parquet-{token}"

    if isinstance(source, list):
        dsk = {
            (name, i): (_from_parquet_single, f, kwargs) for i, f in enumerate(source)
        }
        N = len(source)
    elif "row_groups" in kwargs:
        row_groups = kwargs.pop("row_groups")
        dsk = {
            (name, i): (_from_parquet_rowgroups, source, rg, kwargs)
            for i, rg in enumerate(row_groups)
        }
        N = len(row_groups)

    hlg = HighLevelGraph.from_collections(name, dsk)
    return DaskAwkwardArray(hlg, name, N)
