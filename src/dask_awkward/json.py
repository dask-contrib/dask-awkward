import awkward as ak
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from .collection import AwkwardDaskArray


def _from_json(source, kwargs):
    return ak.from_json(source, **kwargs)


def from_json(source, **kwargs) -> AwkwardDaskArray:
    name = f"from-json-{tokenize(source)}"
    dsk = {(name, i): (_from_json, f, kwargs) for i, f in enumerate(source)}
    hlg = HighLevelGraph.from_collections(name, dsk)
    return AwkwardDaskArray(hlg, name, len(source))
