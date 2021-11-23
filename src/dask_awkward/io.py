from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import awkward as ak

try:
    import ujson as json
except ImportError:
    import json  # type: ignore
from awkward import from_iter as from_iter_v1
from awkward._v2.operations.convert.ak_from_iter import from_iter as from_iter_v2
from dask.base import flatten, tokenize
from dask.bytes import read_bytes
from dask.highlevelgraph import HighLevelGraph

from .core import new_array_object

if TYPE_CHECKING:
    from .core import DaskAwkwardArray


def _from_parquet_single(source: Any, kwargs: dict[Any, Any]) -> Any:
    return ak.from_parquet(source, **kwargs)


def _from_parquet_rowgroups(
    source: Any,
    row_groups: int | list[int],
    kwargs: dict[str, Any],
) -> Any:
    return ak.from_parquet(source, row_groups=row_groups, **kwargs)


def from_parquet(source: Any, **kwargs: Any) -> DaskAwkwardArray:
    token = tokenize(source)
    name = f"from-parquet-{token}"

    if isinstance(source, list):
        dsk = {
            (name, i): (_from_parquet_single, f, kwargs) for i, f in enumerate(source)
        }
        npartitions = len(source)
    elif "row_groups" in kwargs:
        row_groups = kwargs.pop("row_groups")
        dsk = {
            (name, i): (_from_parquet_rowgroups, source, rg, kwargs)  # type: ignore
            for i, rg in enumerate(row_groups)
        }
        npartitions = len(row_groups)

    hlg = HighLevelGraph.from_collections(name, dsk)
    return new_array_object(hlg, name, None, npartitions=npartitions)


def _chunked_to_json_v2(chunks):
    return from_iter_v2(json.loads(ch) for ch in chunks.split(b"\n") if ch)


def _chunked_to_json_v1(chunks):
    return from_iter_v1(json.loads(ch) for ch in chunks.split(b"\n") if ch)


def _from_json(
    source: str | list[str],
    concrete: Callable,
    delimiter: bytes = b"\n",
    blocksize: int | str = "128 MiB",
) -> DaskAwkwardArray:
    token = tokenize(source, delimiter, blocksize)
    name = f"from-json-{token}"
    _, chunks = read_bytes(
        source,
        delimiter=delimiter,
        blocksize=blocksize,
        sample=None,
    )
    chunks = list(flatten(chunks))
    dsk = {(name, i): (concrete, d.key) for i, d in enumerate(chunks)}
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=chunks)
    return new_array_object(hlg, name, None, npartitions=len(chunks))


from_json_v1 = partial(_from_json, concrete=_chunked_to_json_v1)
from_json_v2 = partial(_from_json, concrete=_chunked_to_json_v2)
from_json = from_json_v1
