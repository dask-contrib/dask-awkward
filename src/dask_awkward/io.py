from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

from awkward._v2.highlevel import Array as Array_v2
from awkward._v2.operations.convert.ak_from_iter import from_iter as from_iter_v2
from dask.base import flatten, tokenize
from dask.bytes import read_bytes
from dask.highlevelgraph import HighLevelGraph

from .core import new_array_object

if TYPE_CHECKING:
    from .core import DaskAwkwardArray


def is_file_path(source: Any) -> bool:
    try:
        return os.path.isfile(source)
    except ValueError:
        return False


# NOTE: split this into two sep functions
def _chunked_to_json_v2(source):
    if isinstance(source, str) and is_file_path(source):
        import fsspec

        # try to read JSON object in a single file.
        try:
            with fsspec.open(source) as f:
                return Array_v2(json.load(f))
        # if exception we assume to have line delimited JSON.
        except ValueError:
            with fsspec.open(source) as f:
                return from_iter_v2(json.loads(line) for line in f)
    else:
        return from_iter_v2(json.loads(ch) for ch in source.split(b"\n") if ch)


# def _chunked_to_json_v1(source):
#     if isinstance(source, str) and is_file_path(source):
#         x = ak.from_json(source)
#         if isinstance(x, ak.Record):
#             return ak.Array([x])
#         return x
#     return from_iter_v1(json.loads(ch) for ch in source.split(b"\n") if ch)


def _from_json(
    source: str | list[str],
    concrete: Callable,
    delimiter: bytes | None = b"\n",
    blocksize: int | str = "128 MiB",
) -> DaskAwkwardArray:
    token = tokenize(source, delimiter, blocksize)
    name = f"from-json-{token}"

    if isinstance(source, (list, tuple)) and delimiter is None:
        dsk = {(name, i): (concrete, s) for i, s in enumerate(source)}
        deps = set()
        n = len(dsk)
    else:
        _, chunks = read_bytes(
            source,
            delimiter=delimiter,
            blocksize=blocksize,
            sample=None,
        )
        chunks = list(flatten(chunks))
        dsk = {(name, i): (concrete, d.key) for i, d in enumerate(chunks)}
        deps = chunks
        n = len(deps)

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    return new_array_object(hlg, name, None, npartitions=n)


# from_json_v1 = partial(_from_json, concrete=_chunked_to_json_v1)
from_json_v2 = partial(_from_json, concrete=_chunked_to_json_v2)
from_json = from_json_v2
