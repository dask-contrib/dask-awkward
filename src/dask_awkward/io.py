from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

from awkward._v2.highlevel import ak
from awkward._v2.operations.convert.ak_from_iter import from_iter
from dask.base import flatten, tokenize
from dask.bytes import read_bytes
from dask.highlevelgraph import HighLevelGraph

from .core import new_array_object

if TYPE_CHECKING:
    from .core import Array


def is_file_path(source: Any) -> bool:
    try:
        return os.path.isfile(source)
    except (ValueError, TypeError):
        return False


def _from_json_single_object_in_file(source) -> ak.Array:
    with open(source) as f:
        return ak.Array([json.load(f)])


def _from_json_line_by_line(source) -> ak.Array:
    with open(source) as f:
        return from_iter(json.loads(line) for line in f)


def _from_json_bytes(source) -> ak.Array:
    # return from_iter(json.loads(ch) for ch in source.split(b"\n") if ch)
    return from_iter(
        json.loads(ch) for ch in io.TextIOWrapper(io.BytesIO(source)) if ch
    )


def from_json(
    source: str | list[str],
    blocksize: int | str | None = None,
    delimiter: bytes | None = None,
    one_obj_per_file: bool = False,
) -> Array:
    token = tokenize(source, delimiter, blocksize, one_obj_per_file)
    name = f"from-json-{token}"

    # allow either blocksize or delimieter being not-None to trigger
    # line deliminated JSON reading.
    if blocksize is not None and delimiter is None:
        delimiter = b"\n"
    elif blocksize is None and delimiter == b"\n":
        blocksize = "128 MiB"

    # if delimiter is None and blocksize is None we are expecting to
    # read a single file or a list of files.
    if delimiter is None and blocksize is None:
        if is_file_path(source):
            source = [source]  # type: ignore
        concrete = (
            _from_json_single_object_in_file
            if one_obj_per_file
            else _from_json_line_by_line
        )
        dsk = {(name, i): (concrete, s) for i, s in enumerate(source)}
        deps = set()
        n = len(dsk)
    elif delimiter is not None and blocksize is not None:
        _, chunks = read_bytes(
            source,
            delimiter=delimiter,
            blocksize=blocksize,
            sample=None,
        )
        chunks = list(flatten(chunks))
        dsk = {(name, i): (_from_json_bytes, d.key) for i, d in enumerate(chunks)}
        deps = chunks
        n = len(deps)
    else:
        raise TypeError("Incompatible combination of arguments.")

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    return new_array_object(hlg, name, None, npartitions=n)
