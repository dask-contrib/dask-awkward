from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

import awkward._v2 as ak
import fsspec
from dask.base import tokenize
from dask.bytes.core import read_bytes
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from fsspec.utils import infer_compression

from .core import new_array_object

if TYPE_CHECKING:
    from .core import Array

__all__ = ["from_json"]


def is_file_path(source: Any) -> bool:
    try:
        return os.path.isfile(source)
    except (ValueError, TypeError):
        return False


class FromJsonWrapper:
    def __init__(self, *, compression: str | None = None):
        self.compression = compression


class FromJsonLineDelimitedWrapper(FromJsonWrapper):
    def __init__(self, *, compression: str | None = None):
        super().__init__(compression=compression)

    def __call__(self, source: str) -> ak.Array:
        with fsspec.open(source, mode="rt", compression=self.compression) as f:
            return ak.from_iter(json.loads(line) for line in f)


class FromJsonSingleObjInFileWrapper(FromJsonWrapper):
    def __init__(self, *, compression: str | None = None):
        super().__init__(compression=compression)

    def __call__(self, source: str) -> ak.Array:
        with fsspec.open(source, mode="r", compression=self.compression) as f:
            return ak.Array([json.load(f)])


def _from_json_bytes(source) -> ak.Array:
    return ak.from_iter(
        json.loads(ch) for ch in io.TextIOWrapper(io.BytesIO(source)) if ch
    )


def from_json(
    urlpath: str | list[str],
    blocksize: int | str | None = None,
    delimiter: bytes | None = None,
    one_obj_per_file: bool = False,
    compression: str | None = "infer",
) -> Array:
    token = tokenize(urlpath, delimiter, blocksize, one_obj_per_file)
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
        if is_file_path(urlpath):
            urlpath = [urlpath]  # type: ignore

        urlpath = fsspec.get_fs_token_paths(urlpath)[2]

        if compression == "infer":
            compression = infer_compression(urlpath[0])

        if one_obj_per_file:
            f: FromJsonWrapper = FromJsonSingleObjInFileWrapper(compression=compression)
        else:
            f = FromJsonLineDelimitedWrapper(compression=compression)

        dsk = {(name, i): (f, s) for i, s in enumerate(urlpath)}
        deps = set()
        n = len(dsk)

    elif delimiter is not None and blocksize is not None:
        _, chunks = read_bytes(
            urlpath,
            delimiter=delimiter,
            blocksize=blocksize,
            sample=None,
        )
        chunks = list(flatten(chunks))
        dsk = {(name, i): (_from_json_bytes, d.key) for i, d in enumerate(chunks)}  # type: ignore
        deps = chunks
        n = len(deps)
    else:
        raise TypeError("Incompatible combination of arguments.")

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    return new_array_object(hlg, name, None, npartitions=n)
