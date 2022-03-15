from __future__ import annotations

import io
import os
import warnings
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
from dask.utils import parse_bytes
from fsspec.utils import infer_compression

from dask_awkward.core import new_array_object

if TYPE_CHECKING:
    from dask.delayed import Delayed

    from dask_awkward.core import Array

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


def derive_json_meta(
    source: str,
    compression: str | None = "infer",
    sample_rows: int = 5,
    bytechunks: str | int = "16 KiB",
    force_by_lines: bool = False,
) -> ak.Array:
    if compression == "infer":
        compression = infer_compression(source)

    bytechunks = parse_bytes(bytechunks)

    # when the data is uncompressed we read `bytechunks` number of
    # bytes then split on a newline bytes, and use the first
    # `sample_rows` number of lines.
    if compression is None and not force_by_lines:
        try:
            with fsspec.open(source, mode="rb") as f:
                byteread = f.read(bytechunks)
                lines = [
                    json.loads(ln) for ln in byteread.split(b"\n")[:sample_rows] if ln
                ]
                return ak.Array(ak.from_iter(lines).layout.typetracer.forget_length())
        except ValueError:
            # we'll get a ValueError if we can't decode the JSON from
            # the bytes that we grabbed.
            warnings.warn(
                f"Couldn't determine metadata from reading first {bytechunks} "
                f"of the dataset; will read the first {sample_rows} instead. "
                "Try increasing the value of `bytechunks` or decreasing `sample_rows` "
                "to remove this warning."
            )

    # for compressed data (or if explicitly asked for with
    # force_by_lines set to True) we read the first `sample_rows`
    # number of rows after opening the compressed file.
    with fsspec.open(source, mode="rt", compression=compression) as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if i >= sample_rows:
                break
        return ak.Array(ak.from_iter(lines).layout.typetracer.forget_length())


def from_json(
    urlpath: str | list[str],
    blocksize: int | str | None = None,
    delimiter: bytes | None = None,
    one_obj_per_file: bool = False,
    compression: str | None = "infer",
    meta: ak.Array | None = None,
    derive_meta_kwargs: dict[str, Any] | None = None,
) -> Array:
    """Create an Awkward Array collection from JSON data.

    There are three styles supported for reading JSON data:

    1. Line delimited style: file(s) with one JSON object per line.
       The function argument defaults are setup to handle this style.
       This method assumes newline characters are not embedded in JSON
       values.
    2. Single JSON object per file (this requires `one_obj_per_file`
       to be set to ``True``.
    3. Reading some number of bytes at a time. If at least one of
       `blocksize` or `delimiter` are defined, Dask's
       :py:func:`~dask.bytes.read_bytes` function will be used to
       lazily read bytes (`blocksize` bytes per partition) and split
       on `delimiter`). This method assumes line delimited JSON
       without newline characters embedded in JSON values.

    Parameters
    ----------
    urlpath : str | list[str]
        The source of the JSON dataset.
    blocksize : int | str, optional
        If defined, each partition will be created from a block of
        JSON data of this size.
    delimiter : bytes, optional
        If defined (not ``None``), the byte character to split on when
        reading `blocksizes`. If this is ``None`` but `blocksize` is
        defined, the default byte charater is ``b"\\n"``.
    one_obj_per_file : bool
        If ``True`` each file will be considered a single JSON object.
    compression : str, optional
        Compression of the files in the dataset.
    meta : Any, optional
        The metadata for the collection. If ``None`` (the default),
        them metadata will be determined by scanning the beginning of
        the dataset.
    derive_meta_kwargs : dict[str, Any], optional
        Dictionary of arguments to be passed to `derive_json_meta` for
        determining the collection metadata if `meta` is ``None``.

    Returns
    -------
    Array
        The resulting Dask Awkward Array collection.

    Examples
    --------
    >>> import dask_awkard as dak

    """
    token = tokenize(urlpath, delimiter, blocksize, one_obj_per_file)
    name = f"from-json-{token}"

    # allow either blocksize or delimieter being not-None to trigger
    # line deliminated JSON reading.
    if blocksize is not None and delimiter is None:
        delimiter = b"\n"
    elif blocksize is None and delimiter == b"\n":
        blocksize = "128 MiB"

    # if delimiter is None and blocksize is None we are expecting to
    # read a single file or a list of files. The list of files are
    # expected to be line delimited (one JSON object per line)
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

        dsk: dict[tuple[str, int], tuple[Any, ...]] = {
            (name, i): (f, s) for i, s in enumerate(urlpath)
        }
        deps: set[Any] = set()
        n = len(dsk)

        if meta is None:
            meta_read_kwargs = derive_meta_kwargs or {}
            meta = derive_json_meta(urlpath[0], **meta_read_kwargs)

    # if a `delimiter` and `blocksize` are defined we use Dask's
    # `read_bytes` function to get delayed chunks of bytes.
    elif delimiter is not None and blocksize is not None:
        _, bytechunks = read_bytes(
            urlpath,
            delimiter=delimiter,
            blocksize=blocksize,  # type: ignore
            sample=None,  # type: ignore
        )
        flat_chunks: list[Delayed] = list(flatten(bytechunks))
        dsk = {
            (name, i): (_from_json_bytes, delayed_chunk.key)
            for i, delayed_chunk in enumerate(flat_chunks)
        }
        deps = set(flat_chunks)
        n = len(deps)

    else:
        raise TypeError("Incompatible combination of arguments.")

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    return new_array_object(hlg, name, meta=meta, npartitions=n)
