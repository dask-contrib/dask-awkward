from __future__ import annotations

import abc
import math
import warnings
from typing import TYPE_CHECKING, Any

try:
    import ujson as json
except ImportError:
    import json  # type: ignore[no-redef]

import awkward as ak
from awkward.forms.form import Form
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.bytes.core import read_bytes
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
from fsspec.core import get_fs_token_paths, url_to_fs
from fsspec.utils import infer_compression

from dask_awkward.lib.core import (
    map_partitions,
    new_array_object,
    new_scalar_object,
    typetracer_array,
)
from dask_awkward.lib.io.io import from_map

if TYPE_CHECKING:
    from awkward.contents.content import Content
    from fsspec.spec import AbstractFileSystem

    from dask_awkward.lib.core import Array, Scalar

__all__ = ("from_json", "to_json")


class _FromJsonFn:
    def __init__(
        self,
        *args: Any,
        storage: AbstractFileSystem,
        compression: str | None = None,
        schema: dict | None = None,
        form: Form | None = None,
        original_form: Form | None = None,
        **kwargs: Any,
    ) -> None:
        self.compression = compression
        self.storage = storage
        self.schema = schema
        self.args = args
        self.kwargs = kwargs
        self.form = form
        self.original_form = original_form

    @abc.abstractmethod
    def __call__(self, source: Any) -> ak.Array:
        ...


class _FromJsonLineDelimitedFn(_FromJsonFn):
    def __init__(
        self,
        *args: Any,
        storage: AbstractFileSystem,
        compression: str | None = None,
        schema: dict | None = None,
        form: Form | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            storage=storage,
            compression=compression,
            schema=schema,
            form=form,
            **kwargs,
        )

    def __call__(self, source: str) -> ak.Array:
        with self.storage.open(source, mode="rt", compression=self.compression) as f:
            return ak.from_json(f.read(), line_delimited=True, schema=self.schema)

    def project_columns(self, columns, original_form: Form | None = None):
        if self.form is not None:
            form = self.form.select_columns(columns)
            schema = layout_to_jsonschema(form.length_zero_array(highlevel=False))

            return _FromJsonLineDelimitedFn(
                schema=schema,
                form=form,
                storage=self.storage,
                compression=self.compression,
                original_form=original_form,
            )

        return self


class _FromJsonSingleObjInFileFn(_FromJsonFn):
    def __init__(
        self,
        *args: Any,
        storage: AbstractFileSystem,
        compression: str | None = None,
        schema: dict | None = None,
        form: Form | None = None,
        original_form: Form | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            storage=storage,
            compression=compression,
            schema=schema,
            form=form,
            original_form=original_form,
            **kwargs,
        )

    def __call__(self, source: str) -> ak.Array:
        with self.storage.open(source, mode="rb", compression=self.compression) as f:
            return ak.from_json(f, schema=self.schema, **self.kwargs)


class _FromJsonBytesFn:
    def __init__(
        self,
        schema: dict | None = None,
        form: Form | None = None,
        original_form: Form | None = None,
    ) -> None:
        self.schema = schema
        self.form = form
        self.original_form = original_form

    def __call__(self, source: bytes) -> ak.Array:
        return ak.from_json(source, line_delimited=True, schema=self.schema)


def derive_json_meta(
    storage: AbstractFileSystem,
    source: str,
    schema: dict | None = None,
    compression: str | None = "infer",
    sample_rows: int = 5,
    bytechunks: str | int = "16 KiB",
    force_by_lines: bool = False,
    one_obj_per_file: bool = False,
) -> ak.Array:
    if compression == "infer":
        compression = infer_compression(source)

    bytechunks = parse_bytes(bytechunks)

    if one_obj_per_file:
        fn = _FromJsonSingleObjInFileFn(
            storage=storage,
            compression=compression,
            schema=schema,
        )
        return typetracer_array(fn(source))

    # when the data is uncompressed we read `bytechunks` number of
    # bytes then split on a newline bytes, and use the first
    # `sample_rows` number of lines.
    if compression is None and not force_by_lines:
        try:
            bytes = storage.cat(source, start=0, end=bytechunks)
            lines = [json.loads(ln) for ln in bytes.split(b"\n")[:sample_rows]]
            return typetracer_array(ak.from_iter(lines))
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
    with storage.open(source, mode="rt", compression=compression) as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if i >= sample_rows:
                break
        return typetracer_array(ak.from_iter(lines))


def _from_json_files(
    *,
    urlpath: str | list[str],
    meta: ak.Array,
    schema: dict | None = None,
    one_obj_per_file: bool = False,
    compression: str | None = "infer",
    behavior: dict | None = None,
    derive_meta_kwargs: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> Array:
    fs, fstoken, urlpaths = get_fs_token_paths(
        urlpath,
        mode="rb",
        storage_options=storage_options,
    )
    if meta is None:
        meta_read_kwargs = derive_meta_kwargs or {}
        meta = derive_json_meta(
            fs,
            urlpaths[0],
            schema=schema,
            one_obj_per_file=one_obj_per_file,
            **meta_read_kwargs,
        )

    token = tokenize(fstoken, one_obj_per_file, compression, meta)

    if compression == "infer":
        compression = infer_compression(urlpaths[0])

    if one_obj_per_file:
        f: _FromJsonFn = _FromJsonSingleObjInFileFn(
            storage=fs,
            compression=compression,
            schema=schema,
        )
    else:
        f = _FromJsonLineDelimitedFn(
            storage=fs,
            compression=compression,
            schema=schema,
            form=meta.layout.form,
        )

    return from_map(
        f, urlpaths, label="from-json", token=token, meta=meta, behavior=behavior
    )


def _from_json_bytes(
    *,
    urlpath: str | list[str],
    meta: ak.Array,
    schema: dict | None,
    blocksize: int | str,
    delimiter: Any,
    behavior: dict | None,
    storage_options: dict[str, Any] | None,
) -> Array:
    token = tokenize(urlpath, delimiter, blocksize, meta)
    name = f"from-json-{token}"
    storage_options = storage_options or {}
    _, bytechunks = read_bytes(
        urlpath,
        delimiter=delimiter,
        blocksize=blocksize,
        sample="0",
        **storage_options,
    )
    flat_chunks = list(flatten(bytechunks))
    f = _FromJsonBytesFn(schema=schema)
    dsk = {
        (name, i): (f, delayed_chunk.key) for i, delayed_chunk in enumerate(flat_chunks)
    }
    deps = flat_chunks
    n = len(deps)

    # doesn't work because flat_chunks elements are remaining delayed objects.
    # return from_map(
    #     _from_json_bytes,
    #     flat_chunks,
    #     label="from-json",
    #     token=token,
    #     produces_tasks=True,
    #     deps=flat_chunks,
    # )

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    return new_array_object(hlg, name, meta=meta, behavior=behavior, npartitions=n)


def from_json(
    urlpath: str | list[str],
    schema: dict | None = None,
    # nan_string: str | None = None,
    # posinf_string: str | None = None,
    # neginf_string: str | None = None,
    # complex_record_fields: tuple[str, str] | None = None,
    # buffersize: int = 65536,
    # initial: int = 1024,
    # resize: float = 1.5,
    highlevel: bool = True,
    *,
    blocksize: int | str | None = None,
    delimiter: bytes | None = None,
    one_obj_per_file: bool = False,
    compression: str | None = "infer",
    meta: ak.Array | None = None,
    behavior: dict | None = None,
    derive_meta_kwargs: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> Array:
    """Create an Awkward Array collection from JSON data.

    There are three styles supported for reading JSON data:

    1. Line delimited style: file(s) with one JSON object per line.
       The function argument defaults are setup to handle this style.
       This method assumes newline characters are not embedded in JSON
       values.
    2. Single JSON object per file (this requires `one_obj_per_file`
       to be set to ``True``. These objects *must* be arrays.
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
        JSON bytes of this size. If `delimiter` is defined (not
        ``None``) but this value remains ``None``, a default value of
        ``128 MiB`` will be used.
    delimiter : bytes, optional
        If defined (not ``None``), this will be the byte(s) to split
        on when reading `blocksizes`. If this is ``None`` but
        `blocksize` is defined (not ``None``), the default byte
        charater will be the newline (``b"\\n"``).
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
    storage_options : dict[str, Any], optional
        Storage options passed to fsspec.

    Returns
    -------
    Array
        The resulting Dask Awkward Array collection.

    Examples
    --------
    One partition per file:

    >>> import dask_awkard as dak
    >>> a = dak.from_json("dataset*.json")

    One partition ber 200 MB of JSON data:

    >>> a = dak.from_json("dataset*.json", blocksize="200 MB")

    Same as previous call (explicit definition of the delimiter):

    >>> a = dak.from_json(
    ...     "dataset*.json", blocksize="200 MB", delimiter=b"\\n",
    ... )

    """

    if not highlevel:
        raise ValueError("dask-awkward only supports highlevel awkward Arrays.")

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
        return _from_json_files(
            urlpath=urlpath,
            meta=meta,
            schema=schema,
            one_obj_per_file=one_obj_per_file,
            compression=compression,
            behavior=behavior,
            derive_meta_kwargs=derive_meta_kwargs,
            storage_options=storage_options,
        )

    # if a `delimiter` and `blocksize` are defined we use Dask's
    # `read_bytes` function to get delayed chunks of bytes.
    elif delimiter is not None and blocksize is not None:
        return _from_json_bytes(
            urlpath=urlpath,
            meta=meta,
            schema=schema,
            delimiter=delimiter,
            blocksize=blocksize,
            behavior=behavior,
            storage_options=storage_options,
        )

    # otherwise the arguments are bad
    else:
        raise TypeError("Incompatible combination of arguments.")  # pragma: no cover


class _ToJsonFn:
    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
        npartitions: int,
        compression: str | None,
        **kwargs: Any,
    ) -> None:
        self.fs = fs
        self.path = path
        if not self.fs.exists(path):
            self.fs.mkdir(path)
        self.zfill = math.ceil(math.log(npartitions, 10))
        self.kwargs = kwargs
        self.compression = compression
        if self.compression == "infer":
            self.compression = infer_compression(self.path)

    def __call__(self, array: ak.Array, block_index: tuple[int]) -> None:
        part = str(block_index[0]).zfill(self.zfill)
        filename = f"part{part}.json"
        if self.compression is not None and self.compression != "infer":
            compression = "gz" if self.compression == "gzip" else self.compression
            filename = f"{filename}.{compression}"

        thispath = self.fs.sep.join([self.path, filename])
        with self.fs.open(thispath, mode="wt", compression=self.compression) as f:
            ak.to_json(array, f, line_delimited=True, **self.kwargs)

        return None


def to_json(
    array: Array,
    path: str,
    storage_options: dict[str, Any] | None = None,
    compute: bool = False,
    compression: str | None = None,
    **kwargs: Any,
) -> Scalar:
    """Write data to line delimited JSON.

    Parameters
    ----------
    array : Array
        dask-awkward Array collection to write to disk.
    path : str
         Root directory of location to write to.
    storage_options : dict[str, Any], optional
        filesystem-spec storage options.
    compute : bool
        If ``True`` immediately compute this collection.
    compression : str, optional
        filesystem-spec compression algorithm to use.
    **kwargs : Any
        Additional arguments passed to ``ak.to_json``

    Returns
    -------
    None or dask_awkward.Scalar
        Scalar which provides a lazy compute if `compute` is
        ``False``, or ``None`` if `compute` is ``True``.

    """
    storage_options = storage_options or {}
    fs, _ = url_to_fs(path, **storage_options)
    nparts = array.npartitions
    map_res = map_partitions(
        _ToJsonFn(
            fs,
            path,
            npartitions=nparts,
            compression=compression,
            **kwargs,
        ),
        array,
        BlockIndex((nparts,)),
        label="to-json-on-block",
        meta=array._meta,
    )
    map_res.dask.layers[map_res.name].annotations = {"ak_output": True}
    name = f"to-json-{tokenize(array, path)}"
    dsk = {(name, 0): (lambda *_: None, map_res.__dask_keys__())}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=(map_res,))
    res = new_scalar_object(graph, name=name, meta=None)
    if compute:
        res.compute()
    return res


def layout_to_jsonschema(
    layout: Content,
    existing_schema: dict | None = None,
    title: str = "untitled",
    description: str = "Auto generated by dask-awkward",
    required: bool = False,
) -> dict:
    """Convert awkward array Layout to a JSON Schema dictionary."""
    if existing_schema is None:
        existing_schema = {
            "title": title,
            "description": description,
            "type": "object",
            "properties": {},
        }
    if layout.is_record:
        existing_schema["type"] = "object"
        existing_schema["properties"] = {}
        if required:
            existing_schema["required"] = layout.fields
        for field in layout.fields:
            existing_schema["properties"][field] = {"type": None}
            layout_to_jsonschema(layout[field], existing_schema["properties"][field])
    elif (layout.parameters or {}) == {"__array__": "string"}:
        existing_schema["type"] = "string"
    elif layout.is_list:
        existing_schema["type"] = "array"
        existing_schema["items"] = {}
        layout_to_jsonschema(layout.content, existing_schema["items"])
    elif layout.is_option:
        layout_to_jsonschema(layout.content, existing_schema)
    elif layout.is_numpy:
        if layout.dtype.kind == "i":
            existing_schema["type"] = "integer"
        elif layout.dtype.kind == "f":
            existing_schema["type"] = "number"
    return existing_schema


def form_to_dict(form):
    """Convert awkward array Layout to a JSON Schema dictionary."""
    if form.is_record:
        out = {}
        for field, content in zip(form.fields, form.contents):
            out[field] = form_to_dict(content)
        return out
    elif form.parameters.get("__array__") == "string":
        return "string"
    elif form.is_list:
        return [form_to_dict(form.content)]
    elif hasattr(form, "content"):
        return form_to_dict(form.content)
    else:
        return form.dtype


def ak_schema_repr(arr):
    import yaml

    return yaml.dump(arr.layout.form)
