from __future__ import annotations

import abc
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

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


def _fs_token_paths(
    source: Any,
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[AbstractFileSystem, str, list[str]]:
    fs, token, paths = get_fs_token_paths(source, storage_options=storage_options)

    # if paths is length 1, check to see if it's a directory and
    # wildcard search for JSON files. Otherwise, just keep whatever
    # has already been found.
    if len(paths) == 1 and fs.isdir(paths[0]):
        paths = list(filter(lambda s: ".json" in s, fs.ls(paths[0])))

    return fs, token, paths


class FromJsonFn:
    def __init__(
        self,
        *args: Any,
        storage: AbstractFileSystem,
        compression: str | None = None,
        schema: str | dict | list | None = None,
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


class FromJsonLineDelimitedFn(FromJsonFn):
    def __init__(
        self,
        *args: Any,
        storage: AbstractFileSystem,
        compression: str | None = None,
        schema: str | dict | list | None = None,
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
            return ak.from_json(
                f.read(),
                line_delimited=True,
                schema=self.schema,
                **self.kwargs,
            )

    def project_columns(
        self,
        columns: Sequence[str],
        original_form: Form | None = None,
    ) -> FromJsonLineDelimitedFn:
        # if a schema is already defined we are not going to project
        # columns; we will keep the schema that already exists.
        if self.schema is not None:
            return self

        if self.form is not None:
            form = self.form.select_columns(columns)
            schema = layout_to_jsonschema(form.length_zero_array(highlevel=False))

            return FromJsonLineDelimitedFn(
                schema=schema,
                form=form,
                storage=self.storage,
                compression=self.compression,
                original_form=original_form,
                **self.kwargs,
            )

        return self


class FromJsonBytesFn:
    def __init__(
        self,
        schema: str | dict | list | None = None,
        form: Form | None = None,
        original_form: Form | None = None,
        **kwargs: Any,
    ) -> None:
        self.schema = schema
        self.form = form
        self.original_form = original_form
        self.kwargs = kwargs

    def __call__(self, source: bytes) -> ak.Array:
        return ak.from_json(
            source,
            line_delimited=True,
            schema=self.schema,
            **self.kwargs,
        )


def meta_from_bytechunks(
    fs: AbstractFileSystem,
    paths: list[str],
    bytechunks: str | int = "16 KiB",
    **kwargs: Any,
):
    bytechunks = parse_bytes(bytechunks)
    bytes = fs.cat(paths[0], start=0, end=bytechunks)
    rfind = bytes.rfind(b"\n")
    if rfind > 0:
        bytes = bytes[:rfind]
    array = ak.from_json(bytes, line_delimited=True, **kwargs)
    return typetracer_array(array)


def meta_from_line_by_line(
    fs: AbstractFileSystem,
    paths: list[str],
    compression: str | None,
    sample_rows: int = 10,
    **kwargs: Any,
):
    with fs.open(paths[0], mode="rt", compression=compression) as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if i >= sample_rows:
                break
        array = ak.from_json("\n".join(lines), line_delimited=True, **kwargs)
        return typetracer_array(array)


def _from_json_files(
    fs: AbstractFileSystem,
    token: str,
    paths: list[str],
    *,
    schema: str | dict | list | None = None,
    compression: str | None = "infer",
    behavior: dict | None = None,
    **kwargs: Any,
) -> Array:
    if compression == "infer":
        compression = infer_compression(paths[0])

    if not compression:
        meta = meta_from_bytechunks(fs, paths, **kwargs)
    else:
        meta = meta_from_line_by_line(fs, paths, compression, **kwargs)

    token = tokenize(compression, meta, kwargs)

    f = FromJsonLineDelimitedFn(
        storage=fs,
        compression=compression,
        schema=schema,
        form=meta.layout.form,
        **kwargs,
    )

    return from_map(
        f,
        paths,
        label="from-json",
        token=token,
        meta=meta,
        behavior=behavior,
    )


def _from_json_bytes(
    *,
    source: str | list[str],
    schema: str | dict | list | None = None,
    blocksize: str,
    delimiter: Any,
    behavior: dict | None,
    storage_options: dict[str, Any] | None,
    **kwargs: Any,
) -> Array:
    token = tokenize(source, delimiter, blocksize)
    name = f"from-json-{token}"
    storage_options = storage_options or {}
    _, bytechunks = read_bytes(
        source,
        delimiter=delimiter,
        blocksize=blocksize,
        sample="0",
        **storage_options,
    )
    flat_chunks = list(flatten(bytechunks))
    f = FromJsonBytesFn(schema=schema, **kwargs)
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
    return new_array_object(hlg, name, meta=None, behavior=behavior, npartitions=n)


def from_json(
    source: str | list[str],
    *,
    line_delimited: bool = True,
    schema: str | dict | list | None = None,
    nan_string: str | None = None,
    posinf_string: str | None = None,
    neginf_string: str | None = None,
    complex_record_fields: tuple[str, str] | None = None,
    buffersize: int = 65536,
    initial: int = 1024,
    resize: float = 8,
    highlevel: bool = True,
    behavior: dict | None = None,
    blocksize: str | None = None,
    delimiter: bytes | None = None,
    compression: str | None = "infer",
    storage_options: dict[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("dask-awkward only supports highlevel awkward Arrays.")

    fs, token, paths = _fs_token_paths(source, storage_options=storage_options)

    # allow either blocksize or delimieter being not-None to trigger
    # line deliminated JSON reading.
    if blocksize is not None and delimiter is None:
        delimiter = b"\n"
    elif blocksize is None and delimiter == b"\n":
        blocksize = "128 MiB"

    if blocksize is None and delimiter is None:
        return _from_json_files(
            fs=fs,
            token=token,
            paths=paths,
            schema=schema,
            compression=compression,
            nan_string=nan_string,
            posinf_string=posinf_string,
            neginf_string=neginf_string,
            complex_record_fields=complex_record_fields,
            buffersize=buffersize,
            initial=initial,
            resize=resize,
            behavior=behavior,
        )
        # if a `delimiter` and `blocksize` are defined we use Dask's
    # `read_bytes` function to get delayed chunks of bytes.
    elif delimiter is not None and blocksize is not None:
        return _from_json_bytes(
            source=source,
            schema=schema,
            delimiter=delimiter,
            blocksize=blocksize,
            behavior=behavior,
            nan_string=nan_string,
            posinf_string=posinf_string,
            neginf_string=neginf_string,
            complex_record_fields=complex_record_fields,
            buffersize=buffersize,
            initial=initial,
            resize=resize,
            storage_options=storage_options,
        )

    # otherwise the arguments are bad
    else:
        raise TypeError("Incompatible combination of arguments.")  # pragma: no cover


class ToJsonFn:
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
        self.compression = compression
        if self.compression == "infer":
            self.compression = infer_compression(self.path)
        self.kwargs = kwargs

    def __call__(self, array: ak.Array, block_index: tuple[int]) -> None:
        part = str(block_index[0]).zfill(self.zfill)
        filename = f"part{part}.json"
        if self.compression is not None and self.compression != "infer":
            ext = self.compression
            if ext == "gzip":
                ext = "gz"
            if ext == "zstd":
                ext = "zst"
            filename = f"{filename}.{ext}"

        thispath = self.fs.sep.join([self.path, filename])
        with self.fs.open(thispath, mode="wt", compression=self.compression) as f:
            ak.to_json(array, f, **self.kwargs)

        return None


@overload
def to_json(
    array: Array,
    path: str,
    *,
    line_delimited: bool | str = True,
    num_indent_spaces: int | None = None,
    num_readability_spaces: int = 0,
    nan_string: str | None = None,
    posinf_string: str | None = None,
    neginf_string: str | None = None,
    complex_record_fields: tuple[str, str] | None = None,
    convert_bytes: Callable | None = None,
    convert_other: Callable | None = None,
    storage_options: dict[str, Any] | None = None,
    compression: str | None = None,
    compute: Literal[False] = False,
) -> Scalar:
    ...


@overload
def to_json(
    array: Array,
    path: str,
    *,
    line_delimited: bool | str = True,
    num_indent_spaces: int | None = None,
    num_readability_spaces: int = 0,
    nan_string: str | None = None,
    posinf_string: str | None = None,
    neginf_string: str | None = None,
    complex_record_fields: tuple[str, str] | None = None,
    convert_bytes: Callable | None = None,
    convert_other: Callable | None = None,
    storage_options: dict[str, Any] | None = None,
    compression: str | None = None,
    compute: Literal[True] = True,
) -> None:
    ...


def to_json(
    array: Array,
    path: str,
    *,
    line_delimited: bool | str = True,
    num_indent_spaces: int | None = None,
    num_readability_spaces: int = 0,
    nan_string: str | None = None,
    posinf_string: str | None = None,
    neginf_string: str | None = None,
    complex_record_fields: tuple[str, str] | None = None,
    convert_bytes: Callable | None = None,
    convert_other: Callable | None = None,
    storage_options: dict[str, Any] | None = None,
    compression: str | None = None,
    compute: bool = False,
) -> Scalar | None:
    """Store Array collection in JSON text.

    Parameters
    ----------
    array : Array
        Collection to store in JSON format
    path : str
        Root directory to save data; interpreted by filesystem-spec
        (can be a remote filesystem path, for example an s3 bucket:
        ``"s3://bucket/data"``).
    line_delimited : bool | str
        See docstring for :py:func:`ak.to_json`.
    num_indent_spaces : int, optional
        See docstring for :py:func:`ak.to_json`.
    num_readability_spaces : int
        See docstring for :py:func:`ak.to_json`.
    nan_string : str, optional
        See docstring for :py:func:`ak.to_json`.
    posinf_string : str, optional
        See docstring for :py:func:`ak.to_json`.
    neginf_string : str, optional
        See docstring for :py:func:`ak.to_json`.
    complex_record_fields : tuple[str, str], optional
        See docstring for :py:func:`ak.to_json`.
    convert_bytes : Callable, optional
        See docstring for :py:func:`ak.to_json`.
    convert_other : Callable, optional
        See docstring for :py:func:`ak.to_json`.
    storage_options : dict[str, Any], optional
        Options passed to ``fsspec``.
    compression : str, optional
        Compress JSON data via ``fsspec``
    compute : bool
        Immediately compute the collection.

    Returns
    -------
    Scalar or None
        Computable Scalar object if ``compute`` is ``False``,
        otherwise returns ``None``.

    Examples
    --------

    >>> import dask_awkward as dak
    >>> print("Hello, world!")

    """
    storage_options = storage_options or {}
    fs, _ = url_to_fs(path, **storage_options)
    nparts = array.npartitions
    map_res = map_partitions(
        ToJsonFn(
            fs,
            path,
            npartitions=nparts,
            compression=compression,
            line_delimited=line_delimited,
            num_indent_spaces=num_indent_spaces,
            num_readability_spaces=num_readability_spaces,
            nan_string=nan_string,
            posinf_string=posinf_string,
            neginf_string=neginf_string,
            complex_record_fields=complex_record_fields,
            convert_bytes=convert_bytes,
            convert_other=convert_other,
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
        return None
    return res


def array_param_is_string_or_bytestring(layout: Content) -> bool:
    params = layout.parameters or {}
    return params == {"__array__": "string"} or params == {"__array__": "bytestring"}


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
    elif (layout.parameters or {}) == {"__array__": "categorical"}:
        existing_schema["enum"] = layout.content.to_list()
        existing_schema["type"] = layout_to_jsonschema(layout.content)["type"]
    elif array_param_is_string_or_bytestring(layout):
        existing_schema["type"] = "string"
    elif layout.is_list:
        existing_schema["type"] = "array"
        if layout.is_regular:
            existing_schema["minItems"] = layout.size
            existing_schema["maxItems"] = layout.size
        existing_schema["items"] = {}
        layout_to_jsonschema(layout.content, existing_schema["items"])
    elif layout.is_option:
        layout_to_jsonschema(layout.content, existing_schema)
    elif layout.is_numpy:
        if layout.dtype.kind == "i":
            existing_schema["type"] = "integer"
        elif layout.dtype.kind == "f":
            existing_schema["type"] = "number"
    elif layout.is_indexed:
        pass
    elif layout.is_unknown:
        existing_schema["type"] = [
            "array",
            "boolean",
            "object",
            "null",
            "number",
            "string",
        ]
    elif layout.is_union:
        existing_schema["type"] = [
            layout_to_jsonschema(content)["type"] for content in layout.contents
        ]
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
