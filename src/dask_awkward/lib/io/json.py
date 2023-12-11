from __future__ import annotations

import abc
import logging
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import awkward as ak
import dask
from awkward.forms.form import Form
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
from fsspec.core import get_fs_token_paths, url_to_fs
from fsspec.utils import infer_compression, read_block

from dask_awkward.layers.layers import AwkwardMaterializedLayer
from dask_awkward.lib.core import (
    Array,
    Scalar,
    map_partitions,
    new_scalar_object,
    typetracer_array,
)
from dask_awkward.lib.io.columnar import ColumnProjectionMixin
from dask_awkward.lib.io.io import (
    _bytes_with_sample,
    _BytesReadingInstructions,
    from_map,
)

if TYPE_CHECKING:
    from awkward.contents.content import Content
    from fsspec.spec import AbstractFileSystem


log = logging.getLogger(__name__)


class FromJsonFn(ColumnProjectionMixin):
    def __init__(
        self,
        *,
        storage: AbstractFileSystem,
        form: Form,
        compression: str | None = None,
        schema: str | dict | list | None = None,
        behavior: Mapping | None = None,
        **kwargs: Any,
    ) -> None:
        self.compression = compression
        self.storage = storage
        self.schema = schema
        self.kwargs = kwargs
        self.form = form
        self.behavior = behavior

    @abc.abstractmethod
    def __call__(self, source: Any) -> ak.Array:
        ...

    @property
    def use_optimization(self) -> bool:
        return (
            "json"
            in dask.config.get(
                "awkward.optimization.columns-opt-formats",
                default=[],
            )
            and self.schema is None
        )

    def project_columns(self, columns):
        form = self.form.select_columns(columns)
        assert form is not None
        schema = layout_to_jsonschema(form.length_zero_array(highlevel=False))

        return type(self)(
            schema=schema,
            form=self.form,
            storage=self.storage,
            compression=self.compression,
            behavior=self.behavior,
            **self.kwargs,
        )


class FromJsonLineDelimitedFn(FromJsonFn):
    def __init__(
        self,
        *,
        storage: AbstractFileSystem,
        form: Form,
        compression: str | None = None,
        schema: str | dict | list | None = None,
        behavior: Mapping | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            storage=storage,
            compression=compression,
            schema=schema,
            form=form,
            behavior=behavior,
            **kwargs,
        )

    def __call__(self, source: str) -> ak.Array:
        with self.storage.open(source, mode="rt", compression=self.compression) as f:
            array = ak.from_json(
                f.read(),
                line_delimited=True,
                schema=self.schema,
                **self.kwargs,
            )
        log.debug("columns read from disk: %s" % str(array.layout.form.columns()))
        assert isinstance(array, ak.Array)
        return array
        # return ak.Array(unproject_layout(self.original_form, array.layout))


class FromJsonSingleObjPerFile(FromJsonFn):
    def __init__(
        self,
        *,
        storage: AbstractFileSystem,
        form: Form,
        compression: str | None = None,
        schema: str | dict | list | None = None,
        behavior: Mapping | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            storage=storage,
            compression=compression,
            schema=schema,
            form=form,
            behavior=behavior,
            **kwargs,
        )

    def __call__(self, source: str) -> ak.Array:
        with self.storage.open(source, mode="rt", compression=self.compression) as f:
            array = ak.Array(
                [
                    ak.from_json(
                        f.read(),
                        line_delimited=False,
                        schema=self.schema,
                        **self.kwargs,
                    )
                ]
            )
        log.debug("columns read from disk: %s" % str(array.layout.form.columns()))
        assert isinstance(array, ak.Array)
        return array
        # return ak.Array(unproject_layout(self.original_form, array.layout))


class FromJsonBytesFn(FromJsonFn):
    def __init__(
        self,
        *,
        storage: AbstractFileSystem,
        form: Form,
        compression: str | None = None,
        schema: str | dict | list | None = None,
        behavior: Mapping | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            storage=storage,
            compression=compression,
            schema=schema,
            behavior=behavior,
            form=form,
            **kwargs,
        )

    def __call__(self, instructions: _BytesReadingInstructions) -> ak.Array:
        with instructions.fs.open(
            instructions.path, compression=instructions.compression
        ) as f:
            if instructions.offset == 0 and instructions.length is None:
                bytestring = f.read()
            else:
                bytestring = read_block(
                    f,
                    instructions.offset,
                    instructions.length,
                    instructions.delimiter,
                )

        array = ak.from_json(
            bytestring,
            line_delimited=True,
            schema=self.schema,
            **self.kwargs,
        )
        log.debug("columns read from disk: %s" % str(array.layout.form.columns()))
        assert isinstance(array, ak.Array)
        return array
        # return ak.Array(unproject_layout(self.original_form, array.layout))


def meta_from_single_file(
    *,
    fs: AbstractFileSystem,
    paths: list[str],
    compression: str | None,
    **kwargs: Any,
) -> ak.Array:
    with fs.open(paths[0], compression=compression) as f:
        array = ak.Array([ak.from_json(f.read(), line_delimited=False, **kwargs)])
    return typetracer_array(array)


def meta_from_bytechunks(
    *,
    fs: AbstractFileSystem,
    paths: list[str],
    sample_bytes: str | int,
    **kwargs: Any,
) -> ak.Array:
    sample_bytes = parse_bytes(sample_bytes)
    bytes = fs.cat(paths[0], start=0, end=sample_bytes)
    rfind = bytes.rfind(b"\n")
    if rfind > 0:
        bytes = bytes[:rfind]
    array = ak.from_json(bytes, line_delimited=True, **kwargs)
    assert isinstance(array, ak.Array)
    return typetracer_array(array)


def meta_from_line_by_line(
    *,
    fs: AbstractFileSystem,
    paths: list[str],
    compression: str | None,
    sample_rows: int | None,
    **kwargs: Any,
) -> ak.Array:
    if sample_rows is not None:
        lines = []
        with fs.open(paths[0], mode="rt", compression=compression) as f:
            for i, line in enumerate(f):
                lines.append(line)
                if i >= sample_rows:
                    break
        array = ak.from_json("\n".join(lines), line_delimited=True, **kwargs)
    else:
        with fs.open(paths[0], mode="rt", compression=compression) as f:
            array = ak.from_json(
                f.read(),
                line_delimited=True,
                **kwargs,
            )
    assert isinstance(array, ak.Array)
    return typetracer_array(array)


def _from_json_files(
    *,
    fs: AbstractFileSystem,
    token: str,
    paths: list[str],
    schema: str | dict | list | None,
    compression: str | None,
    sample_rows: int | None = 150,
    sample_bytes: str | int = "256 KiB",
    **kwargs: Any,
) -> Array:
    if compression == "infer":
        compression = infer_compression(paths[0])

    if not compression:
        meta = meta_from_bytechunks(
            fs=fs,
            paths=paths,
            sample_bytes=sample_bytes,
            **kwargs,
        )
    else:
        meta = meta_from_line_by_line(
            fs=fs,
            paths=paths,
            compression=compression,
            sample_rows=sample_rows,
            **kwargs,
        )

    token = tokenize(compression, meta, kwargs)

    f = FromJsonLineDelimitedFn(
        storage=fs,
        compression=compression,
        schema=schema,
        form=meta.layout.form,
        **kwargs,
    )

    return cast(
        Array,
        from_map(
            f,
            paths,
            label="from-json-files",
            token=token,
            meta=meta,
        ),
    )


def _from_json_sopf(
    *,
    fs: AbstractFileSystem,
    token: str,
    paths: list[str],
    schema: str | dict | list | None,
    compression: str | None,
    **kwargs: Any,
) -> Array:
    if compression == "infer":
        compression = infer_compression(paths[0])

    meta = meta_from_single_file(
        fs=fs,
        paths=paths,
        compression=compression,
        **kwargs,
    )
    token = tokenize(token, compression, meta, kwargs)

    f = FromJsonSingleObjPerFile(
        storage=fs,
        compression=compression,
        schema=schema,
        form=meta.layout.form,
        **kwargs,
    )

    return cast(
        Array,
        from_map(
            f,
            paths,
            label="from-json-sopf",
            token=token,
            meta=meta,
        ),
    )


def _from_json_bytes(
    fs: AbstractFileSystem,
    token: str,
    paths: list[str],
    *,
    schema: str | dict | list | None = None,
    compression: str | None = "infer",
    delimiter: bytes = b"\n",
    not_zero: bool = False,
    blocksize: str | int = "128 MiB",
    sample_bytes: str | int = "10 kiB",
    **kwargs: Any,
) -> Array:
    if compression == "infer":
        compression = infer_compression(paths[0])

    token = tokenize(
        fs,
        token,
        paths,
        schema,
        compression,
        delimiter,
        not_zero,
        blocksize,
        sample_bytes,
        kwargs,
    )

    bytes_ingredients, the_sample_bytes = _bytes_with_sample(
        fs=fs,
        paths=paths,
        compression=compression,
        delimiter=delimiter,
        not_zero=not_zero,
        blocksize=blocksize,
        sample=sample_bytes,
    )

    sample_array = ak.from_json(the_sample_bytes, line_delimited=True, **kwargs)
    assert isinstance(sample_array, ak.Array)
    meta = typetracer_array(sample_array)

    fn = FromJsonBytesFn(
        storage=fs,
        compression=compression,
        schema=schema,
        form=meta.layout.form,
        **kwargs,
    )

    return cast(
        Array,
        from_map(
            fn,
            list(flatten(bytes_ingredients)),
            label="from-json-bytes",
            token=token,
            meta=meta,
        ),
    )


def json_fs_token_paths(
    source: Any,
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[AbstractFileSystem, str, list[str]]:
    fs, token, paths = get_fs_token_paths(source, storage_options=storage_options)

    # if paths is length 1, check to see if it's a directory and
    # wildcard search for JSON files. Otherwise, just keep whatever
    # has already been found.
    if len(paths) == 1 and fs.isdir(paths[0]):
        paths = list(filter(lambda s: ".json" in s, fs.find(paths[0])))

    return fs, token, paths


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
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
    blocksize: int | str | None = None,
    delimiter: bytes | None = None,
    compression: str | None = "infer",
    storage_options: dict[str, Any] | None = None,
    meta_sample_rows: int | None = 100,
    meta_sample_bytes: int | str = "10 kiB",
) -> Array:
    """Create an Array collection from JSON data.

    See :func:`ak.from_json` for more information.

    Parameters
    ----------
    source : str | list[str]
        Local or remote directory or list of files containing JSON
        data to load. May contain glob patterns (passed to ``fsspec``).
    line_delimited : bool
        If ``True`` (the default) treat each line in the file as a
        JSON object, if ``False``, entire files will be treated as
        single objects.
    schema : str | dict | list, optional
        If defined the schema will be used by the parser to skip type
        discovery. If not defined (``None``, the default),
        dask-awkward's optimization capabilities will potentially be
        used to generate a JSONSchema that contains the minimal
        necessary parts of the JSON data that should be used to build
        an Array to complete the desired computation. See
        dask-awkward's optimization documentation for more
        information.
    nan_string : str, optional
        See :func:`ak.from_json`
    posinf_string : str, optional
        See :func:`ak.from_json`
    neginf_string : str, optional
        See :func:`ak.from_json`
    complex_record_fields : tuple[str, str], optional
        See :func:`ak.from_json`
    buffersize : int
        See :func:`ak.from_json`
    initial : int
        See :func:`ak.from_json`
    resize : float
        See :func:`ak.from_json`
    highlevel : bool
        Argument specific to awkward-array that is always ``True`` for
        dask-awkward.
    behavior : dict, optional
        See :func:`ak.from_json`
    blocksize : int, str, optional
        If ``None`` (default), the collection will be partitioned on a
        per-file bases. If defined, this sets the size (in bytes) of
        each partition. Can be a string of the form ``"10 MiB"``.
    delimiter : bytes, optional
        Delimiter to use for separating blocks; if ``blocksize`` is
        defined but this argument is not defined, the default is the
        bytestring newline: ``b"\\n"``.
    compression : str, optional
        The compression of the dataset (default is to infer based on
        file suffix)
    storage_options : dict[str, Any], optional
        Storage options based to ``fsspec``.
    meta_sample_rows : int, optional
        Number of rows to sample from files for determining metadata.
        When reading files partitioned on a per-file basis this will
        be the number of lines extracted from the first file to
        determine the collection's metadata.
    meta_sample_bytes : int | str
        Number of bytes to sample from files for determining metadata.
        When reading file partitioned on a blocksize basis this will
        be the number of bytes sampled from the first partition to
        determine the collection's metadata.

    Returns
    -------
    Array
        Resulting collection.

    Examples
    --------
    An example where data is stored in an S3 data; this will grab all
    JSON files under the path with blocksizes of 50 MB and we sample
    the first 10 MB to determine metadata:

    >>> import dask_awkward as dak
    >>> ds = dak.from_json(
    ...     "s3://path/to/data",
    ...     blocksize="50 MB",
    ...     meta_sample_byes="10 MB",
    ... )

    An example where a JSONSchema is pre-defined. In this case
    dask-awkward's optimization infrastructure will not attempt to
    generate a minimal necessary schema, it will use the one provided:

    >>> import dask_awkward as dak
    >>> my_schema = ...
    >>> ds = dak.from_json(["file1.json", "file2.json"], schema=my_schema)

    An example where each discovered file will be treated as a single
    JSON object when creating the Array collection:

    >>> import dask_awkward as dak
    >>> ds = dak.from_json("/path/to/files/**.json", line_delimited=False)

    """
    if not highlevel:
        raise ValueError("dask-awkward only supports highlevel awkward Arrays.")

    fs, token, paths = json_fs_token_paths(source, storage_options=storage_options)

    if len(paths) == 0:
        raise OSError("%s resolved to no files" % source)

    # allow either blocksize or delimieter being not-None to trigger
    # line deliminated JSON reading.
    if blocksize is not None and delimiter is None:
        delimiter = b"\n"
    elif blocksize is None and delimiter == b"\n":
        blocksize = "128 MiB"

    # if line delimited is False we use the single object per file
    # implementation.
    if not line_delimited:
        return _from_json_sopf(
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

    # if we are not using blocksize and delimiter we are partitioning
    # by file.
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
            sample_rows=meta_sample_rows,
            sample_bytes=meta_sample_bytes,
        )

    # if a `delimiter` and `blocksize` are defined we use the byte
    # reading implementation
    elif delimiter is not None and blocksize is not None:
        return _from_json_bytes(
            fs=fs,
            token=token,
            paths=paths,
            schema=schema,
            delimiter=delimiter,
            blocksize=blocksize,
            sample_bytes=meta_sample_bytes,
            behavior=behavior,
            nan_string=nan_string,
            posinf_string=posinf_string,
            neginf_string=neginf_string,
            complex_record_fields=complex_record_fields,
            buffersize=buffersize,
            initial=initial,
            resize=resize,
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


def to_json(
    array: Array,
    path: str,
    line_delimited: bool = True,
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
    compute: bool = True,
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
    line_delimited : bool
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
    graph = HighLevelGraph.from_collections(
        name,
        AwkwardMaterializedLayer(dsk, previous_layer_names=[map_res.name]),
        dependencies=(map_res,),
    )
    res = new_scalar_object(graph, name=name, dtype="f8")
    if compute:
        res.compute()
        return None
    return res


@overload
def json_type(original: str, add_null: Literal[False] = False) -> str:
    ...


@overload
def json_type(original: str, add_null: Literal[True]) -> list[str]:
    ...


@overload
def json_type(original: str, add_null: bool) -> str | list[str]:
    ...


@overload
def json_type(original: list[str], add_null: bool) -> list[str]:
    ...


def json_type(original: str | list[str], add_null: bool = False) -> str | list[str]:
    if isinstance(original, str):
        if add_null:
            return [original, "null"]
        else:
            return original
    elif isinstance(original, list):
        if add_null:
            return original + ["null"]
        else:
            return original


def array_param_is_string_or_bytestring(layout: Content) -> bool:
    params = layout.parameters or {}
    return params == {"__array__": "string"} or params == {"__array__": "bytestring"}


def layout_to_jsonschema(
    layout: Content,
    existing_schema: dict | None = None,
    title: str = "untitled",
    description: str = "Auto generated by dask-awkward",
    required: bool = False,
    is_option: bool = False,
) -> dict:
    """Convert awkward array Layout to a JSON Schema dictionary."""
    if existing_schema is None:
        existing_schema = {
            "title": title,
            "description": description,
            "type": "object",
            "properties": {},
        }
    if layout.is_option:
        layout_to_jsonschema(layout.content, existing_schema, is_option=True)
    elif layout.is_record:
        existing_schema["type"] = json_type("object", add_null=is_option)
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
        existing_schema["type"] = json_type("string", add_null=is_option)
    elif layout.is_list:
        existing_schema["type"] = json_type("array", add_null=is_option)
        if layout.is_regular:
            existing_schema["minItems"] = layout.size
            existing_schema["maxItems"] = layout.size
        existing_schema["items"] = {}
        layout_to_jsonschema(layout.content, existing_schema["items"])
    elif layout.is_numpy:
        if layout.dtype.kind == "i":
            existing_schema["type"] = json_type("integer", add_null=is_option)
        elif layout.dtype.kind == "f":
            existing_schema["type"] = json_type("number", add_null=is_option)
        elif layout.dtype.kind == "b":
            existing_schema["type"] = json_type("boolean", add_null=is_option)
    elif layout.is_indexed:
        pass
    elif layout.is_unknown:
        existing_schema["type"] = "null"
    elif layout.is_union:
        existing_schema["type"] = [
            layout_to_jsonschema(content)["type"] for content in layout.contents
        ]
    return existing_schema
