from __future__ import annotations

import abc
import itertools
import logging
import math
import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import awkward as ak
import awkward.operations.ak_from_parquet as ak_from_parquet
import dask
from awkward.forms.form import Form
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.highlevelgraph import HighLevelGraph
from fsspec import AbstractFileSystem
from fsspec.core import get_fs_token_paths, url_to_fs

from dask_awkward.layers.layers import AwkwardMaterializedLayer
from dask_awkward.lib.core import Array, Scalar, map_partitions, new_scalar_object
from dask_awkward.lib.io.columnar import ColumnProjectionMixin
from dask_awkward.lib.io.io import from_map
from dask_awkward.lib.unproject_layout import unproject_layout

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


T = TypeVar("T")


def report_failure(exception, *args, **kwargs):
    return ak.Array(
        [
            {
                "columns": [],
                "args": [repr(a) for a in args],
                "kwargs": [[k, repr(v)] for k, v in kwargs.items()],
                "exception": type(exception).__name__,
                "message": str(exception),
            }
        ]
    )


def report_success(columns, *args, **kwargs):
    return ak.Array(
        [
            {
                "columns": columns,
                "args": [repr(a) for a in args],
                "kwargs": [[k, repr(v)] for k, v in kwargs.items()],
                "exception": None,
                "message": None,
            }
        ]
    )


class FromParquetFn(ColumnProjectionMixin):
    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        form: Any,
        listsep: str = "list.item",
        unnamed_root: bool = False,
        original_form: Form | None = None,
        report: bool = False,
        allowed_exceptions: tuple[type[BaseException], ...] = (OSError,),
        behavior: Mapping | None = None,
        attrs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.fs = fs
        self.form = form
        self.listsep = listsep
        self.unnamed_root = unnamed_root
        self.columns = self.form.columns(self.listsep)
        if self.unnamed_root:
            self.columns = [f".{c}" for c in self.columns]
        self.original_form = original_form
        self.report = report
        self.allowed_exceptions = allowed_exceptions
        self.behavior = behavior
        self.attrs = attrs
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def project_columns(self, columns):
        ...

    @property
    def return_report(self) -> bool:
        return self.report

    @property
    def use_optimization(self) -> bool:
        return "parquet" in dask.config.get(
            "awkward.optimization.columns-opt-formats",
            default=[],
        )

    def __repr__(self) -> str:
        s = (
            "\nFromParquetFn(\n"
            f"  form={repr(self.form)}\n"
            f"  listsep={self.listsep}\n"
            f"  unnamed_root={self.unnamed_root}\n"
            f"  columns={self.columns}\n"
            f"  behavior={self.behavior}\n"
        )
        for key, val in self.kwargs.items():
            s += f"  {key}={val}\n"
        s = f"{s})"
        return s

    def __str__(self) -> str:
        return self.__repr__()


class FromParquetFileWiseFn(FromParquetFn):
    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        form: Any,
        listsep: str = "list.item",
        unnamed_root: bool = False,
        original_form: Form | None = None,
        behavior: Mapping | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fs=fs,
            form=form,
            listsep=listsep,
            unnamed_root=unnamed_root,
            original_form=original_form,
            behavior=behavior,
            **kwargs,
        )

    def read_fn(self, source: Any) -> Any:
        layout = ak_from_parquet._load(
            [source],
            parquet_columns=self.columns,
            subrg=[None],
            subform=self.form,
            highlevel=False,
            fs=self.fs,
            behavior=self.behavior,
            attrs=self.attrs,
            **self.kwargs,
        )
        return ak.Array(
            unproject_layout(self.original_form, layout),
            attrs=self.attrs,
            behavior=self.behavior,
        )

    def __call__(self, *args, **kwargs):
        source = args[0]
        if self.return_report:
            try:
                result = self.read_fn(source)
                return result, report_success(self.columns, source)
            except self.allowed_exceptions as err:
                return self.mock_empty(), report_failure(err, source)

        return self.read_fn(source)

    def project_columns(self, columns):
        return FromParquetFileWiseFn(
            fs=self.fs,
            form=self.form.select_columns(columns),
            listsep=self.listsep,
            unnamed_root=self.unnamed_root,
            original_form=self.form,
            report=self.report,
            attrs=self.attrs,
            behavior=self.behavior,
            **self.kwargs,
        )


class FromParquetFragmentWiseFn(FromParquetFn):
    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        form: Any,
        listsep: str = "list.item",
        unnamed_root: bool = False,
        original_form: Form | None = None,
        behavior: Mapping | None = None,
        attrs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fs=fs,
            form=form,
            listsep=listsep,
            unnamed_root=unnamed_root,
            original_form=original_form,
            behavior=behavior,
            attrs=attrs,
            **kwargs,
        )

    def __call__(self, pair: Any) -> ak.Array:
        subrg, source = pair
        if isinstance(subrg, int):
            subrg = [[subrg]]
        layout = ak_from_parquet._load(
            [source],
            parquet_columns=self.columns,
            subrg=subrg,
            subform=self.form,
            highlevel=False,
            fs=self.fs,
            behavior=self.behavior,
            attrs=self.attrs,
            **self.kwargs,
        )
        return ak.Array(
            unproject_layout(self.original_form, layout),
            behavior=self.behavior,
            attrs=self.attrs,
        )

    def project_columns(self, columns):
        return FromParquetFragmentWiseFn(
            fs=self.fs,
            form=self.form.select_columns(columns),
            unnamed_root=self.unnamed_root,
            original_form=self.form,
            report=self.report,
            behavior=self.behavior,
            attrs=self.attrs,
            **self.kwargs,
        )


def from_parquet(
    path: str | list[str],
    *,
    columns: str | list[str] | None = None,
    max_gap: int = 64_000,
    max_block: int = 256_000_000,
    footer_sample_size: int = 1_000_000,
    generate_bitmasks: bool = False,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
    ignore_metadata: bool = True,
    scan_files: bool = False,
    split_row_groups: bool | None = False,
    storage_options: dict[str, Any] | None = None,
    report: bool = False,
) -> Array | tuple[Array, Array]:
    """Create an Array collection from a Parquet dataset.

    See :func:`ak.from_parquet` for more information.

    Parameters
    ----------
    path
        Local directory containing parquet files, remote URL directory
        containing Parquet files, or explicit list of Parquet files,
        passed to fsspec for resolution. May contain glob patterns.
    columns
        See :func:`ak.from_parquet`
    max_gap
        See :func:`ak.from_parquet`
    max_block
        See :func:`ak.from_parquet`
    footer_sample_size
        See :func:`ak.from_parquet`
    generate_bitmasks
        See :func:`ak.from_parquet`
    highlevel
        Argument specific to awkward-array that is always ``True`` for
        dask-awkward.
    behavior
        See :func:`ak.from_parquet`
    ignore_metadata
        If ``True``, ignore Parquet metadata file (if it exists).
    scan_files
        Scan files when parsing metadata.
    split_row_groups
        If True, each row group becomes a partition. If False, each
        file becomes a partition. If None, the existence of a
        ``_metadata`` file and ignore_metadata=False implies True,
        else ``False``.
    storage_options
        Storage options passed to fsspec.

    Returns
    -------
    Array
        Collection represented by the Parquet data on disk.

    """
    if not highlevel:
        raise ValueError("dask-awkward only supports highlevel=True")

    fs, token, paths = get_fs_token_paths(
        path,
        mode="rb",
        storage_options=storage_options,
    )
    label = "from-parquet"
    token = tokenize(
        token,
        paths,
        columns,
        max_gap,
        max_block,
        footer_sample_size,
        generate_bitmasks,
        behavior,
        ignore_metadata,
        scan_files,
        split_row_groups,
        behavior,
        attrs,
    )

    (
        parquet_columns,
        subform,
        actual_paths,
        fs,
        subrg,
        row_counts,
        metadata,
    ) = ak_from_parquet.metadata(
        path,
        storage_options,
        row_groups=None,
        columns=columns,
        ignore_metadata=ignore_metadata,
        scan_files=scan_files,
    )

    listsep = "list.item"
    unnamed_root = False
    for c in parquet_columns:
        if ".list.element" in c:
            listsep = "list.element"
            break
        if c.startswith("."):
            unnamed_root = True

    if split_row_groups is None:
        split_row_groups = row_counts is not None and len(row_counts) > 1

    if split_row_groups is False or subrg is None:
        # file-wise
        return from_map(
            FromParquetFileWiseFn(
                fs=fs,
                form=subform,
                listsep=listsep,
                unnamed_root=unnamed_root,
                max_gap=max_gap,
                max_block=max_block,
                footer_sample_size=footer_sample_size,
                generate_bitmasks=generate_bitmasks,
                behavior=behavior,
                attrs=attrs,
                report=report,
            ),
            actual_paths,
            label=label,
            token=token,
        )
    else:
        # row-group wise
        if set(subrg) == {None}:
            rgs_paths = {path: 0 for path in actual_paths}
            for i in range(metadata.num_row_groups):
                fp = metadata.row_group(i).column(0).file_path
                rgs_path = [p for p in rgs_paths if fp in p][
                    0
                ]  # returns 1st if fp is empty
                rgs_paths[rgs_path] += 1

            subrg = [list(range(rgs_paths[_])) for _ in actual_paths]

        rgs = [metadata.row_group(i) for i in range(metadata.num_row_groups)]
        divisions = [0] + list(
            itertools.accumulate([rg.num_rows for rg in rgs], operator.add)
        )
        pairs = []

        for isubrg, path in zip(subrg, actual_paths):
            pairs.extend([(irg, path) for irg in isubrg])

        return cast(
            Array,
            from_map(
                FromParquetFragmentWiseFn(
                    fs=fs,
                    form=subform,
                    listsep=listsep,
                    unnamed_root=unnamed_root,
                    max_gap=max_gap,
                    max_block=max_block,
                    footer_sample_size=footer_sample_size,
                    generate_bitmasks=generate_bitmasks,
                    behavior=behavior,
                    attrs=attrs,
                ),
                pairs,
                label=label,
                token=token,
                divisions=tuple(divisions),
            ),
        )


def _metadata_file_from_data_files(path_list, fs, out_path):
    """
    Aggregate _metadata and _common_metadata from data files

    Maybe only used in testing

    (similar to fastparquet's merge)

    path_list: list[str]
        Input data files
    fs: AbstractFileSystem instance
    out_path: str
        Root directory of the dataset
    """
    import pyarrow.parquet as pq

    meta = None
    out_path = out_path.rstrip("/")
    for path in path_list:
        assert path.startswith(out_path)
        with fs.open(path, "rb") as f:
            _meta = pq.ParquetFile(f).metadata
        _meta.set_file_path(path[len(out_path) + 1 :])
        if meta:
            meta.append_row_groups(_meta)
        else:
            meta = _meta
    _write_metadata(fs, out_path, meta)


def _metadata_file_from_metas(fs, out_path, *metas):
    """Agregate metadata from arrow objects and write"""
    meta = metas[0]
    for _meta in metas[1:]:
        meta.append_row_groups(_meta)
    _write_metadata(fs, out_path, meta)


def _write_metadata(fs, out_path, meta):
    """Output metadata files"""
    metadata_path = "/".join([out_path, "_metadata"])
    with fs.open(metadata_path, "wb") as fil:
        meta.write_metadata_file(fil)
    metadata_path = "/".join([out_path, "_metadata"])
    with fs.open(metadata_path, "wb") as fil:
        meta.write_metadata_file(fil)


class _ToParquetFn:
    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
        npartitions: int,
        prefix: str | None = None,
        storage_options: dict | None = None,
        **kwargs: Any,
    ):
        self.fs = fs
        self.path = path
        self.prefix = prefix
        self.zfill = math.ceil(math.log(npartitions, 10))
        self.storage_options = storage_options
        self.fs.mkdirs(self.path, exist_ok=True)
        self.protocol = (
            self.fs.protocol
            if isinstance(self.fs.protocol, str)
            else self.fs.protocol[0]
        )
        self.kwargs = kwargs

    def __call__(self, data, block_index):
        filename = f"part{str(block_index[0]).zfill(self.zfill)}.parquet"
        if self.prefix is not None:
            filename = f"{self.prefix}-{filename}"
        filename = f"{self.protocol}://{self.path}/{filename}"
        return ak.to_parquet(
            data, filename, **self.kwargs, storage_options=self.storage_options
        )


def to_parquet(
    array: Array,
    destination: str,
    list_to32: bool = False,
    string_to32: bool = True,
    bytestring_to32: bool = True,
    emptyarray_to: Any | None = None,
    categorical_as_dictionary: bool = False,
    extensionarray: bool = False,
    count_nulls: bool = True,
    compression: str | dict | None = "zstd",
    compression_level: int | dict | None = None,
    row_group_size: int | None = 64 * 1024 * 1024,
    data_page_size: int | None = None,
    parquet_flavor: Literal["spark"] | None = None,
    parquet_version: Literal["1.0"] | Literal["2.4"] | Literal["2.6"] = "2.4",
    parquet_page_version: Literal["1.0"] | Literal["2.0"] = "1.0",
    parquet_metadata_statistics: bool | dict = True,
    parquet_dictionary_encoding: bool | dict = False,
    parquet_byte_stream_split: bool | dict = False,
    parquet_coerce_timestamps: Literal["ms"] | Literal["us"] | None = None,
    parquet_old_int96_timestamps: bool | None = None,
    parquet_compliant_nested: bool = False,
    parquet_extra_options: dict | None = None,
    storage_options: dict[str, Any] | None = None,
    write_metadata: bool = False,
    compute: bool = True,
    prefix: str | None = None,
) -> Scalar | None:
    """Write data to Parquet format.

    This will create one output file per partition.

    See the documentation for :func:`ak.to_parquet` for more
    information; there are many optional function arguments that are
    described in that documentation.

    Parameters
    ----------
    array
        The :obj:`dask_awkward.Array` collection to write to disk.
    destination
        Where to store the output; this can be a local filesystem path
        or a remote filesystem path.
    list_to32
        See :func:`ak.to_parquet`
    string_to32
        See :func:`ak.to_parquet`
    bytestring_to32
        See :func:`ak.to_parquet`
    emptyarray_to
        See :func:`ak.to_parquet`
    categorical_as_dictionary
        See :func:`ak.to_parquet`
    extensionarray
        See :func:`ak.to_parquet`
    count_nulls
        See :func:`ak.to_parquet`
    compression
        See :func:`ak.to_parquet`
    compression_level
        See :func:`ak.to_parquet`
    row_group_size
        See :func:`ak.to_parquet`
    data_page_size
        See :func:`ak.to_parquet`
    parquet_flavor
        See :func:`ak.to_parquet`
    parquet_version
        See :func:`ak.to_parquet`
    parquet_page_version
        See :func:`ak.to_parquet`
    parquet_metadata_statistics
        See :func:`ak.to_parquet`
    parquet_dictionary_encoding
        See :func:`ak.to_parquet`
    parquet_byte_stream_split
        See :func:`ak.to_parquet`
    parquet_coerce_timestamps
        See :func:`ak.to_parquet`
    parquet_old_int96_timestamps
        See :func:`ak.to_parquet`
    parquet_compliant_nested
        See :func:`ak.to_parquet`
    parquet_extra_options
        See :func:`ak.to_parquet`
    storage_options
        Storage options passed to ``fsspec``.
    write_metadata
        Write Parquet metadata.
    compute
        If ``True``, immediately compute the result (write data to
        disk). If ``False`` a Scalar collection will be returned such
        that ``compute`` can be explicitly called.
    prefix
        An addition prefix for output files. If ``None`` all parts
        inside the destination directory will be named
        ``"partN.parquet"``; if defined, the names will be
        ``f"{prefix}-partN.parquet"``.

    Returns
    -------
    Scalar | None
        If ``compute`` is ``False`` a :obj:`dask_awkward.Scalar`
        object is returned such that it can be computed later. If
        ``compute`` is ``True``, the collection is immediately
        computed (and data will be written to disk) and ``None`` is
        returned.

    Examples
    --------

    >>> import awkward as ak
    >>> import dask_awkward as dak
    >>> a = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    >>> d = dak.from_awkward(a, npartitions=2)
    >>> d.npartitions
    2
    >>> dak.to_parquet(d, "/tmp/my-output", prefix="data")
    >>> import os
    >>> os.listdir("/tmp/my-output")
    ['data-part0.parquet', 'data-part1.parquet']


    """
    # TODO options we need:
    #  - byte stream split for floats if compression is not None or lzma
    #  - partitioning
    #  - dict encoding always off
    fs, path = url_to_fs(destination, **(storage_options or {}))
    name = f"write-parquet-{tokenize(fs, array, destination)}"

    map_res = map_partitions(
        _ToParquetFn(
            fs=fs,
            path=path,
            npartitions=array.npartitions,
            prefix=prefix,
            list_to32=list_to32,
            string_to32=string_to32,
            bytestring_to32=bytestring_to32,
            emptyarray_to=emptyarray_to,
            categorical_as_dictionary=categorical_as_dictionary,
            extensionarray=extensionarray,
            count_nulls=count_nulls,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
            data_page_size=data_page_size,
            parquet_flavor=parquet_flavor,
            parquet_version=parquet_version,
            parquet_page_version=parquet_page_version,
            parquet_metadata_statistics=parquet_metadata_statistics,
            parquet_dictionary_encoding=parquet_dictionary_encoding,
            parquet_byte_stream_split=parquet_byte_stream_split,
            parquet_coerce_timestamps=parquet_coerce_timestamps,
            parquet_old_int96_timestamps=parquet_old_int96_timestamps,
            parquet_compliant_nested=parquet_compliant_nested,
            parquet_extra_options=parquet_extra_options,
        ),
        array,
        BlockIndex((array.npartitions,)),
        label="to-parquet",
        meta=array._meta,
    )
    map_res.dask.layers[map_res.name].annotations = {"ak_output": True}

    dsk = {}
    if write_metadata:
        final_name = name + "-metadata"
        dsk[(final_name, 0)] = (_metadata_file_from_metas, fs, path) + tuple(
            map_res.__dask_keys__()
        )
    else:
        final_name = name + "-finalize"
        dsk[(final_name, 0)] = (lambda *_: None, map_res.__dask_keys__())
    graph = HighLevelGraph.from_collections(
        final_name,
        AwkwardMaterializedLayer(dsk, previous_layer_names=[map_res.name]),
        dependencies=[map_res],
    )
    out = new_scalar_object(graph, final_name, dtype="f8")
    if compute:
        out.compute()
        return None
    else:
        return out
