from __future__ import annotations

import abc
import itertools
import logging
import math
import operator
from typing import Any, Sequence

import awkward as ak
import fsspec
from awkward.forms.form import Form
from awkward.operations import ak_from_parquet, to_arrow_table
from awkward.operations.ak_from_parquet import _load
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.highlevelgraph import HighLevelGraph
from fsspec import AbstractFileSystem
from fsspec.core import get_fs_token_paths

from dask_awkward.lib.core import (
    Array,
    Scalar,
    map_partitions,
    new_scalar_object,
    typetracer_array,
)
from dask_awkward.lib.io.io import from_map
from dask_awkward.lib.unproject_layout import unproject_layout

log = logging.getLogger(__name__)


class _FromParquetFn:
    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        schema: Any,
        listsep: str = "list.item",
        unnamed_root: bool = False,
        original_form: Form | None = None,
    ) -> None:
        self.fs = fs
        self.schema = schema
        self.listsep = listsep
        self.unnamed_root = unnamed_root
        self.columns = self.schema.columns(self.listsep)
        if self.unnamed_root:
            self.columns = [f".{c}" for c in self.columns]
        self.original_form = original_form

    @abc.abstractmethod
    def __call__(self, source: Any) -> ak.Array:
        ...

    @abc.abstractmethod
    def project_columns(
        self,
        columns: Sequence[str] | None,
        orignal_form: Form | None = None,
    ) -> _FromParquetFn:
        ...

    def __repr__(self) -> str:
        s = (
            "\nFromParquetFn(\n"
            f"  schema={repr(self.schema)}\n"
            f"  listsep={self.listsep}\n"
            f"  unnamed_root={self.unnamed_root}\n"
            f"  columns={self.columns}\n"
        )
        return s

    def __str__(self) -> str:
        return self.__repr__()


class _FromParquetFileWiseFn(_FromParquetFn):
    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        schema: Any,
        listsep: str = "list.item",
        unnamed_root: bool = False,
        original_form: Form | None = None,
    ) -> None:
        super().__init__(
            fs=fs,
            schema=schema,
            listsep=listsep,
            unnamed_root=unnamed_root,
            original_form=original_form,
        )

    def __call__(self, source: Any) -> Any:
        array = _file_to_partition(
            source,
            self.fs,
            self.columns,
            self.schema,
        )
        return ak.Array(unproject_layout(self.original_form, array.layout))

    def project_columns(
        self,
        columns: Sequence[str] | None,
        original_form: Form | None = None,
    ) -> _FromParquetFileWiseFn:
        if columns is None:
            return self

        new_schema = self.schema.select_columns(columns)
        new = _FromParquetFileWiseFn(
            fs=self.fs,
            schema=new_schema,
            listsep=self.listsep,
            unnamed_root=self.unnamed_root,
            original_form=original_form,
        )

        log.debug(f"project_columns received: {columns}")
        log.debug(f"new schema is {repr(new_schema)}")
        log.debug(f"new schema columns are: {new_schema.columns(self.listsep)}")
        log.debug(new)

        return new


class _FromParquetFragmentWiseFn(_FromParquetFn):
    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        schema: Any,
        listsep: str = "list.item",
        unnamed_root: bool = False,
        original_form: Form | None = None,
    ) -> None:
        super().__init__(
            fs=fs,
            schema=schema,
            listsep=listsep,
            unnamed_root=unnamed_root,
            original_form=original_form,
        )

    def __call__(self, pair: Any) -> ak.Array:
        subrg, source = pair
        if isinstance(subrg, int):
            subrg = [[subrg]]
        array = _file_to_partition(
            source,
            self.fs,
            self.columns,
            self.schema,
            subrg=subrg,
        )
        return ak.Array(unproject_layout(self.original_form, array.layout))

    def project_columns(
        self,
        columns: Sequence[str] | None,
        original_form: Form | None = None,
    ) -> _FromParquetFragmentWiseFn:
        if columns is None:
            return self
        return _FromParquetFragmentWiseFn(
            fs=self.fs,
            schema=self.schema.select_columns(columns),
            unnamed_root=self.unnamed_root,
            original_form=original_form,
        )


def from_parquet(
    path: Any,
    storage_options: dict | None = None,
    ignore_metadata: bool = True,
    scan_files: bool = False,
    columns: Sequence[str] | None = None,
    filters: Any | None = None,
    split_row_groups: Any | None = None,
) -> Array:
    """Create an Array collection from a Parquet dataset.

    Parameters
    ----------
    url : str
        Location of data, including protocol (e.g. ``s3://``)
    storage_options : dict
        For creating filesystem (see ``fsspec`` documentation).
    ignore_metadata : bool
        Ignore parquet metadata associated with the input dataset (the
        ``_metadata`` file).
    scan_files : bool
        TBD
    columns : list[str], optional
        Select columns to load
    filters : list[list[tuple]], optional
        Parquet-style filters for excluding row groups based on column statistics
    split_row_groups: bool, optional
        If True, each row group becomes a partition. If False, each
        file becomes a partition. If None, the existence of a
        ``_metadata`` file and ignore_metadata=False implies True,
        else False.

    Returns
    -------
    Array
        Array collection from the parquet dataset.

    """
    fs, tok, paths = get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    label = "from-parquet"
    token = tokenize(
        tok, paths, ignore_metadata, columns, filters, scan_files, split_row_groups
    )

    # same as ak_metadata_from_parquet
    results = ak_from_parquet.metadata(
        path,
        storage_options,
        row_groups=None,
        columns=columns,
        ignore_metadata=ignore_metadata,
        scan_files=scan_files,
    )
    parquet_columns, subform, actual_paths, fs, subrg, row_counts, metadata = results

    listsep = "list.item"
    unnamed_root = False
    for c in parquet_columns:
        if ".list.element." in c:
            listsep = "list.element"
            break
        if c.startswith("."):
            unnamed_root = True

    if split_row_groups is None:
        split_row_groups = row_counts is not None and len(row_counts) > 1

    meta = ak.Array(
        subform.length_zero_array(highlevel=False).to_typetracer(forget_length=True)
    )

    if split_row_groups is False or subrg is None:
        # file-wise
        return from_map(
            _FromParquetFileWiseFn(
                fs=fs,
                schema=subform,
                listsep=listsep,
                unnamed_root=unnamed_root,
            ),
            actual_paths,
            label=label,
            token=token,
            meta=typetracer_array(meta),
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
        return from_map(
            _FromParquetFragmentWiseFn(
                fs=fs,
                schema=subform,
                listsep=listsep,
                unnamed_root=unnamed_root,
            ),
            pairs,
            label=label,
            token=token,
            divisions=tuple(divisions),
            meta=typetracer_array(meta),
        )


def _file_to_partition(path, fs, columns, schema, subrg=None):
    """read a whole parquet file to awkward"""
    return _load(
        actual_paths=[path],
        fs=fs,
        parquet_columns=columns,
        subrg=subrg or [None],
        footer_sample_size=2**15,
        max_gap=2**10,
        max_block=2**22,
        generate_bitmasks=False,
        metadata=None,
        highlevel=True,
        subform=schema,
        behavior=None,
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


def _write_partition(
    data,
    path,  # dataset root
    fs,
    filename,  # relative path within the dataset
    # partition_on=Fa,  # must be top-level leaf (i.e., a simple column)
    return_metadata=False,  # whether making global _metadata
    compression=None,  # TBD
    head=False,  # is this the first piece
    # custom_metadata=None,
):
    import pyarrow.parquet as pq

    t = to_arrow_table(
        data,
        list_to32=True,
        string_to32=True,
        bytestring_to32=True,
        categorical_as_dictionary=True,
        extensionarray=False,
    )
    md_list = []
    with fs.open(fs.sep.join([path, filename]), "wb") as fil:
        pq.write_table(
            t,
            fil,
            compression=compression,
            metadata_collector=md_list,
        )

    # Return the schema needed to write global _metadata
    if return_metadata:
        _meta = md_list[0]
        _meta.set_file_path(filename)
        d = {"meta": _meta}
        if head:
            # Only return schema if this is the "head" partition
            d["schema"] = t.schema
        return [d]
    else:
        return []


class _ToParquetFn:
    def __init__(
        self,
        fs: AbstractFileSystem,
        path: Any,
        return_metadata: bool = False,
        compression: Any | None = None,
        head: Any | None = None,
        npartitions: int | None = None,
        prefix: str | None = None,
    ):
        self.fs = fs
        self.path = path
        self.return_metadata = return_metadata
        self.compression = compression
        self.head = head
        self.prefix = prefix
        self.zfill = (
            math.ceil(math.log(npartitions, 10)) if npartitions is not None else 1
        )

        self.fs.mkdirs(self.path, exist_ok=True)

    def __call__(self, data, block_index):
        filename = f"part{str(block_index[0]).zfill(self.zfill)}.parquet"
        if self.prefix is not None:
            filename = f"{self.prefix}-{filename}"
        return _write_partition(
            data,
            self.path,
            self.fs,
            filename,
            return_metadata=self.return_metadata,
            compression=self.compression,
            head=self.head,
        )


def to_parquet(
    data: Array,
    path: Any,
    storage_options: dict[str, Any] | None = None,
    write_metadata: bool = False,
    compute: bool = True,
    prefix: str | None = None,
) -> Scalar | None:
    """Write data to Parquet format.

    Parameters
    ----------
    data : dask_awkward.Array
        Array to write to parquet.
    path : str
        Root directory of location to write to
    storage_options : dict
        Arguments to pass to fsspec for creating the filesystem (see
        ``fsspec`` documentation).
    write_metadata : bool
        Whether to create _metadata and _common_metadata files
    compute : bool
        Whether to immediately start writing or to return the dask
        collection which can be computed at the user's discression.

    Returns
    -------
    None or dask_awkward.Scalar
        If `compute` is ``False``, a :py:class:`dask_awkward.Scalar`
        representing the process will be returned, if `compute` is
        ``True`` then the return is ``None``.
    """
    # TODO options we need:
    #  - compression per data type or per leaf column ("path.to.leaf": "zstd" format)
    #  - byte stream split for floats if compression is not None or lzma
    #  - partitioning
    #  - parquet 2 for full set of time and int types
    #  - v2 data page (for possible later fastparquet implementation)
    #  - dict encoding always off
    fs, _ = fsspec.core.url_to_fs(path, **(storage_options or {}))
    name = f"write-parquet-{tokenize(fs, data, path)}"

    map_res = map_partitions(
        _ToParquetFn(fs, path=path, npartitions=data.npartitions, prefix=prefix),
        data,
        BlockIndex((data.npartitions,)),
        label="to-parquet",
        meta=data._meta,
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
    graph = HighLevelGraph.from_collections(final_name, dsk, dependencies=[map_res])
    out = new_scalar_object(graph, final_name, meta=None)
    if compute:
        out.compute()
        return None
    else:
        return out
