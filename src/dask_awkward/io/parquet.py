import itertools
import math
import operator

import awkward._v2 as ak
import awkward._v2.forms as forms
import fsspec
import numpy as np
import pyarrow
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from awkward._v2.operations.convert import from_arrow, from_buffers, to_arrow_table
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from fsspec.core import get_fs_token_paths

from dask_awkward.core import map_partitions, new_array_object, new_scalar_object


def _parquet_schema_to_form(schema):
    """Helpre for arrow parq schema->ak form"""

    def maybe_nullable(field, content):
        if field.nullable:
            if isinstance(content, forms.EmptyForm):
                return forms.IndexedOptionForm(
                    "i64",
                    content,
                    form_key="",
                )
            else:
                return forms.ByteMaskedForm(
                    "i8",
                    content,
                    valid_when=True,
                    form_key="",
                )
        else:
            return content

    def contains_record(form):
        if isinstance(form, forms.RecordForm):
            return True
        elif isinstance(form, forms.ListOffsetForm):
            return contains_record(form.content)
        else:
            return False

    def recurse(arrow_type, path):
        if isinstance(arrow_type, pyarrow.StructType):
            names = []
            contents = []
            for index in range(arrow_type.num_fields):
                field = arrow_type[index]
                names.append(field.name)
                content = maybe_nullable(
                    field, recurse(field.type, path + (field.name,))
                )
                contents.append(content)
            assert len(contents) != 0
            return forms.RecordForm(contents, names)

        elif isinstance(arrow_type, pyarrow.ListType):
            field = arrow_type.value_field
            content = maybe_nullable(
                field, recurse(field.type, path + ("list", "item"))
            )
            return forms.ListOffsetForm("i32", content, form_key="")

        elif isinstance(arrow_type, pyarrow.LargeListType):
            field = arrow_type.value_field
            content = maybe_nullable(
                field, recurse(field.type, path + ("list", "item"))
            )
            return forms.ListOffsetForm("i64", content, form_key="")

        elif arrow_type == pyarrow.string():
            return forms.ListOffsetForm(
                "i32",
                forms.NumpyForm("uint8"),
                parameters={"__array__": "string"},
                form_key="",
            )

        elif arrow_type == pyarrow.large_string():
            return forms.ListOffsetForm(
                "i64",
                forms.NumpyForm("uint8"),
                parameters={"__array__": "string"},
                form_key="",
            )

        elif arrow_type == pyarrow.binary():
            return forms.ListOffsetForm(
                "i32",
                forms.NumpyForm("uint8"),
                parameters={"__array__": "bytestring"},
                form_key="",
            )

        elif arrow_type == pyarrow.large_binary():
            return forms.ListOffsetForm(
                "i64",
                forms.NumpyForm("uint8"),
                parameters={"__array__": "bytestring"},
                form_key="",
            )

        elif isinstance(arrow_type, pyarrow.DataType):
            if arrow_type == pyarrow.null():
                return forms.EmptyForm(form_key="")
            else:
                dtype = np.dtype(arrow_type.to_pandas_dtype())
                # return forms.Form.from_numpy(dtype).with_form_key(col(path))
                return forms.numpyform.NumpyForm(str(dtype))

        else:
            raise NotImplementedError

    contents = []
    for index, name in enumerate(schema.names):
        field = schema.field(index)
        content = maybe_nullable(field, recurse(field.type, (name,)))
        contents.append(content)
    assert len(contents) != 0
    return forms.RecordForm(contents, schema.names)


def _read_metadata(path, fs, partition_base_dir=None, schema=None):
    return pa_ds.dataset(
        path,
        filesystem=fs,
        format="parquet",
        partition_base_dir=partition_base_dir,
        schema=schema,
    )


def read_parquet(
    path,
    storage_options=None,
    ignore_metadata=False,
    columns=None,
    filters=None,
    split_row_groups=None,
):
    """
    url: str
        location of data, including protocol
    storage_options: dict
        for creating filesystem
    columns: list[str] or None
        Select columns to load
    filters: list[list[tuple]]
        parquet-style filters for excluding row groups based on column statistics
    split_row_groups: bool | int
        If True, each row group becomes a partition. If False, each file becomes
        a partition. If int, at least this many row groups become a partition.
        If None, the existence of a `_metadata` file implies True, else False.
        The values True and 1 ar equivalent.
    """
    fs, tok, paths = get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    name = "read-parquet-" + tokenize(
        tok, ignore_metadata, columns, filters, split_row_groups
    )
    if len(paths) == 1:
        path = paths[0]
        # single file or directory
        if not ignore_metadata and fs.isfile("/".join([path, "_metadata"])):
            # dataset with global metadata
            metadata = _read_metadata(path, fs)
            if split_row_groups is None:
                # default to one row-group per partition
                split_row_groups = 1
            elif split_row_groups is False:
                # would need to pick out files from set of row-groups
                raise NotImplementedError
        elif fs.isfile(path):
            # single file
            metadata = _read_metadata(path, fs)
            if split_row_groups is None:
                # default to one row-group per partition
                split_row_groups = 1
            elif split_row_groups is False:
                # would need to pick out files from set of row-groups
                raise NotImplementedError
        else:
            # read dir as set of files
            if split_row_groups is None:
                # default to one file per partition
                split_row_groups = False
            allfiles = fs.find(path)
            common_file = [f for f in allfiles if f.endswith("_common_metadata")]
            allfiles = [f for f in allfiles if f.endswith(("parq", "parquet"))]
            if split_row_groups is False:
                # read whole files, no scan
                # reproduce partitioning here?
                if common_file:
                    metadata = _read_metadata(common_file, fs)
                else:
                    metadata = _read_metadata(allfiles[0], fs)
            else:
                # metadata from all files
                metadata = _read_metadata(allfiles, fs, partition_base_dir=path)

    else:
        # list of data files
        allfiles = paths
        if split_row_groups is None:
            # default to one file per partition
            split_row_groups = False
        if split_row_groups is False:
            # metadata from first file
            metadata = _read_metadata(paths[0], fs)
        else:
            metadata = _read_metadata(paths, fs)

    form = _parquet_schema_to_form(metadata.schema)
    meta = from_buffers(
        form,
        length=0,
        container={"": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
        buffer_key="",
    )

    if split_row_groups is False:
        # file-wise
        dsk = {
            (name, i): (_file_to_partition, path, fs, columns, filters, metadata.schema)
            for i, path in enumerate(allfiles)
        }
        divisions = (None,) * (len(dsk) + 1)
    else:
        # organise row-groups into fragments
        frags = list(metadata.get_fragments())
        rgs = sum((frag.row_groups for frag in frags), [])
        frags2 = sum(
            (
                [_frag_subset(frag, [i]) for i in range(len(frag.row_groups))]
                for frag in frags
            ),
            [],
        )
        dsk = {
            (name, i): (_fragment_to_partition, frag, columns, filters, metadata.schema)
            for i, frag in enumerate(frags2)
        }
        divisions = [0] + list(
            itertools.accumulate([rg.num_rows for rg in rgs], operator.add)
        )
    arr = new_array_object(
        HighLevelGraph.from_collections("read-parquet", MaterializedLayer(dsk)),
        name=name,
        meta=ak.Array(meta.layout.typetracer),
        divisions=divisions,
    )

    return arr


def _frag_subset(old_frag, row_groups):
    """Create new fragment with row-group subset.

    Used by `ArrowDatasetEngine` only.
    """
    return old_frag.format.make_fragment(
        old_frag.path,
        old_frag.filesystem,
        old_frag.partition_expression,
        row_groups=row_groups,
    )


def _file_to_partition(path, fs, columns, filters, schema):
    """read a whole parquet file to awkward"""
    ds = _read_metadata(path, fs)
    table = ds.to_table(
        use_threads=False,
        columns=columns,
        filter=pq._filters_to_expression(filters) if filters else None,
        # schema=schema
    )
    return from_arrow(table)


def _fragment_to_partition(frag, columns, filters, schema):
    """read one or more row-groups to awkward"""
    table = frag.to_table(
        use_threads=False,
        schema=schema,
        columns=columns,
        filter=pq._filters_to_expression(filters) if filters else None,
    )
    return from_arrow(table)


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
    t = to_arrow_table(data)
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


class WritePartitionWrapper:
    def __init__(
        self,
        fs,
        path,
        return_metadata=False,
        compression=None,
        head=None,
        npartitions=None,
    ):
        self.fs = fs
        self.path = path
        self.return_metadata = return_metadata
        self.compression = compression
        self.head = head
        self.zfill = (
            math.ceil(math.log(npartitions, 10)) if npartitions is not None else 1
        )

    def __call__(self, data, block_index):
        filename = f"part{str(block_index[0]).zfill(self.zfill)}.parquet"
        return _write_partition(
            data,
            self.path,
            self.fs,
            filename,
            return_metadata=self.return_metadata,
            compression=self.compression,
            head=self.head,
        )


def to_parquet(data, path, storage_options=None, write_metadata=False, compute=True):
    """Write data to parquet format

    Parameters
    ----------
    data: DaskAwrkardArray
    path: str
        Root directory of location to write to
    storage_options: dict
        arguments to pass to fsspec for creating the filesystem
    write_metadata: bool
        Whether to create _metadata and _common_metadata files
    compute: bool
        Whether to immediately start writing or to return the dask
        collection which can be computed at the user's discression.

    Returns
    -------
    If compute=False, a dask Scalar representing the process
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
        WritePartitionWrapper(fs, path=path, npartitions=data.npartitions),
        data,
        BlockIndex((data.npartitions,)),
        label="to-parquet",
        meta=data._meta,
    )

    dsk = {}
    if write_metadata:
        final_name = name + "-metadata"
        dsk[final_name] = (_metadata_file_from_metas, fs, path) + tuple(
            map_res.__dask_keys__()
        )
    else:
        final_name = name + "-finalize"
        dsk[final_name] = (lambda *_: None, map_res.__dask_keys__())
    graph = HighLevelGraph.from_collections(final_name, dsk, dependencies=[map_res])
    out = new_scalar_object(graph, final_name, meta=None)
    if compute:
        out.compute()
    else:
        return out
