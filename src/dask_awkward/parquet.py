# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from dask.dataframe.io.parquet.arrow import ArrowDatasetEngine
from dask.dataframe.io.parquet.utils import natural_sort_key
from dask.dataframe.io.parquet.core import (
    DataFrameIOLayer,
    HighLevelGraph,
    ParquetFunctionWrapper
)
from dask.base import tokenize
import pyarrow as pa
from fsspec import get_fs_token_paths
from .core import DaskAwkwardArray, new_array_object
import awkward  # implicitly v1
from awkward._v2.operations.convert import from_arrow
import awkward as ak
import toolz


class AkParquetEngine(ArrowDatasetEngine):

    @classmethod
    def read_metadata(
            cls,
            fs,
            paths,
            categories=None,
            index=None,
            gather_statistics=None,
            filters=None,
            split_row_groups=None,
            chunksize=None,
            aggregate_files=None,
            ignore_metadata_file=False,
            metadata_task_size=0,
            **kwargs,
    ):

        # Stage 1: Collect general dataset information
        dataset_info = cls._collect_dataset_info(
            paths,
            fs,
            categories,
            index,
            gather_statistics,
            filters,
            split_row_groups,
            chunksize,
            aggregate_files,
            ignore_metadata_file,
            metadata_task_size,
            **kwargs.get("dataset", {}),
        )

        # Stage 2: Generate output `meta`
        meta = dataset_info['physical_schema']

        # Stage 3: Generate parts and stats
        dataset_info["index_cols"] = []
        parts, stats, common_kwargs = cls._construct_collection_plan(dataset_info)

        # Add `common_kwargs` and `aggregation_depth` to the first
        # element of `parts`. We can return as a separate element
        # in the future, but should avoid breaking the API for now.
        if len(parts):
            parts[0]["common_kwargs"] = common_kwargs
            parts[0]["aggregation_depth"] = dataset_info["aggregation_depth"]

        return (meta, stats, parts, dataset_info["index"])

    @classmethod
    def read_partition(
        cls,
        fs,
        pieces,
        columns,
        index,
        categories=(),
        partitions=(),
        filters=None,
        schema=None,
        **kwargs,
    ):
        columns_and_parts = columns.copy()
        if not isinstance(partitions, (list, tuple)):
            if columns_and_parts and partitions:
                for part_name in partitions.partition_names:
                    if part_name in columns:
                        columns.remove(part_name)
                    else:
                        columns_and_parts.append(part_name)
                columns = columns or None

        # Always convert pieces to list
        if not isinstance(pieces, list):
            pieces = [pieces]

        tables = []
        multi_read = len(pieces) > 1
        for piece in pieces:

            if isinstance(piece, str):
                # `piece` is a file-path string
                path_or_frag = piece
                row_group = None
                partition_keys = None
            else:
                # `piece` contains (path, row_group, partition_keys)
                (path_or_frag, row_group, partition_keys) = piece

            # Convert row_group to a list and be sure to
            # check if msgpack converted it to a tuple
            if isinstance(row_group, tuple):
                row_group = list(row_group)
            if not isinstance(row_group, list):
                row_group = [row_group]

            # Read in arrow table and convert to pandas
            arrow_table = cls._read_table(
                path_or_frag,
                fs,
                row_group,
                columns,
                schema,
                filters,
                partitions,
                partition_keys,
                **kwargs,
            )
            if multi_read:
                tables.append(arrow_table)

        if multi_read:
            arrow_table = pa.concat_tables(tables)

        return arrow_table


class MyParquetFunctionWrapper(ParquetFunctionWrapper):

    def __call__(self, part):

        if not isinstance(part, list):
            part = [part]

        return read_parquet_part(
            self.fs,
            self.engine,
            self.meta,
            [(p["piece"], p.get("kwargs", {})) for p in part],
            self.columns,
            self.index,
            self.common_kwargs,
        )



def ak_read_parquet(path,
                    columns=None,
                    filters=None,
                    categories=None,
                    index=None,
                    storage_options=None,
                    gather_statistics=None,
                    ignore_metadata_file=False,
                    metadata_task_size=None,
                    split_row_groups=None,
                    chunksize=None,
                    aggregate_files=None,
                    **kwargs,):
    if columns is not None:
        columns = list(columns)

    label = "read-parquet-"
    output_name = label + tokenize(
        path,
        columns,
        kwargs
    )
    fs, _, paths = get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )

    engine = AkParquetEngine
    paths = sorted(paths, key=natural_sort_key)  # numeric rather than glob ordering

    if chunksize or (
            split_row_groups and int(split_row_groups) > 1 and aggregate_files
    ):
        # Require `gather_statistics=True` if `chunksize` is used,
        # or if `split_row_groups>1` and we are aggregating files.
        if gather_statistics is False:
            raise ValueError("read_parquet options require gather_statistics=True")
        gather_statistics = True

    read_metadata_result = engine.read_metadata(
        fs,
        paths,
        categories=False,
        index=False,
        gather_statistics=False,
        filters=filters,
        split_row_groups=split_row_groups,
        chunksize=chunksize,
        aggregate_files=aggregate_files,
        ignore_metadata_file=ignore_metadata_file,
        metadata_task_size=metadata_task_size,
        **kwargs,
    )

    meta, statistics, parts, index = read_metadata_result[:4]
    if columns is None:
        columns = list(meta.names)
    common_kwargs = {}
    aggregation_depth = False
    if len(parts):
        # For now, `common_kwargs` and `aggregation_depth`
        # may be stored in the first element of `parts`
        common_kwargs = parts[0].pop("common_kwargs", {})
        aggregation_depth = parts[0].pop("aggregation_depth", aggregation_depth)

    # Create Blockwise layer
    layer = DataFrameIOLayer(
        output_name,
        columns,
        parts,
        MyParquetFunctionWrapper(
            engine,
            fs,
            None,  # meta
            columns,
            index,
            kwargs,
            common_kwargs,
        ),
        label=output_name,
    )
    graph = HighLevelGraph({output_name: layer}, {output_name: set()})
    return new_array_object(graph, output_name, npartitions=len(parts))


def read_parquet_part(fs, engine, meta, part, columns, index, kwargs):
    """Read a part of a parquet dataset

    This function is used by `read_parquet`."""
    if isinstance(part, list):
        if len(part) == 1 or part[0][1] or not check_multi_support(engine):
            # Part kwargs expected
            func = engine.read_partition
            tables = [
                func(fs, rg, columns.copy(), index, **toolz.merge(kwargs, kw))
                for (rg, kw) in part
            ]
            table = pa.concat_tables(tables)
        else:
            # No part specific kwargs, let engine read
            # list of parts at once
            table = engine.read_partition(
                fs, [p[0] for p in part], columns.copy(), index, **kwargs
            )
    else:
        # NOTE: `kwargs` are the same for all parts, while `part_kwargs` may
        #       be different for each part.
        table, part_kwargs = part
        df = engine.read_partition(
            fs, rg, columns, index, **toolz.merge(kwargs, part_kwargs)
        )
    return from_arrow(table)
