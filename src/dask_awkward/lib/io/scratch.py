# Module that serves as a testing ground for IO things.
# It has zero promised API stability
# It is partially tested (at best)
# It is not included in coverage

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, Any

import awkward as ak
import fsspec
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.lib.core import map_partitions, new_scalar_object, typetracer_array
from dask_awkward.lib.io.io import from_map

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

    from dask_awkward.lib.core import Array, Scalar

__all__ = (
    "from_parquet",
    "to_parquet",
)


class _BaseFromParquetFn:
    @abc.abstractmethod
    def project_columns(self, columns: list[str]) -> Any:
        ...

    @abc.abstractmethod
    def __call__(self, source: Any) -> ak.Array:
        ...


class _FromParquetFn(_BaseFromParquetFn):
    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        columns: Any = None,
        **kwargs: Any,
    ) -> None:
        self.is_parquet_read = 1
        self.fs = fs
        self.columns = columns
        self.kwargs = kwargs

    def project_columns(self, columns: list[str]) -> _BaseFromParquetFn:

        # this commented section targets a scenario where we have to
        # pass parquet columns to the parquet read function.
        #
        # ak_columns = meta.layout.form.columns()
        # indicator = "list.item"
        # pq_columns = meta.layout.form.columns(list_indicator=indicator)
        # keep = [
        #     pq_col
        #     for pq_col, ak_col in zip(pq_columns, ak_columns)
        #     if ak_col in columns
        # ]

        return _FromParquetFn(
            fs=self.fs,
            columns=columns,
            **self.kwargs,
        )

    def __call__(self, source: Any) -> ak.Array:
        source = fsspec.utils._unstrip_protocol(source, self.fs)
        return ak.from_parquet(
            source,
            columns=self.columns,
            storage_options=self.fs.storage_options,
            **self.kwargs,
        )


def from_parquet(
    urlpath: str | list[str],
    columns: list[str] | None = None,
    meta: ak.Array | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Array:
    fs, token, paths = fsspec.get_fs_token_paths(
        urlpath,
        storage_options=storage_options,
    )

    fn = _FromParquetFn(fs=fs, columns=columns, **kwargs)
    meta = typetracer_array(fn(paths[0]))
    token = tokenize(urlpath, columns, meta, token)

    return from_map(
        fn,
        paths,
        label="from-parquet",
        token=token,
        meta=meta,
    )


class _ToParquetFn:
    def __init__(
        self,
        name: str,
        fs: AbstractFileSystem | None,
        npartitions: int | None = None,
    ) -> None:
        parts = name.split(".")
        self.suffix = parts[-1]
        self.name = "".join(parts[:-1])
        self.fs = fs
        self.zfill = (
            math.ceil(math.log(npartitions, 10)) if npartitions is not None else 1
        )

    def __call__(self, array: ak.Array, block_index: tuple[int]) -> None:
        part = str(block_index[0]).zfill(self.zfill)
        name = f"{self.name}.{part}.{self.suffix}"
        ak.to_parquet(array, name)
        return None


def to_parquet(
    array: Array,
    where: str,
    compute: bool = False,
) -> Scalar:
    nparts = array.npartitions
    write_res = map_partitions(
        _ToParquetFn(where, None, npartitions=nparts),
        array,
        BlockIndex((nparts,)),
        label="to-parquet-on-block",
        meta=array._meta,
    )
    name = f"to-parquet-{tokenize(array, where)}"
    dsk = {name: (lambda *_: None, write_res.__dask_keys__())}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=(write_res,))
    res = new_scalar_object(graph, name=name, meta=None)
    if compute:
        res.compute()
    return res
