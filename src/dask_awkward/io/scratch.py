# Module that serves as a testing ground for IO things.
# It has zero promised API stability
# It is partially tested (at best)
# It is not included in coverage

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import awkward._v2 as ak
import fsspec
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.core import map_partitions, new_scalar_object
from dask_awkward.io.io import from_map

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

    from dask_awkward.core import Array, Scalar


class FromParquetWrapper:
    def __init__(self, *, storage: AbstractFileSystem, **kwargs: Any) -> None:
        self.fs = storage
        self.kwargs = kwargs

    def __call__(self, source: Any) -> ak.Array:
        source = fsspec.utils._unstrip_protocol(source, self.fs)
        return ak.from_parquet(
            source,
            storage_options=self.fs.storage_options,
            **self.kwargs,
        )


def from_parquet(
    urlpath: str | list[str],
    meta: ak.Array | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
):
    fs, token, paths = fsspec.get_fs_token_paths(
        urlpath,
        storage_options=storage_options,
    )

    return from_map(
        FromParquetWrapper(storage=fs, **kwargs),
        paths,
        label="from-parquet",
        token=tokenize(urlpath, meta, token),
    )


class ToParquetOnBlock:
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
) -> Scalar | None:
    nparts = array.npartitions
    write_res = map_partitions(
        ToParquetOnBlock(where, None, npartitions=nparts),
        array,
        BlockIndex((nparts,)),
        label="to-parquet",
        meta=array._meta,
    )
    name = f"to-parquet-{tokenize(array, where)}"
    dsk = {name: (lambda *_: None, write_res.__dask_keys__())}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=(write_res,))
    res = new_scalar_object(graph, name=name, meta=None)
    if compute:
        return res.compute()
    return res
