from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import awkward as ak1
import awkward._v2 as ak
import fsspec
from awkward._v2.tmp_for_testing import v2_to_v1
from dask.base import tokenize
from dask.blockwise import BlockIndex

from dask_awkward.core import map_partitions
from dask_awkward.io.io import from_map

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

    from dask_awkward.core import Array


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
        name = f"{self.name}.part-{part}.{self.suffix}"
        ak1.to_parquet(ak1.Array(v2_to_v1(array.layout)), name)
        return None


def to_parquet(
    array: Array,
    where: str,
    compute: bool = False,
) -> Array | None:
    nparts = array.npartitions
    res = map_partitions(
        ToParquetOnBlock(where, None, npartitions=nparts),
        array,
        BlockIndex((nparts,)),
        name="to-parquet",
        meta=array._meta,
    )
    if compute:
        return res.compute()
    return res


class FromParquetWrapper:
    def __init__(self, *, storage: AbstractFileSystem) -> None:
        self.storage = storage

    def __call__(self, part: Any) -> ak.Array:
        source = part
        source = fsspec.utils._unstrip_protocol(source, self.storage)
        return ak.from_parquet(source)


def from_parquet(
    urlpath: str | list[str],
    meta: ak.Array | None = None,
    storage_options: dict[str, Any] | None = None,
):
    fs, token, paths = fsspec.get_fs_token_paths(
        urlpath,
        storage_options=storage_options,
    )
    return from_map(
        FromParquetWrapper(storage=fs),
        paths,
        label="from-parquet",
        token=tokenize(urlpath, meta, token),
    )
