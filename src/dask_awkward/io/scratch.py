from __future__ import annotations

from typing import TYPE_CHECKING, Any

import awkward as ak1
import awkward._v2 as ak
from awkward._v2.tmp_for_testing import v2_to_v1
from dask.blockwise import BlockIndex

from dask_awkward.core import map_partitions

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

    from dask_awkward.core import Array


class ToParquetOnBlock:
    def __init__(self, name: str, fs: AbstractFileSystem | None) -> None:
        parts = name.split(".")
        self.suffix = parts[-1]
        self.name = "".join(parts[:-1])
        self.fs = fs

    def __call__(self, array: ak.Array, block_index: tuple[int]) -> None:
        part = block_index[0]
        name = f"{self.name}.part{part}.{self.suffix}"
        ak1.to_parquet(ak1.Array(v2_to_v1(array.layout)), name)
        return None


def to_parquet(
    array: Array,
    where: str,
    compute: bool = False,
) -> Array | None:
    res = map_partitions(
        ToParquetOnBlock(where, None),
        array,
        BlockIndex((array.npartitions,)),
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
        return ak.from_parquet(source)


def from_parquet(
    urlpath: str | list[str],
    meta: ak.Array | None = None,
    storage_options: dict[str, Any] | None = None,
):
    pass
