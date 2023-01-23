from __future__ import annotations

from typing import Any

from dask.highlevelgraph import HighLevelGraph


def necessary_columns(obj: Any) -> dict[str, list[str]]:
    import dask_awkward.lib.optimize as o

    dsk = obj if isinstance(obj, HighLevelGraph) else obj.dask
    return o._necessary_columns(dsk)
