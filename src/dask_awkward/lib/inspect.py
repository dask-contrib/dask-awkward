from __future__ import annotations

from typing import Any, Literal

import dask_awkward.lib.optimize as o
from dask.highlevelgraph import HighLevelGraph
from dask_awkward.lib.core import Array, Record, Scalar


def necessary_columns(obj):
    dsk = obj if isinstance(obj, HighLevelGraph) else obj.dask
    return o._necessary_columns(dsk)
