from ._version import version  # noqa
from .core import map_partitions
from .io import from_json, from_parquet
from .routines import count, count_nonzero, flatten, max, min, num, sum

__version__ = version  # noqa


__all__ = (
    # top level methods
    "count",
    "count_nonzero",
    "flatten",
    "max",
    "min",
    "num",
    "sum",
    # collection specific
    "map_partitions",
    # io
    "from_json",
    "from_parquet",
)
