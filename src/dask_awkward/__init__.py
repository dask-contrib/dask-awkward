from ._version import version  # noqa
from .core import map_partitions
from .routines import count, flatten, max, min, num, sum
from .io import from_json, from_parquet

__version__ = version  # noqa


__all__ = (
    # top level methods
    "count",
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
