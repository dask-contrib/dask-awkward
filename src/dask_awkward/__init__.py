from ._version import version  # noqa
from .core import count, flatten, max, min, num, sum
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
    # io
    "from_json",
    "from_parquet",
)
