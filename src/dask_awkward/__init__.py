from ._version import version
from .core import DaskAwkwardArray
from .core import _type as type
from .core import fields, from_awkward, map_partitions
from .io import from_json

__version__ = version


__all__ = (
    "DaskAwkwardArray",
    "fields",
    "from_awkward",
    "from_json",
    "map_partitions",
    "type",
)
