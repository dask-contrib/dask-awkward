from ._version import version
from .core import Array, Record, Scalar
from .core import _type as type
from .core import from_awkward, map_partitions
from .describe import fields
from .io import from_json
from .reducers import *
from .structure import *

__version__ = version
