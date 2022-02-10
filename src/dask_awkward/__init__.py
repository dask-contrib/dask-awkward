from ._version import version
from .core import Array
from .core import _type as type
from .core import fields, from_awkward, map_partitions, typetracer_array
from .io import from_json
from .reducers import *
from .structure import *

__version__ = version
