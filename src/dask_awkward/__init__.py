from dask_awkward import config  # isort:skip; load awkward config

from dask_awkward._version import version
from dask_awkward.core import Array, Record, Scalar
from dask_awkward.core import _type as type
from dask_awkward.core import from_awkward, map_partitions
from dask_awkward.describe import fields
from dask_awkward.io import *
from dask_awkward.reducers import *
from dask_awkward.structure import *

__version__ = version
