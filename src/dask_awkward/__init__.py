from dask_awkward import config  # isort:skip; load awkward config

from dask_awkward._version import version
from dask_awkward.core import Array, Record, Scalar
from dask_awkward.core import _type as type
from dask_awkward.core import map_partitions
from dask_awkward.describe import fields
from dask_awkward.io import (
    from_awkward,
    from_dask_array,
    from_json,
    to_dask_array,
    to_delayed,
)
from dask_awkward.reducers import (
    all,
    any,
    argmax,
    argmin,
    corr,
    count,
    count_nonzero,
    covar,
    linear_fit,
    max,
    mean,
    min,
    moment,
    prod,
    ptp,
    softmax,
    std,
    sum,
    var,
)
from dask_awkward.structure import (
    argcartesian,
    argcombinations,
    argsort,
    broadcast_arrays,
    cartesian,
    combinations,
    concatenate,
    copy,
    fill_none,
    firsts,
    flatten,
    from_regular,
    full_like,
    is_none,
    isclose,
    local_index,
    mask,
    nan_to_num,
    num,
    ones_like,
    packed,
    pad_none,
    ravel,
    run_lengths,
    singletons,
    sort,
    strings_astype,
    to_regular,
    unflatten,
    unzip,
    values_astype,
    where,
    with_field,
    with_name,
    with_parameter,
    without_parameters,
    zeros_like,
    zip,
)

__version__ = version
