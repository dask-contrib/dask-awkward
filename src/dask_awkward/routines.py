from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import awkward as ak
from dask.utils import derived_from

from .core import TrivialPartitionwiseOp, pw_reduction_with_agg_to_scalar

if TYPE_CHECKING:
    from .core import DaskAwkwardArray

_count_trivial = TrivialPartitionwiseOp(ak.count, axis=1)
_flatten_trivial = TrivialPartitionwiseOp(ak.flatten, axis=1)
_max_trivial = TrivialPartitionwiseOp(ak.max, axis=1)
_min_trivial = TrivialPartitionwiseOp(ak.min, axis=1)
_num_trivial = TrivialPartitionwiseOp(ak.num, axis=1)
_sum_trivial = TrivialPartitionwiseOp(ak.sum, axis=1)


@derived_from(ak)
def count(a, axis: Optional[int] = None, **kwargs):
    if axis is not None and axis == 1:
        return _count_trivial(a, axis=axis, **kwargs)
    elif axis is None:
        trivial_result = _count_trivial(a, axis=1, **kwargs)
        return pw_reduction_with_agg_to_scalar(trivial_result, ak.sum, ak.sum)
    elif axis == 0 or axis == -1 * a.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


@derived_from(ak)
def flatten(a: DaskAwkwardArray, axis: int = 1, **kwargs):
    return _flatten_trivial(a, axis=axis, **kwargs)


def _min_or_max(f, a, axis, **kwargs):
    # translate negative axis (a.ndim currently raises)
    if axis is not None and axis < 0:
        axis = a.ndim + axis + 1
    # get the correct trivial callable
    tf = _min_trivial if f == ak.min else _max_trivial
    # generate collection based on axis
    if axis == 1:
        return tf(a, axis=axis, **kwargs)
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(a, f, f, **kwargs)
    elif axis == 0 or axis == -1 * a.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


@derived_from(ak)
def max(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    return _min_or_max(ak.max, a, axis, **kwargs)


@derived_from(ak)
def min(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    return _min_or_max(ak.min, a, axis, **kwargs)


@derived_from(ak)
def num(a: DaskAwkwardArray, axis: int = 1, **kwargs):
    raise NotImplementedError("function is still TODO")


@derived_from(ak)
def sum(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    raise NotImplementedError("function is still TODO")
