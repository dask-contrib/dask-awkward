from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import awkward as ak
from dask.utils import derived_from

from .core import TrivialPartitionwiseOp, pw_reduction_with_agg_to_scalar

if TYPE_CHECKING:
    from .core import DaskAwkwardArray

####################################
# all awkward.operations.reducers.py
####################################
# def count(array, axis=None, keepdims=False, mask_identity=False):
# def count_nonzero(array, axis=None, keepdims=False, mask_identity=False):
# def sum(array, axis=None, keepdims=False, mask_identity=False):
# def prod(array, axis=None, keepdims=False, mask_identity=False):
# def any(array, axis=None, keepdims=False, mask_identity=False):
# def all(array, axis=None, keepdims=False, mask_identity=False):
# def min(array, axis=None, keepdims=False, initial=None, mask_identity=True):
# def max(array, axis=None, keepdims=False, initial=None, mask_identity=True):
# def ptp(arr, axis=None, keepdims=False, mask_identity=True):
# def argmin(array, axis=None, keepdims=False, mask_identity=True):
# def argmax(array, axis=None, keepdims=False, mask_identity=True):
# def moment(x, n, weight=None, axis=None, keepdims=False, mask_identity=True):
# def mean(x, weight=None, axis=None, keepdims=False, mask_identity=True):
# def var(x, weight=None, ddof=0, axis=None, keepdims=False, mask_identity=True):
# def std(x, weight=None, ddof=0, axis=None, keepdims=False, mask_identity=True):
# def covar(x, y, weight=None, axis=None, keepdims=False, mask_identity=True):
# def corr(x, y, weight=None, axis=None, keepdims=False, mask_identity=True):
# def linear_fit(x, y, weight=None, axis=None, keepdims=False, mask_identity=True):
# def softmax(x, axis=None, keepdims=False, mask_identity=False):
####################################

_count_trivial = TrivialPartitionwiseOp(ak.count, axis=1)
_max_trivial = TrivialPartitionwiseOp(ak.max, axis=1)
_min_trivial = TrivialPartitionwiseOp(ak.min, axis=1)
_sum_trivial = TrivialPartitionwiseOp(ak.sum, axis=1)


@derived_from(ak)
def count(array, axis: Optional[int] = None, **kwargs):
    if axis is not None and axis == 1:
        return _count_trivial(array, axis=axis, **kwargs)
    elif axis is None:
        trivial_result = _count_trivial(array, axis=1, **kwargs)
        return pw_reduction_with_agg_to_scalar(trivial_result, ak.sum, ak.sum)
    elif axis == 0 or axis == -1 * array.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


def _min_or_max(f, array, axis, **kwargs):
    # translate negative axis (array.ndim currently raises)
    if axis is not None and axis < 0:
        axis = array.ndim + axis + 1
    # get the correct trivial callable
    tf = _min_trivial if f == ak.min else _max_trivial
    # generate collection based on axis
    if axis == 1:
        return tf(array, axis=axis, **kwargs)
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(array, f, f, **kwargs)
    elif axis == 0 or axis == -1 * array.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


@derived_from(ak)
def max(array: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    return _min_or_max(ak.max, array, axis, **kwargs)


@derived_from(ak)
def min(array: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    return _min_or_max(ak.min, array, axis, **kwargs)


@derived_from(ak)
def sum(array: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    if axis is not None and axis < 0:
        axis = array.ndim + axis + 1
    if axis == 1:
        return _sum_trivial(array, **kwargs)
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(array, ak.sum, ak.sum, **kwargs)
    elif axis == 0:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be none or an integer")


# Non reduction routines


_flatten_trivial = TrivialPartitionwiseOp(ak.flatten, axis=1)
_num_trivial = TrivialPartitionwiseOp(ak.num, axis=1)


@derived_from(ak)
def num(array: DaskAwkwardArray, axis: int = 1, **kwargs):
    return _num_trivial(array, axis=axis, **kwargs)


@derived_from(ak)
def flatten(array: DaskAwkwardArray, axis: int = 1, **kwargs):
    return _flatten_trivial(array, axis=axis, **kwargs)
