from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union

import awkward._v2 as ak
import numpy as np

from dask_awkward.core import (
    DaskAwkwardNotImplemented,
    TrivialPartitionwiseOp,
    compatible_partitions,
    incompatible_partitions_msg,
    map_partitions,
    pw_reduction_with_agg_to_scalar,
)
from dask_awkward.utils import borrow_docstring

if TYPE_CHECKING:
    from dask_awkward.core import Array, Scalar

    LazyResult = Union[Array, Scalar]

__all__ = (
    "all",
    "any",
    "argmax",
    "argmin",
    "corr",
    "count",
    "count_nonzero",
    "covar",
    "linear_fit",
    "max",
    "mean",
    "min",
    "moment",
    "prod",
    "ptp",
    "softmax",
    "std",
    "sum",
    "var",
)


@borrow_docstring(ak.all)
def all(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    if axis and axis >= 1:
        return map_partitions(
            ak.all,
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.any)
def any(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    if axis and axis >= 1:
        return map_partitions(
            ak.any,
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.argmax)
def argmax(array, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
    if axis and axis >= 1:
        return map_partitions(
            ak.argmax,
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.argmin)
def argmin(array, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
    if axis and axis >= 1:
        return map_partitions(
            ak.argmin,
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.corr)
def corr(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    if not compatible_partitions(x, y):
        raise ValueError(incompatible_partitions_msg("corr", x, y))
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.count)
def count(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    if axis and axis >= 1:
        return map_partitions(
            ak.count,
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    elif axis is None:
        trivial_result = map_partitions(
            ak.count,
            array,
            axis=1,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
        return pw_reduction_with_agg_to_scalar(
            trivial_result,
            ak.sum,
            agg=ak.sum,
            dtype=np.int64,
        )
    elif axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be None or an integer.")


@borrow_docstring(ak.count_nonzero)
def count_nonzero(
    array, axis=None, keepdims=False, mask_identity=False, flatten_records=False
):
    if axis is not None and axis == 1:
        return map_partitions(
            ak.count_nonzero,
            array,
            axis=1,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    elif axis is None:
        trivial_result = map_partitions(
            ak.count_nonzero,
            array,
            axis=1,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
        return pw_reduction_with_agg_to_scalar(
            trivial_result,
            ak.sum,
            agg=ak.sum,
            dtype=np.int64,
        )
    elif axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be None or an integer.")


@borrow_docstring(ak.covar)
def covar(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    if not compatible_partitions(x, y):
        raise ValueError(incompatible_partitions_msg("covar", x, y))
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.linear_fit)
def linear_fit(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    if not compatible_partitions(x, y):
        raise ValueError(incompatible_partitions_msg("linear_fit", x, y))
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.max)
def max(
    array,
    axis=None,
    keepdims=False,
    initial=None,
    mask_identity=True,
    flatten_records=False,
):
    return _min_or_max(
        ak.max,
        array,
        axis=axis,
        keepdims=keepdims,
        initial=initial,
        mask_identity=mask_identity,
        flatten_records=flatten_records,
    )


@borrow_docstring(ak.mean)
def mean(
    array,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    if axis and axis >= 1:
        return map_partitions(
            ak.mean,
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.min)
def min(
    array,
    axis=None,
    keepdims=False,
    initial=None,
    mask_identity=True,
    flatten_records=False,
):
    return _min_or_max(
        ak.min,
        array,
        axis,
        keepdims=keepdims,
        initial=initial,
        mask_identity=mask_identity,
        flatten_records=flatten_records,
    )


@borrow_docstring(ak.moment)
def moment(
    x,
    n,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.prod)
def prod(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.ptp)
def ptp(arr, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.softmax)
def softmax(x, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.std)
def std(
    x,
    weight=None,
    ddof=0,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    if weight is not None:
        raise DaskAwkwardNotImplemented("dak.std with weights is not supported yet.")

    if axis == 1:
        return map_partitions(
            ak.std,
            x,
            weight=weight,
            ddof=ddof,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.sum)
def sum(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    if axis is not None and axis < 0:
        axis = array.ndim + axis + 1
    if axis == 1:
        return map_partitions(
            ak.sum,
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(array, ak.sum, agg=ak.sum)
    elif axis == 0:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be none or an integer")


@borrow_docstring(ak.var)
def var(
    x,
    weight=None,
    ddof=0,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


_min_trivial = TrivialPartitionwiseOp(ak.min, axis=1)
_max_trivial = TrivialPartitionwiseOp(ak.max, axis=1)


def _min_or_max(
    f: Callable,
    array: Array,
    axis: int | None = None,
    **kwargs: Any,
) -> LazyResult:
    # translate negative axis (array.ndim currently raises)
    if axis is not None and axis < 0 and array.ndim is not None:
        axis = array.ndim + axis + 1
    # get the correct trivial callable
    tf = _min_trivial if f == ak.min else _max_trivial
    # generate collection based on axis
    if axis == 1:
        return tf(array, axis=axis, **kwargs)
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(array, f, agg=f, **kwargs)
    elif array.ndim is not None and (axis == 0 or axis == -1 * array.ndim):
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be None or an integer.")
