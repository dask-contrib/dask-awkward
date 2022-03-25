from __future__ import annotations

import awkward._v2 as ak
import numpy as np

from dask_awkward.core import (
    DaskAwkwardNotImplemented,
    IncompatiblePartitions,
    compatible_partitions,
    map_partitions,
    pw_reduction_with_agg_to_scalar,
)
from dask_awkward.utils import borrow_docstring

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
            output_divisions=1,
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
            output_divisions=1,
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
            output_divisions=1,
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
            output_divisions=1,
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
        raise IncompatiblePartitions("corr", x, y)
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.count)
def count(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    if axis and axis >= 1:
        return map_partitions(
            ak.count,
            array,
            output_divisions=1,
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
            ak.sum,
            ak.sum,
            trivial_result,
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
            output_divisions=1,
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
            ak.sum,
            ak.sum,
            trivial_result,
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
        raise IncompatiblePartitions("covar", x, y)
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
        raise IncompatiblePartitions("linear_fit", x, y)
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
    if axis and axis >= 1:
        return map_partitions(
            ak.max,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    if axis is None:
        return pw_reduction_with_agg_to_scalar(
            ak.max,
            ak.max,
            array,
            axis=1,
            agg_kwargs={"axis": None},
        )
    else:
        raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


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
            output_divisions=1,
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
    if axis and axis >= 1:
        return map_partitions(
            ak.min,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    if axis is None:
        return pw_reduction_with_agg_to_scalar(
            ak.min,
            ak.min,
            array,
            axis=1,
            agg_kwargs={"axis": None},
        )
    else:
        raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


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
            output_divisions=1,
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
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(ak.sum, ak.sum, array)
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
