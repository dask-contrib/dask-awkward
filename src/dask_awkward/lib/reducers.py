from __future__ import annotations

from typing import TYPE_CHECKING, Any

import awkward as ak
import numpy as np
from awkward._typetracer import UnknownScalar

from dask_awkward.lib.core import map_partitions, total_reduction_to_scalar
from dask_awkward.utils import DaskAwkwardNotImplemented, borrow_docstring

if TYPE_CHECKING:
    from dask_awkward.lib.core import Array

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
def all(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Any:
    if axis and axis != 0:
        return map_partitions(
            ak.all,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.any)
def any(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Any:
    if axis and axis != 0:
        return map_partitions(
            ak.any,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.argmax)
def argmax(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = True,
) -> Any:
    if axis and axis >= 1:
        return map_partitions(
            ak.argmax,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.argmin)
def argmin(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = True,
) -> Any:
    if axis and axis >= 1:
        return map_partitions(
            ak.argmin,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.corr)
def corr(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=False,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.count)
def count(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Any:
    if axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            ak.count,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    elif axis is None:
        return total_reduction_to_scalar(
            label="count",
            array=array,
            meta=UnknownScalar(np.dtype(int)),
            chunked_fn=ak.count,
            chunked_kwargs={"axis": 1},
            comb_fn=ak.sum,
            comb_kwargs={"axis": None},
            agg_fn=ak.sum,
            agg_kwargs={"axis": None},
        )
    else:
        raise ValueError("axis must be None or an integer.")


@borrow_docstring(ak.count_nonzero)
def count_nonzero(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Any:
    if axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            ak.count_nonzero,
            array,
            output_divisions=1,
            axis=1,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    elif axis is None:
        return total_reduction_to_scalar(
            label="count_nonzero",
            array=array,
            meta=UnknownScalar(np.dtype(int)),
            chunked_fn=ak.count_nonzero,
            chunked_kwargs={"axis": 1},
            comb_fn=ak.sum,
            comb_kwargs={"axis": None},
            agg_fn=ak.sum,
            agg_kwargs={"axis": None},
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
    mask_identity=False,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.linear_fit)
def linear_fit(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=False,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.max)
def max(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    initial: float | None = None,
    mask_identity: bool = True,
) -> Any:
    if axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            ak.max,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            mask_identity=mask_identity,
        )
    if axis is None:
        return total_reduction_to_scalar(
            label="max",
            array=array,
            chunked_fn=ak.max,
            chunked_kwargs={
                "axis": None,
                "mask_identity": mask_identity,
            },
            meta=ak.max(array._meta, axis=None),
        )
    else:
        raise DaskAwkwardNotImplemented(f"axis={axis} is a TODO")


@borrow_docstring(ak.mean)
def mean(
    array,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=False,
):
    if axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            ak.mean,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.min)
def min(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    initial: float | None = None,
    mask_identity: bool = True,
) -> Any:
    if axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            ak.min,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            mask_identity=mask_identity,
        )
    if axis is None:
        return total_reduction_to_scalar(
            label="min",
            array=array,
            chunked_fn=ak.min,
            chunked_kwargs={
                "axis": None,
                "mask_identity": mask_identity,
            },
            meta=ak.max(array._meta, axis=None),
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
    mask_identity=False,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.prod)
def prod(array, axis=None, keepdims=False, mask_identity=False):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.ptp)
def ptp(arr, axis=None, keepdims=False, mask_identity=True):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.softmax)
def softmax(x, axis=None, keepdims=False, mask_identity=False):
    raise DaskAwkwardNotImplemented("TODO")


class _StdFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.std(array, **self.kwargs)


@borrow_docstring(ak.std)
def std(
    x,
    weight=None,
    ddof=0,
    axis=None,
    keepdims=False,
    mask_identity=False,
):
    if weight is not None:
        raise DaskAwkwardNotImplemented("weight argument is not supported.")
    if axis is None or axis == 0 or axis == -1 * x.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            _StdFn(
                ddof=ddof,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            ),
            x,
            output_divisions=1,
        )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.sum)
def sum(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Any:
    if axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            ak.sum,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        )
    elif axis is None:
        return total_reduction_to_scalar(
            label="sum",
            array=array,
            chunked_fn=ak.sum,
            chunked_kwargs={
                "axis": None,
                "mask_identity": mask_identity,
            },
            meta=ak.max(array._meta, axis=None),
        )
    elif axis == 0:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be none or an integer")


class _VarFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.var(array, **self.kwargs)


@borrow_docstring(ak.var)
def var(
    x,
    weight=None,
    ddof=0,
    axis=None,
    keepdims=False,
    mask_identity=False,
):
    if weight is not None:
        raise DaskAwkwardNotImplemented("weight argument is not supported.")
    if axis is None or axis == 0 or axis == -1 * x.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    if axis and axis != 0:
        return map_partitions(
            _VarFn(
                ddof=ddof,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            ),
            x,
            output_divisions=1,
        )
    raise DaskAwkwardNotImplemented("TODO")
