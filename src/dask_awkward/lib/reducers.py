from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import awkward as ak

from dask_awkward.lib.core import map_partitions, non_trivial_reduction
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
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="all",
            array=array,
            reducer=ak.all,
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.all,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.any)
def any(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="any",
            array=array,
            reducer=ak.any,
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.any,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.argmax)
def argmax(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="argmax",
            array=array,
            reducer=ak.argmax,
            is_positional=True,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.argmax,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.argmin)
def argmin(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="argmin",
            array=array,
            reducer=ak.argmin,
            is_positional=True,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.argmin,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.corr)
def corr(
    x: Array,
    y: Array,
    weight: Array | int | float | complex | None = None,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.count)
def count(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="count",
            array=array,
            reducer=ak.count,
            combiner=ak.sum,
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.count,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.count_nonzero)
def count_nonzero(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="count_nonzero",
            array=array,
            reducer=ak.count_nonzero,
            combiner=ak.sum,
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.count_nonzero,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.covar)
def covar(
    x: Array,
    y: Array,
    weight: Array | int | float | complex | None = None,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.linear_fit)
def linear_fit(
    x: Array,
    y: Array,
    weight: Array | int | float | complex | None = None,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    raise DaskAwkwardNotImplemented("TODO")


class _MaxFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array, **kwargs):
        return ak.max(array, **self.kwargs, **kwargs)


@borrow_docstring(ak.max)
def max(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    initial: float | None = None,
    mask_identity: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="max",
            array=array,
            reducer=_MaxFn(initial=initial),
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            _MaxFn(initial=initial),
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.mean)
def mean(
    array: Array,
    weight: Array | int | float | complex | None = None,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if weight is not None:
        raise DaskAwkwardNotImplemented(
            f"weight={weight} is not supported for this array yet."
        )

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
            behavior=behavior,
            attrs=attrs,
        )
    raise DaskAwkwardNotImplemented("TODO")


class _MinFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array, **kwargs):
        return ak.min(array, **self.kwargs, **kwargs)


@borrow_docstring(ak.min)
def min(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    initial: float | None = None,
    mask_identity: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="min",
            array=array,
            reducer=_MinFn(initial=initial),
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            _MinFn(initial=initial),
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.moment)
def moment(
    x: Array,
    n: int,
    weight: Array | int | float | complex | None = None,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.prod)
def prod(
    array: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="prod",
            array=array,
            reducer=ak.prod,
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.prod,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


@borrow_docstring(ak.ptp)
def ptp(
    arr: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.softmax)
def softmax(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    raise DaskAwkwardNotImplemented("TODO")


class _StdFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.std(array, **self.kwargs)


@borrow_docstring(ak.std)
def std(
    x: Array,
    weight: Array | int | float | complex | None = None,
    ddof: int = 0,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
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
                behavior=behavior,
                attrs=attrs,
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
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis is None or axis == 0 or axis == -1 * array.ndim:
        return non_trivial_reduction(
            axis=axis,
            label="sum",
            array=array,
            reducer=ak.sum,
            is_positional=False,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )
    else:
        return map_partitions(
            ak.sum,
            array,
            output_divisions=1,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            behavior=behavior,
            attrs=attrs,
        )


class _VarFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.var(array, **self.kwargs)


@borrow_docstring(ak.var)
def var(
    x: Array,
    weight: Array | int | float | complex | None = None,
    ddof: int = 0,
    axis: int | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
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
                behavior=behavior,
                attrs=attrs,
            ),
            x,
            output_divisions=1,
        )
    raise DaskAwkwardNotImplemented("TODO")
