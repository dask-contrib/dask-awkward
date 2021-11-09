from __future__ import annotations

from typing import TYPE_CHECKING

import awkward as ak
from dask.utils import derived_from

from .core import _TrivialPartitionwiseOp, pw_reduction_with_agg_to_scalar

if TYPE_CHECKING:
    from typing import Any, Callable, Union

    from .core import DaskAwkwardArray, Scalar

    LazyResult = Union[DaskAwkwardArray, Scalar]

####################################
# all awkward.operations.reducers.py
####################################
# def prod(array, axis=None, keepdims=False, mask_identity=False):
# def any(array, axis=None, keepdims=False, mask_identity=False):
# def all(array, axis=None, keepdims=False, mask_identity=False):
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

_count_trivial = _TrivialPartitionwiseOp(ak.count, axis=1)
_count_nonzero_trivial = _TrivialPartitionwiseOp(ak.count_nonzero, axis=1)
_max_trivial = _TrivialPartitionwiseOp(ak.max, axis=1)
_min_trivial = _TrivialPartitionwiseOp(ak.min, axis=1)
_sum_trivial = _TrivialPartitionwiseOp(ak.sum, axis=1)


@derived_from(ak)
def count(
    array: DaskAwkwardArray,
    axis: int | None = None,
    **kwargs: Any,
) -> LazyResult:
    if axis == 1:
        return _count_trivial(array, axis=axis, **kwargs)
    elif axis is None:
        trivial_result = _count_trivial(array, axis=1, **kwargs)
        return pw_reduction_with_agg_to_scalar(trivial_result, ak.sum, ak.sum)
    elif axis == 0 or axis == -1 * array.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


@derived_from(ak)
def count_nonzero(
    array: DaskAwkwardArray,
    axis: int | None = None,
    **kwargs: Any,
) -> LazyResult:
    if axis is not None and axis == 1:
        return _count_nonzero_trivial(array, axis=1, **kwargs)
    elif axis is None:
        trivial_result = _count_nonzero_trivial(array, axis=1, **kwargs)
        return pw_reduction_with_agg_to_scalar(trivial_result, ak.sum, ak.sum)
    elif axis == 0 or axis == -1 * array.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


def _min_or_max(
    f: Callable,
    array: DaskAwkwardArray,
    axis: int | None = None,
    **kwargs: Any,
) -> LazyResult:
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
def max(array: DaskAwkwardArray, axis: int | None = None, **kwargs: Any) -> LazyResult:
    return _min_or_max(ak.max, array, axis, **kwargs)


@derived_from(ak)
def min(array: DaskAwkwardArray, axis: int | None = None, **kwargs: Any) -> LazyResult:
    return _min_or_max(ak.min, array, axis, **kwargs)


@derived_from(ak)
def sum(array: DaskAwkwardArray, axis: int | None = None, **kwargs: Any) -> LazyResult:
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


#####################################
# all awkward.operations.structure.py
#####################################
# def copy(array):
# def mask(array, mask, valid_when=True, highlevel=True, behavior=None):
# def num(array, axis=1, highlevel=True, behavior=None):
# def run_lengths(array, highlevel=True, behavior=None):
# def zip(
# def unzip(array):
# def to_regular(array, axis=1, highlevel=True, behavior=None):
# def from_regular(array, axis=1, highlevel=True, behavior=None):
# def with_name(array, name, highlevel=True, behavior=None):
# def with_field(base, what, where=None, highlevel=True, behavior=None):
# def with_parameter(array, parameter, value, highlevel=True, behavior=None):
# def without_parameters(array, highlevel=True, behavior=None):
# def zeros_like(array, highlevel=True, behavior=None, dtype=None):
# def ones_like(array, highlevel=True, behavior=None, dtype=None):
# def full_like(array, fill_value, highlevel=True, behavior=None, dtype=None):
# def broadcast_arrays(*arrays, **kwargs):
# def concatenate(
# def where(condition, *args, **kwargs):
# def flatten(array, axis=1, highlevel=True, behavior=None):
# def unflatten(array, counts, axis=0, highlevel=True, behavior=None):
# def ravel(array, highlevel=True, behavior=None):
# def _pack_layout(layout):
# def packed(array, highlevel=True, behavior=None):
# def local_index(array, axis=-1, highlevel=True, behavior=None):
# def sort(array, axis=-1, ascending=True, stable=True, highlevel=True, behavior=None):
# def argsort(array, axis=-1, ascending=True, stable=True, highlevel=True, behavior=None):
# def pad_none(array, target, axis=1, clip=False, highlevel=True, behavior=None):
# def _fill_none_deprecated(array, value, highlevel=True, behavior=None):
# def fill_none(array, value, axis=ak._util.MISSING, highlevel=True, behavior=None):
# def is_none(array, axis=0, highlevel=True, behavior=None):
# def singletons(array, highlevel=True, behavior=None):
# def firsts(array, axis=1, highlevel=True, behavior=None):
# def cartesian(
# def argcartesian(
# def combinations(
# def argcombinations(
# def partitions(array):
# def partitioned(arrays, highlevel=True, behavior=None):
# def repartition(array, lengths, highlevel=True, behavior=None):
# def virtual(
# def materialized(array, highlevel=True, behavior=None):
# def with_cache(array, cache, highlevel=True, behavior=None):
# def size(array, axis=None):
# def atleast_1d(*arrays):
# def nan_to_num(
# def isclose(
# def values_astype(array, to, highlevel=True, behavior=None):
# def strings_astype(array, to, highlevel=True, behavior=None):
####################################


_flatten_trivial = _TrivialPartitionwiseOp(ak.flatten, axis=1)
_num_trivial = _TrivialPartitionwiseOp(ak.num, axis=1)


@derived_from(ak)
def num(array: DaskAwkwardArray, axis: int = 1, **kwargs: Any) -> LazyResult:
    return _num_trivial(array, axis=axis, **kwargs)


@derived_from(ak)
def flatten(array: DaskAwkwardArray, axis: int = 1, **kwargs: Any) -> LazyResult:
    return _flatten_trivial(array, axis=axis, **kwargs)


###################################
# all awkward.operations.convert.py
###################################
# def from_numpy(
# def to_numpy(array, allow_missing=True):
# def from_cupy(array, regulararray=False, highlevel=True, behavior=None):
# def to_cupy(array):
# def from_jax(array, regulararray=False, highlevel=True, behavior=None):
# def to_jax(array):
# def kernels(*arrays):
# def to_kernels(array, kernels, highlevel=True, behavior=None):
# def from_iter(
# def to_list(array):
# def from_json(
# def to_json(
# def from_awkward0(
# def to_awkward0(array, keep_layout=False):
# def to_layout(
# def regularize_numpyarray(array, allow_empty=True, highlevel=True, behavior=None):
# def to_arrow(
# def to_arrow_table(
# def from_arrow(array, highlevel=True, behavior=None):
# def to_parquet(
# def from_parquet(
# def to_buffers(
# def from_buffers(
# def to_pandas(
# def from_numpy(
# def to_numpy(array, allow_missing=True):
# def from_cupy(array, regulararray=False, highlevel=True, behavior=None):
# def to_cupy(array):
# def from_jax(array, regulararray=False, highlevel=True, behavior=None):
# def to_jax(array):
# def kernels(*arrays):
# def to_kernels(array, kernels, highlevel=True, behavior=None):
# def from_iter(
# def to_list(array):
# def from_json(
# def to_json(
# def from_awkward0(
# def to_awkward0(array, keep_layout=False):
# def to_layout(
# def regularize_numpyarray(array, allow_empty=True, highlevel=True, behavior=None):
# def to_arrow(
# def to_arrow_table(
# def from_arrow(array, highlevel=True, behavior=None):
# def to_parquet(
# def from_parquet(
# def to_buffers(
# def from_buffers(
# def to_pandas(
###################################
