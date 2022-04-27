from __future__ import annotations

from typing import Any

import awkward._v2 as ak
import numpy as np

from dask_awkward.core import (
    DaskAwkwardNotImplemented,
    IncompatiblePartitions,
    compatible_partitions,
    map_partitions,
    new_known_scalar,
    pw_reduction_with_agg_to_scalar,
)
from dask_awkward.utils import borrow_docstring

__all__ = (
    "argcartesian",
    "argcombinations",
    "argsort",
    "broadcast_arrays",
    "cartesian",
    "combinations",
    "concatenate",
    "copy",
    "fill_none",
    "firsts",
    "flatten",
    "from_regular",
    "full_like",
    "isclose",
    "is_none",
    "local_index",
    "mask",
    "nan_to_num",
    "num",
    "ones_like",
    "packed",
    "pad_none",
    "ravel",
    "run_lengths",
    "singletons",
    "sort",
    "strings_astype",
    "to_regular",
    "unflatten",
    "unzip",
    "values_astype",
    "where",
    "with_field",
    "with_name",
    "with_parameter",
    "without_parameters",
    "zeros_like",
    "zip",
)


@borrow_docstring(ak.argcartesian)
def argcartesian(
    arrays,
    axis: int | None = 1,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel: bool = True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.argcombinations)
def argcombinations(
    array,
    n,
    replacement=False,
    axis: int | None = 1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel: bool = True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.argsort)
def argsort(
    array,
    axis: int | None = -1,
    ascending: bool = True,
    stable: bool = True,
    highlevel: bool = True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.broadcast_arrays)
def broadcast_arrays(*arrays, **kwargs):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.cartesian)
def cartesian(
    arrays,
    axis: int | None = 1,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel: bool = True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.combinations)
def combinations(
    array,
    n,
    replacement: bool = False,
    axis: int | None = 1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel: bool = True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.concatenate)
def concatenate(
    arrays,
    axis: int | None = 0,
    merge: bool = True,
    mergebool: bool = True,
    highlevel: bool = True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.copy)
def copy(array):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.fill_none)
def fill_none(array, value, axis=-1, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.firsts)
def firsts(array, axis: int | None = 1, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.flatten)
def flatten(array, axis: int | None = 1, highlevel: bool = True, behavior=None):
    return map_partitions(
        ak.flatten,
        array,
        axis=axis,
        highlevel=highlevel,
        behavior=behavior,
    )


@borrow_docstring(ak.from_regular)
def from_regular(array, axis: int | None = 1, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.full_like)
def full_like(array, fill_value, highlevel: bool = True, behavior=None, dtype=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.isclose)
def isclose(
    a, b, rtol=1e-05, atol=1e-08, equal_nan=False, highlevel: bool = True, behavior=None
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.is_none)
def is_none(array, axis=0, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.local_index)
def local_index(array, axis=-1, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.mask)
def mask(array, mask, valid_when=True, highlevel: bool = True, behavior=None):
    if not compatible_partitions(array, mask):
        raise IncompatiblePartitions("mask", array, mask)
    return map_partitions(
        ak.mask,
        array,
        mask,
        valid_when=valid_when,
        highlevel=highlevel,
        behavior=behavior,
    )


@borrow_docstring(ak.nan_to_num)
def nan_to_num(
    array,
    copy: bool = True,
    nan: float = 0.0,
    posinf: Any | None = None,
    neginf: Any | None = None,
    highlevel: bool = True,
    behavior: Any | None = None,
):
    return map_partitions(
        ak.nan_to_num,
        array,
        output_partitions=1,
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
        highlevel=highlevel,
        behavior=behavior,
    )


@borrow_docstring(ak.num)
def num(
    array: Any,
    axis: int | None = 1,
    highlevel: bool = True,
    behavior: Any | None = None,
) -> Any:
    if axis and axis >= 1:
        return map_partitions(
            ak.num,
            array,
            axis=axis,
            highlevel=highlevel,
            behavior=behavior,
        )
    if axis == 0:
        if array.known_divisions:
            return new_known_scalar(array.divisions[-1], dtype=int)
        else:
            return pw_reduction_with_agg_to_scalar(
                func=ak.num,
                agg=ak.sum,
                array=array,
                axis=0,
                dtype=np.int64,
                agg_kwargs={"axis": None},
                highlevel=highlevel,
                behavior=behavior,
            )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.ones_like)
def ones_like(array, highlevel: bool = True, behavior=None, dtype=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.packed)
def packed(array, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.pad_none)
def pad_none(
    array,
    target,
    axis: int | None = 1,
    clip=False,
    highlevel: bool = True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.ravel)
def ravel(array, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.run_lengths)
def run_lengths(array, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.singletons)
def singletons(array, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.sort)
def sort(
    array, axis=-1, ascending=True, stable=True, highlevel: bool = True, behavior=None
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.strings_astype)
def strings_astype(array, to, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.to_regular)
def to_regular(array, axis: int | None = 1, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.unflatten)
def unflatten(array, counts, axis=0, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.unzip)
def unzip(array, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.values_astype)
def values_astype(array, to, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.where)
def where(condition, *args, **kwargs):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.with_field)
def with_field(base, what, where=None, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.with_name)
def with_name(array, name, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.with_parameter)
def with_parameter(array, parameter, value, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.without_parameters)
def without_parameters(array, highlevel: bool = True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.zeros_like)
def zeros_like(array, highlevel: bool = True, behavior=None, dtype=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.zip)
def zip(
    arrays,
    depth_limit=None,
    parameters=None,
    with_name=None,
    highlevel: bool = True,
    behavior=None,
    right_broadcast=False,
):
    raise DaskAwkwardNotImplemented("TODO")
