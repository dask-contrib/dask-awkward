from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import awkward as ak
import numpy as np
from awkward._typetracer import UnknownScalar

from dask_awkward.lib.core import (
    Array,
    compatible_partitions,
    map_partitions,
    new_known_scalar,
    total_reduction_to_scalar,
)
from dask_awkward.utils import (
    DaskAwkwardNotImplemented,
    IncompatiblePartitions,
    borrow_docstring,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from dask_awkward.typing import AwkwardDaskCollection

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
    "pad_none",
    "ravel",
    "run_lengths",
    "singletons",
    "sort",
    "strings_astype",
    "to_packed",
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
    axis=1,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.argcombinations)
def argcombinations(
    array,
    n,
    replacement=False,
    axis=1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.argsort)
def argsort(
    array,
    axis=-1,
    ascending=True,
    stable=True,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.broadcast_arrays)
def broadcast_arrays(*arrays, **kwargs):
    raise DaskAwkwardNotImplemented("TODO")


class _CartesianFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *arrays):
        return ak.cartesian(list(arrays), **self.kwargs)


@borrow_docstring(ak.cartesian)
def cartesian(
    arrays,
    axis=1,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    if axis == 1:
        fn = _CartesianFn(
            axis=axis,
            nested=nested,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
        )
        return map_partitions(fn, *arrays, label="cartesian", output_divisions=1)
    raise DaskAwkwardNotImplemented("TODO")


class _CombinationsFn:
    def __init__(self, n: int, axis: int, **kwargs: Any):
        self.n = n
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.combinations(array, self.n, axis=self.axis, **self.kwargs)


@borrow_docstring(ak.combinations)
def combinations(
    array: Array,
    n: int,
    replacement: bool = False,
    axis: int = 1,
    fields: list[str] | None = None,
    parameters: dict | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    if fields is not None and len(fields) != n:
        raise ValueError("if provided, the length of 'fields' must be 'n'")

    if axis != 0:
        fn = _CombinationsFn(
            n=n,
            replacement=replacement,
            axis=axis,
            fields=fields,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
        )
        return map_partitions(
            fn,
            array,
            label="combinations",
            output_divisions=1,
        )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.concatenate)
def concatenate(
    arrays,
    axis=0,
    merge=True,
    mergebool=True,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.copy)
def copy(array):
    raise DaskAwkwardNotImplemented("TODO")


class _FillNoneFn:
    def __init__(self, value, **kwargs):
        self.value = value
        self.kwargs = kwargs

    def __call__(self, arr):
        return ak.fill_none(arr, self.value, **self.kwargs)


@borrow_docstring(ak.fill_none)
def fill_none(
    array: Array,
    value: Any,
    axis: int = -1,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    fn = _FillNoneFn(value, axis=axis, highlevel=highlevel, behavior=behavior)
    return map_partitions(fn, array, label="fill-none", output_divisions=1)


class _FirstsFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.firsts(array, **self.kwargs)


@borrow_docstring(ak.firsts)
def firsts(
    array: Array,
    axis: int = 1,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> AwkwardDaskCollection:
    if axis == 1:
        return map_partitions(
            _FirstsFn(
                axis=axis,
                highlevel=highlevel,
                behavior=behavior,
            ),
            array,
            label="firsts",
            output_divisions=1,
        )
    elif axis == 0:
        return array[0]
    raise DaskAwkwardNotImplemented("TODO")


class _FlattenFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array: ak.Array) -> ak.Array:
        return ak.flatten(array, **self.kwargs)


@borrow_docstring(ak.flatten)
def flatten(
    array: Array,
    axis: int | None = 1,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        _FlattenFn(
            axis=axis,
            highlevel=highlevel,
            behavior=behavior,
        ),
        array,
        label="flatten",
        output_divisions=None,
    )


@borrow_docstring(ak.from_regular)
def from_regular(array, axis=1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.full_like)
def full_like(array, fill_value, highlevel=True, behavior=None, dtype=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.isclose)
def isclose(
    a, b, rtol=1e-05, atol=1e-08, equal_nan=False, highlevel=True, behavior=None
):
    raise DaskAwkwardNotImplemented("TODO")


class _IsNoneFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.is_none(array, **self.kwargs)


@borrow_docstring(ak.is_none)
def is_none(array, axis=0, highlevel=True, behavior=None):
    fn = _IsNoneFn(axis=axis, highlevel=highlevel, behavior=behavior)
    return map_partitions(fn, array, label="is-none", output_divisions=1)


@borrow_docstring(ak.local_index)
def local_index(array, axis=-1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.mask)
def mask(array, mask, valid_when=True, highlevel=True, behavior=None):
    # if not compatible_partitions(array, mask):
    #     raise IncompatiblePartitions("mask", array, mask)
    # return map_partitions(
    #     ak.mask,
    #     array,
    #     mask,
    #     valid_when=valid_when,
    #     highlevel=highlevel,
    #     behavior=behavior,
    # )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.nan_to_num)
def nan_to_num(
    array: Array,
    copy: bool = True,
    nan: float = 0.0,
    posinf: Any | None = None,
    neginf: Any | None = None,
    highlevel: bool = True,
    behavior: Any | None = None,
) -> Array:
    # return map_partitions(
    #     ak.nan_to_num,
    #     array,
    #     output_partitions=1,
    #     copy=copy,
    #     nan=nan,
    #     posinf=posinf,
    #     neginf=neginf,
    #     highlevel=highlevel,
    #     behavior=behavior,
    # )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.num)
def num(
    array: Any,
    axis: int = 1,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Any:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if axis and axis != 0:
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
            return total_reduction_to_scalar(
                label="num",
                array=array,
                meta=UnknownScalar(np.dtype(int)),
                chunked_fn=ak.num,
                chunked_kwargs={"axis": 0},
                comb_fn=ak.sum,
                comb_kwargs={"axis": None},
            )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.ones_like)
def ones_like(
    array: Array,
    highlevel: bool = True,
    behavior: dict | None = None,
    dtype: DTypeLike | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        ak.ones_like,
        array,
        output_divisions=1,
        label="ones-like",
        behavior=behavior,
        dtype=dtype,
    )


@borrow_docstring(ak.to_packed)
def to_packed(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


class _PadNoneFn:
    def __init__(self, target, axis, **kwargs):
        self.target = target
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.pad_none(
            array,
            target=self.target,
            axis=self.axis,
            **self.kwargs,
        )


@borrow_docstring(ak.pad_none)
def pad_none(
    array,
    target,
    axis=1,
    clip=False,
    highlevel=True,
    behavior=None,
):
    if axis == 0:
        DaskAwkwardNotImplemented("axis=0 for pad_none is not supported")
    return map_partitions(
        _PadNoneFn(
            target=target,
            axis=axis,
            clip=clip,
            behavior=behavior,
        ),
        array,
        label="pad-none",
        output_divisions=1,
    )


@borrow_docstring(ak.ravel)
def ravel(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.run_lengths)
def run_lengths(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.singletons)
def singletons(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.sort)
def sort(array, axis=-1, ascending=True, stable=True, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.strings_astype)
def strings_astype(array, to, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.to_regular)
def to_regular(array, axis=1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.unflatten)
def unflatten(array, counts, axis=0, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.unzip)
def unzip(
    array: Array, highlevel: bool = True, behavior: dict | None = None
) -> tuple(Array):
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    fields = ak.fields(array._meta)
    if len(fields) == 0:
        return (array, )
    else:
        return tuple(array[field] for field in fields)


@borrow_docstring(ak.values_astype)
def values_astype(array, to, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


class _WhereFn:
    def __init__(
        self,
        mergebool: bool = True,
        highlevel: bool = True,
        behavior: dict | None = None,
    ) -> None:
        self.mergebool = mergebool
        self.highlevel = highlevel
        self.behavior = behavior

    def __call__(self, condition: ak.Array, x: ak.Array, y: ak.Array) -> ak.Array:
        return ak.where(
            condition,
            x,
            y,
            mergebool=self.mergebool,
            highlevel=self.highlevel,
            behavior=self.behavior,
        )


@borrow_docstring(ak.where)
def where(
    condition: Array,
    x: Array,
    y: Array,
    mergebool: bool = True,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if not compatible_partitions(condition, x, y):
        raise IncompatiblePartitions("where", condition, x, y)
    return map_partitions(
        _WhereFn(mergebool=mergebool, highlevel=highlevel, behavior=behavior),
        condition,
        x,
        y,
        label="where",
    )


class _WithFieldFn:
    def __init__(
        self,
        where: str | None = None,
        highlevel: bool = True,
        behavior: dict | None = None,
    ) -> None:
        self.where = where
        self.highlevel = highlevel
        self.behavior = behavior

    def __call__(self, base: ak.Array, what: ak.Array) -> ak.Array:
        return ak.with_field(
            base,
            what,
            where=self.where,
            highlevel=self.highlevel,
            behavior=self.behavior,
        )


@borrow_docstring(ak.with_field)
def with_field(base, what, where=None, highlevel=True, behavior=None):
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if not compatible_partitions(base, what):
        raise IncompatiblePartitions("with_field", base, what)
    return map_partitions(
        _WithFieldFn(where=where, highlevel=highlevel, behavior=behavior),
        base,
        what,
        label="with-field",
        output_divisions=1,
    )


class _WithNameFn:
    def __init__(self, name: str, behavior: dict | None = None) -> None:
        self.name = name
        self.behavior = behavior

    def __call__(self, array: ak.Array) -> ak.Array:
        return ak.with_name(array, self.name, behavior=self.behavior)


@borrow_docstring(ak.with_name)
def with_name(
    array: Array,
    name: str,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        _WithNameFn(name=name, behavior=behavior),
        array,
        label="with-name",
        output_divisions=1,
    )


class _WithParameterFn:
    def __init__(self, parameter, value, behavior):
        self.parameter = parameter
        self.value = value
        self.behavior = behavior

    def __call__(self, array):
        return ak.with_parameter(
            array,
            parameter=self.parameter,
            value=self.value,
            behavior=self.behavior,
        )


@borrow_docstring(ak.with_parameter)
def with_parameter(
    array: Array,
    parameter: str,
    value: Any,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        _WithParameterFn(parameter=parameter, value=value, behavior=behavior),
        array,
        label="with-parameter",
        output_divisions=1,
    )


class _WithoutParameterFn:
    def __init__(self, behavior):
        self.behavior = behavior

    def __call__(self, array):
        return ak.without_parameters(array, behavior=self.behavior)


@borrow_docstring(ak.without_parameters)
def without_parameters(
    array: Array,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        _WithoutParameterFn(behavior=behavior),
        array,
        label="without-parameters",
        output_divisions=1,
    )


@borrow_docstring(ak.zeros_like)
def zeros_like(
    array: Array,
    highlevel: bool = True,
    behavior: dict | None = None,
    dtype: DTypeLike | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        ak.zeros_like,
        array,
        output_divisions=1,
        label="zeros-like",
        behavior=behavior,
        dtype=dtype,
    )


class _ZipDictInputFn:
    def __init__(self, keys: Sequence[str], **kwargs: Any) -> None:
        self.keys = keys
        self.kwargs = kwargs

    def __call__(self, *parts: ak.Array) -> ak.Array:
        return ak.zip(
            {k: p for k, p in builtins.zip(self.keys, list(parts))},
            **self.kwargs,
        )


class _ZipListInputFn:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, *parts: Any) -> ak.Array:
        return ak.zip(list(parts), **self.kwargs)


@borrow_docstring(ak.zip)
def zip(
    arrays: dict | list | tuple,
    depth_limit: int | None = None,
    parameters: dict | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: dict | None = None,
    right_broadcast: bool = False,
    optiontype_outside_record: bool = False,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if isinstance(arrays, dict):
        keys, colls, metadict = [], [], {}
        for k, coll in arrays.items():
            keys.append(k)
            colls.append(coll)
            metadict[k] = coll._meta

        meta = ak.zip(
            metadict,
            depth_limit=depth_limit,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            right_broadcast=right_broadcast,
            optiontype_outside_record=optiontype_outside_record,
        )

        return map_partitions(
            _ZipDictInputFn(
                keys,
                depth_limit=depth_limit,
                parameters=parameters,
                with_name=with_name,
                highlevel=highlevel,
                behavior=behavior,
                right_broadcast=right_broadcast,
                optiontype_outside_record=optiontype_outside_record,
            ),
            *colls,
            label="zip",
            meta=meta,
        )

    elif isinstance(arrays, (list, tuple)):
        fn = _ZipListInputFn(
            depth_limit=depth_limit,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            right_broadcast=right_broadcast,
            optiontype_outside_record=optiontype_outside_record,
        )
        return map_partitions(
            fn,
            *arrays,
            label="zip",
        )

    else:
        raise DaskAwkwardNotImplemented(
            "only sized iterables are supported by dak.zip (dict, list, or tuple)"
        )
