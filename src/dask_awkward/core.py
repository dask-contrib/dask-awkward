from __future__ import annotations

import keyword
import operator
import warnings
from functools import cached_property, partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Hashable, Mapping, Sequence, TypeVar

import awkward._v2 as ak
import dask.config
import numpy as np
from awkward._v2._typetracer import MaybeNone, OneOf, TypeTracerArray, UnknownScalar
from awkward._v2.highlevel import _dir_pattern
from dask.base import DaskMethodsMixin
from dask.base import compute as dask_compute
from dask.base import dont_optimize, is_dask_collection, tokenize
from dask.blockwise import blockwise as dask_blockwise
from dask.context import globalmethod
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import IndexCallable, funcname, key_split
from numpy.lib.mixins import NDArrayOperatorsMixin

from dask_awkward.optimize import optimize
from dask_awkward.utils import (
    empty_typetracer,
    is_empty_slice,
    normalize_single_outer_inner_index,
)

if TYPE_CHECKING:
    from awkward._v2.contents.content import Content
    from awkward._v2.forms.form import Form
    from awkward._v2.types.type import Type
    from dask.array.core import Array as DaskArray
    from dask.blockwise import Blockwise
    from dask.delayed import Delayed
    from numpy.typing import DTypeLike


T = TypeVar("T")


class DaskAwkwardNotImplemented(NotImplementedError):
    NOT_SUPPORTED_MSG = """

If you would like this unsupported call to be supported by
dask-awkward please open an issue at:
https://github.com/ContinuumIO/dask-awkward."""

    def __init__(self, msg: str | None = None) -> None:
        msg = f"{msg or ''}{self.NOT_SUPPORTED_MSG}"
        super().__init__(msg)


class IncompatiblePartitions(ValueError):
    def __init__(self, name, *args) -> None:
        msg = self.divisions_msg(name, *args)
        super().__init__(msg)

    @staticmethod
    def divisions_msg(name: str, *args: Any) -> str:
        msg = f"The inputs to {name} are incompatibly partitioned\n"
        for i, arg in enumerate(args):
            msg += f"- arg{i} divisions: {arg.divisions}\n"
        return msg


def _finalize_array(
    results: Sequence[Number | ak.Array | ak.Record],
) -> Number | ak.Array:
    if any(isinstance(r, ak.Array) for r in results):
        return ak.concatenate(results)
    elif len(results) == 1 and isinstance(results[0], int):
        return results[0]
    elif all(isinstance(r, (int, np.integer)) for r in results):
        return ak.Array(results)
    else:
        msg = (
            "Unexpected results of a computation.\n "
            f"results: {results}"
            f"type of first result: {type(results[0])}"
        )
        raise RuntimeError(msg)


def _finalize_scalar(results: Sequence[T]) -> T:
    return results[0]


class Scalar(DaskMethodsMixin):
    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        meta: Any | None = None,
        known_value: Any | None = None,
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._meta: Any | None = self._check_meta(meta)
        self._known_value: Any | None = known_value

    def __dask_graph__(self) -> HighLevelGraph:
        return self._dask

    def __dask_keys__(self) -> list[str]:
        return [self._name]

    def __dask_layers__(self) -> tuple[str, ...]:
        if isinstance(self._dask, HighLevelGraph) and len(self._dask.layers) == 1:
            return tuple(self._dask.layers)
        return (self.name,)

    def __dask_tokenize__(self) -> Hashable:
        return self.name

    __dask_optimize__ = globalmethod(
        optimize, key="awkward_scalar_optimize", falsey=dont_optimize
    )

    __dask_scheduler__ = staticmethod(threaded_get)

    def __dask_postcompute__(self) -> tuple[Callable, tuple]:
        return _finalize_scalar, ()

    def __dask_postpersist__(self) -> tuple[Callable, tuple]:
        return self._rebuild, ()

    def _rebuild(
        self,
        dsk: HighLevelGraph,
        *,
        rename: Mapping[str, str] | None = None,
    ) -> Any:
        name = self._name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self._meta, self.known_value)

    def __reduce__(self):
        return (Scalar, (self.dask, self.name, self._meta, self.known_value))

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def name(self) -> str:
        return self._name

    def _check_meta(self, m: Any | None) -> Any | None:
        if m is not None and not isinstance(m, (MaybeNone, UnknownScalar, OneOf)):
            raise TypeError(f"meta must be a typetracer module object, not a {type(m)}")
        return m

    @property
    def dtype(self) -> np.dtype | None:
        try:
            if self._meta is not None:
                return self._meta.dtype
        except AttributeError:
            pass
        return None

    @staticmethod
    def from_known(s: Any, dtype: DTypeLike | None = None) -> Scalar:
        return new_known_scalar(s, dtype=dtype)

    def __repr__(self) -> str:  # pragma: no cover
        return self.__str__()

    def __str__(self) -> str:
        dt = str(self.dtype) or "Unknown"
        if self.known_value is not None:
            return (
                f"dask.awkward<{key_split(self.name)}, "
                "type=Scalar, "
                f"dtype={dt}, "
                f"known_value={self.known_value}>"
            )
        return f"dask.awkward<{key_split(self.name)}, type=Scalar, dtype={dt}>"

    @property
    def known_value(self) -> Any | None:
        return self._known_value

    def to_delayed(self, optimize_graph: bool = True) -> Delayed:
        from dask.delayed import Delayed

        dsk = self.__dask_graph__()
        if optimize_graph:
            dsk = self.__dask_optimize__(dsk, self.__dask_keys__())
        return Delayed(self.name, dsk)


def new_scalar_object(dsk: HighLevelGraph, name: str, *, meta: Any) -> Scalar:
    return Scalar(dsk, name, meta, known_value=None)


def new_known_scalar(s: Any, dtype: DTypeLike | None = None) -> Scalar:
    name = tokenize(s)
    if dtype is None:
        if isinstance(s, (int, np.integer)):
            dtype = np.dtype(int)
        elif isinstance(s, (float, np.floating)):
            dtype = np.dtype(float)
        else:
            dtype = np.dtype(type(s))
    llg = {name: s}
    hlg = HighLevelGraph.from_collections(name, llg, dependencies=())
    return Scalar(hlg, name, meta=UnknownScalar(dtype), known_value=s)


class Record(Scalar):
    def __init__(self, dsk: HighLevelGraph, name: str, meta: Any | None = None) -> None:
        super().__init__(dsk, name, meta)

    def _check_meta(self, m: Any | None) -> Any | None:
        if m is not None and not isinstance(m, ak.Record):
            raise TypeError(f"meta must be a Record typetracer object, not a {type(m)}")
        return m

    @property
    def npartitions(self):
        """Records are unpartitioned by definition."""
        return 1

    def __getitem__(self, key: str) -> Any:
        token = tokenize(self, key)
        name = f"getitem-{token}"
        new_meta = self._meta[key]  # type: ignore

        # first check for array type return
        if isinstance(new_meta, ak.Array):
            graphlayer = {(name, 0): (operator.getitem, self.name, key)}
            hlg = HighLevelGraph.from_collections(
                name,
                graphlayer,
                dependencies=[self],  # type: ignore
            )
            return new_array_object(hlg, name, meta=new_meta, npartitions=1)

        # then check for scalar (or record) type
        graphlayer = {name: (operator.getitem, self.name, key)}  # type: ignore
        hlg = HighLevelGraph.from_collections(
            name,
            graphlayer,
            dependencies=[self],  # type: ignore
        )
        if isinstance(new_meta, ak.Record):
            return new_record_object(hlg, name, meta=new_meta)
        else:
            return new_scalar_object(hlg, name, meta=new_meta)

    def __getattr__(self, attr: str) -> Any:
        if attr not in (self.fields or []):
            raise AttributeError(f"{attr} not in fields.")
        try:
            return self.__getitem__(attr)
        except (IndexError, KeyError):
            raise AttributeError(f"{attr} not in fields.")

    def __str__(self) -> str:
        return f"dask.awkward<{key_split(self.name)}, type=Record>"

    def __reduce__(self):
        return (Record, (self.dask, self.name, self._meta))

    @property
    def fields(self) -> list[str] | None:
        if self._meta is not None:
            return ak.fields(self._meta)
        return None

    def _ipython_key_completions_(self) -> list[str]:
        if self._meta is not None:
            return self._meta._ipython_key_completions_()
        return []

    def __dir__(self) -> list[str]:
        fields = [] if self._meta is None else self._meta._layout.fields
        return sorted(
            set(
                [x for x in dir(type(self)) if not x.startswith("_")]
                + dir(super())
                + [
                    x
                    for x in fields
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )


def new_record_object(dsk: HighLevelGraph, name: str, *, meta: Any) -> Record:
    return Record(dsk, name, meta)


class Array(DaskMethodsMixin, NDArrayOperatorsMixin):
    """Partitioned, lazy, and parallel Awkward Array Dask collection.

    The class constructor is not intended for users. Instead use
    factory functions like :py:func:`dask_awkward.from_parquet`,
    :py:func:`dask_awkward.from_json`, etc.

    Within dask-awkward the ``new_array_object`` factory function is
    used for creating new instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        meta: ak.Array | None,
        divisions: tuple[int | None, ...],
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._divisions = divisions
        if meta is None:
            self._meta: ak.Array = empty_typetracer()
        elif not isinstance(meta, (ak.Array, TypeTracerArray)):
            raise TypeError("meta must be an instance of an Awkward Array.")
        self._meta = meta

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> list[tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> Hashable:
        return self.name

    def __dask_postcompute__(self) -> tuple[Callable, tuple]:
        return _finalize_array, ()

    def __dask_postpersist__(self) -> tuple[Callable, tuple]:
        return self._rebuild, ()

    __dask_optimize__ = globalmethod(
        optimize, key="awkward_array_optimize", falsey=dont_optimize
    )

    __dask_scheduler__ = staticmethod(threaded_get)

    def _rebuild(
        self,
        dsk: HighLevelGraph,
        *,
        rename: Mapping[str, str] | None = None,
    ) -> Array:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return Array(dsk, name, self._meta, divisions=self.divisions)

    def __len__(self) -> int:
        if self.known_divisions:
            pass
        else:
            self.eager_compute_divisions()
        return self.divisions[-1]  # type: ignore

    def _shorttypestr(self, max: int = 10) -> str:
        return str(_type(self))[0:max]

    def _typestr(self, max: int = 0) -> str:
        tstr = str(_type(self))
        if max and len(tstr) > max:
            tstr = f"{tstr[0:max]} ... }}"
        return f"var * {tstr}"

    def __str__(self) -> str:
        return (
            f"dask.awkward<{key_split(self.name)}, "
            f"npartitions={self.npartitions}"
            ">"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.__str__()

    def reset_meta(self) -> None:
        self._meta = empty_typetracer()

    # def _ipython_display_(self) -> None:
    #     if self._meta is None:
    #         return None
    #
    #     import json
    #
    #     from IPython.display import display_json
    #
    #     display_json(json.loads(self._meta.form.to_json()), raw=True)

    def _ipython_key_completions_(self) -> list[str]:
        if self._meta is not None:
            return self._meta._ipython_key_completions_()
        return []

    def __dir__(self) -> list[str]:
        fields = [] if self._meta is None else self._meta._layout.fields
        return sorted(
            set(
                [x for x in dir(type(self)) if not x.startswith("_")]
                + dir(super())
                + [
                    x
                    for x in fields
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def keys(self) -> list[tuple[str, int]]:
        return self.__dask_keys__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def ndim(self) -> int | None:
        return ndim(self)

    @property
    def divisions(self) -> tuple[int | None, ...]:
        return self._divisions

    @property
    def known_divisions(self) -> bool:
        return len(self.divisions) > 0 and None not in self.divisions

    @property
    def npartitions(self) -> int:
        return len(self.divisions) - 1

    @property
    def layout(self) -> Content:
        if self._meta is not None:
            return self._meta.layout
        raise ValueError("This collections meta is None; unknown layout.")

    @property
    def _typetracer(self) -> ak.Array:
        return self._meta

    @property
    def fields(self) -> list[str]:
        return ak.fields(self._meta)

    @property
    def form(self) -> Form:
        return self._meta.layout.form

    @cached_property
    def keys_array(self) -> np.ndarray:
        return np.array(self.__dask_keys__(), dtype=object)

    def _partitions(self, index: Any) -> Array:
        if not isinstance(index, tuple):
            index = (index,)
        token = tokenize(self, index)
        from dask.array.slicing import normalize_index

        raw = normalize_index(index, (self.npartitions,))
        index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in raw)  # type: ignore
        name = f"partitions-{token}"
        new_keys = self.keys_array[index].tolist()
        dsk = {(name, i): tuple(key) for i, key in enumerate(new_keys)}
        graph = HighLevelGraph.from_collections(
            name,
            dsk,
            dependencies=[self],  # type: ignore
        )

        # if a single partition was requested we trivially know the new divisions.
        if len(raw) == 1 and isinstance(raw[0], int) and self.known_divisions:  # type: ignore
            new_divisions = (
                0,
                self.divisions[raw[0] + 1] - self.divisions[raw[0]],  # type: ignore
            )
        # otherwise nullify the known divisions
        else:
            new_divisions = (None,) * (len(new_keys) + 1)  # type: ignore

        return new_array_object(
            graph, name, meta=self._meta, divisions=tuple(new_divisions)
        )

    @property
    def partitions(self) -> IndexCallable:
        """Get a specific partition or slice of partitions.

        Returns
        -------
        dask.utils.IndexCallable

        Examples
        --------
        >>> import dask_awkward as dak
        >>> import awkward._v2 as ak
        >>> aa = ak.Array([[1, 2, 3], [], [2]])
        >>> a = dak.from_awkward(aa, npartitions=3)
        >>> a
        dask.awkward<from-awkward, npartitions=3>
        >>> a.partitions[0]
        dask.awkward<partitions, npartitions=1>
        >>> a.partitions[0:2]
        dask.awkward<partitions, npartitions=2>
        >>> a.partitions[2].compute()
        <Array [[2]] type='1 * var * int64'>

        """
        return IndexCallable(self._partitions)

    def _getitem_trivial_map_partitions(
        self,
        where: Any,
        meta: Any | None = None,
    ) -> Any:
        if meta is None and self._meta is not None:
            if isinstance(where, tuple):
                metad = to_meta(where)
                meta = self._meta[metad]
            else:
                m = to_meta([where])[0]
                meta = self._meta[m]
        return self.map_partitions(
            operator.getitem, where, meta=meta, output_divisions=1
        )

    def _getitem_outer_boolean_lazy_array(self, where: Array | tuple[Any, ...]) -> Any:
        ba = where if isinstance(where, Array) else where[0]
        if not compatible_partitions(self, ba):
            raise IncompatiblePartitions("getitem", self, ba)

        new_meta: Any | None = None
        if self._meta is not None:
            if isinstance(where, tuple):
                raise DaskAwkwardNotImplemented(
                    "tuple style input boolean selection is not supported."
                )
            elif isinstance(where, Array):
                new_meta = self._meta[where._meta]
                return self.map_partitions(
                    operator.getitem,
                    where,
                    meta=new_meta,
                )

    def _getitem_outer_str_or_list(self, where: str | list | tuple[Any, ...]) -> Any:
        new_meta: Any | None = None
        if self._meta is not None:
            if isinstance(where, tuple):
                if not isinstance(where[0], (str, list)):
                    raise TypeError("Expected where[0] to be a string or list")
                metad = to_meta(where)
                new_meta = self._meta[metad]
            elif isinstance(where, (str, list)):
                new_meta = self._meta[where]
        return self._getitem_trivial_map_partitions(where, meta=new_meta)

    def _getitem_outer_int(self, where: int | tuple[Any, ...]) -> Any:
        self._divisions = calculate_known_divisions(self)

        new_meta: Any | None = None
        # multiple objects passed to getitem. collections passed in
        # the tuple of objects have not been tested!
        if isinstance(where, tuple):
            if not isinstance(where[0], int):
                raise TypeError("Expected where[0] to be and integer.")
            pidx, outer_where = normalize_single_outer_inner_index(
                self.divisions, where[0]  # type: ignore
            )
            partition = self.partitions[pidx]
            rest = where[1:]
            where = (outer_where, *rest)
            if partition._meta is not None:
                metad = to_meta(where)
                new_meta = partition._meta[metad]
        # single object passed to getitem
        elif isinstance(where, int):
            pidx, where = normalize_single_outer_inner_index(self.divisions, where)  # type: ignore
            partition = self.partitions[pidx]
            if partition._meta is not None:
                new_meta = partition._meta[where]

        # if we know a new array is going to be made, just call the
        # trivial inner on the new partition.
        if isinstance(new_meta, ak.Array):
            result = partition._getitem_trivial_map_partitions(where, meta=new_meta)
            result._divisions = (0, None)
            return result

        # otherwise make sure we have one of the other potential results.
        if not isinstance(new_meta, (ak.Record, UnknownScalar, OneOf, MaybeNone)):
            raise DaskAwkwardNotImplemented("Key type not supported for this array.")

        token = tokenize(partition, where)
        name = f"getitem-{token}"
        dsk = {
            name: (
                lambda x, gikey: operator.getitem(x, gikey),
                partition.__dask_keys__()[0],
                where,
            )
        }
        hlg = HighLevelGraph.from_collections(
            name,
            dsk,
            dependencies=[partition],  # type: ignore
        )
        if isinstance(new_meta, ak.Record):
            return new_record_object(hlg, name, meta=new_meta)
        else:
            return new_scalar_object(hlg, name, meta=new_meta)

    def _getitem_tuple(self, where: tuple[Any, ...]) -> Array:
        if isinstance(where[0], int):
            return self._getitem_outer_int(where)

        elif isinstance(where[0], str):
            return self._getitem_outer_str_or_list(where)

        elif isinstance(where[0], list):
            return self._getitem_outer_str_or_list(where)

        elif isinstance(where[0], slice) and is_empty_slice(where[0]):
            return self._getitem_trivial_map_partitions(where)

        # boolean array
        elif isinstance(where[0], Array):
            try:
                dtype = where[0].layout.dtype.type
            except AttributeError:
                dtype = where[0].layout.content.dtype.type
            if issubclass(dtype, (np.bool_, bool)):
                return self._getitem_outer_boolean_lazy_array(where)

        raise DaskAwkwardNotImplemented(
            f"Array.__getitem__ doesn't support multi object: {where}"
        )

    def _getitem_single(self, where: Any) -> Array:

        # a single string
        if isinstance(where, str):
            return self._getitem_outer_str_or_list(where)

        elif isinstance(where, list):
            return self._getitem_outer_str_or_list(where)

        # a single integer
        elif isinstance(where, int):
            return self._getitem_outer_int(where)

        elif isinstance(where, Array):
            try:
                dtype = where.layout.dtype.type
            except AttributeError:
                dtype = where.layout.content.dtype.type
            if issubclass(dtype, (np.bool_, bool)):
                return self._getitem_outer_boolean_lazy_array(where)

        # an empty slice
        elif is_empty_slice(where):
            return self

        # a single ellipsis
        elif where is Ellipsis:
            return self._getitem_trivial_map_partitions(where)

        raise DaskAwkwardNotImplemented(f"__getitem__ doesn't support where={where}.")

    def __getitem__(self, where: Any) -> Any:
        """Select items from the collection.

        Heavily under construction.

        Arguments
        ---------
        where : many types supported
            Selection criteria.

        Returns
        -------
        Array | Record | Scalar
            Resulting collection.

        """

        # don't accept lists containing integers.
        if isinstance(where, list):
            if any(isinstance(k, int) for k in where):
                # this is something we'll likely never support so we
                # do not use the DaskAwkwardNotImplemented exception.
                raise RuntimeError("Lists containing integers are not supported.")

        if isinstance(where, tuple):
            return self._getitem_tuple(where)

        return self._getitem_single(where)

    def __getattr__(self, attr: str) -> Any:
        if attr not in (self.fields or []):
            raise AttributeError(f"{attr} not in fields.")
        try:
            return self.__getitem__(attr)
        except (IndexError, KeyError):
            raise AttributeError(f"{attr} not in fields.")

    def map_partitions(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        """Map a function across all partitions of the collection.

        Parameters
        ----------
        func : Callable
            Function to call on all partitions.
        *args : Collections and function arguments
            Additional arguments passed to `func` after the
            collection, if arguments are Array collections
            they must be compatibly partitioned with the object this
            method is being called from.
        **kwargs : Any
            Additional keyword arguments passed to the `func`.

        Returns
        -------
        dask_awkward.Array
            The new collection.

        See Also
        --------
        dask_awkward.map_partitions

        """
        return map_partitions(func, self, *args, **kwargs)

    def eager_compute_divisions(self) -> None:
        self._divisions = calculate_known_divisions(self)

    def clear_divisions(self) -> None:
        """Clear the divisions of a Dask Awkward Collection."""
        self._divisions = (None,) * (self.npartitions + 1)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise RuntimeError("Array ufunc supports only method == '__call__'")

        new_meta = None

        # divisions need to be compat. (identical for now?)

        inputs_meta = []
        for inp in inputs:
            # if input is a Dask Awkward Array collection, grab it's meta
            if isinstance(inp, Array):
                inputs_meta.append(inp._meta)
            # if input is a concrete Awkward Array, grab it's typetracer
            elif isinstance(inp, ak.Array):
                inputs_meta.append(ak.Array(inp.layout.typetracer.forget_length()))
            # otherwise pass along
            else:
                inputs_meta.append(inp)

        # compute new meta from inputs
        new_meta = ufunc(*inputs_meta)

        return map_partitions(
            ufunc,
            *inputs,
            meta=new_meta,
            output_divisions=1,
            **kwargs,
        )

    def to_delayed(self, optimize_graph: bool = True) -> list[Delayed]:
        from dask_awkward.io import to_delayed

        return to_delayed(self, optimize_graph=optimize_graph)

    def to_dask_array(self) -> DaskArray:
        from dask_awkward.io import to_dask_array

        return to_dask_array(self)


def _first_partition(array: Array) -> ak.Array:
    """Compute the first partition of an Array collection.

    Parameters
    ----------
    array : dask_awkward.Array
        Awkward collection.

    Returns
    -------
    ak.Array
        Concrete awkward array for the first partition of the Dask
        awkward array.

    """
    with dask.config.set({"awkward.compute-unknown-meta": False}):
        (computed,) = dask_compute(
            array.partitions[0],
            traverse=False,
            optimize_graph=True,
            scheduler="threads",
        )
        return computed


def _get_typetracer(array: Array) -> ak.Array:
    """Obtain the awkward type tracer associated with an array.

    This will compute the first partition of the array collection if
    necessary.

    Parameters
    ----------
    array : dask_awkward.Array
        The collection.

    Returns
    -------
    Array
        Awkward high level array wrapping the typetracer (metadata).

    """
    if array._meta is not None:
        return array._meta
    first_part = _first_partition(array)
    if not isinstance(first_part, ak.Array):
        raise TypeError(f"Should have an ak.Array type, got {type(first_part)}")
    return ak.Array(first_part.layout.typetracer.forget_length())


def new_array_object(
    dsk: HighLevelGraph,
    name: str,
    *,
    meta: ak.Array | None = None,
    npartitions: int | None = None,
    divisions: tuple[int | None, ...] | None = None,
) -> Array:
    """Instantiate a new Array collection object.

    Parameters
    ----------
    dsk : dask.highlevelgraph.HighLevelGraph
        Graph backing the collection.
    name : str
        Unique name for the collection.
    meta : Array, optional
        Collection metadata; this is an awkward-array type tracer.
    npartitions : int, optional
        Total number of partitions; if used `divisions` will be a
        tuple of length `npartitions` + 1 with all elements``None``.
    divisions : tuple[int | None, ...], optional
        Tuple identifying the locations of the divisions between the
        partitions.

    Returns
    -------
    Array
        Resulting collection.

    """
    if divisions is None and npartitions is not None:
        divisions = (None,) * (npartitions + 1)
    elif divisions is not None and npartitions is not None:
        raise ValueError("Only one of either divisions or npartitions must be defined.")
    elif divisions is None and npartitions is None:
        raise ValueError("One of either divisions or npartitions must be defined.")

    array = Array(dsk, name, meta, divisions)  # type: ignore

    if meta is None:
        if dask.config.get("awkward.compute-unknown-meta"):
            try:
                array._meta = _get_typetracer(array)
            except (AttributeError, AssertionError, TypeError):
                array._meta = empty_typetracer()
    else:
        if not isinstance(meta, (ak.Array, TypeTracerArray)):
            msg = (
                "meta should be an awkward Array or TypeTracerArray object.\n"
                f"got: {type(meta)}"
            )
            raise ValueError(msg)

    return array


def partitionwise_layer(
    func: Callable,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Blockwise:
    """Create a partitionwise graph layer.

    Parameters
    ----------
    func : Callable
        Function to apply on all partitions.
    name : str
        Name for the layer.
    *args : Any
        Arguments that will be passed to `func`.
    **kwargs : Any
        Keyword arguments that will be passed to `func`.

    Returns
    -------
    dask.blockwise.Blockwise
        The Dask HighLevelGraph Blockwise layer.

    """
    pairs: list[Any] = []
    numblocks: dict[Any, int | tuple[int, ...]] = {}
    for arg in args:
        if isinstance(arg, Array):
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = (arg.npartitions,)
        elif is_dask_collection(arg):
            raise DaskAwkwardNotImplemented(
                "Use of Array with other Dask collections is currently unsupported."
            )
        else:
            pairs.extend([arg, None])
    return dask_blockwise(
        func,
        name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )


def map_partitions(
    func: Callable,
    *args: Any,
    name: str | None = None,
    meta: Any | None = None,
    output_divisions: int | None = None,
    **kwargs: Any,
) -> Array:
    """Map a callable across all partitions of a collection.

    Parameters
    ----------
    func : Callable
        Function to apply on all partitions.
    *args : Collections and function arguments
        Arguments passed to the function, if arguments are
        Array collections they must be compatibly
        partitioned.
    name : str, optional
        Name for the Dask graph layer; if left to ``None`` (default),
        the name of the function will be used.
    meta : Any, optional
        Metadata (typetracer) information of the result (if known).
    output_divisions : int, optional
        If ``None`` (the default), the divisions of the output will be
        assumed unknown. If defined, the output divisions will be
        multiplied by a factor of `output_divisions`. A value of 1
        means constant divisions (e.g. a string based slice). Any
        value greater than 1 means the divisions were expanded by some
        operation. This argument is mainly for internal library
        function implementations.
    **kwargs : Any
        Additional keyword arguments passed to the `func`.

    Returns
    -------
    dask_awkward.Array
        The new collection.

    """
    token = tokenize(func, *args, **kwargs)
    name = name or funcname(func)
    name = f"{name}-{token}"
    lay = partitionwise_layer(func, name, *args, **kwargs)
    deps = [a for a in args if is_dask_collection(a)] + [
        v for _, v in kwargs.items() if is_dask_collection(v)
    ]

    if meta is None:
        if dask.config.get("awkward.compute-unknown-meta"):
            metas = to_meta(args)
            try:
                meta = func(*metas, **kwargs)
            except Exception:
                warnings.warn(
                    "metadata could not be determined; "
                    "a compute on the first partition will occur."
                )

    hlg = HighLevelGraph.from_collections(
        name,
        lay,
        dependencies=deps,  # type: ignore
    )

    if output_divisions is not None:
        if output_divisions == 1:
            new_divisions = args[0].divisions
        else:
            new_divisions = tuple(
                map(lambda x: x * output_divisions, args[0].divisions)
            )
        return new_array_object(
            hlg,
            name=name,
            meta=meta,
            divisions=new_divisions,
        )
    else:
        return new_array_object(
            hlg,
            name=name,
            meta=meta,
            npartitions=args[0].npartitions,
        )


def pw_reduction_with_agg_to_scalar(
    func: Callable,
    agg: Callable,
    array: Array,
    *,
    dtype: Any | None = None,
    agg_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Scalar:
    """Partitionwise operation with aggregation to scalar.

    Parameters
    ----------
    array : dask_awkward.Array
        Awkward array collection.
    func : Callable
        Function to apply on all partitions.
    agg : Callable
        Function to aggregate the result on each partition.
    name : str, optional
        Name for the computation, if ``None`` the name of `func` will
        be used.
    **kwargs : Any
        Keyword arguments passed to `func`.

    Returns
    -------
    Scalar
        Resulting scalar Dask collection.

    """
    if agg_kwargs is None:
        agg_kwargs = {}
    token = tokenize(array)
    namefunc = func.__name__
    nameagg = agg.__name__
    namefunc = f"{namefunc}-{token}"
    nameagg = f"{nameagg}-{token}"
    func = partial(func, **kwargs)
    agg = partial(agg, **agg_kwargs)
    dsk = {(namefunc, i): (func, k) for i, k in enumerate(array.__dask_keys__())}
    dsk[nameagg] = (agg, list(dsk.keys()))  # type: ignore
    hlg = HighLevelGraph.from_collections(
        nameagg,
        dsk,
        dependencies=[array],  # type: ignore
    )
    meta = UnknownScalar(np.dtype(dtype)) if dtype is not None else None
    return new_scalar_object(hlg, name=nameagg, meta=meta)


def calculate_known_divisions(array: Array) -> tuple[int, ...]:
    """Determine the divisions of a collection.

    This function triggers an immediate computation.

    Parameters
    ----------
    array : dask_awkward.Array
        Awkard array collection.

    Returns
    -------
    tuple[int, ...]
        Locations (indices) of division boundaries.

    """
    with dask.config.set({"awkward.compute-unknown-meta": False}):
        # if more than 1 partition use cumulative sum
        if array.npartitions > 1:
            nums = array.map_partitions(len).compute()
            cs = list(np.cumsum(nums))
            return tuple([0, *cs])

        # if only 1 partition just get it's length
        return (0, array.map_partitions(len).compute())


def _type(array: Array) -> Type | None:
    """Get the type object associated with an array.

    Parameters
    ----------
    array : dask_awkward.Array
        The collection.

    Returns
    -------
    Type
        The awkward type object of the array; if the array does not
        contain metadata ``None`` is returned.

    """
    if array._meta is not None:
        return array._meta.layout.form.type
    return None


def ndim(array: Array) -> int | None:
    """Number of dimensions before reaching a numeric type or a record.

    Parameters
    ----------
    array : dask_awkward.Array
        The collection

    Returns
    -------
    int or None
        Number of dimensions as an integer, or ``None`` if the
        collection does not contain metadata.

    """
    if array._meta is not None:
        return array._meta.ndim
    return None


def is_awkward_collection(obj: Any) -> bool:
    """Check if an object is a Dask Awkward collection.

    Parameters
    ----------
    obj : Any
        The object of interest.

    Returns
    -------
    bool
        True if `obj` is an Awkward Dask collection.

    """
    return isinstance(obj, (Array, Record, Scalar))


def is_typetracer(obj: Any) -> bool:
    """Check if an object is an Awkward typetracer.

    Typetracers can be one of these categories:
    - Array
    - Record
    - UnknownScalar
    - MaybeNone
    - OneOf

    Parameters
    ----------
    obj : Any
        The object to test.

    Returns
    -------
    bool
        True if the `obj` is a typetracer like object.

    """
    # array typetracer
    if isinstance(obj, ak.Array):
        if not obj.layout.nplike.known_shape and not obj.layout.nplike.known_data:
            return True
    # record typetracer
    if isinstance(obj, ak.Record):
        if (
            not obj.layout.array.nplike.known_shape
            and not obj.layout.array.nplike.known_data
        ):
            return True
    # scalar-like typetracer
    if isinstance(obj, (UnknownScalar, MaybeNone, OneOf)):
        return True
    return False


def meta_or_identity(obj: Any) -> Any:
    """Retrieve the meta of an object or simply pass through.

    Parameters
    ----------
    obj : Any
        The object of interest.

    Returns
    -------
    Any
        If `obj` is an Awkward Dask collection it is `obj._meta`; if
        not we simply return `obj`.

    Examples
    --------
    >>> import awkward._v2 as ak
    >>> import dask_awkward as dak
    >>> from dask_awkward.core import meta_or_identity
    >>> x = ak.from_iter([[1, 2, 3], [4]])
    >>> x = dak.from_awkward(x, npartitions=2)
    >>> x
    dask.awkward<from-awkward, npartitions=2>
    >>> meta_or_identity(x)
    <Array-typetracer type='?? * var * int64'>
    >>> meta_or_identity(5)
    5
    >>> meta_or_identity("foo")
    'foo'

    """
    if is_awkward_collection(obj):
        return obj._meta
    return obj


def to_meta(objects: Sequence[Any]) -> tuple[Any, ...]:
    """In a sequence convert Dask Awkward collections to their metas.

    Parameters
    ----------
    objects : Sequence[Any]
        Sequence of objects.

    Returns
    -------
    tuple[Any, ...]
        The sequence of objects where collections have been replaced
        with their metadata.

    """
    return tuple(map(meta_or_identity, objects))


def typetracer_array(a: ak.Array | Array) -> ak.Array | None:
    """Retrieve the typetracer Array from a concrete or lazy instance.

    Parameters
    ----------
    a : ak.Array | Array
        Array of interest.

    Returns
    -------
    ak.Array | None
        Typetracer array associated with `a`.

    """
    if isinstance(a, Array):
        return a._typetracer
    else:
        return ak.Array(a.layout.typetracer.forget_length())


def compatible_partitions(*args: Array) -> bool:
    """Check if all arguments are compatibly partitioned.

    In operations where the blocks of multiple collections are used
    simultaneously, we need the collections to be equally partitioned.

    If the first argument has known divisions, other collections with
    known divisions will be tested against the first arguments
    divisions.

    Parameters
    ----------
    *args : Array
        Array collections of interest.

    Returns
    -------
    bool
        ``True`` if the collections appear to be equally partitioned.

    """
    a = args[0]

    for arg in args[1:]:
        if a.npartitions != arg.npartitions:
            return False

    if a.known_divisions:
        for arg in args[1:]:
            if arg.known_divisions:
                if a.divisions != arg.divisions:
                    return False

    return True


# class TrivialPartitionwiseOp:
#     def __init__(
#         self,
#         func: Callable,
#         *,
#         name: str | None = None,
#         **kwargs: Any,
#     ) -> None:
#         self._func = func
#         self.name = func.__name__ if name is None else name
#         self._kwargs = kwargs

#     def __call__(self, collection: Array, **kwargs: Any) -> Array:
#         # overwrite any saved kwargs in self._kwargs
#         for k, v in kwargs.items():
#             self._kwargs[k] = v
#         try:
#             new_meta = self._func(collection._meta, **kwargs)
#         except NotImplementedError:
#             new_meta = None
#         return map_partitions(
#             self._func, collection, name=self.name, meta=new_meta, **self._kwargs
#         )
