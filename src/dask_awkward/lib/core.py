from __future__ import annotations

import keyword
import operator
import sys
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import cached_property, partial
from numbers import Number
from typing import TYPE_CHECKING, Any, TypeVar

import awkward as ak
import dask.config
import numpy as np
from awkward._nplikes.typetracer import (
    MaybeNone,
    OneOf,
    TypeTracerArray,
    is_unknown_scalar,
)
from awkward.highlevel import _dir_pattern
from dask.base import DaskMethodsMixin, dont_optimize, is_dask_collection, tokenize
from dask.blockwise import BlockwiseDep
from dask.blockwise import blockwise as dask_blockwise
from dask.context import globalmethod
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import IndexCallable, funcname, key_split
from numpy.lib.mixins import NDArrayOperatorsMixin
from tlz import first

from dask_awkward.lib.optimize import all_optimizations
from dask_awkward.typing import AwkwardDaskCollection
from dask_awkward.utils import (
    DaskAwkwardNotImplemented,
    IncompatiblePartitions,
    hyphenize,
    is_empty_slice,
)

if TYPE_CHECKING:
    from awkward.contents.content import Content
    from awkward.forms.form import Form
    from awkward.types.arraytype import ArrayType
    from awkward.types.type import Type
    from dask.array.core import Array as DaskArray
    from dask.bag.core import Bag as DaskBag
    from dask.blockwise import Blockwise
    from numpy.typing import DTypeLike


T = TypeVar("T")


def _finalize_array(results: Sequence[Any]) -> Any:
    # special cases for length 1 results
    if len(results) == 1:
        if isinstance(results[0], (int, ak.Array)):
            return results[0]

    # a sequence of arrays that need to be concatenated.
    elif any(isinstance(r, ak.Array) for r in results):
        return ak.concatenate(results)

    # sometimes we just check the length of partitions so all results
    # will be integers, just make an array out of that.
    elif isinstance(results, tuple) and all(
        isinstance(r, (int, np.integer)) for r in results
    ):
        return ak.Array(list(results))

    # sometimes all partition results will be None (some write-to-disk
    # operations)
    elif all(r is None for r in results):
        return None

    else:
        msg = (
            "Unexpected results of a computation.\n "
            f"results: {results}"
            f"type of first result: {type(results[0])}"
        )
        raise RuntimeError(msg)


class Scalar(DaskMethodsMixin):
    """Single partition Dask collection representing a lazy Scalar.

    The class constructor is not intended for users. Instances of this
    class will be results from awkward operations.

    Within dask-awkward the ``new_scalar_object`` and
    ``new_known_scalar`` factory functions are used for creating new
    instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        meta: Any,
        known_value: Any | None = None,
    ) -> None:
        if not isinstance(dsk, HighLevelGraph):
            dsk = HighLevelGraph.from_collections(name, dsk, dependencies=())  # type: ignore
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._meta: Any = self._check_meta(meta)
        self._known_value: Any | None = known_value

    def __dask_graph__(self) -> HighLevelGraph:
        return self._dask

    def __dask_keys__(self) -> list[Hashable]:
        return [self.key]

    def __dask_layers__(self) -> tuple[str, ...]:
        return (self.name,)

    def __dask_tokenize__(self) -> Hashable:
        return self.name

    __dask_optimize__ = globalmethod(
        all_optimizations, key="awkward_scalar_optimize", falsey=dont_optimize
    )

    __dask_scheduler__ = staticmethod(threaded_get)

    def __dask_postcompute__(self) -> tuple[Callable, tuple]:
        return first, ()

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
            raise ValueError("rename= unsupported in dask-awkward")
        return type(self)(dsk, name, self._meta, self.known_value)

    def __reduce__(self):
        return (Scalar, (self.dask, self.name, self._meta, self.known_value))

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def name(self) -> str:
        return self._name

    @property
    def key(self) -> Hashable:
        return (self._name, 0)

    def _check_meta(self, m: Any) -> Any | None:
        if isinstance(m, (MaybeNone, OneOf)) or is_unknown_scalar(m):
            return m
        raise TypeError(f"meta must be a typetracer object, not a {type(m)}")

    @property
    def dtype(self) -> np.dtype | None:
        try:
            if self._meta is not None:
                return self._meta.dtype
        except AttributeError:
            pass
        return None

    @property
    def npartitions(self) -> int:
        """Scalar and Records are unpartitioned by definition."""
        return 1

    @property
    def fields(self) -> list[str]:
        return []

    @property
    def layout(self) -> Any:
        raise TypeError("Scalars do not have a layout.")

    @property
    def divisions(self) -> tuple[None, None]:
        """Scalar and Records do not have divisions by definition."""
        return (None, None)

    @staticmethod
    def from_known(s: Any, dtype: DTypeLike | None = None) -> Scalar:
        """Create a scalar from a known value."""
        return new_known_scalar(s, dtype=dtype)

    def __repr__(self) -> str:  # pragma: no cover
        return self.__str__()

    def __str__(self) -> str:
        dt = self.dtype or "Unknown"
        if self.known_value is not None:
            return (
                f"dask.awkward<{key_split(self.name)}, "
                "type=Scalar, "
                f"dtype={dt}, "
                f"known_value={self.known_value}>"
            )
        return f"dask.awkward<{key_split(self.name)}, type=Scalar, dtype={dt}>"

    def __getitem__(self, _: Any) -> Any:
        raise RuntimeError("Scalars do not support __getitem__")

    def __getattr__(self, _: str) -> Any:
        raise RuntimeError("Scalars do not support __getattr__")

    @property
    def known_value(self) -> Any | None:
        return self._known_value

    def to_delayed(self, optimize_graph: bool = True) -> Delayed:
        """Convert Scalar collection into a Delayed collection.

        Parameters
        ----------
        optimize_graph : bool
            If ``True`` optimize the existing task graph before
            converting to delayed.

        Returns
        -------
        Delayed
            Resulting Delayed collection object.

        """
        dsk = self.__dask_graph__()
        layer = self.__dask_layers__()[0]
        if optimize_graph:
            layer = f"delayed-{self.name}"
            dsk = self.__dask_optimize__(dsk, self.__dask_keys__())
            dsk = HighLevelGraph.from_collections(layer, dsk, dependencies=())
        return Delayed(self.key, dsk, layer=layer)


def new_scalar_object(dsk: HighLevelGraph, name: str, *, meta: Any) -> Scalar:
    """Instantiate a new scalar collection.

    Parameters
    ----------
    dsk : HighLevelGraph
        Dask highlevel task graph.
    name : str
        Name for the collection.
    meta : Any
        Awkward typetracer metadata.

    Returns
    -------
    Scalar
        Resulting collection.

    """
    if meta is None:
        meta = TypeTracerArray._new(dtype=np.dtype(None), shape=())
    return Scalar(dsk, name, meta, known_value=None)


def new_known_scalar(
    s: Any,
    dtype: DTypeLike | None = None,
    label: str | None = None,
) -> Scalar:
    """Instantiate a Scalar with a known value.

    Parameters
    ----------
    s : Any
        Python object.
    dtype : DTypeLike, optional
        NumPy dtype associated with the object, if undefined the dtype
        will be assigned via NumPy's interpretation of the object.
    label : str, optional
        Label for the task graph; if undefined "known-scalar" will be
        used.

    Returns
    -------
    Scalar
        Resulting collection.

    Examples
    --------
    >>> from dask_awkward.core import new_known_scalar
    >>> a = new_known_scalar(5, label="five")
    >>> a
    dask.awkward<five, type=Scalar, dtype=int64, known_value=5>
    >>> a.compute()
    5

    """
    label = label or "known-scalar"
    name = f"{label}-{tokenize(s)}"
    if dtype is None:
        if isinstance(s, (int, np.integer)):
            dtype = np.dtype(int)
        elif isinstance(s, (float, np.floating)):
            dtype = np.dtype(float)
        else:
            dtype = np.dtype(type(s))
    else:
        dtype = np.dtype(dtype)
    llg = {(name, 0): s}
    hlg = HighLevelGraph.from_collections(name, llg, dependencies=())
    return Scalar(
        hlg, name, meta=TypeTracerArray._new(dtype=dtype, shape=()), known_value=s
    )


class Record(Scalar):
    """Single partition Dask collection representing a lazy Awkward Record.

    The class constructor is not intended for users. Instances of this
    class will be results from awkward operations.

    Within dask-awkward the ``new_record_object`` factory function is
    used for creating new instances.

    """

    def __init__(self, dsk: HighLevelGraph, name: str, meta: Any | None = None) -> None:
        super().__init__(dsk, name, meta)

    def _check_meta(self, m: Any | None) -> Any | None:
        if not isinstance(m, ak.Record):
            raise TypeError(f"meta must be a Record typetracer object, not a {type(m)}")
        return m

    def __getitem__(self, where: str) -> AwkwardDaskCollection:
        token = tokenize(self, where)
        new_name = f"{where}-{token}"
        new_meta = self._meta[where]

        # first check for array type return
        if isinstance(new_meta, ak.Array):
            graphlayer = {(new_name, 0): (operator.getitem, self.key, where)}
            hlg = HighLevelGraph.from_collections(
                new_name,
                graphlayer,
                dependencies=[self],
            )
            return new_array_object(hlg, new_name, meta=new_meta, npartitions=1)

        # then check for scalar (or record) type
        graphlayer = {(new_name, 0): (operator.getitem, self.key, where)}
        hlg = HighLevelGraph.from_collections(
            new_name,
            graphlayer,
            dependencies=[self],
        )
        if isinstance(new_meta, ak.Record):
            return new_record_object(hlg, new_name, meta=new_meta)
        else:
            return new_scalar_object(hlg, new_name, meta=new_meta)

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
    def fields(self) -> list[str]:
        if self._meta is None:
            raise TypeError("metadata is missing; cannot determine fields.")
        return ak.fields(self._meta)

    @property
    def layout(self) -> Any:
        return self._meta.layout

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
    """Instantiate a new record collection.

    Parameters
    ----------
    dsk : HighLevelGraph
        Dask high level graph.
    name : str
        Name for the collection.
    meta : Any
        Awkward typetracer as metadata

    Returns
    -------
    Record
        Resulting collection.

    """
    return Record(dsk, name, meta)


def _outer_int_getitem_fn(x: Any, gikey: str) -> Any:
    return x[gikey]


class Array(DaskMethodsMixin, NDArrayOperatorsMixin):
    """Partitioned, lazy, and parallel Awkward Array Dask collection.

    The class constructor is not intended for users. Instead use
    factory functions like :py:func:`~dask_awkward.from_parquet`,
    :py:func:`~dask_awkward.from_json`, etc.

    Within dask-awkward the ``new_array_object`` factory function is
    used for creating new instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        meta: ak.Array,
        divisions: tuple[int | None, ...],
    ) -> None:
        self._dask: HighLevelGraph = dsk
        if hasattr(dsk, "layers"):
            # i.e., NOT matrializes/persisted state
            # output typetracer
            list(dsk.layers.values())[-1]._meta = meta  # type: ignore
        self._name: str = name
        self._divisions: tuple[int | None, ...] = divisions
        self._meta: ak.Array = meta

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> list[Hashable]:
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
        all_optimizations, key="awkward_array_optimize", falsey=dont_optimize
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
            raise ValueError("rename= unsupported in dask-awkward")
        return Array(dsk, name, self._meta, divisions=self.divisions)

    def reset_meta(self) -> None:
        """Assign an empty typetracer array as the collection metadata."""
        self._meta = empty_typetracer()

    def __len__(self) -> int:
        if not self.known_divisions:
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

    def __iter__(self):
        raise NotImplementedError(
            "Iteration over a Dask Awkward collection is not supported.\n"
            "A suggested alternative: define a function which iterates over\n"
            "an awkward array and use that function with map_partitions."
        )

    def _ipython_display_(self):
        return self._meta._ipython_display_()

    def _ipython_canary_method_should_not_exist_(self):
        return self._meta._ipython_canary_method_should_not_exist_()

    def _repr_mimebundle_(self):
        return self._meta._repr_mimebundle_()

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
        """High level task graph associated with the collection."""
        return self._dask

    @property
    def keys(self) -> list[Hashable]:
        """Task graph keys."""
        return self.__dask_keys__()

    @property
    def name(self) -> str:
        """Name of the collection."""
        return self._name

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        assert self._meta is not None
        return self._meta.ndim

    @property
    def divisions(self) -> tuple[int | None, ...]:
        """Location of the collections partition boundaries."""
        return self._divisions

    @property
    def known_divisions(self) -> bool:
        """True of the divisions are known (absence of ``None`` in the tuple)."""
        return len(self.divisions) > 0 and None not in self.divisions

    @property
    def npartitions(self) -> int:
        """Total number of partitions."""
        return len(self.divisions) - 1

    @property
    def layout(self) -> Content:
        """awkward Array layout associated with the eventual computed result."""
        if self._meta is not None:
            return self._meta.layout
        raise ValueError("This collection's meta is None; unknown layout.")

    @property
    def behavior(self) -> dict:
        """awkward Array behavior dictionary."""
        if self._meta is not None:
            return self._meta.behavior
        raise ValueError(
            "This collection's meta is None; no behavior property available."
        )

    @property
    def fields(self) -> list[str]:
        """Record field names (if any)."""
        return ak.fields(self._meta)

    @property
    def form(self) -> Form:
        """awkward Array form associated with the eventual computed result."""
        if self._meta is not None:
            return self._meta.layout.form
        raise ValueError("This collection's meta is None; unknown form.")

    @property
    def type(self) -> ArrayType:
        """awkward Array type associated with the eventual computed result."""
        t = ak.types.ArrayType(
            self._meta._layout.form.type_from_behavior(self._meta._behavior),
            0,
        )
        t._length = "??"
        return t

    @cached_property
    def keys_array(self) -> np.ndarray:
        """NumPy array of task graph keys."""
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
            dependencies=[self],
        )

        # if a single partition was requested we trivially know the new divisions.
        if len(raw) == 1 and isinstance(raw[0], int) and self.known_divisions:
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
        >>> import awkward as ak
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
        label: str | None = None,
    ) -> Any:
        if meta is None and self._meta is not None:
            if isinstance(where, tuple):
                metad = to_meta(where)
                meta = self._meta[metad]
            else:
                m = to_meta([where])[0]
                meta = self._meta[m]
        return self.map_partitions(
            operator.getitem,
            where,
            meta=meta,
            output_divisions=1,
            label=label,
        )

    def _getitem_outer_bool_or_int_lazy_array(
        self, where: Array | tuple[Any, ...]
    ) -> Any:
        ba = where if isinstance(where, Array) else where[0]
        if not compatible_partitions(self, ba):
            raise IncompatiblePartitions("getitem", self, ba)

        new_meta: Any | None = None
        if self._meta is not None:
            if isinstance(where, tuple):
                raise DaskAwkwardNotImplemented(
                    "tuple style input boolean/int selection is not supported."
                )
            elif isinstance(where, Array):
                new_meta = self._meta[where._meta]
                return self.map_partitions(
                    operator.getitem,
                    where,
                    meta=new_meta,
                )

    def _getitem_outer_str_or_list(
        self,
        where: str | list | tuple[Any, ...],
        label: str | None = None,
    ) -> Any:
        new_meta: Any | None = None
        if self._meta is not None:
            if isinstance(where, tuple):
                if not isinstance(where[0], (str, list)):
                    raise TypeError("Expected where[0] to be a string or list")
                metad = to_meta(where)
                new_meta = self._meta[metad]
            elif isinstance(where, (str, list)):
                new_meta = self._meta[where]
        return self._getitem_trivial_map_partitions(where, meta=new_meta, label=label)

    def _getitem_outer_int(self, where: int | tuple[Any, ...]) -> Any:
        if where == 0 or (isinstance(where, tuple) and where[0] == 0):
            pass
        elif not self.known_divisions:
            self.eager_compute_divisions()

        new_meta: Any | None = None
        # multiple objects passed to getitem. collections passed in
        # the tuple of objects have not been tested!
        if isinstance(where, tuple):
            if not isinstance(where[0], int):
                raise TypeError("Expected where[0] to be and integer.")
            if where[0] == 0:
                pidx, outer_where = 0, 0
            else:
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
            if where == 0:
                pidx, where = 0, 0
            else:
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
        if not isinstance(new_meta, (ak.Record, TypeTracerArray, OneOf, MaybeNone)):
            raise DaskAwkwardNotImplemented("Key type not supported for this array.")

        token = tokenize(partition, where)
        name = f"getitem-{token}"
        dsk = {
            (name, 0): (
                _outer_int_getitem_fn,
                partition.__dask_keys__()[0],
                where,
            )
        }
        hlg = HighLevelGraph.from_collections(
            name,
            dsk,
            dependencies=[partition],
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
            if issubclass(dtype, (np.bool_, bool, np.int64, np.int32, int)):
                return self._getitem_outer_bool_or_int_lazy_array(where)

        raise DaskAwkwardNotImplemented(
            f"Array.__getitem__ doesn't support multi object: {where}"
        )

    def _getitem_single(self, where: Any) -> Array:
        # a single string
        if isinstance(where, str):
            return self._getitem_outer_str_or_list(where, label=where)

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
            if issubclass(dtype, (np.bool_, bool, np.int64, np.int32, int)):
                return self._getitem_outer_bool_or_int_lazy_array(where)

        # an empty slice
        elif is_empty_slice(where):
            return self

        # a single ellipsis
        elif where is Ellipsis:
            return self

        elif self.npartitions == 1:
            return self.map_partitions(operator.getitem, where)

        raise DaskAwkwardNotImplemented(f"__getitem__ doesn't support where={where}.")

    def __getitem__(self, where: Any) -> AwkwardDaskCollection:
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

    def _call_behavior_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._meta, method_name):
            return self.map_partitions(
                _BehaviorMethodFn(method_name, **kwargs),
                *args,
                label=hyphenize(method_name),
            )
        raise AttributeError(
            f"Method {method_name} is not available to this collection."
        )

    def _call_behavior_property(self, property_name: str) -> Any:
        if hasattr(self._meta, property_name):
            return self.map_partitions(
                _BehaviorPropertyFn(property_name),
                label=hyphenize(property_name),
            )
        raise AttributeError(
            f"Property {property_name} is not available to this collection."
        )

    def _maybe_behavior_method(self, attr: str) -> bool:
        try:
            res = getattr(self._meta, attr)
            return callable(res)
        except AttributeError:
            return False

    def _maybe_behavior_property(self, attr: str) -> bool:
        try:
            res = getattr(self._meta, attr)
            return not callable(res)
        except AttributeError:
            return False

    def __getattr__(self, attr: str) -> Any:
        if attr not in (self.fields or []):
            # check for possible behavior method
            if self._maybe_behavior_method(attr):

                def wrapper(*args, **kwargs):
                    return self._call_behavior_method(attr, *args, **kwargs)

                return wrapper
            # check for possible behavior property
            elif self._maybe_behavior_property(attr):
                return self._call_behavior_property(attr)

            raise AttributeError(f"{attr} not in fields.")
        try:
            # at this point attr is either a field or we'll have to
            # raise an exception.
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
        """Force a compute of the divisions."""
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
                inputs_meta.append(typetracer_array(inp))
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

    def __array__(self, *args, **kwargs):
        raise NotImplementedError

    def to_delayed(self, optimize_graph: bool = True) -> list[Delayed]:
        """Convert the collection to a list of delayed objects.

        One dask.delayed.Delayed object per partition.

        Parameters
        ----------
        optimize_graph : bool
            If True the task graph associated with the collection will
            be optimized before conversion to the list of Delayed
            objects.

        See Also
        --------
        dask_awkward.to_delayed

        Returns
        -------
        list[Delayed]
            List of delayed objects (one per partition).

        """
        from dask_awkward.lib.io.io import to_delayed

        return to_delayed(self, optimize_graph=optimize_graph)

    def to_dask_array(self, optimize_graph: bool = True) -> DaskArray:
        from dask_awkward.lib.io.io import to_dask_array

        return to_dask_array(self, optimize_graph=optimize_graph)

    def to_parquet(
        self,
        path: str,
        storage_options: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        from dask_awkward.lib.io.parquet import to_parquet

        return to_parquet(self, path, storage_options=storage_options, **kwargs)

    def to_dask_bag(self) -> DaskBag:
        from dask_awkward.lib.io.io import to_dask_bag

        return to_dask_bag(self)


def compute_typetracer(dsk: HighLevelGraph, name: str) -> ak.Array:
    if dask.config.get("awkward.compute-unknown-meta"):
        key = (name, 0)
        return typetracer_array(Delayed(key, dsk.cull({key}), layer=name).compute())
    return empty_typetracer()


def new_array_object(
    dsk: HighLevelGraph,
    name: str,
    *,
    meta: ak.Array | None = None,
    behavior: dict | None = None,
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
        Collection metadata; this is an awkward-array type tracer. If
        `meta` is ``None``, the first partition of the task graph
        (`dsk`) will be computed by default to determine the
        typetracer for the new Array. If the configuration option
        ``awkward.compute-unknown-meta`` is set to ``False``,
        undefined `meta` will be assigned an empty typetracer.
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
    if divisions is None:
        if npartitions is not None:
            divs: tuple[int | None, ...] = (None,) * (npartitions + 1)
        else:
            raise ValueError("One of either divisions or npartitions must be defined.")
    else:
        if npartitions is not None:
            raise ValueError(
                "Only one of either divisions or npartitions can be defined."
            )
        divs = divisions

    if meta is None:
        actual_meta = compute_typetracer(dsk, name)
    else:
        if not isinstance(meta, ak.Array):
            raise TypeError(
                f"meta must be an instance of an Awkward Array, not {type(meta)}."
            )
        actual_meta = meta

    if behavior is not None:
        actual_meta.behavior = behavior

    return Array(dsk, name, actual_meta, divs)


def partitionwise_layer(
    func: Callable,
    name: str,
    *args: Any,
    opt_touch_all: bool = False,
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
        elif isinstance(arg, BlockwiseDep):
            if len(arg.numblocks) == 1:
                pairs.extend([arg, "i"])
            elif len(arg.numblocks) == 2:
                pairs.extend([arg, "ij"])
        elif is_dask_collection(arg):
            raise DaskAwkwardNotImplemented(
                "Use of Array with other Dask collections is currently unsupported."
            )
        else:
            pairs.extend([arg, None])
    layer = dask_blockwise(
        func,
        name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )
    if opt_touch_all:
        layer._opt_touch_all = True
    return layer


def map_partitions(
    fn: Callable,
    *args: Any,
    label: str | None = None,
    token: str | None = None,
    meta: Any | None = None,
    output_divisions: int | None = None,
    opt_touch_all: bool = False,
    **kwargs: Any,
) -> Array:
    """Map a callable across all partitions of any number of collections.

    Parameters
    ----------
    fn : Callable
        Function to apply on all partitions.
    *args : Collections and function arguments
        Arguments passed to the function. Partitioned arguments (i.e.
        Dask collections) will have `fn` applied to each partition.
        Array collection arguments they must be compatibly
        partitioned.
    label : str, optional
        Label for the Dask graph layer; if left to ``None`` (default),
        the name of the function will be used.
    token : str, optional
        Provide an already defined token. If ``None`` a new token will
        be generated.
    meta : Any, optional
        Metadata (typetracer) array for the result (if known). If
        unknown, `fn` will be applied to the metadata of the `args`;
        if that call fails, the first partition of the new collection
        will be used to compute the new metadata **if** the
        ``awkward.compute-known-meta`` configuration setting is
        ``True``. If the configuration setting is ``False``, an empty
        typetracer will be assigned as the metadata.
    output_divisions : int, optional
        If ``None`` (the default), the divisions of the output will be
        assumed unknown. If defined, the output divisions will be
        multiplied by a factor of `output_divisions`. A value of 1
        means constant divisions (e.g. a string based slice). Any
        value greater than 1 means the divisions were expanded by some
        operation. This argument is mainly for internal library
        function implementations.
    opt_touch_all : bool
        Touch all layers in this graph during typetracer based
        optimization.
    **kwargs : Any
        Additional keyword arguments passed to the `fn`.

    Returns
    -------
    dask_awkward.Array
        The new collection.

    Examples
    --------
    >>> import dask_awkward as dak
    >>> a = [[1, 2, 3], [4]]
    >>> b = [[5, 6, 7], [8]]
    >>> c = dak.from_lists([a, b])
    >>> c
    dask.awkward<from-lists, npartitions=2>
    >>> c.compute()
    <Array [[1, 2, 3], [4], [5, 6, 7], [8]] type='4 * var * int64'>
    >>> c2 = dak.map_partitions(np.add, c, c)
    >>> c2
    dask.awkward<add, npartitions=2>
    >>> c2.compute()
    <Array [[2, 4, 6], [8], [10, 12, 14], [16]] type='4 * var * int64'>

    Multiplying `c` (a Dask collection) with `a` (a regular Python
    list object) will multiply each partition of `c` by `a`:

    >>> d = dak.map_partitions(np.multiply, c, a)
    dask.awkward<multiply, npartitions=2>
    >>> d.compute()
    <Array [[1, 4, 9], [16], [5, 12, 21], [32]] type='4 * var * int64'>

    This is effectively the same as `d = c * a`

    """
    token = token or tokenize(fn, *args, meta, **kwargs)
    label = label or funcname(fn)
    name = f"{label}-{token}"
    lay = partitionwise_layer(
        fn,
        name,
        *args,
        opt_touch_all=opt_touch_all,
        **kwargs,
    )
    deps = [a for a in args if is_dask_collection(a)] + [
        v for _, v in kwargs.items() if is_dask_collection(v)
    ]

    if meta is None:
        metas = to_meta(args)
        try:
            meta = fn(*metas, **kwargs)
        except Exception:
            if dask.config.get("awkward.compute-unknown-meta"):
                extras = (
                    f"function call: {fn}\n"
                    f"metadata: {metas}\n"
                    f"kwargs: {kwargs}\n"
                )
                warnings.warn(
                    "metadata could not be determined; "
                    "a compute on the first partition will occur.\n"
                    f"{extras}",
                    UserWarning,
                )
            pass

    hlg = HighLevelGraph.from_collections(
        name,
        lay,
        dependencies=deps,
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


def _from_iter(obj):
    """Try to run ak.from_iter, but have fallbacks.

    This function first tries to call ak.form_iter on the input (which
    should be some iterable). We expect a list of Scalar typetracers
    to fail, so if the call fails due to ValueError or TypeError then
    we manually do some typetracer operations to return the proper
    representation of the input iterable-of-typetracers.

    """
    try:
        return ak.from_iter(obj)
    except (ValueError, TypeError):
        first_obj = obj[0]

        if isinstance(first_obj, MaybeNone):
            first_obj = first_obj.content

        return ak.Array(
            ak.Array(first_obj)
            .layout.form.length_one_array()
            .layout.to_typetracer(forget_length=True)
        )


def total_reduction_to_scalar(
    *,
    label: str,
    array: Array,
    meta: Any,
    chunked_fn: Callable,
    comb_fn: Callable | None = None,
    agg_fn: Callable | None = None,
    token: str | None = None,
    dtype: Any | None = None,
    split_every: int | bool | None = None,
    chunked_kwargs: dict[str, Any] | None = None,
    comb_kwargs: dict[str, Any] | None = None,
    agg_kwargs: dict[str, Any] | None = None,
) -> Scalar:
    from dask.layers import DataFrameTreeReduction

    chunked_kwargs = chunked_kwargs or {}
    token = token or tokenize(
        array,
        chunked_fn,
        comb_fn,
        agg_fn,
        label,
        dtype,
        split_every,
        chunked_kwargs,
        comb_kwargs,
        agg_kwargs,
    )
    name_comb = f"{label}-combine-{token}"
    name_agg = f"{label}-agg-{token}"

    comb_kwargs = comb_kwargs or chunked_kwargs
    agg_kwargs = agg_kwargs or comb_kwargs

    comb_fn = comb_fn or chunked_fn
    agg_fn = agg_fn or comb_fn

    chunked_fn = partial(chunked_fn, **chunked_kwargs)
    comb_fn = partial(comb_fn, **comb_kwargs)
    agg_fn = partial(agg_fn, **agg_kwargs)

    chunked_result = map_partitions(
        chunked_fn,
        array,
        meta=empty_typetracer(),
    )

    if split_every is None:
        split_every = 8
    elif split_every is False:
        split_every = sys.maxsize
    else:
        pass

    dftr = DataFrameTreeReduction(
        name=name_agg,
        name_input=chunked_result.name,
        npartitions_input=chunked_result.npartitions,
        concat_func=_from_iter,
        tree_node_func=comb_fn,
        finalize_func=agg_fn,
        split_every=split_every,
        tree_node_name=name_comb,
    )

    graph = HighLevelGraph.from_collections(
        name_agg, dftr, dependencies=(chunked_result,)
    )
    return new_scalar_object(graph, name_agg, meta=meta)


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
    num = map_partitions(ak.num, array, axis=0, meta=empty_typetracer())

    # if only 1 partition things are simple
    if array.npartitions == 1:
        return (0, num.compute())

    # if more than 1 partition cumulative sum required
    cs = list(np.cumsum(num.compute()))
    return tuple([0, *cs])


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


def ndim(array: Array) -> int:
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
    return array.ndim


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
    - TypeTracerArray
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
    # array/record typetracer
    if isinstance(obj, (ak.Array, ak.Record)):
        backend = obj.layout.backend

        if not backend.nplike.known_data:
            return True
    # scalar-like typetracer
    elif is_unknown_scalar(obj) or isinstance(obj, (MaybeNone, OneOf)):
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
    >>> import awkward as ak
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


def typetracer_array(a: ak.Array | Array) -> ak.Array:
    """Retrieve the typetracer Array from a concrete or lazy instance.

    Parameters
    ----------
    a : ak.Array | Array
        Array of interest.

    Returns
    -------
    ak.Array
        Typetracer array associated with `a`.

    """
    if isinstance(a, Array):
        return a._meta
    elif isinstance(a, ak.Array):
        return ak.Array(a.layout.to_typetracer(forget_length=True))
    else:
        msg = (
            "`a` should be an awkward array or a Dask awkward collection.\n"
            f"Got type {type(a)}"
        )
        raise TypeError(msg)


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


def compatible_divisions(*args: Array) -> bool:
    if not all(a.known_divisions for a in args):
        return False
    for arg in args[1:]:
        if arg.divisions != args[0].divisions:
            return False
    return True


def empty_typetracer() -> ak.Array:
    """Instantiate a typetracer array with unknown length.

    Returns
    -------
    ak.Array
        Length-less typetracer array (content-less array).

    """
    a = ak.Array([])
    return ak.Array(a.layout.to_typetracer(forget_length=True))


class _BehaviorMethodFn:
    def __init__(self, attr: str, **kwargs: Any) -> None:
        self.attr = attr
        self.kwargs = kwargs

    def __call__(self, coll: ak.Array, *args: Any) -> ak.Array:
        return getattr(coll, self.attr)(*args, **self.kwargs)


class _BehaviorPropertyFn:
    def __init__(self, attr: str) -> None:
        self.attr = attr

    def __call__(self, coll: ak.Array) -> ak.Array:
        return getattr(coll, self.attr)


def normalize_single_outer_inner_index(
    divisions: tuple[int, ...], index: int
) -> tuple[int, int]:
    """Determine partition index and inner index for some divisions.

    Parameters
    ----------
    divisions : tuple[int, ...]
        The divisions of a Dask awkward collection.
    index : int
        The overall index (for the complete collection).

    Returns
    -------
    partition_index : int
        Which partition in the collection.
    new_index : int
        Which inner index in the determined partition.

    Examples
    --------
    >>> from dask_awkward.utils import normalize_single_outer_inner_index
    >>> divisions = (0, 3, 6, 9)
    >>> normalize_single_outer_inner_index(divisions, 0)
    (0, 0)
    >>> normalize_single_outer_inner_index(divisions, 5)
    (1, 2)
    >>> normalize_single_outer_inner_index(divisions, 8)
    (2, 2)

    """
    if index < 0:
        index = divisions[-1] + index
    if len(divisions) == 2:
        return (0, index)
    partition_index = int(np.digitize(index, divisions)) - 1
    new_index = index - divisions[partition_index]
    return (partition_index, new_index)


def typetracer_from_form(form: Form) -> ak.Array:
    """Create a typetracer Array from an awkward form.

    Parameters
    ----------
    form : awkward.form.Form
        Form that the resulting Array will have.

    Returns
    -------
    awkward.Array
        Resulting highlevel typetracer Array

    """
    layout = form.length_zero_array(highlevel=False)
    return ak.Array(layout.to_typetracer(forget_length=True))
