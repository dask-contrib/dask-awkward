from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from dask_awkward.lib.core import Array

T = TypeVar("T")
P = ParamSpec("P")


class DaskAwkwardNotImplemented(NotImplementedError):
    NOT_SUPPORTED_MSG = """

If you would like this unsupported call to be supported by
dask-awkward please open an issue at:
https://github.com/dask-contrib/dask-awkward."""

    def __init__(self, msg: str | None = None) -> None:
        msg = f"{msg or ''}{self.NOT_SUPPORTED_MSG}"
        super().__init__(msg)


class IncompatiblePartitions(ValueError):
    def __init__(self, name: str, *args: Array) -> None:
        msg = self.divisions_msg(name, *args)
        super().__init__(msg)

    @staticmethod
    def divisions_msg(name: str, *args: Array) -> str:
        msg = f"The inputs to {name} are incompatibly partitioned\n"
        for i, arg in enumerate(args):
            msg += f"- arg{i} divisions: {arg.divisions}\n"
        return msg


class ConcretizationTypeError(TypeError):
    """
    This error occurs when a ``dask_awkward.Array`` is used in a context that requires a concrete
    value.


    There are several reasons why this error might occur:

    Examples
    --------

    - When a ``dask_awkward.Array`` is used in a conditional statement:

    >>> import dask_awkward as dak
    >>> import awkward as ak
    >>> dask_arr = dak.from_awkward(ak.Array([1]), npartitions=1)
    >>> if dask_arr > 2:
    >>>     dask_arr += 1
    Traceback (most recent call last): ...
    dask_awkward.utils.ConcretizationTypeError: A dask_awkward.Array is encountered in a computation where a concrete value is expected. If you intend to convert the dask_awkward.Array to a concrete value, use the `.compute()` method. The __bool__() method was called on dask.awkward<greater, npartitions=1>.

    - When a ``dask_awkward.Array`` is cast to a Python type:

    >>> import dask_awkward as dak
    >>> import awkward as ak
    >>> dask_arr = dak.from_awkward(ak.Array([1]), npartitions=1)
    >>> int(dask_arr)
    Traceback (most recent call last): ...
    dask_awkward.utils.ConcretizationTypeError: A dask_awkward.Array is encountered in a computation where a concrete value is expected. If you intend to convert the dask_awkward.Array to a concrete value, use the `.compute()` method. The __int__() method was called on dask.awkward<from-awkward, npartitions=1>.

    These errors can be resolved by explicitely converting the tracer to a concrete value:

    >>> import dask_awkward as dak
    >>> dask_arr = dak.from_awkward(ak.Array([1]), npartitions=1)
    >>> bool(dask_arr.compute())
    True
    """

    def __init__(self, msg: str):
        self.message = "A dask_awkward.Array is encountered in a computation where a concrete value is expected. "
        self.message += "If you intend to convert the dask_awkward.Array to a concrete value, use the `.compute()` method. "
        self.message += msg
        super().__init__(self.message)


class LazyInputsDict(Mapping):
    """Dictionary with lazy key value pairs

    Parameters
    ----------
    inputs : list[Any]
        The list of dictionary values.

    """

    def __init__(self, inputs: list[Any], **kwargs: Any) -> None:
        self.inputs = inputs
        self.kwargs = kwargs

    def __len__(self):
        return len(self.inputs)

    def __iter__(self):
        return (self[k] for k in self.keys())

    def __getitem__(self, i: tuple[int]) -> Any:
        return self.inputs[i[0]]

    def __contains__(self, k: Any) -> bool:
        if isinstance(k, tuple):
            if isinstance(k[0], int):
                return k[0] >= 0 and k[0] < len(self)
        return False

    def keys(self):
        return ((i,) for i in range(len(self.inputs)))


def borrow_docstring(original: Callable) -> Callable:
    def wrapper(method: Callable[P, T]) -> Callable[P, T]:
        method.__doc__ = (
            f"Partitioned version of ak.{original.__name__}\n" f"{original.__doc__}"
        )
        return method

    return wrapper


def hyphenize(x: str) -> str:
    """Replace underscores with hyphens.

    Returns
    -------
    str
        Resulting strings with hyphens replacing underscores.

    """
    return x.replace("_", "-")


def is_empty_slice(s: Any) -> bool:
    """Check if a slice is empty.

    Parameters
    ----------
    s : Any
        Slice of interest

    Returns
    -------
    result : bool
        True if the slice is empty

    Examples
    --------
    >>> from dask_awkward.utils import is_empty_slice
    >>> is_empty_slice(slice(1, 5, None))
    False
    >>> is_empty_slice(slice(None, None, 2))
    False
    >>> is_empty_slice(slice(None, None, None))
    True

    """
    if not isinstance(s, slice):
        return False
    if s.start is not None:
        return False
    if s.stop is not None:
        return False
    if s.step is not None:
        return False
    return True


def first(seq: Iterable[T]) -> T:
    """Get the first element of a sequence.

    Parameters
    ----------
    seq : Sequence
        The sequence of interest.

    Returns
    -------
    Any
        The first element of `seq`.

    """
    return next(iter(seq))


def second(seq: Iterable[T]) -> T:
    the_iter = iter(seq)
    next(the_iter)
    return next(the_iter)


def not_field_access_like(entry: Any) -> bool:
    """Test field-access-likeness of a getitem argument.

    Field accesses are strings or lists-of-strings, for example:

    - ``"foo"``
    - ``["foo", "bar"]``

    Parameters
    ----------
    entry : Any
        Thing to test.

    Returns
    -------
    bool
        True if ENTRY is _not_ field access like, otherwise False.

    Examples
    --------
    >>> not_field_access_like(0)
    True
    >>> not_field_access_like("foo")
    False
    >>> not_field_access_like(["foo", "bar"])
    False
    >>> not_field_access_like(["foo", 0])
    True

    """
    if isinstance(entry, str):
        return False
    if isinstance(entry, (list, tuple)) and all(isinstance(x, str) for x in entry):
        return False
    return True


def field_access_to_front(seq: Sequence[Any]) -> tuple[tuple[Any, ...], int]:
    """Move field access to the front of a sequence.

    We have multiargument getitem calls we want to bring the field
    access calls to the front. For example

    >>> a[0, "foo"]

    Is the same as

    >>> a["foo", 0]

    But the latter starts with something that is trivially
    map-partitionable. This function helps us write out the logic for
    getitem calls.

    Parameters
    ----------
    seq : Sequence[Any]
        Sequence to reorder.

    Returns
    -------
    tuple[Any, ...]
        Reordered sequence with field accesses brought to the front.
    int
        Total number of field accesses.

    Examples
    --------
    >>> where = [0, ["foo", "bar"], "x"]
    >>> new, n = field_access_to_front(where)
    >>> new
    [["foo", "bar"], "x", 0]
    >>> n
    2

    """
    new_args = tuple(sorted(seq, key=not_field_access_like))
    n_field_accesses = sum(map(lambda x: not not_field_access_like(x), new_args))
    return new_args, n_field_accesses
