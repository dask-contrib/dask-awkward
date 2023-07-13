from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from dask_awkward.lib.core import Array


T = TypeVar("T")


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


def borrow_docstring(original: Callable[..., T]) -> Callable[..., T]:
    def wrapper(method):
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
