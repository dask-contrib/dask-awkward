from __future__ import annotations

from typing import Any, Callable

import awkward._v2 as ak
import numpy as np


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


def borrow_docstring(original: Callable) -> Callable:
    def wrapper(method):
        method.__doc__ = (
            f"Partitioned version of ak.{original.__name__}\n" f"{original.__doc__}"
        )
        return method

    return wrapper


def empty_typetracer() -> ak.Array:
    a = ak.Array([])
    return ak.Array(a.layout.typetracer.forget_length())
