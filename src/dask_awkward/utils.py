from __future__ import annotations

from typing import Any

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
    int
        Which partition in the collection.
    int
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
    bool
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
