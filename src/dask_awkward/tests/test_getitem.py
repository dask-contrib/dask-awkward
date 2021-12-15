from __future__ import annotations

import pytest

from ..utils import assert_eq
from .helpers import (  # noqa: F401
    line_delim_records_file,
    load_records_eager,
    load_records_lazy,
)


def test_multi_string(line_delim_records_file) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)
    caa = load_records_eager(line_delim_records_file)
    assert_eq(
        daa["analysis"][["x1", "y2"]],
        caa["analysis"][["x1", "y2"]],
    )


def test_single_string(line_delim_records_file) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)
    caa = load_records_eager(line_delim_records_file)
    assert_eq(daa["analysis"], caa["analysis"])


def test_list_with_ints_raise(line_delim_records_file) -> None:  # noqa: F811
    daa = load_records_lazy(line_delim_records_file)
    with pytest.raises(NotImplementedError, match="Lists containing integers"):
        assert daa[[1, 2]]
