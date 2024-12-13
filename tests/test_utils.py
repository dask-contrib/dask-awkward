from __future__ import annotations

import pytest

from dask_awkward.lib.utils import typetracer_nochecks
from dask_awkward.utils import (
    LazyInputsDict,
    field_access_to_front,
    hyphenize,
    is_empty_slice,
)


def test_is_empty_slice() -> None:
    assert is_empty_slice(slice(None, None, None))
    assert not is_empty_slice(slice(0, 10, 2))
    assert not is_empty_slice(slice(0, 10, None))
    assert not is_empty_slice(slice(None, None, -1))
    assert not is_empty_slice(slice(2, None, -1))
    assert not is_empty_slice(slice(None, 5, -1))


def test_lazyfilesdict() -> None:
    inputs = ["f1.json", "f2.json"]
    lfd = LazyInputsDict(inputs)
    assert len(lfd) == 2
    assert (0,) in lfd
    assert (1,) in lfd
    assert (2,) not in lfd
    assert lfd[(0,)] == "f1.json"
    assert lfd[(1,)] == "f2.json"
    assert list(lfd) == inputs
    assert (5,) not in lfd
    assert "a" not in lfd


def test_hyphenize() -> None:
    assert hyphenize("with_name") == "with-name"
    assert hyphenize("with_a_name") == "with-a-name"
    assert hyphenize("ok") == "ok"


@pytest.mark.parametrize(
    "pairs",
    [
        (
            (1, 3, 2, "z", "a"),
            ("z", "a", 1, 3, 2),
            2,
        ),
        (
            ("a", 1, 2, ["1", "2"]),
            ("a", ["1", "2"], 1, 2),
            2,
        ),
        (
            (0, ["a", "b", "c"]),
            (["a", "b", "c"], 0),
            1,
        ),
        (
            ("hello", "abc"),
            ("hello", "abc"),
            2,
        ),
        (
            (1, 2, slice(None, None, 2), 3),
            (1, 2, slice(None, None, 2), 3),
            0,
        ),
        (
            (0, ["a", 0], ["a", "b"]),
            (["a", "b"], 0, ["a", 0]),
            1,
        ),
    ],
)
def test_field_access_to_front(pairs):
    res = field_access_to_front(pairs[0])
    assert res[0] == pairs[1]
    assert res[1] == pairs[2]


def test_nocheck_context():
    from awkward._nplikes.typetracer import TypeTracerArray

    old = getattr(TypeTracerArray, "runtime_typechecks", True)
    with typetracer_nochecks():
        assert not TypeTracerArray.runtime_typechecks
    new = getattr(TypeTracerArray, "runtime_typechecks", True)
    assert old == new
