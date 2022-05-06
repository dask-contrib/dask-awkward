from __future__ import annotations

from dask_awkward.utils import (
    LazyInputsDict,
    hyphenize,
    is_empty_slice,
    normalize_single_outer_inner_index,
)


def test_normalize_single_outer_inner_index() -> None:
    divisions = (0, 12, 14, 20, 23, 24)
    indices = [0, 1, 2, 8, 12, 13, 14, 15, 17, 20, 21, 22]
    results = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 8),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    for i, r in zip(indices, results):
        res = normalize_single_outer_inner_index(divisions, i)
        assert r == res

    divisions = (0, 12)  # type: ignore
    indices = [0, 2, 3, 6, 8, 11]
    results = [
        (0, 0),
        (0, 2),
        (0, 3),
        (0, 6),
        (0, 8),
        (0, 11),
    ]
    for i, r in zip(indices, results):
        res = normalize_single_outer_inner_index(divisions, i)
        assert r == res


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
    assert not (5,) in lfd
    assert "a" not in lfd


def test_hyphenize() -> None:
    assert hyphenize("with_name") == "with-name"
    assert hyphenize("with_a_name") == "with-a-name"
    assert hyphenize("ok") == "ok"
