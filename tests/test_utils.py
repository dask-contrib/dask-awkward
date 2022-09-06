from __future__ import annotations

from dask_awkward.utils import LazyInputsDict, hyphenize, is_empty_slice


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
