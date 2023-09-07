from __future__ import annotations

import awkward as ak
import awkward.operations.str as akstr
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq

text1 = [
    "a sentence",
    "one two three",
    "one,two,three",
    "abc 123 def 456",
    "this is a test ok ok ok",
    "aaaaaaaaaaa bbbbbbbbbbbbb",
    "ccccccccccccccccccccccccc",
    "ddd ddd ddd ddd ddd ddd",
    "asdf jkl",
]

text2 = [
    "abc 123",
    "456 789 10,11,12",
    "dddddddd",
    "jjjjjjjjjjj",
    "lllllllll",
    "oooooooooooo iiiiiiii nnnnnn",
    "nlm abc 456",
]


def test_form_text(tmp_path_factory: pytest.TempPathFactory) -> None:
    p = tmp_path_factory.mktemp("from_text")
    with (p / "file1.txt").open("wt") as f:
        f.write("\n".join(text1))
    with (p / "file2.txt").open("wt") as f:
        f.write("\n".join(text2))

    daa = dak.from_text([str(p / "file1.txt"), str(p / "file2.txt")])
    caa = ak.concatenate([ak.Array(text1), ak.Array(text2)])

    assert daa.npartitions == 2

    daa_split = daa.map_partitions(akstr.split_whitespace).map_partitions(
        akstr.is_numeric
    )
    caa_split = akstr.is_numeric(akstr.split_whitespace(caa))
    assert_eq(daa_split, caa_split)

    assert daa_split[-1].compute().tolist() == [False, False, True]

    daa_split = daa.map_partitions(akstr.capitalize)
    caa_split = akstr.capitalize(caa)
    assert_eq(daa_split, caa_split)
