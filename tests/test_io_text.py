from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq

text1 = """a sentence
one two three
one,two,three
abc 123 def 456
this is a test\t ok ok ok
aaaaaaaaaaa bbbbbbbbbbbbb
ccccccccccccccccccccccccc
ddd ddd ddd ddd ddd ddd
asdf jkl"""

text2 = """abc 123
456 789 10,11,12
dddddddd
jjjjjjjjjjj
lllllllll
oooooooooooo iiiiiiii nnnnnn
nlm abc 456"""


def test_form_text(tmp_path_factory: pytest.TempPathFactory) -> None:
    p = tmp_path_factory.mktemp("from_text")
    with (p / "file1.txt").open("w") as f:
        print(text1, file=f)
    with (p / "file2.txt").open("w") as f:
        print(text2, file=f)

    daa = dak.from_text(str(p / "*.txt"))
    caa = ak.concatenate([ak.Array(text1.split("\n")), ak.Array(text2.split("\n"))])

    assert daa.npartitions == 2

    daa_split = daa.map_partitions(ak.str.split_whitespace).map_partitions(
        ak.str.is_numeric
    )
    caa_split = ak.str.is_numeric(ak.str.split_whitespace(caa))
    assert_eq(daa_split, caa_split)

    daa_split = daa.map_partitions(ak.str.capitalize)
    caa_split = ak.str.capitalize(caa)
    assert_eq(daa_split, caa_split)
