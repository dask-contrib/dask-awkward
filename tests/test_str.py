from __future__ import annotations

import awkward as ak
import pytest

pytest.importorskip("pyarrow")

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq

lines1 = [
    "this is line one",
    "123",
    "      1234" "   123      ",
    "this is line two",
]

lines2 = [
    "aaaaaaaaaaa",
    "bbbbbbbbbbbbbbbb",
    "cccccccccccccccccccc",
    "ddddddddddddddddddddddddd",
]

lines3 = [
    "          a          ",
    "         aaa         ",
    "        aaaaa        ",
    "       aaaaaaa       ",
    "      aaaaaaaaa      ",
    "       aaaaaaa       ",
    "        aaaaa        ",
    "         aaa         ",
    "          a          ",
]

caa = ak.from_iter(lines1 + lines2 + lines3)
daa = dak.from_lists([lines1, lines2, lines3])


def test_sanity():
    assert_eq(caa, daa)


def test_capitalize() -> None:
    assert_eq(
        dak.str.capitalize(daa),
        ak.str.capitalize(caa),
    )


def test_center() -> None:
    assert_eq(
        dak.str.center(daa, 5),
        ak.str.center(caa, 5),
    )


def test_count_substring() -> None:
    assert_eq(
        dak.str.count_substring(daa, "aa"),
        ak.str.count_substring(caa, "aa"),
    )


def test_count_substring_regex() -> None:
    a = ak.str.count_substring_regex(daa, r"aa\s+")
    assert isinstance(a, dak.Array)
    b = ak.str.count_substring_regex(caa, r"aa\s+")
    assert_eq(a, b)


def test_ends_with() -> None:
    assert_eq(ak.str.ends_with(daa, "123"), ak.str.ends_with(caa, "123"))


def test_extract_regex() -> None:
    pass


def test_find_substring() -> None:
    pass


def test_find_substring_regex() -> None:
    pass


def test_index_in() -> None:
    pass


def test_is_alnum() -> None:
    pass


def test_is_alpha() -> None:
    pass


def test_is_ascii() -> None:
    pass


def test_is_decimal() -> None:
    pass


def test_is_digit() -> None:
    pass


def test_is_in() -> None:
    pass


def test_is_lower() -> None:
    pass


def test_is_numeric() -> None:
    pass


def test_is_printable() -> None:
    pass


def test_is_space() -> None:
    pass


def test_is_title() -> None:
    pass


def test_is_upper() -> None:
    pass


def test_join() -> None:
    pass


def test_join_element_wise() -> None:
    pass


def test_length() -> None:
    pass


def test_lower() -> None:
    pass


def test_lpad() -> None:
    pass


def test_ltrim() -> None:
    pass


def test_ltrim_whitespace() -> None:
    pass


def test_match_like() -> None:
    pass


def test_match_substring() -> None:
    pass


def test_match_substring_regex() -> None:
    pass


def test_repeat() -> None:
    pass


def test_replace_slice() -> None:
    pass


def test_replace_substring() -> None:
    pass


def test_replace_substring_regex() -> None:
    pass


def test_reverse() -> None:
    pass


def test_rpad() -> None:
    pass


def test_rtrim() -> None:
    pass


def test_rtrim_whitespace() -> None:
    pass


def test_slice() -> None:
    pass


def test_split_pattern() -> None:
    pass


def test_split_pattern_regex() -> None:
    pass


def test_split_whitespace() -> None:
    pass


def test_starts_with() -> None:
    pass


def test_swapcase() -> None:
    pass


def test_title() -> None:
    pass


def test_to_categorical() -> None:
    pass


def test_trim() -> None:
    pass


def test_trim_whitespace() -> None:
    pass


def test_upper() -> None:
    pass
