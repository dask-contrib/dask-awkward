from __future__ import annotations

import pytest

pytest.importorskip("pyarrow")

import awkward as ak
import awkward.operations.str as akstr

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq

lines1 = [
    "this is line one",
    "123",
    "5",
    " ",
    "      12.34" "   123      ",
    "this is line two",
    "THIS IS LINE THREE",
    "OKOKOK",
    "42.52",
    "OKOK",
]

lines2 = [
    "1",
    "aaaaaaaaaaa",
    "bbbbbbbbbbbbbbbb",
    "CCC",
    " ",
    "OK",
    "DDDDDDDDDDDDDDDDDDDDDD",
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
        akstr.capitalize(caa),
    )


def test_center() -> None:
    assert_eq(
        dak.str.center(daa, 5),
        akstr.center(caa, 5),
    )


def test_count_substring() -> None:
    assert_eq(
        dak.str.count_substring(daa, "aa"),
        akstr.count_substring(caa, "aa"),
    )


def test_count_substring_regex() -> None:
    a = akstr.count_substring_regex(daa, r"aa\s+")
    assert isinstance(a, dak.Array)
    b = akstr.count_substring_regex(caa, r"aa\s+")
    assert_eq(a, b)


def test_ends_with() -> None:
    assert_eq(akstr.ends_with(daa, "123"), akstr.ends_with(caa, "123"))


def test_extract_regex() -> None:
    pass


def test_find_substring() -> None:
    assert_eq(akstr.find_substring(daa, r"bbb"), akstr.find_substring(caa, r"bbb"))


def test_find_substring_regex() -> None:
    assert_eq(
        akstr.find_substring_regex(daa, r"aa\s+"),
        akstr.find_substring_regex(caa, r"aa\s+"),
    )


def test_index_in() -> None:
    assert_eq(
        akstr.index_in(daa, [" aaa ", "123"]),
        akstr.index_in(caa, [" aaa ", "123"]),
    )


def test_is_alnum() -> None:
    assert_eq(akstr.is_alnum(daa), akstr.is_alnum(caa))


def test_is_alpha() -> None:
    assert_eq(akstr.is_alpha(daa), akstr.is_alpha(caa))


def test_is_ascii() -> None:
    assert_eq(akstr.is_ascii(daa), akstr.is_ascii(caa))


def test_is_decimal() -> None:
    assert_eq(akstr.is_decimal(daa), akstr.is_decimal(caa))


def test_is_digit() -> None:
    assert_eq(akstr.is_digit(daa), akstr.is_digit(caa))


def test_is_in() -> None:
    assert_eq(
        akstr.is_in(daa, ["CCC", "1"]),
        akstr.is_in(caa, ["CCC", "1"]),
    )


def test_is_lower() -> None:
    assert_eq(akstr.is_lower(daa), akstr.is_lower(caa))


def test_is_numeric() -> None:
    assert_eq(akstr.is_numeric(daa), akstr.is_numeric(caa))


def test_is_printable() -> None:
    assert_eq(akstr.is_printable(daa), akstr.is_printable(caa))


def test_is_space() -> None:
    assert_eq(akstr.is_space(daa), akstr.is_space(caa))


def test_is_title() -> None:
    pass


def test_is_upper() -> None:
    assert_eq(akstr.is_upper(daa), akstr.is_upper(caa))


def test_join() -> None:
    pass


def test_join_element_wise() -> None:
    pass


def test_length() -> None:
    assert_eq(akstr.length(daa), akstr.length(caa))


def test_lower() -> None:
    assert_eq(akstr.lower(daa), akstr.lower(caa))


def test_lpad() -> None:
    assert_eq(akstr.lpad(daa, 3), akstr.lpad(caa, 3))


def test_ltrim() -> None:
    assert_eq(
        akstr.ltrim(akstr.ltrim(daa, "th"), "   "),
        akstr.ltrim(akstr.ltrim(caa, "th"), "   "),
    )


def test_ltrim_whitespace() -> None:
    assert_eq(akstr.ltrim_whitespace(daa), akstr.ltrim_whitespace(caa))


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
    assert_eq(akstr.reverse(daa), akstr.reverse(caa))


def test_rpad() -> None:
    assert_eq(akstr.rpad(daa, 5, "j"), akstr.rpad(caa, 5, "j"))


def test_rtrim() -> None:
    assert_eq(
        akstr.rtrim(akstr.rtrim(daa, "OK"), "   "),
        akstr.rtrim(akstr.rtrim(caa, "OK"), "   "),
    )


def test_rtrim_whitespace() -> None:
    assert_eq(akstr.rtrim_whitespace(daa), akstr.rtrim_whitespace(caa))


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
    assert_eq(akstr.swapcase(daa), akstr.swapcase(caa))


def test_title() -> None:
    pass


def test_to_categorical() -> None:
    pass


def test_trim() -> None:
    pass


def test_trim_whitespace() -> None:
    pass


def test_upper() -> None:
    assert_eq(akstr.upper(daa), akstr.upper(caa))
