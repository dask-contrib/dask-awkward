from __future__ import annotations

import functools

import awkward.operations.str as akstr

from dask_awkward.lib.core import Array, map_partitions


def always_highlevel(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not kwargs.get("highlevel", True):
            raise ValueError("dask-awkward supports only highlevel awkward arrays.")
        return fn(*args, **kwargs)

    return wrapper


@always_highlevel
def capitalize(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.capitalize,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def center(
    array: Array,
    width,
    padding=" ",
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.center,
        array,
        width=width,
        padding=padding,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def count_substring(
    array: Array,
    pattern: str | bytes,
    *,
    ignore_case: bool = False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.count_substring,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def count_substring_regex(
    array: Array,
    pattern: str | bytes,
    *,
    ignore_case: bool = False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.count_substring_regex,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def ends_with(
    array: Array,
    pattern: str | bytes,
    *,
    ignore_case: bool = False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.ends_with,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def extract_regex(
    array: Array,
    pattern: bytes | str,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.extract_regex,
        array,
        pattern=pattern,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def find_substring(
    array: Array,
    pattern,
    *,
    ignore_case=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.find_substring,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def find_substring_regex(
    array: Array,
    pattern,
    *,
    ignore_case=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.find_substring_regex,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def index_in(
    array: Array,
    value_set,
    *,
    skip_nones=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.index_in,
        array,
        value_set=value_set,
        skip_nones=skip_nones,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_alnum(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_alnum,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_alpha(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_alpha,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_ascii(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_ascii,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_decimal(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_decimal,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_digit(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_digit,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_in(
    array: Array,
    value_set,
    *,
    skip_nones=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_in,
        array,
        value_set=value_set,
        skip_nones=skip_nones,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_lower(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_lower,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_numeric(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_numeric,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_printable(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_printable,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_space(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_space,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_title(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_title,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def is_upper(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.is_upper,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def join(
    array: Array,
    separator,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.join,
        array,
        separator=separator,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def join_element_wise(
    *arrays,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.join_element_wise,
        *arrays,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def length(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.length,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def lower(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.lower,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def lpad(
    array: Array,
    width,
    padding=" ",
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.lpad,
        array,
        width=width,
        padding=padding,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def ltrim(
    array: Array,
    characters,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.ltrim,
        array,
        characters=characters,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def ltrim_whitespace(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.ltrim_whitespace,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def match_like(
    array: Array,
    pattern,
    *,
    ignore_case=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.match_like,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def match_substring(
    array: Array,
    pattern,
    *,
    ignore_case=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.match_substring,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def match_substring_regex(
    array: Array,
    pattern,
    *,
    ignore_case=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.match_substring_regex,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def repeat(
    array: Array,
    num_repeats,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.repeat,
        array,
        num_repeats=num_repeats,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def replace_slice(
    array: Array,
    start,
    stop,
    replacement,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.replace_slice,
        array,
        start=start,
        stop=stop,
        replacement=replacement,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def replace_substring(
    array: Array,
    pattern,
    replacement,
    *,
    max_replacements=None,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.replace_substring,
        array,
        pattern=pattern,
        replacement=replacement,
        max_replacements=max_replacements,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def replace_substring_regex(
    array: Array,
    pattern,
    replacement,
    *,
    max_replacements=None,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.replace_substring_regex,
        array,
        pattern=pattern,
        replacement=replacement,
        max_replacements=max_replacements,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def reverse(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.reverse,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def rpad(
    array: Array,
    width,
    padding=" ",
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.rpad,
        array,
        width=width,
        padding=padding,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def rtrim(
    array: Array,
    characters,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.rtrim,
        array,
        characters=characters,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def rtrim_whitespace(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.rtrim_whitespace,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def slice(
    array: Array,
    start,
    stop=None,
    step=1,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.slice,
        array,
        start=start,
        stop=stop,
        step=step,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def split_pattern(
    array: Array,
    pattern,
    *,
    max_splits=None,
    reverse=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.split_pattern,
        array,
        pattern=pattern,
        max_splits=max_splits,
        reverse=reverse,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def split_pattern_regex(
    array: Array,
    pattern,
    *,
    max_splits=None,
    reverse=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.split_pattern_regex,
        array,
        pattern=pattern,
        max_splits=max_splits,
        reverse=reverse,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def split_whitespace(
    array: Array,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.split_whitespace,
        array,
        max_splits=max_splits,
        reverse=reverse,
        behavior=behavior,
    )


@always_highlevel
def starts_with(
    array: Array,
    pattern,
    *,
    ignore_case=False,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.starts_with,
        array,
        pattern=pattern,
        ignore_case=ignore_case,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def swapcase(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.swapcase,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def title(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.title,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def to_categorical(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.to_categorical,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def trim(
    array: Array,
    characters,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.trim,
        array,
        characters=characters,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def trim_whitespace(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.trim_whitespace,
        array,
        behavior=behavior,
        output_divisions=1,
    )


@always_highlevel
def upper(
    array: Array,
    *,
    highlevel: bool = True,
    behavior: dict | None = None,
) -> Array:
    return map_partitions(
        akstr.upper,
        array,
        behavior=behavior,
        output_divisions=1,
    )
