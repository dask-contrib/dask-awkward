from __future__ import annotations

__all__ = ("trace_form_structure", "buffer_keys_required_to_compute_shapes")

import copy
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
    from awkward.forms import Form


KNOWN_LENGTH_ATTRIBUTES = frozenset(("mask",))
UNKNOWN_LENGTH_ATTRIBUTES = frozenset(("offsets", "starts", "stops", "index", "tags"))
DATA_ATTRIBUTES = frozenset(("data",))
METADATA_ATTRIBUTES = UNKNOWN_LENGTH_ATTRIBUTES | KNOWN_LENGTH_ATTRIBUTES


class FormStructure(TypedDict):
    form_key_to_parent_key: dict[str, str]
    form_key_to_buffer_keys: dict[str, tuple[str, ...]]
    form_key_to_path: dict[str, tuple[str, ...]]


def trace_form_structure(form: Form, buffer_key: Callable) -> FormStructure:
    form_key_to_parent_key: dict[str, str] = {}
    form_key_to_buffer_keys: dict[str, tuple[str, ...]] = {}
    form_key_to_path: dict[str, tuple[str, ...]] = {}

    def impl_with_parent(form: Form, parent_form: Form, path: tuple[str, ...]):
        # Associate child form key with parent form key
        form_key_to_parent_key[form.form_key] = parent_form.form_key
        return impl(form, path)

    def impl(form: Form, column_path: tuple[str, ...]):
        # Keep track of which buffer keys are associated with which form-keys
        form_key_to_buffer_keys[form.form_key] = tuple(
            form.expected_from_buffers(recursive=False, buffer_key=buffer_key)
        )
        # Store columnar path
        form_key_to_path[form.form_key] = column_path

        if form.is_union:
            for _i, entry in enumerate(form.contents):
                impl_with_parent(entry, form, column_path)
        elif form.is_indexed:
            impl_with_parent(form.content, form, column_path)
        elif form.is_list:
            impl_with_parent(form.content, form, column_path)
        elif form.is_option:
            impl_with_parent(form.content, form, column_path)
        elif form.is_record:
            for field in form.fields:
                impl_with_parent(form.content(field), form, column_path + (field,))
        elif form.is_unknown or form.is_numpy:
            pass
        else:
            raise AssertionError(form)

    impl(form, ())

    return {
        "form_key_to_parent_key": form_key_to_parent_key,
        "form_key_to_buffer_keys": form_key_to_buffer_keys,
        "form_key_to_path": form_key_to_path,
    }


T = TypeVar("T")


def walk_parents(node: T, parentage: dict[T, T | None]) -> Iterator[T]:
    while (node := parentage.get(node)) is not None:
        yield node


def buffer_keys_required_to_compute_shapes(
    parse_buffer_key: Callable[[str], tuple[str, str]],
    shape_buffers: Iterable[str],
    form_key_to_parent_key: dict[str, str],
    form_key_to_buffer_keys: dict[str, str],
):
    # Buffers needing known shapes must traverse all the way up the tree.
    for buffer_key in shape_buffers:
        form_key, attribute = parse_buffer_key(buffer_key)

        # For impacted form keys above this node
        for impacted_form_key in walk_parents(form_key, form_key_to_parent_key):
            # Identify the associated buffers
            for impacted_buffer_key in form_key_to_buffer_keys[impacted_form_key]:
                _, other_attribute = parse_buffer_key(impacted_buffer_key)

                # Would the omission of this key lead to unknown lengths?
                if other_attribute in UNKNOWN_LENGTH_ATTRIBUTES:
                    # If so, touch the buffers
                    yield impacted_buffer_key


def render_buffer_key(form: Form, form_key: str, attribute: str) -> str:
    return f"{form_key}-{attribute}"


def parse_buffer_key(buffer_key: str) -> tuple[str, str]:
    return buffer_key.rsplit("-", maxsplit=1)


def form_with_unique_keys(form: Form, key: str) -> Form:
    def impl(form: Form, key: str):
        # Set form key
        form.form_key = key

        # If the form is a record we need to loop over all fields in the
        # record and set form that include the field name; this will keep
        # recursing as well.
        if form.is_record:
            for field in form.fields:
                full_key = f"{key}.{field}"
                impl(form.content(field), full_key)

        elif form.is_union:
            for i, entry in enumerate(form.contents):
                impl(entry, f"{key}#{i}")

        # NumPy like array is easy
        elif form.is_numpy or form.is_unknown:
            pass

        # Anything else grab the content and keep recursing
        else:
            impl(form.content, f"{key}.content")

    form = copy.deepcopy(form)
    impl(form, key)
    return form
