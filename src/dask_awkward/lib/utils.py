from __future__ import annotations

__all__ = ("trace_form_structure", "buffer_keys_required_to_compute_shapes")

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING, TypedDict, TypeVar

import awkward as ak

if TYPE_CHECKING:
    from awkward.forms import Form

KNOWN_LENGTH_ATTRIBUTES = frozenset(("mask",))
UNKNOWN_LENGTH_ATTRIBUTES = frozenset(("offsets", "starts", "stops", "index", "tags"))
DATA_ATTRIBUTES = frozenset(("data",))
METADATA_ATTRIBUTES = UNKNOWN_LENGTH_ATTRIBUTES | KNOWN_LENGTH_ATTRIBUTES


class FormStructure(TypedDict):
    form_key_to_form: MutableMapping[str, Form]
    form_key_to_parent_form_key: MutableMapping[str, str | None]
    form_key_to_path: MutableMapping[str, tuple[str, ...]]
    form_key_to_buffer_keys: MutableMapping[str, tuple[str, ...]]


def trace_form_structure(form: Form, buffer_key: Callable) -> FormStructure:
    form_key_to_form: MutableMapping[str, Form] = {}
    form_key_to_parent_form_key: MutableMapping[str, str | None] = {}
    form_key_to_path: MutableMapping[str, tuple[str, ...]] = {}
    form_key_to_buffer_keys: MutableMapping[str, tuple[str, ...]] = {}

    def impl_with_parent(
        form: Form,
        parent_form: Form | None,
        column_path: tuple[str, ...],
    ) -> None:
        # Associate child form key with parent form key
        form_key_to_parent_form_key[form.form_key] = (
            None if parent_form is None else parent_form.form_key
        )
        # Keep track of column-level path
        form_key_to_path[form.form_key] = column_path
        # Identify each form with a form key
        form_key_to_form[form.form_key] = form
        # Pre-compute the buffer keys for each form
        form_key_to_buffer_keys[form.form_key] = form.expected_from_buffers(
            recursive=False, buffer_key=buffer_key
        )
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
                next_column_path = column_path + (field,)
                # Recurse
                impl_with_parent(form.content(field), form, next_column_path)
        elif form.is_unknown or form.is_numpy:
            pass
        else:
            raise AssertionError(form)

    impl_with_parent(form, None, ())

    return {
        "form_key_to_form": form_key_to_form,
        "form_key_to_parent_form_key": form_key_to_parent_form_key,
        "form_key_to_path": form_key_to_path,
        "form_key_to_buffer_keys": form_key_to_buffer_keys,
    }


T = TypeVar("T")


def walk_bijective_graph(node: T, graph: Mapping[T, T | None]) -> Iterator[T]:
    while (node := graph.get(node)) is not None:  # type: ignore[assignment]
        yield node


def walk_graph_breadth_first(
    node: T, graph: Mapping[T, Iterable[T] | None]
) -> Iterator[T]:
    children = graph.get(node)
    if children is None:
        return
    yield from children
    for node in children:
        yield from walk_graph_breadth_first(node, graph)


def walk_graph_depth_first(
    node: T, graph: Mapping[T, Iterable[T] | None]
) -> Iterator[T]:
    children = graph.get(node)
    if children is None:
        return
    for node in children:
        yield node
        yield from walk_graph_depth_first(node, graph)


def buffer_keys_required_to_compute_shapes(
    parse_buffer_key: Callable[[str], tuple[str, str]],
    shape_buffers: Iterable[str],
    form_key_to_parent_key: Mapping[str, str | None],
    form_key_to_buffer_keys: Mapping[str, Iterable[str]],
) -> Iterable[str]:
    # Buffers needing known shapes must traverse all the way up the tree.
    for buffer_key in shape_buffers:
        form_key, attribute = parse_buffer_key(buffer_key)

        # For impacted form keys above this node
        for impacted_form_key in walk_bijective_graph(form_key, form_key_to_parent_key):
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
    head, tail = buffer_key.rsplit("-", maxsplit=1)
    return head, tail


def form_with_unique_keys(form: Form, key: str) -> Form:
    def impl(form: Form, key: str) -> None:
        # Set form key
        form.form_key = key

        # If the form is a record we need to loop over all fields in the
        # record and set form that include the field name; this will keep
        # recursing as well.
        if form.is_record:
            for field in form.fields:
                impl(form.content(field), f"{key}.{field}")

        elif form.is_union:
            for i, entry in enumerate(form.contents):
                impl(entry, f"{key}#{i}")

        # NumPy like array is easy
        elif form.is_numpy or form.is_unknown:
            pass

        # Anything else grab the content and keep recursing
        else:
            impl(form.content, f"{key}.content")

    # Perform a "deep" copy without preserving references
    form = ak.forms.from_dict(form.to_dict())
    impl(form, key)
    return form
