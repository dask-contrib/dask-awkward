from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ("plugin",)

from pickle import PickleBuffer

import awkward as ak
from awkward.typetracer import PlaceholderArray


def _maybe_make_pickle_buffer(buffer: Any) -> PlaceholderArray | PickleBuffer:
    if isinstance(buffer, PlaceholderArray):
        return buffer
    else:
        return PickleBuffer(buffer)


def _without_transient_attrs(attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    return {k: v for k, v in attrs.items() if not k.startswith("@")}


def _unpickle_record_schema_1(
    form_dict: dict,
    length: Any,
    container: Mapping[str, Any],
    behavior: Mapping | None,
    attrs: Mapping[str, Any] | None,
    at: int,
) -> ak.Record:
    array_layout = ak.from_buffers(
        form_dict, length, container, behavior=behavior, attrs=attrs, highlevel=False
    )
    layout = ak.record.Record(array_layout, at)
    return ak.Record(layout, behavior=behavior, attrs=attrs)


def _unpickle_array_schema_1(
    form_dict: dict,
    length: Any,
    container: Mapping[str, Any],
    behavior: Mapping | None,
    attrs: Mapping[str, Any] | None,
) -> ak.Array:
    return ak.from_buffers(
        form_dict, length, container, behavior=behavior, attrs=attrs, highlevel=True
    )


def pickle_record(record: ak.Record, protocol: int) -> tuple:
    layout = ak.to_layout(record, allow_record=True)
    form, length, container = ak.operations.to_buffers(
        layout.array,
        buffer_key="{form_key}-{attribute}",
        form_key="node{id}",
        byteorder="<",
    )

    # For pickle >= 5, we can avoid copying the buffers
    if protocol >= 5:
        container = {k: _maybe_make_pickle_buffer(v) for k, v in container.items()}

    if record.behavior is ak.behavior:
        behavior = None
    else:
        behavior = record.behavior

    if record._attrs is None:
        attrs = record._attrs
    else:
        attrs = _without_transient_attrs(record._attrs)

    return (
        _unpickle_record_schema_1,
        (form.to_dict(), length, container, behavior, attrs, layout.at),
    )


def pickle_array(array: ak.Array, protocol: int) -> tuple:
    layout = ak.to_layout(array, allow_record=False)
    form, length, container = ak.operations.to_buffers(
        layout,
        buffer_key="{form_key}-{attribute}",
        form_key="node{id}",
        byteorder="<",
    )

    # For pickle >= 5, we can avoid copying the buffers
    if protocol >= 5:
        container = {k: _maybe_make_pickle_buffer(v) for k, v in container.items()}

    if array.behavior is ak.behavior:
        behavior = None
    else:
        behavior = array.behavior

        if array._attrs is None:
            attrs = array._attrs
        else:
            attrs = _without_transient_attrs(array._attrs)

    return (
        _unpickle_array_schema_1,
        (form.to_dict(), length, container, behavior, attrs),
    )


def plugin(obj: Any, protocol: int) -> tuple:
    if isinstance(obj, ak.Record):
        return pickle_record(obj, protocol)
    elif isinstance(obj, ak.Array):
        return pickle_array(obj, protocol)
    else:
        return NotImplemented
