from __future__ import annotations

from typing import Any

__all__ = ("plugin",)

from pickle import PickleBuffer

import awkward as ak
from awkward.typetracer import PlaceholderArray


def maybe_make_pickle_buffer(buffer: Any) -> PlaceholderArray | PickleBuffer:
    if isinstance(buffer, PlaceholderArray):
        return buffer
    else:
        return PickleBuffer(buffer)


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
        container = {k: maybe_make_pickle_buffer(v) for k, v in container.items()}

    if record.behavior is ak.behavior:
        behavior = None
    else:
        behavior = record.behavior

    return (
        object.__new__,
        (ak.Record,),
        (form.to_dict(), length, container, behavior, layout.at),
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
        container = {k: maybe_make_pickle_buffer(v) for k, v in container.items()}

    if array.behavior is ak.behavior:
        behavior = None
    else:
        behavior = array.behavior

    return (
        object.__new__,
        (ak.Array,),
        (form.to_dict(), length, container, behavior),
    )


def plugin(obj: Any, protocol: int) -> tuple:
    if isinstance(obj, ak.Record):
        return pickle_record(obj, protocol)
    elif isinstance(obj, ak.Array):
        return pickle_array(obj, protocol)
    else:
        return NotImplemented
