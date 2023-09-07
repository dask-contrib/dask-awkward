from __future__ import annotations

from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
from awkward.forms.listoffsetform import ListOffsetForm
from awkward.forms.numpyform import NumpyForm
from awkward.typetracer import typetracer_from_form
from dask.base import tokenize
from dask.core import flatten
from fsspec.core import get_fs_token_paths
from fsspec.utils import infer_compression, read_block

from dask_awkward.lib.io.io import (
    _bytes_with_sample,
    _BytesReadingInstructions,
    from_map,
)

if TYPE_CHECKING:
    from dask_awkward.lib.core import Array


def _string_array_from_bytestring(bytestring: bytes, delimiter: bytes) -> ak.Array:
    buffer = np.frombuffer(bytestring, dtype=np.uint8)
    array = ak.from_numpy(buffer)
    array = ak.unflatten(array, len(array))
    array = ak.enforce_type(array, "string")
    array_split = ak.str.split_pattern(array, delimiter)
    lines = array_split[0]
    if len(lines) == 0:
        return lines
    if lines[-1] == "":
        lines = lines[:-1]
    return lines


def _from_text_on_block(instructions: _BytesReadingInstructions) -> ak.Array:
    with instructions.fs.open(
        instructions.path, compression=instructions.compression
    ) as f:
        if instructions.offset == 0 and instructions.length is None:
            bytestring = f.read()
        else:
            bytestring = read_block(
                f,
                instructions.offset,
                instructions.length,
                instructions.delimiter,
            )

    return _string_array_from_bytestring(bytestring, instructions.delimiter)


def from_text(
    source: str | list[str],
    blocksize: str | int = "128 MiB",
    delimiter: bytes = b"\n",
    sample_size: str | int = "128 KiB",
    compression: str | None = "infer",
    storage_options: dict | None = None,
) -> Array:
    fs, token, paths = get_fs_token_paths(source, storage_options=storage_options or {})

    token = tokenize(source, token, blocksize, delimiter, sample_size)

    if compression == "infer":
        compression = infer_compression(paths[0])

    bytes_ingredients, _ = _bytes_with_sample(
        fs,
        paths,
        compression,
        delimiter,
        False,
        blocksize,
        False,
    )

    # meta is _always_ an unknown length array of strings.
    meta = typetracer_from_form(
        ListOffsetForm(
            "i64",
            NumpyForm("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        )
    )

    return from_map(
        _from_text_on_block,
        list(flatten(bytes_ingredients)),
        label="from-text",
        token=token,
        meta=meta,
    )
