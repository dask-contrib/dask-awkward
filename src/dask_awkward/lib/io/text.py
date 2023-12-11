from __future__ import annotations

from typing import cast

import awkward as ak
import numpy as np
from awkward.forms.listoffsetform import ListOffsetForm
from awkward.forms.numpyform import NumpyForm
from awkward.typetracer import typetracer_from_form
from dask.base import tokenize
from dask.core import flatten
from fsspec.core import get_fs_token_paths
from fsspec.utils import infer_compression, read_block

from dask_awkward.lib.core import Array
from dask_awkward.lib.io.io import (
    _bytes_with_sample,
    _BytesReadingInstructions,
    from_map,
)


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
    compression: str | None = "infer",
    storage_options: dict | None = None,
) -> Array:
    """Create an Array collection from text data and a delimiter.

    The default behavior of this input function is to create an array
    collection where elements are seperated by newlines.

    Parameters
    ----------
    source : str | list[str]
        Data source as a list of files or a single path (can be remote
        files).
    blocksize : str | int
        Size of each partition in bytes.
    delimiter : bytes
        Delimiter to separate elements of the array (default is
        newline character).
    compression : str, optional
        Compression of the files for reading (default is to infer).
    storage_options : dict, optional
        Storage options passed to the ``fsspec`` filesystem.

    Returns
    -------
    Array
        Resulting collection.

    Examples
    --------
    >>> import dask_awkward as dak
    >>> ds = dak.from_text("s3://path/to/files/*.txt", blocksize="256 MiB")

    """
    fs, token, paths = get_fs_token_paths(source, storage_options=storage_options or {})

    token = tokenize(source, token, blocksize, delimiter, compression)

    if compression == "infer":
        compression = infer_compression(paths[0])

    bytes_ingredients, _ = _bytes_with_sample(
        fs,
        paths=paths,
        compression=compression,
        delimiter=delimiter,
        not_zero=False,
        blocksize=blocksize,
        sample=False,
    )

    # meta is _always_ an unknown length array of strings.
    meta = typetracer_from_form(
        ListOffsetForm(
            "i64",
            NumpyForm("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        )
    )

    return cast(
        Array,
        from_map(
            _from_text_on_block,
            list(flatten(bytes_ingredients)),
            label="from-text",
            token=token,
            meta=meta,
        ),
    )
