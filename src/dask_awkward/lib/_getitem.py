from __future__ import annotations

import copy
import operator
from pprint import pprint

from dask.blockwise import Blockwise
from dask.highlevelgraph import HighLevelGraph, Layer

from dask_awkward.lib.core import Array


def rewind_one_layer(collection):
    hlg = collection.dask
    tsl = hlg._toposort_layers()
    second_to_last_name = tsl[-2]
    last_name = tsl[-1]

    layers = hlg.layers.copy()
    deps = hlg.dependencies.copy()


def last_layer(obj: Array | HighLevelGraph) -> tuple[str, Layer]:
    if isinstance(obj, Array):
        obj = obj.dask

    name = obj._toposort_layers()[-1]
    return name, obj.layers[name]


def is_getitem_layer(layer: Layer) -> bool:
    if isinstance(layer, Blockwise):
        return layer.dsk[layer.output][0] == operator.getitem
    return False


def rewrite_last_getitem(hlg: HighLevelGraph, new_key: str) -> HighLevelGraph:
    lname, llayer = last_layer(hlg)

    if not is_getitem_layer(llayer):
        return hlg

    fn_args = llayer.indices[-1][0]
    pprint(fn_args)
    if isinstance(fn_args, tuple):
        if not all(isinstance(x, str) for x in fn_args):
            return None
        new_indices = (tuple(list(fn_args) + [new_key]), None)
    elif isinstance(fn_args, str):
        new_indices = ((fn_args, new_key), None)

    layers = hlg.layers.copy()
    deps = hlg.dependencies.copy()

    indices = list(copy.deepcopy(llayer.indices))
    indices[-1] = new_indices

    loi = Blockwise(
        output=llayer.output,
        output_indices=llayer.output_indices,
        dsk=llayer.dsk,
        indices=indices,
        numblocks=llayer.numblocks,
        concatenate=llayer.concatenate,
        new_axes=llayer.new_axes,
        output_blocks=llayer.output_blocks,
        annotations=llayer.annotations,
        io_deps=llayer.io_deps,
    )

    layers[lname] = loi

    return HighLevelGraph(layers, deps)
