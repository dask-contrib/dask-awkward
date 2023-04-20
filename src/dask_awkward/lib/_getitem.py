from __future__ import annotations

import copy
import operator

from dask.blockwise import Blockwise
from dask.highlevelgraph import HighLevelGraph, Layer


def last_layer(obj: HighLevelGraph) -> tuple[str, Layer]:
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
    if isinstance(fn_args, tuple):
        if not all(isinstance(x, str) for x in fn_args):
            return hlg
        new_indices = (tuple(list(fn_args) + [new_key]), None)
    elif isinstance(fn_args, (str, int)):
        new_indices = ((fn_args, new_key), None)
    else:
        return hlg

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
