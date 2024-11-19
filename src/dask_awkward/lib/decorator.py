from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import awkward as ak
from dask.highlevelgraph import HighLevelGraph
from dask.typing import DaskCollection

from dask_awkward.lib.core import (
    _map_partitions_prepare,
    _to_packed_fn_args,
    dak_cache,
    empty_typetracer,
    new_array_object,
    partitionwise_layer,
)


def _single_return_map_partitions(
    hlg: HighLevelGraph,
    name: str,
    meta: tp.Any,
    npartitions: int,
) -> tp.Any:
    # ak.Array (this is dak.map_partitions case)
    if isinstance(meta, ak.Array):
        return new_array_object(
            hlg,
            name=name,
            meta=meta,
            npartitions=npartitions,
        )

    # TODO: np.array
    # from dask.utils import is_arraylike, is_dataframe_like, is_index_like, is_series_like
    #
    # elif is_arraylike(meta):
    # this doesn't work yet, because the graph/chunking is not correct
    #
    # import numpy as np
    # from dask.array.core import new_da_object
    # meta = meta[None, ...]
    # first = (np.nan,) * npartitions
    # rest = ((-1,),) * (meta.ndim - 1)
    # chunks = (first, *rest)
    # return new_da_object(hlg, name=name, meta=meta, chunks=chunks)

    # TODO: dataframe, series, index
    # elif (
    #     is_dataframe_like(meta)
    #     or is_series_like(meta)
    #     or is_index_like(meta)
    # ): pass

    # don't know? -> put it in a bag
    else:
        from dask.bag.core import Bag

        return Bag(dsk=hlg, name=name, npartitions=npartitions)


def _multi_return_map_partitions(
    hlg: HighLevelGraph,
    name: str,
    meta: tp.Any,
    npartitions: int,
) -> tp.Any:
    # single-return case, this is equal to `dak.map_partitions`
    # but supports other DaskCollections in addition
    if not isinstance(meta, tuple):
        return _single_return_map_partitions(
            hlg=hlg,
            name=name,
            meta=meta,
            npartitions=npartitions,
        )
    # multi-return case
    else:
        from operator import itemgetter
        from typing import cast

        # create tmp dask collection for HLG creation
        tmp = new_array_object(
            hlg, name=name, meta=empty_typetracer(), npartitions=npartitions
        )

        ret = []
        for i, m_pick in enumerate(meta):
            # add a "select/pick" layer
            # to get the ith element of the output
            ith_name = f"{name}-pick-{i}th"

            if ith_name in dak_cache:
                hlg_pick, m_pick = dak_cache[ith_name]
            else:
                lay_pick = partitionwise_layer(itemgetter(i), ith_name, tmp)
                hlg_pick = HighLevelGraph.from_collections(
                    name=ith_name,
                    layer=lay_pick,
                    dependencies=[cast(DaskCollection, tmp)],
                )
                dak_cache[ith_name] = hlg_pick, m_pick

            # nested return case -> recurse
            if isinstance(m_pick, tuple):
                ret.append(
                    _multi_return_map_partitions(
                        hlg=hlg_pick,
                        name=ith_name,
                        meta=m_pick,
                        npartitions=npartitions,
                    )
                )
            else:
                ret.append(
                    _single_return_map_partitions(
                        hlg=hlg_pick,
                        name=ith_name,
                        meta=m_pick,
                        npartitions=npartitions,
                    )
                )
        return tuple(ret)


@dataclass
class mapfilter:
    """Map a callable across all partitions of any number of collections.
    This decorator is a convenience wrapper around the `dak.map_partitions` function.

    It serves the following purposes:
        - Turn multiple operations into a single node in the Dask graph
        - Explicitly touch columns if necessarily without interacting with the typetracer

    Caveats:
        - The function must use pure eager awkward inside (no delayed operations)
        - The function must return a single argument, i.e. an awkward array
        - The function must be emberassingly parallel

    Parameters
    ----------
    base_fn : Callable
        Function to apply on all partitions, this will get wrapped to
        handle kwargs, including dask collections.
    label : str, optional
        Label for the Dask graph layer; if left to ``None`` (default),
        the name of the function will be used.
    token : str, optional
        Provide an already defined token. If ``None`` a new token will
        be generated.
    meta : Any, optional
        Metadata (typetracer) array for the result (if known). If
        unknown, `fn` will be applied to the metadata of the `args`;
        if that call fails, the first partition of the new collection
        will be used to compute the new metadata **if** the
        ``awkward.compute-known-meta`` configuration setting is
        ``True``. If the configuration setting is ``False``, an empty
        typetracer will be assigned as the metadata.
    traverse : bool
        Unpack basic python containers to find dask collections.
    needs: dict, optional
        If ``None`` (the default), nothing is touched in addition to the
        standard typetracer report. In certain cases, it is necessary to
        touch additional objects **explicitly** to get the correct typetracer report.
        For this, provide a dictionary that maps input argument that's an array to
        the columns/slice of that array that should be touched.
    out_like: tp.Any, optional
        If ``None`` (the default), the output will be computed through the default
        typetracing pass. If a ak.Array is provided, the output will be mocked for the typetracing
        pass as the provided array. This is useful for cases where the output can not be
        computed through the default typetracing pass.


    Returns
    -------
    dask_awkward.Array
        The new collection.

    Examples
    --------
    >>> from coffea.nanoevents import NanoEventsFactory
    >>> from coffea.processor.decorator import mapfilter
    >>> events, report = NanoEventsFactory.from_root(
            {"https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root": "Events"},
            metadata={"dataset": "Test"},
            uproot_options={"allow_read_errors_with_report": True},
            steps_per_file=2,
        ).events()
    >>> @mapfilter
        def process(events):
            # do an emberassing parallel computation
            # only eager awkward is allowed here
            import awkward as ak

            jets = events.Jet
            jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
            return events[ak.num(jets) == 2]
    >>> selected = process(events)
    >>> print(process(events).dask)  # collapsed into a single node (2.)
    HighLevelGraph with 3 layers.
    <dask.highlevelgraph.HighLevelGraph object at 0x11700d640>
    0. from-uproot-0e54dc3659a3c020608e28b03f22b0f4
    1. from-uproot-971b7f00ce02a189422528a5044b08fb
    2. <dask-awkward.lib.core.ArgsKwargsPackedFunction ob-c9ee010d2e5671a2805f6d5106040d55
    >>> print(process.base_fn(events).dask) # call the function as it is (many nodes in the graph); `base_fn` is the function that is wrapped
    HighLevelGraph with 13 layers.
    <dask.highlevelgraph.HighLevelGraph object at 0x136e3d910>
    0. from-uproot-0e54dc3659a3c020608e28b03f22b0f4
    1. from-uproot-971b7f00ce02a189422528a5044b08fb
    2. Jet-efead9353042e606d7ffd59845f4675d
    3. eta-f31547c2a94efc053977790a489779be
    4. absolute-74ced100c5db654eb0edd810542f724a
    5. less-b33e652814e0cd5157b3c0885087edcb
    6. pt-f50c15fa409e60152de61957d2a4a0d8
    7. greater-da496609d36631ac857bb15eba6f0ba6
    8. bitwise-and-a501c0ff0f5bcab618514603d4f78eec
    9. getitem-fc20cad0c32130756d447fc749654d11
    10. <dask-awkward.lib.core.ArgsKwargsPackedFunction ob-0d3090f1c746eafd595782bcacd30d69
    11. equal-a4642445fb4e5da0b852c2813966568a
    12. getitem-f951afb4c4d4b527553f5520f6765e43

    # if you want to touch additional objects explicitly, because they are not touched by the standard typetracer (i.e. due to 'opaque' operations)
    # you can provide a dict of slices that should be touched directly to the decorator, e.g.:
    >>> from functools import partial
    >>> @partial(mapfilter, needs={"events": [("Electron", "pt"), ("Electron", "eta")]})
        def process(events):
            # do an emberassing parallel computation
            # only eager awkward is allowed here
            import awkward as ak

            jets = events.Jet
            jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
            return events[ak.num(jets) == 2]
    >>> selected = process(events)
    >>> print(dak.necessary_columns(selected))
    {'from-uproot-0e54dc3659a3c020608e28b03f22b0f4': frozenset({'Electron_eta', 'Jet_eta', 'nElectron', 'Jet_pt', 'Electron_pt', 'nJet'})}

    """

    base_fn: tp.Callable
    label: str | None = None
    token: str | None = None
    meta: tp.Any | None = None
    traverse: bool = True
    # additional options that are not available in dak.map_partitions
    needs: tp.Mapping | None = None
    out_like: tp.Any = None

    def __post_init__(self) -> None:
        if self.needs is not None and not isinstance(self.needs, tp.Mapping):
            # this is reachable, mypy doesn't understand this
            msg = (  # type: ignore[unreachable]
                "'needs' argument must be a mapping where the keys "
                "point to input argument dask_awkward arrays and the values "
                "to columns/slices that should be touched explicitly, "
                f"got '{self.needs!r}' instead.\n\n"
                "Exemplary usage:\n"
                "\n@partial(mapfilter, needs={'array': ['col1', 'ecol2']})"
                "\ndef process(array: ak.Array) -> ak.Array:"
                "\n  return array.col1 + array.col2"
            )
            raise ValueError(msg)

    def wrapped_fn(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        import inspect

        ba = inspect.signature(self.base_fn).bind(*args, **kwargs)
        in_arguments = ba.arguments
        if self.needs is not None:
            tobe_touched = set()
            for arg in self.needs.keys():
                if arg in in_arguments:
                    tobe_touched.add(arg)
                else:
                    msg = f"Argument '{arg}' is not present in the function signature."
                    raise ValueError(msg)
            for arg in tobe_touched:
                array = in_arguments[arg]
                if not isinstance(array, ak.Array):
                    raise ValueError(
                        f"Can only touch columns of an awkward array, got {array}."
                    )
                if ak.backend(array) == "typetracer":
                    # touch the objects explicitly
                    for slce in self.needs[arg]:
                        ak.typetracer.touch_data(array[slce])
        if self.out_like is not None:
            # check if we're in the typetracing step
            if any(
                ak.backend(array) == "typetracer" for array in in_arguments.values()
            ):
                # mock the output as the specified type
                if isinstance(self.out_like, (tuple, list)):
                    output = []
                    for out in self.out_like:
                        if isinstance(out, ak.Array):
                            if not ak.backend(out) == "typetracer":
                                out = ak.Array(
                                    out.layout.to_typetracer(forget_length=True)
                                )
                            output.append(out)
                        else:
                            output.append(out)
                    return tuple(output)
                else:
                    if isinstance(self.out_like, ak.Array):
                        if not ak.backend(self.out_like) == "typetracer":
                            return ak.Array(
                                self.out_like.layout.to_typetracer(forget_length=True)
                            )
                        return self.out_like
                    else:
                        raise ValueError(
                            "out_like must be an awkward array in the single return value case."
                        )
        return self.base_fn(*args, **kwargs)

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        fn, arg_flat_deps_expanded, kwarg_flat_deps = _to_packed_fn_args(
            self.wrapped_fn, *args, traverse=self.traverse, **kwargs
        )

        hlg, meta, deps, name = _map_partitions_prepare(
            fn,
            *arg_flat_deps_expanded,
            *kwarg_flat_deps,
            label=self.label,
            token=self.token,
            meta=self.meta,
            output_divisions=None,
        )

        # check consistent partitioning
        # needs to be implemented
        # how to get the (correct) partitioning from the deps (any dask collection)?
        if len(deps) == 0:
            raise ValueError("Need at least one input that is a dask collection.")
        elif len(deps) == 1:
            npart = deps[0].npartitions
        else:
            npart = deps[0].npartitions
            if not all(dep.npartitions == npart for dep in deps):
                msg = "All inputs must have the same partitioning, got:"
                for dep in deps:
                    npartitions = dep.npartitions
                    msg += f"\n{dep} = {npartitions=}"
                raise ValueError(msg)

        return _multi_return_map_partitions(
            hlg=hlg,
            name=name,
            meta=meta,
            npartitions=npart,
        )
