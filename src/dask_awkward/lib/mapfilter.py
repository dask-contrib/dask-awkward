from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import awkward as ak
from dask.highlevelgraph import HighLevelGraph
from dask.typing import DaskCollection

from dask_awkward.lib.core import Array as DakArray
from dask_awkward.lib.core import (
    _map_partitions_prepare,
    _to_packed_fn_args,
    dak_cache,
    empty_typetracer,
    new_array_object,
    partitionwise_layer,
    to_meta,
    typetracer_array,
)


def _single_return_map_partitions(
    hlg: HighLevelGraph,
    name: str,
    meta: tp.Any,
    npartitions: int,
) -> tp.Any:
    from dask.utils import (
        is_arraylike,
        is_dataframe_like,
        is_index_like,
        is_series_like,
    )

    # ak.Array (this is dak.map_partitions case)
    if isinstance(meta, ak.Array):
        # convert to typetracer if not already
        # this happens when the user provides a concrete array (e.g. np.array)
        # and then wraps it with ak.Array as a return type
        if not ak.backend(meta) == "typetracer":
            meta = ak.to_backend(meta, "typetracer")
        return new_array_object(
            hlg,
            name=name,
            meta=meta,
            npartitions=npartitions,
        )
    # TODO: array, dataframe, series, index
    elif (
        is_arraylike(meta)
        or is_dataframe_like(meta)
        or is_series_like(meta)
        or is_index_like(meta)
    ):
        msg = (
            f"{meta=} is not (yet) supported as return type. If possible, "
            "you can convert it to ak.Array, or wrap it with a python container."
        )
        raise NotImplementedError(msg)
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
            ret.append(
                _single_return_map_partitions(
                    hlg=hlg_pick,
                    name=ith_name,
                    meta=m_pick,
                    npartitions=npartitions,
                )
            )
        return tuple(ret)


def _compare_return_vals(left: tp.Any, right: tp.Any) -> None:
    def cmp(left, right):
        msg = (
            "The provided 'meta' does not match "
            "the output type inferred from the pre-run step; "
            "got {right}, but expected {left}.".format(left=left, right=right)
        )
        if isinstance(left, ak.Array):
            if left.layout.form != right.layout.form:
                raise ValueError(msg)

        else:
            if left != right:
                raise ValueError(msg)

    if isinstance(left, tuple) and isinstance(right, tuple):
        for left_, right_ in zip(left, right):
            cmp(left_, right_)
    else:
        cmp(left, right)


class UntraceableFunctionError(Exception): ...


def _func_args(fun: tp.Callable, *args: tp.Any, **kwargs: tp.Any) -> tp.Mapping:
    import inspect

    ba = inspect.signature(fun).bind(*args, **kwargs)
    return ba.arguments


def reports2needs(reports: tp.Mapping) -> dict:
    import ast
    from collections import defaultdict

    needs = defaultdict(list)
    for arg, report in reports.items():
        # this should maybe be differently treated?
        keys = set(report.shape_touched) | set(report.data_touched)
        for key in keys:
            slce = ast.literal_eval(key)
            # only strings are actual slice paths to columns,
            # `None` or `ints` are path values to non-record array types,
            # see: https://github.com/scikit-hep/awkward/pull/3311
            slce = tuple(it for it in slce if isinstance(it, str))
            needs[arg].append(slce)
    return needs


def _replace_arrays_with_typetracers(meta: tp.Any) -> tp.Any:
    def _to_tracer(meta: tp.Any) -> tp.Any:
        if isinstance(meta, ak.Array):
            if not ak.backend(meta) == "typetracer":
                meta = typetracer_array(meta)
        elif isinstance(meta, DakArray):
            meta = to_meta([meta])
        return meta

    if isinstance(meta, tuple):
        meta = tuple(map(_to_tracer, meta))
    else:
        meta = _to_tracer(meta)
    return meta


def prerun(
    fun: tp.Callable, *args: tp.Any, **kwargs: tp.Any
) -> tuple[tp.Any, tp.Mapping]:
    in_arguments = _func_args(fun, *args, **kwargs)

    # replace ak.Arrays with typetracers and store the reports
    reports = {}
    fun_kwargs = {}
    args_metas = {arg: to_meta([val])[0] for arg, val in in_arguments.items()}

    # can't typetrace if no ak.Arrays are present
    ak_arrays = tuple(filter(lambda x: isinstance(x, ak.Array), args_metas.values()))
    if not ak_arrays:
        return None, {}

    def render_buffer_key(
        form: ak.forms.Form,
        form_key: str,
        attribute: str,
    ) -> str:
        return form_key

    # prepare function arguments
    for arg, val in args_metas.items():
        if isinstance(val, ak.Array):
            if not ak.backend(val) == "typetracer":
                val = typetracer_array(val)
            tracer, report = ak.typetracer.typetracer_with_report(
                val.layout.form_with_key_path(root=()),
                highlevel=True,
                behavior=val.behavior,
                attrs=val.attrs,
                buffer_key=render_buffer_key,
            )
            reports[arg] = report
            fun_kwargs[arg] = tracer
        else:
            fun_kwargs[arg] = val

    # try to run the function once with type tracers
    try:
        out = fun(**fun_kwargs)
    except Exception as err:
        import traceback

        # get line number of where the error occurred in the provided function
        # traceback 0: this function, 1: the provided function, >1: the rest of the stack
        tb = traceback.extract_tb(err.__traceback__)
        line_number = tb[1].lineno

        # add also the reports of the typetracer to the error message,
        # and format them as 'needs' wants it to be
        needs = dict(reports2needs(reports=reports))

        msg = (
            f"This wrapped function '{fun}' is not traceable. "
            f"An error occurred at line {line_number}.\n"
            "'mapfilter' can circumvent this by providing the 'needs' and "
            "'meta' arguments to the decorator.\n"
            "\n- 'needs': mapping where the keys point to input argument "
            "dask_awkward arrays and the values to columns/slices that "
            "should be touched explicitly. The typetracing step could "
            "determine the following necessary columns/slices.\n\n"
            f"Typetracer reported the following 'needs':\n"
            f"{needs}\n"
            "\n- 'meta': value(s) of what the wrapped function would "
            "return. For arrays, only the shape and type matter."
        )
        raise UntraceableFunctionError(msg) from err
    return out, reports


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
    """

    base_fn: tp.Callable
    label: str | None = None
    token: str | None = None
    meta: tp.Any | None = None
    traverse: bool = True
    # additional options that are not available in dak.map_partitions
    needs: tp.Mapping | None = None

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
        in_arguments = _func_args(self.base_fn, *args, **kwargs)
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

        if self.meta is not None:
            ak_arrays = [
                arg for arg in in_arguments.values() if isinstance(arg, ak.Array)
            ]
            if all(ak.backend(arr) == "typetracer" for arr in ak_arrays):
                # if the meta is known, we can use it to skip the tracing step
                return self.meta
        return self.base_fn(*args, **kwargs)

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        fn, arg_flat_deps_expanded, kwarg_flat_deps = _to_packed_fn_args(
            self.wrapped_fn, *args, traverse=self.traverse, **kwargs
        )

        arg_flat_deps_expanded = _replace_arrays_with_typetracers(
            arg_flat_deps_expanded
        )
        kwarg_flat_deps = _replace_arrays_with_typetracers(kwarg_flat_deps)
        meta = _replace_arrays_with_typetracers(self.meta)
        in_typetracing_mode = arg_flat_deps_expanded or kwarg_flat_deps or meta

        try:
            hlg, meta, deps, name = _map_partitions_prepare(
                fn,
                *arg_flat_deps_expanded,
                *kwarg_flat_deps,
                label=self.label,
                token=self.token,
                meta=meta,
                output_divisions=None,
            )
        except Exception as err:
            # if there's a problem with typetracing, we can report it and recommend a 'prerun'
            if in_typetracing_mode:
                fn_args = _func_args(self.base_fn, *args, **kwargs)
                sig_str = ", ".join(f"{k}={v}" for k, v in fn_args.items())
                msg = (
                    f"Failed to trace the function '{self.base_fn}'. "
                    "You can use 'needs' and 'meta' to circumvent this step. "
                    "For this, it might be helpful to do a pre-run of the function:"
                    f"\n\n\tfrom dask_awkward.lib.mapfilter import prerun"
                    f"\n\n\tprerun({self.base_fn.__name__}, {sig_str})"
                    f"\n\nThis may help to infer the correct `needs` for `mapfilter`."
                )
                raise UntraceableFunctionError(msg) from err
            # otherwise, just raise the error - whatever it is
            else:
                raise err from None

        # check consistent partitioning
        if len(deps) == 0:
            raise ValueError("Need at least one input that is a dask collection.")
        elif len(deps) == 1:
            npart = deps[0].npartitions
        else:
            npart = deps[0].npartitions
            if not all(dep.npartitions == npart for dep in deps):
                msg = "All inputs must have the same number of partitions, got:"
                for dep in deps:
                    npartitions = dep.npartitions
                    msg += f"\n{dep}: {npartitions=}"
                raise ValueError(msg)

        return _multi_return_map_partitions(
            hlg=hlg,
            name=name,
            meta=meta,
            npartitions=npart,
        )
