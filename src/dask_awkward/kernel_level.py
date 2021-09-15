"""
This module implements Dask task graphs at Awkward Array's kernel level.

Graph nodes are 1D array and kernel calls, rather than user-level functions (ak.*).
"""

import numbers

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def _run(symbols, graph, which):
    function, *delayed_args = graph[which]
    args = []
    for x in delayed_args:
        if (
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[0], str)
            and isinstance(x[1], int)
        ):
            if x not in symbols:
                symbols[x] = _run(symbols, graph, x)
            args.append(symbols[x])
        else:
            args.append(x)

    return function(*args)


def run(graph, which):
    symbols = {}
    return _run(symbols, graph, which)


class DaskNode:
    def graph(self, nplike):
        graph = {}
        self._graph_node(nplike, graph)
        return graph

    @property
    def id(self):
        return (self._id_name(""), id(self))

    @property
    def raw_id(self):
        return (self._id_name(":raw"), id(self))

    @staticmethod
    def _graph_args(nplike, graph, args, kwargs):
        out = []
        for arg in args:
            if isinstance(arg, DaskNode):
                arg._graph_node(nplike, graph)
                out.append(arg.raw_id)
            else:
                out.append(arg)

        num = len(out)
        kws = []
        if kwargs is not None:
            for kw, arg in kwargs.items():
                kws.append(kw)
                if isinstance(arg, DaskNode):
                    arg._graph_node(nplike, graph)
                    out.append(arg.raw_id)
                else:
                    out.append(arg)

        return out, num, kws


class DaskNodeKernelCall(DaskNode):
    def __init__(self, name_and_types, args):
        self.name_and_types = name_and_types
        self.args = args
        self.error_handler = None

        for argdir, arg in zip(ak._cpu_kernels.kernel[name_and_types].dir, args):
            if argdir != "in":
                arg.mutations.append(self)

    def handle_error(self, error_handler):
        self.error_handler = error_handler

    def _id_name(self, suffix):
        n = [self.name_and_types[0]]
        ts = [str(np.dtype(t)) for t in self.name_and_types[1:]]
        return ":".join(n + ts) + suffix

    def _graph_node(self, nplike, graph):
        self_id = self.id
        if self_id not in graph:
            graph[self_id] = None   # prevent infinite recursion

            args, _, _ = self._graph_args(nplike, graph, self.args, None)
            kernel = nplike[self.name_and_types]

            if self.error_handler is None:
                graph[self_id] = (kernel,) + tuple(args)
            else:
                raw_id = self.raw_id
                graph[raw_id] = (kernel,) + tuple(args)
                graph[self_id] = (self.error_handler, raw_id)


def mutations(target, *steps):
    return target


class DaskNodeCall(DaskNode):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        self.mutations = []

        name = self._id_name("")
        if name in propagate_shape_dtype:
            self.shape, self.dtype = propagate_shape_dtype[name](*args, **kwargs)
            if self.dtype is not None:
                self.dtype = np.dtype(self.dtype)
        else:
            self.shape, self.dtype = None, None

    @property
    def nplike(self):
        return DaskTrace.instance()

    def __len__(self):
        if self.shape is None:
            raise TypeError("delayed object is not known to be an array")

    def __getattr__(self, name):
        return DaskTraceMethodCall(self, name)


class DaskNodeModuleCall(DaskNodeCall):
    def __init__(self, args, kwargs, name, submodule):
        self.name = name
        self.submodule = submodule
        super().__init__(args, kwargs)

        out = kwargs.get("out", None)
        if out is not None:
            out.mutations.append(self)

    def _id_name(self, suffix):
        return "".join(x + "." for x in self.submodule) + self.name + suffix

    def _graph_node(self, nplike, graph):
        self_id = self.id
        if self_id not in graph:
            graph[self_id] = None   # prevent infinite recursion

            args, num, kws = self._graph_args(nplike, graph, self.args, self.kwargs)

            module = nplike
            for x in self.submodule:
                module = getattr(module, x)

            direct_function = getattr(module, self.name)
            if len(kws) == 0:
                function = direct_function
            else:
                function = lambda *a: direct_function(*a[:num], **dict(kws, a[num:]))

            if len(self.mutations) == 0:
                graph[self_id] = (function,) + tuple(args)

            else:
                raw_id = self.raw_id
                graph[raw_id] = (function,) + tuple(args)

                mutargs = []
                for mut in self.mutations:
                    mut._graph_node(nplike, graph)
                    mutargs.append(mut.id)
                graph[self_id] = (mutations, raw_id) + tuple(mutargs)


class DaskNodeMethodCall(DaskNodeCall):
    def __init__(self, args, kwargs, node, name):
        self.node = node
        self.name = name
        super().__init__(args, kwargs)

    def _id_name(self, suffix):
        return "ndarray." + self.name + suffix

    def _graph_node(self, nplike, graph):
        self_id = self.id
        if self_id not in graph:
            graph[self_id] = None   # prevent infinite recursion

            name = self.name
            self_arg, _, _ = self._graph_args(nplike, graph, (self.node,), None)
            args, num, kws = self._graph_args(nplike, graph, self.args, self.kwargs)

            if len(kws) == 0:
                function = lambda s, *a: getattr(s, name)(*a)
            else:
                function = lambda s, *a: getattr(s, name)(*a[:num], **dict(kws, a[num:]))

            if len(self.mutations) == 0:
                graph[self_id] = (function, self_arg[0]) + tuple(args)

            else:
                raw_id = self.raw_id
                graph[raw_id] = (function, self_arg[0]) + tuple(args)

                mutargs = []
                for mut in self.mutations:
                    mut._graph_node(nplike, graph)
                    mutargs.append(mut.id)
                graph[self_id] = (mutations, raw_id) + tuple(mutargs)


class DaskTraceKernelCall:
    def __init__(self, name_and_types):
        self.name_and_types = name_and_types

    def __call__(self, *args):
        return DaskNodeKernelCall(self.name_and_types, args)


class DaskTraceModuleCall:
    def __init__(self, name, submodule):
        self.name = name
        self.submodule = submodule

    def __call__(self, *args, **kwargs):
        return DaskNodeModuleCall(args, kwargs, self.name, submodule=self.submodule)


class DaskTraceMethodCall:
    def __init__(self, node, name):
        self.node = node
        self.name = name

    def __call__(self, *args, **kwargs):
        return DaskNodeMethodCall(args, kwargs, self.node, self.name)


class DaskTrace(ak.nplike.NumpyLike):
    def __init__(self):
        self._module = None

    def __getitem__(self, name_and_types):
        return DaskTraceKernelCall(name_and_types)

    def asarray(self, data):
        raise NotImplementedError

    def __getattribute__(self, name):
        if name == "ma":
            return DaskTraceSubmodule(("ma",))
        elif name == "char":
            return DaskTraceSubmodule(("char",))
        else:
            return DaskTraceModuleCall(name, ())


class DaskTraceSubmodule:
    def __init__(self, submodule):
        self.submodule = submodule

    def __getattr__(self, name):
        return DaskTraceModuleCall(name, self.submodule)


propagate_shape_dtype = {}

UNSPECIFIED = object()


def propagate_array(data, dtype=UNSPECIFIED, copy=UNSPECIFIED):
    shape = getattr(data, "shape", UNSPECIFIED)
    if shape is UNSPECIFIED:
        shape = (len(data),)
    if dtype is UNSPECIFIED:
        dtype = getattr(data, "dtype", UNSPECIFIED)
        if dtype is UNSPECIFIED:
            raise TypeError("delayed array dtype must be specified")
    return shape, dtype


propagate_shape_dtype["array"] = propagate_array


def propagate_asarray(data, dtype=UNSPECIFIED, order="C"):
    shape = getattr(data, "shape", UNSPECIFIED)
    if shape is UNSPECIFIED:
        shape = (len(data),)
    if dtype is UNSPECIFIED:
        dtype = getattr(data, "dtype", UNSPECIFIED)
        if dtype is UNSPECIFIED:
            raise TypeError("delayed array dtype must be specified")
    return shape, dtype


propagate_shape_dtype["asarray"] = propagate_asarray


def propagate_ascontiguousarray(data, dtype=UNSPECIFIED):
    shape = getattr(data, "shape", UNSPECIFIED)
    if shape is UNSPECIFIED:
        shape = (len(data),)
    if dtype is UNSPECIFIED:
        dtype = getattr(data, "dtype", UNSPECIFIED)
        if dtype is UNSPECIFIED:
            raise TypeError("delayed array dtype must be specified")
    return shape, dtype


propagate_shape_dtype["ascontiguousarray"] = propagate_ascontiguousarray


def propagate_frombuffer(data, dtype=np.float64):
    shape = (len(data) // dtype.itemsize,)
    return shape, dtype


propagate_shape_dtype["frombuffer"] = propagate_frombuffer


def propagate_zeros(shape, dtype=np.float64):
    if isinstance(shape, numbers.Integral):
        shape = (shape,)
    return shape, dtype


propagate_shape_dtype["zeros"] = propagate_zeros


def propagate_ones(shape, dtype=np.float64):
    if isinstance(shape, numbers.Integral):
        shape = (shape,)
    return shape, dtype


propagate_shape_dtype["ones"] = propagate_ones


def propagate_empty(shape, dtype=np.float64):
    if isinstance(shape, numbers.Integral):
        shape = (shape,)
    return shape, dtype


propagate_shape_dtype["empty"] = propagate_empty


def propagate_full(shape, value, dtype=UNSPECIFIED):
    if isinstance(shape, numbers.Integral):
        shape = (shape,)
    if dtype is UNSPECIFIED:
        dtype = numpy.array(value).dtype
    return shape, dtype


propagate_shape_dtype["full"] = propagate_full


def propagate_zeros_like(array):
    return array.shape, array.dtype


propagate_shape_dtype["zeros_like"] = propagate_zeros_like


def propagate_ones_like(array):
    return array.shape, array.dtype


propagate_shape_dtype["ones_like"] = propagate_ones_like


def propagate_full_like(array):
    return array.shape, array.dtype


propagate_shape_dtype["full_like"] = propagate_full_like


def propagate_arange(arg1, arg2=UNSPECIFIED, arg3=UNSPECIFIED, dtype=np.int64):
    if (
        isinstance(arg1, numbers.Integral)
        and arg2 is UNSPECIFIED
        and arg3 is UNSPECIFIED
    ):
        shape = (arg1,)
    elif (
        isinstance(arg1, numbers.Integral)
        and isinstance(arg2, numbers.Integral)
        and arg3 is UNSPECIFIED
    ):
        shape = (arg2 - arg1,)
    elif (
        isinstance(arg1, numbers.Integral)
        and isinstance(arg2, numbers.Integral)
        and isinstance(arg3, numbers.Integral)
    ):
        shape = ((arg2 - arg1) // arg3,)
    return shape, dtype


propagate_shape_dtype["arange"] = propagate_arange


# FIXME: meshgrid returns a tuple


def propagate_searchsorted(haystack, needle, side="left"):
    return needle.shape, np.int64


propagate_shape_dtype["searchsorted"] = propagate_searchsorted


def propagate_argsort(array):
    return array.shape, np.int64


propagate_shape_dtype["argsort"] = propagate_argsort


# FIXME: broadcast_arrays returns a tuple


def propagate_add(array1, array2, out=None):
    shape = getattr(array1, "shape", None)
    if shape is None or all(x is None for x in shape):
        if getattr(array2, "shape", None) is not None:
            shape = array2
    dtype = (numpy.array([], array1.dtype) + numpy.array([], array2.dtype)).dtype
    return shape, dtype


propagate_shape_dtype["add"] = propagate_add


def propagate_cumsum(array, out=None):
    return array.shape, array.dtype


propagate_shape_dtype["cumsum"] = propagate_cumsum


def propagate_cumprod(array, out=None):
    return array.shape, array.dtype


propagate_shape_dtype["cumprod"] = propagate_cumprod
