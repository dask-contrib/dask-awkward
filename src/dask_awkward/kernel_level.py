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

    @staticmethod
    def _graph_item(nplike, graph, item):
        if isinstance(item, DaskNode):
            item._graph_node(nplike, graph)
            return item.raw_id
        else:
            return item

    @staticmethod
    def _graph_args(nplike, graph, args, kwargs):
        out = []
        for arg in args:
            out.append(DaskNode._graph_item(nplike, graph, arg))

        num = len(out)
        kws = []
        if kwargs is not None:
            for kw, arg in kwargs.items():
                kws.append(kw)
                out.append(DaskNode._graph_item(nplike, graph, arg))

        return out, num, kws


class DaskNodeKernelCall(DaskNode):
    def __init__(self, name_and_types, args):
        self.name_and_types = name_and_types
        self.args = args
        self.error_handler = None

        for argdir, arg in zip(ak._cpu_kernels.kernel[name_and_types].dir, args):
            if argdir != "in":
                arg.mutations.append(self)

    @property
    def raw_id(self):
        if self.error_handler is None:
            return (self._id_name(""), id(self))
        else:
            return (self._id_name(":raw"), id(self))

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


def prohibit(predicate, consequence):
    if predicate:
        consequence()


class DaskNodeCall(DaskNode):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        self.mutations = []

        name = self._id_name("")
        if name == ".__getitem__":
            if (
                isinstance(self.node._ndim, tuple)
                and isinstance(self.node._type, tuple)
                and isinstance(self.args[0], numbers.Integral)
            ):
                self._ndim = self.node._ndim[self.args[0]]
                self._type = self.node._type[self.args[0]]
            elif self.node._ndim is not None and self.node._type is not None:
                # We only ever do internal __getitem__ with int or slice
                if isinstance(self.args[0], numbers.Integral):
                    self._ndim = self.node._ndim - 1
                else:
                    self._ndim = self.node._ndim
                self._type = self.node._type
                assert self._ndim >= 0
            else:
                self._ndim, self._type = None, None

        elif name in propagate_type:
            self._ndim, self._type = propagate_type[name](*args, **kwargs)

        else:
            self._ndim, self._type = None, None

    @property
    def raw_id(self):
        if len(self.mutations) == 0:
            return (self._id_name(""), id(self))
        else:
            return (self._id_name(":raw"), id(self))

    @property
    def nplike(self):
        return DaskTrace.instance()

    def __len__(self):
        return DaskNodeModuleCall((self,), {}, "shape", ())[0]

    def __getattr__(self, name):
        if name == "ndim":
            if self._ndim is None:
                raise TypeError("delayed computation is not known to be an array")
            return self._ndim

        elif name == "shape":
            return DaskNodeModuleCall((self,), {}, "shape", ())

        elif name == "dtype":
            if not isinstance(self._type, np.dtype):
                raise TypeError("delayed computation is not known to be an array")
            return self._type

        else:
            return DaskTraceMethodCall(self, name)

    def __getitem__(self, where):
        return DaskNodeMethodCall((where,), {}, self, "__getitem__")

    def __add__(self, other):
        return DaskNodeMethodCall((other,), {}, self, "__add__")


class DaskNodeModuleCall(DaskNodeCall):
    def __init__(self, args, kwargs, name, submodule):
        self.name = name
        self.submodule = submodule
        super().__init__(args, kwargs)

        out = kwargs.get("out", None)   # all functions we use are consistent about this
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
                function = lambda *a: direct_function(*a[:num], **dict(zip(kws, a[num:])))

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
        return "." + self.name + suffix

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
                function = lambda s, *a: getattr(s, name)(*a[:num], **dict(zip(kws, a[num:])))

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

    @staticmethod
    def prohibit(*args):
        out = DaskNodeModuleCall(args, {}, "prohibit", ())
        for arg in args:
            if isinstance(arg, DaskNodeCall):
                arg.mutations.append(out)
        return out

    @staticmethod
    def postpone(*args):
        out = DaskNodeModuleCall(args, {}, "postpone", ())
        for arg in args:
            if isinstance(arg, DaskNodeCall):
                arg.mutations.append(out)
        return out

    @staticmethod
    def slice(*args):
        return DaskNodeModuleCall(args, {}, "slice", ())

    _dict = {
        "array_str": lambda array, *args, **kwargs: "[delayed]",
    }

    def __getattribute__(self, name):
        if name in DaskTrace._dict:
            return DaskTrace._dict[name]
        elif name == "ma":
            return DaskTraceSubmodule(("ma",))
        elif name == "char":
            return DaskTraceSubmodule(("char",))
        else:
            return DaskTraceModuleCall(name, ())


DaskTrace._dict["instance"] = DaskTrace.instance
DaskTrace._dict["prohibit"] = DaskTrace.prohibit
DaskTrace._dict["slice"] = DaskTrace.slice


class DaskTraceSubmodule:
    def __init__(self, submodule):
        self.submodule = submodule

    def __getattr__(self, name):
        return DaskTraceModuleCall(name, self.submodule)


propagate_type = {}


def propagate_type_array(data, dtype=None, copy=None):
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        raise TypeError("delayed array ndim must be known")
    if dtype is None:
        dtype = getattr(data, "dtype", None)
        if dtype is None:
            raise TypeError("delayed array dtype must be specified")
    return ndim, np.dtype(dtype)


propagate_type["array"] = propagate_type_array


def propagate_type_asarray(data, dtype=None, order=None):
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        raise TypeError("delayed array ndim must be known")
    if dtype is None:
        dtype = getattr(data, "dtype", None)
        if dtype is None:
            raise TypeError("delayed array dtype must be specified")
    return ndim, np.dtype(dtype)


propagate_type["asarray"] = propagate_type_asarray
propagate_type["ascontiguousarray"] = propagate_type_asarray


def propagate_type_frombuffer(data, dtype=np.float64):
    return 1, np.dtype(dtype)


propagate_type["frombuffer"] = propagate_type_frombuffer


def propagate_type_zeros(shape, dtype=np.float64):
    if isinstance(shape, numbers.Integral):
        shape = (shape,)
    return len(shape), np.dtype(dtype)


propagate_type["zeros"] = propagate_type_zeros
propagate_type["ones"] = propagate_type_zeros
propagate_type["empty"] = propagate_type_zeros


def propagate_type_full(shape, value, dtype=None):
    if isinstance(shape, numbers.Integral):
        shape = (shape,)
    if dtype is None:
        dtype = numpy.array(value).dtype
    return len(shape), np.dtype(dtype)


propagate_type["full"] = propagate_type_full


def propagate_type_zeros_like(array, *args, **kwargs):
    return array.ndim, array.dtype


propagate_type["zeros_like"] = propagate_type_zeros_like
propagate_type["ones_like"] = propagate_type_zeros_like
propagate_type["full_like"] = propagate_type_zeros_like


def propagate_type_arange(*args, dtype=np.int64):
    return 1, np.dtype(dtype)


propagate_type["arange"] = propagate_type_arange


def propagate_type_meshgrid(*arrays, indexing="ij"):
    return tuple([(x.ndim, x.dtype) for x in arrays])


propagate_type["meshgrid"] = propagate_type_meshgrid


def propagate_type_searchsorted(haystack, needle, side=None):
    return needle.ndim, np.dtype(np.int64)


propagate_type["searchsorted"] = propagate_type_searchsorted


def propagate_type_argsort(array):
    return array.ndim, np.dtype(np.int64)


propagate_type["argsort"] = propagate_type_argsort


def propagate_type_broadcast_arrays(*arrays):
    ndim = max(x.ndim for x in arrays)
    return (ndim,) * len(arrays), tuple([(x.dtype) for x in arrays])


propagate_type["broadcast_arrays"] = propagate_type_broadcast_arrays


def propagate_type_add(array1, array2, out=None):
    ndim = max(array1.ndim, array2.ndim)
    if out is None:
        return ndim, (np.array([], array1.dtype) + np.array([], array2.dtype)).dtype
    else:
        return ndim, out.dtype


propagate_type["add"] = propagate_type_add


def propagate_type_cumsum(array, out=None):
    if out is None:
        return 1, array.dtype
    else:
        return out.ndim, out.dtype


propagate_type["cumsum"] = propagate_type_cumsum
propagate_type["cumprod"] = propagate_type_cumsum


def propagate_type_nonzero(array):
    # Awkward only ever uses this function on 1D arrays
    return 1, np.dtype(np.int64)


propagate_type["nonzero"] = propagate_type_nonzero


# FIXME: next is unique
