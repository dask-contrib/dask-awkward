"""
This module implements Dask task graphs at Awkward Array's kernel level.

Graph nodes are 1D array and kernel calls, rather than user-level functions (ak.*).
"""

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def run(graph, which):
    function, *delayed_args = graph[which]
    args = []
    for x in delayed_args:
        if (
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[0], str)
            and isinstance(x[1], int)
        ):
            args.append(run(graph, x))
        else:
            args.append(x)
    return function(*args)


class DaskNode:
    def graph(self, nplike):
        graph = {}
        self._graph_node(nplike, graph)
        return graph

    @property
    def id(self):
        return (self._id_name(""), id(self))

    @staticmethod
    def _graph_args(nplike, graph, args, kwargs):
        out = []
        for arg in args:
            if isinstance(arg, DaskNode):
                arg._graph_node(nplike, graph)
                out.append(arg.id)
            else:
                out.append(arg)

        num = len(out)
        kws = []
        if kwargs is not None:
            for kw, arg in kwargs.items():
                kws.append(kw)
                if isinstance(arg, DaskNode):
                    arg._graph_node(nplike, graph)
                    out.append(arg.id)
                else:
                    out.append(arg)

        return out, num, kws


class DaskNodeKernelCall(DaskNode):
    def __init__(self, name_and_types, args):
        self.name_and_types = name_and_types
        self.args = args
        self.error_handler = None

    def handle_error(self, error_handler):
        self.error_handler = error_handler

    def _id_name(self, suffix):
        n = [self.name_and_types[0]]
        ts = [str(np.dtype(t)) for t in self.name_and_types[1:]]
        return ":".join(n + ts) + suffix

    def _graph_node(self, nplike, graph):
        self_id = self.id
        if self_id not in graph:
            args, _, _ = self._graph_args(nplike, graph, self.args, None)
            kernel = nplike[self.name_and_types]

            if self.error_handler is None:
                graph[self_id] = (kernel,) + tuple(args)
            else:
                raw_id = (self._id_name(":raw"), id(self))
                graph[raw_id] = (kernel,) + tuple(args)
                graph[self_id] = (self.error_handler, raw_id)


class DaskNodeCall(DaskNode):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        self.mutations = []  # FIXME: any kernel for which this is an "out" is a dependency

    @property
    def nplike(self):
        return DaskTrace.instance()

    def __getattr__(self, name):
        return DaskTraceMethodCall(self, name)


class DaskNodeModuleCall(DaskNodeCall):
    def __init__(self, args, kwargs, name, submodule):
        super().__init__(args, kwargs)
        self.name = name
        self.submodule = submodule

    def _id_name(self, suffix):
        return self.name + suffix

    def _graph_node(self, nplike, graph):
        self_id = self.id
        if self_id not in graph:
            args, num, kws = self._graph_args(nplike, graph, self.args, self.kwargs)

            module = nplike
            for x in self.submodule:
                module = getattr(module, x)

            direct_function = getattr(module, self.name)
            if len(kws) == 0:
                function = direct_function
            else:
                function = lambda *a: direct_function(*a[:num], **dict(kws, a[num:]))

            graph[self_id] = (function,) + tuple(args)


class DaskNodeMethodCall(DaskNodeCall):
    def __init__(self, args, kwargs, node, name):
        super().__init__(args, kwargs)
        self.node = node
        self.name = name

    def _id_name(self):
        return "ndarray." + self.name + suffix

    def _graph_node(self, nplike, graph):
        self_id = self.id
        if self_id not in graph:
            name = self.name
            self_arg, _, _ = self._graph_args(nplike, graph, (self.node,), None)
            args, num, kws = self._graph_args(nplike, graph, self.args, self.kwargs)

            if len(kws) == 0:
                function = lambda s, *a: getattr(s, name)(*a)
            else:
                function = lambda s, *a: getattr(s, name)(*a[:num], **dict(kws, a[num:]))

            graph[self_id] = (function, self_arg[0]) + tuple(args)


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
