from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import MaterializedLayer
from dask.layers import DataFrameTreeReduction

from dask_awkward.utils import LazyInputsDict

if TYPE_CHECKING:
    from awkward import Array as AwkwardArray


class AwkwardBlockwiseLayer(Blockwise):
    """Just like upstream Blockwise, except we override pickling"""

    @classmethod
    def from_blockwise(cls, layer: Blockwise) -> AwkwardBlockwiseLayer:
        ob = object.__new__(cls)
        ob.__dict__.update(layer.__dict__)
        return ob

    def mock(self) -> AwkwardBlockwiseLayer:
        layer = copy.copy(self)
        nb = layer.numblocks
        layer.numblocks = {k: tuple(1 for _ in v) for k, v in nb.items()}
        layer.__dict__.pop("_dims", None)
        return layer

    def __repr__(self) -> str:
        return "Awkward" + super().__repr__()

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        import pickle

        try:
            pickle.dumps(d["_meta"])
        except (ValueError, TypeError, KeyError):
            print("POP META", self)
            d.pop(
                "_meta", None
            )  # must be a typetracer, does not pickle and not needed on scheduler
        return d


T = TypeVar("T")


class ImplementsIOFunction(Protocol):
    def __call__(self, *args, **kwargs) -> AwkwardArray:
        ...


class ImplementsProjection(Protocol):
    @property
    def meta(self) -> AwkwardArray:
        ...

    def prepare_for_projection(self) -> tuple[AwkwardArray, T]:
        ...

    def project(self, state: T) -> ImplementsIOFunction:
        ...


# IO functions may not end up performing buffer projection, so they
# should also support directly returning the result
class ImplementsIOFunctionWithProjection(
    ImplementsProjection, ImplementsIOFunction, Protocol
):
    ...


class IOFunctionWithMeta(ImplementsIOFunctionWithProjection):
    def __init__(self, meta: AwkwardArray, io_func: ImplementsIOFunction):
        self._meta = meta
        self._io_func = io_func

    def __call__(self, *args, **kwargs) -> AwkwardArray:
        return self._io_func(*args, **kwargs)

    @property
    def meta(self):
        return self._meta

    def prepare_for_projection(self) -> tuple[AwkwardArray, None]:
        return self._meta, None

    def project(self, state: None):
        return self._io_func


def io_func_implements_project(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "project")


class AwkwardInputLayer(AwkwardBlockwiseLayer):
    """A layer known to perform IO and produce Awkward arrays

    We specialise this so that we have a way to prune column selection on load
    """

    def __init__(
        self,
        *,
        name: str,
        inputs: Any,
        io_func: ImplementsIOFunction | ImplementsIOFunctionWithProjection,
        label: str | None = None,
        produces_tasks: bool = False,
        creation_info: dict | None = None,
        annotations: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.inputs = inputs
        self.io_func = io_func
        self.label = label
        self.produces_tasks = produces_tasks
        self.annotations = annotations
        self.creation_info = creation_info

        io_arg_map = BlockwiseDepDict(
            mapping=LazyInputsDict(self.inputs),  # type: ignore
            produces_tasks=self.produces_tasks,
        )

        super().__init__(
            output=self.name,
            output_indices="i",
            dsk={name: (self.io_func, blockwise_token(0))},
            indices=[(io_arg_map, "i")],
            numblocks={},
            annotations=None,
        )

    def __repr__(self) -> str:
        return f"AwkwardInputLayer<{self.output}>"

    @property
    def is_projectable(self) -> bool:
        # isinstance(self.io_func, ImplementsProjection)
        return io_func_implements_project(self.io_func)

    def mock(self) -> tuple[AwkwardInputLayer, T]:
        assert self.is_projectable
        new_meta_array, state = self.io_func.prepare_for_projection()

        new_input_layer = AwkwardInputLayer(
            name=self.name,
            inputs=[None][: int(list(self.numblocks.values())[0][0])],
            io_func=lambda *_, **__: new_meta_array,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )
        return new_input_layer, state

    def project(
        self,
        state: T,
    ):
        return AwkwardInputLayer(
            name=self.name,
            inputs=self.inputs,
            io_func=self.io_func.project(state=state),
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )


class AwkwardMaterializedLayer(MaterializedLayer):
    def __init__(
        self,
        mapping: dict,
        *,
        previous_layer_names: list[str],
        fn: Callable | None = None,
        **kwargs: Any,
    ):
        self.previous_layer_names: list[str] = previous_layer_names
        self.fn = fn
        super().__init__(mapping, **kwargs)

    def mock(self) -> MaterializedLayer:
        mapping = self.mapping.copy()
        if not mapping:
            # no partitions at all
            return self
        name = next(iter(mapping))[0]

        # one previous layer name
        #
        # this case is used for mocking repartition or slicing where
        # we maybe have multiple partitions that need to be included
        # in a task.
        if len(self.previous_layer_names) == 1:
            prev_name: str = self.previous_layer_names[0]
            if (name, 0) in mapping:
                task = mapping[(name, 0)]
                task = tuple(
                    (prev_name, 0)
                    if isinstance(v, tuple) and len(v) == 2 and v[0] == prev_name
                    else v
                    for v in task
                )

                # when using Array.partitions we need to mock that we
                # just want the first partition.
                if len(task) == 2 and task[1] > 0:
                    task = (task[0], 0)
                return MaterializedLayer({(name, 0): task})
            return self

        # more than one previous_layer_names
        #
        # this case is needed for dak.concatenate on axis=0; we need
        # the first partition of _each_ of the previous layer names!
        else:
            if self.fn is None:
                raise ValueError(
                    "For multiple previous layers the fn argument cannot be None."
                )
            name0s = tuple((name, 0) for name in self.previous_layer_names)
            task = (self.fn, *name0s)
            return MaterializedLayer({(name, 0): task})

        # failed to cull during column opt
        return self


class AwkwardTreeReductionLayer(DataFrameTreeReduction):
    def mock(self) -> AwkwardTreeReductionLayer:
        return AwkwardTreeReductionLayer(
            name=self.name,
            name_input=self.name_input,
            npartitions_input=1,
            concat_func=self.concat_func,
            tree_node_func=self.tree_node_func,
            finalize_func=self.finalize_func,
            split_every=self.split_every,
            tree_node_name=self.tree_node_name,
        )
