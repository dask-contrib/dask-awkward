from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, Union, cast

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import MaterializedLayer
from dask.layers import DataFrameTreeReduction
from typing_extensions import TypeAlias

from dask_awkward.utils import LazyInputsDict

if TYPE_CHECKING:
    from awkward import Array as AwkwardArray
    from awkward._nplikes.typetracer import TypeTracerReport


BackendT: TypeAlias = Union[Literal["cpu"], Literal["jax"], Literal["cuda"]]


class AwkwardBlockwiseLayer(Blockwise):
    """Just like upstream Blockwise, except we override pickling"""

    has_been_unpickled: bool = False

    @classmethod
    def from_blockwise(cls, layer: Blockwise) -> AwkwardBlockwiseLayer:
        ob = object.__new__(cls)
        ob.__dict__.update(layer.__dict__)
        return ob

    def __getstate__(self) -> dict:
        # Indicator that this layer has been serialised
        state = self.__dict__.copy()
        state["has_been_unpickled"] = True
        return state

    def __repr__(self) -> str:
        return "Awkward" + super().__repr__()


class ImplementsIOFunction(Protocol):
    def __call__(self, *args, **kwargs): ...


T = TypeVar("T")


class ImplementsMockEmpty(ImplementsIOFunction, Protocol):
    def mock_empty(self, backend: BackendT) -> AwkwardArray: ...


class ImplementsReport(ImplementsIOFunction, Protocol):
    @property
    def return_report(self) -> bool: ...


class ImplementsProjection(Protocol[T]):
    def project(self, report: TypeTracerReport, state: T) -> ImplementsIOFunction: ...


class ImplementsNecessaryColumns(ImplementsProjection[T], Protocol):
    def necessary_columns(
        self, report: TypeTracerReport, state: T
    ) -> frozenset[str]: ...


class IOFunctionWithMocking(ImplementsIOFunction):
    def __init__(self, meta: AwkwardArray, io_func: ImplementsIOFunction):
        self._meta = meta
        self._io_func = io_func

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_meta"] = None
        return state

    def __call__(self, *args, **kwargs):
        return self._io_func(*args, **kwargs)


def io_func_implements_projection(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "prepare_for_projection")


def io_func_implements_columnar(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "necessary_columns")


def io_func_implements_report(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "return_report")


class AwkwardTokenizable:

    def __init__(self, ret_val, parent_name):
        self.parent_name = parent_name
        self.ret_val = ret_val

    def __dask_tokenize__(self):
        return ("AwkwardTokenizable", self.parent_name)

    def __call__(self, *_, **__):
        return self.ret_val


class AwkwardInputLayer(AwkwardBlockwiseLayer):
    """A layer known to perform IO and produce Awkward arrays

    We specialise this so that we have a way to prune column selection on load
    """

    def __init__(
        self,
        *,
        name: str,
        inputs: Any,
        io_func: ImplementsIOFunction,
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
        return (
            io_func_implements_projection(self.io_func) and not self.has_been_unpickled
        )

    @property
    def is_columnar(self) -> bool:
        return io_func_implements_columnar(self.io_func)

    def project(
        self,
        report: TypeTracerReport,
        state: T,
    ) -> AwkwardInputLayer:
        assert self.is_projectable
        io_func = cast(ImplementsProjection, self.io_func).project(
            report=report, state=state
        )
        return AwkwardInputLayer(
            name=self.name,
            inputs=self.inputs,
            io_func=io_func,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )

    def necessary_columns(self, report: TypeTracerReport, state: T) -> frozenset[str]:
        assert self.is_columnar
        return cast(ImplementsNecessaryColumns, self.io_func).necessary_columns(
            report=report, state=state
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


class AwkwardTreeReductionLayer(DataFrameTreeReduction): ...
