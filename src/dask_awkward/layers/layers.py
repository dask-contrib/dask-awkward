from __future__ import annotations

from typing import Any, Callable, TypeVar

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token

from dask_awkward.utils import LazyInputsDict

AwkwardIOLayerT = TypeVar("AwkwardIOLayerT", bound="AwkwardIOLayer")


class AwkwardIOLayer(Blockwise):
    def __init__(
        self,
        name: str,
        columns: str | list[str] | None,
        inputs: Any,
        io_func: Callable,
        label: str | None = None,
        produces_tasks: bool = False,
        creation_info: dict | None = None,
        annotations: dict | None = None,
    ) -> None:
        self.name = name
        self._columns = columns
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

    @property
    def columns(self) -> Any:
        return self._columns

    def project_columns(self, columns: list[str]) -> AwkwardIOLayerT:
        pass
