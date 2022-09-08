from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token

from dask_awkward.utils import LazyInputsDict


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
        annotations: Mapping[str, Any] | None = None,
        meta: Any | None = None,
    ) -> None:
        self.name = name
        self._columns = columns
        self.inputs = inputs
        self.io_func = io_func
        self.label = label
        self.produces_tasks = produces_tasks
        self.annotations = annotations
        self.creation_info = creation_info
        self._meta = meta

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

    def to_mock(self):
        return AwkwardIOLayer(
            name=self.name,
            columns=self.columns,
            inputs=[None for _ in self.inputs],
            io_func=lambda *_, **__: self._meta,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
            meta=self._meta,
        )

    def project_columns(self, columns: list[str]) -> AwkwardIOLayer:
        if hasattr(self.io_func, "project_columns"):
            io_func = self.io_func.project_columns(columns)  # type: ignore
        else:
            io_func = self.io_func

        return AwkwardIOLayer(
            name=self.name,
            columns=columns,
            inputs=self.inputs,
            io_func=io_func,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )
