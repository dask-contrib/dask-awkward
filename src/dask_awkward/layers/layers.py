from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token

from dask_awkward.utils import LazyInputsDict


class AwkwardIOLayer(Blockwise):
    def __init__(
        self,
        *,
        name: str,
        columns: str | list[str] | None,
        inputs: Any,
        io_func: Callable,
        meta: Any,
        label: str | None = None,
        produces_tasks: bool = False,
        creation_info: dict | None = None,
        annotations: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.columns = columns
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

    def project_and_mock(self, columns: list[str]) -> AwkwardIOLayer:

        # imported here because it this method should be run _only_ on
        # the Client (which is allowed to import awkward)
        from dask_awkward.lib.core import typetracer_from_form

        new_meta = typetracer_from_form(
            self._meta.layout.form.select_columns(columns),
        )
        return AwkwardIOLayer(
            name=self.name,
            columns=self.columns,
            inputs=[None],
            io_func=lambda *_, **__: new_meta,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
            meta=self._meta,
        )

    def project_columns(self, columns: list[str]) -> AwkwardIOLayer:
        if hasattr(self.io_func, "project_columns"):
            io_func = self.io_func.project_columns(columns)
            return AwkwardIOLayer(
                name=self.name,
                columns=columns,
                inputs=self.inputs,
                io_func=io_func,
                label=self.label,
                produces_tasks=self.produces_tasks,
                creation_info=self.creation_info,
                annotations=self.annotations,
                meta=self._meta,
            )
        return self
