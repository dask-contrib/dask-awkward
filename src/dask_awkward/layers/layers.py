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

    def __repr__(self):
        return f"AwkwardIOLayer<{self.output}>"

    def mock(self) -> Any:

        # imported here because it this method should be run _only_ on
        # the Client (which is allowed to import awkward)
        from dask_awkward.lib.core import typetracer_from_form
        import awkward as ak
        import copy

        def _label_form(form, start):
            if form.is_record:
                for field in form.fields:
                    _label_form(form.content(field), f"{start}.{field}" if start else field)
            elif form.is_numpy:
                form.form_key = start
            else:
                _label_form(form.content, start)

        new_meta = typetracer_from_form(
            copy.deepcopy(self._meta.layout.form)
        )
        form = new_meta.layout.form
        _label_form(form, "")
        new_labelled_meta, report = ak._typetracer.typetracer_with_report(
            form
        )
        return AwkwardIOLayer(
            name=self.name,
            columns=self.columns,
            inputs=[None],
            io_func=lambda *_, **__: ak.Array(new_labelled_meta),
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
            meta=new_labelled_meta,
        ), report

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
