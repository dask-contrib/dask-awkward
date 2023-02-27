from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token

from dask_awkward.utils import LazyInputsDict

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport


class AwkwardInputLayer(Blockwise):
    def __init__(
        self,
        *,
        name: str,
        columns: str | list[str] | None,
        inputs: Any,
        io_func: Callable,
        meta: Any,
        behavior: dict | None,
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
        self._behavior = behavior

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

    def mock(self) -> tuple[AwkwardInputLayer, TypeTracerReport]:
        """Mock the input layer as starting with a dataless typetracer.

        This method is used to create new dask task graphs that
        operate purely on typetracer Arrays (that is, array with
        awkward structure but without real data buffers). This allows
        us to test which parts of a real awkward array will be used in
        a real computation. We do this by running a graph which starts
        with mocked AwkwardInputLayers

        We mock an AwkwardInputLayer in these steps:

        1. Copy the original ``_meta`` form.
        2. Create a new typetracer array from that form.
        3. Take the form from the new typetracer array.
        4. Label the components of the new form.
        5. Pass the new labelled form to the typetracer_with_report
           function from upstream awkward. This creates a report
           object that tells us which buffers in a form get used.
        6. Create a new typetracer array that represents an array that
           would come from a real input layer, and make that the
           result of the input layer.
        7. Return the new layer (which only results in a typetracer
           array) along with the mutable report object.

        When this new layer is added to a dask task graph and that
        graph is computed, the report object will be mutated.
        Inspecting the report object after the compute tells us which
        buffers from the original form would be required for a real
        compute with the same graph.

        Returns
        -------
        AwkwardInputLayer
            Copy of the input layer with data-less input.
        TypeTracerReport
            The mutable report object that is updated upon computation
            of a graph starting with the new AwkwardInputLayer.

        """
        import awkward as ak

        starting_form = copy.deepcopy(self._meta.layout.form)
        starting_layout = starting_form.length_zero_array(highlevel=False)
        new_meta = ak.Array(
            starting_layout.to_typetracer(forget_length=True), behavior=self._behavior
        )
        form = new_meta.layout.form

        def _label_form(form, start):
            if form.is_record:
                for field in form.fields:
                    _label_form(form.content(field), f"{start}.{field}")
            elif form.is_numpy:
                form.form_key = start
            elif form.is_list:
                form.form_key = f"{start}.__list__"
                _label_form(form.content, start)
            else:
                _label_form(form.content, start)

        _label_form(form, self.name)

        new_meta_labelled, report = ak._nplikes.typetracer.typetracer_with_report(form)
        new_meta_array = ak.Array(new_meta_labelled, behavior=self._behavior)
        new_input_layer = AwkwardInputLayer(
            name=self.name,
            columns=self.columns,
            inputs=[None] * int(list(self.numblocks.values())[0][0]),
            io_func=lambda *_, **__: new_meta_array,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
            meta=new_meta_array,
            behavior=self._behavior,
        )
        return new_input_layer, report

    def project_columns(self, columns: list[str]) -> AwkwardInputLayer:
        if hasattr(self.io_func, "project_columns"):
            io_func = self.io_func.project_columns(columns)
            return AwkwardInputLayer(
                name=self.name,
                columns=columns,
                inputs=self.inputs,
                io_func=io_func,
                label=self.label,
                produces_tasks=self.produces_tasks,
                creation_info=self.creation_info,
                annotations=self.annotations,
                meta=self._meta,
                behavior=self._behavior,
            )
        return self
