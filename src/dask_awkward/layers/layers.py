from __future__ import annotations

import copy
import pickle
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import MaterializedLayer

from dask_awkward.utils import LazyInputsDict

if TYPE_CHECKING:
    from awkward.typetracer import TypeTracerReport


class AwkwardBlockwiseLayer(Blockwise):
    """Just like upstream Blockwise, except we override pickling"""

    @classmethod
    def from_blockwise(cls, layer) -> AwkwardBlockwiseLayer:
        ob = object.__new__(cls)
        ob.__dict__.update(layer.__dict__)
        return ob

    def mock(self):
        layer = copy.copy(self)
        nb = layer.numblocks
        layer.numblocks = {k: tuple(1 for _ in v) for k, v in nb.items()}
        layer.__dict__.pop("_dims", None)
        return layer

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        try:
            pickle.dumps(d["_meta"])
        except (ValueError, TypeError, KeyError):
            d.pop(
                "_meta", None
            )  # must be a typetracer, does not pickle and not needed on scheduler
        return d

    def __repr__(self) -> str:
        return "Awkward" + super().__repr__()


class AwkwardInputLayer(AwkwardBlockwiseLayer):
    """A layer known to perform IO and produce Awkward arrays

    We specialise this so that we have a way to prune column selection on load
    """

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

        from dask_awkward.lib._utils import set_form_keys

        starting_form = copy.deepcopy(self._meta.layout.form)
        starting_layout = starting_form.length_zero_array(highlevel=False)
        new_meta = ak.Array(
            starting_layout.to_typetracer(forget_length=True),
            behavior=self._behavior,
        )
        form = new_meta.layout.form

        set_form_keys(form, key=self.name)

        new_meta_labelled, report = ak.typetracer.typetracer_with_report(form)
        new_meta_array = ak.Array(new_meta_labelled, behavior=self._behavior)
        new_input_layer = AwkwardInputLayer(
            name=self.name,
            columns=self.columns,
            inputs=[None][: int(list(self.numblocks.values())[0][0])],
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
            # TODO: make project_columns call sites never pass in an
            # empty list.
            if len(columns) == 0:
                columns = self._meta.fields[:1]

            # original form
            original_form = self._meta.layout.form

            # original columns before column projection
            original_form_columns = original_form.columns()

            # make sure that the requested columns match the order of
            # the original columns; tack on "new" columns that are
            # likely the wildcard columns.
            original = [c for c in original_form_columns if c in columns]
            new = [c for c in columns if c not in original_form_columns]
            columns = original + new

            try:
                io_func = self.io_func.project_columns(
                    columns,
                    original_form=original_form,
                )
            except TypeError:
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


class AwkwardMaterializedLayer(MaterializedLayer):
    def __init__(self, mapping, previous_layer_name, **kwargs):
        self.prev_name = previous_layer_name
        super().__init__(mapping, **kwargs)

    def mock(self):
        mapping = self.mapping.copy()
        if not mapping:
            # no partitions at all
            return self
        name = next(iter(mapping))[0]

        if (name, 0) in mapping:
            task = mapping[(name, 0)]
            task = tuple(
                (self.prev_name, 0)
                if isinstance(v, tuple) and len(v) == 2 and v[0] == self.prev_name
                else v
                for v in task
            )
            return MaterializedLayer({(name, 0): task})

        # failed to cull during column opt
        return self
