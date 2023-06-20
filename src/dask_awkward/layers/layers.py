from __future__ import annotations

import copy
import pickle
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import MaterializedLayer
from dask.layers import DataFrameTreeReduction

from dask_awkward.utils import LazyInputsDict

if TYPE_CHECKING:
    from awkward.typetracer import TypeTracerReport


class AwkwardBlockwiseLayer(Blockwise):
    """Just like upstream Blockwise, except we override pickling"""

    @classmethod
    def from_blockwise(cls, layer: Blockwise) -> AwkwardBlockwiseLayer:
        ob = object.__new__(cls)
        ob.__dict__.update(layer.__dict__)
        return ob

    def mock(self) -> tuple[AwkwardBlockwiseLayer, Any | None]:
        layer = copy.copy(self)
        nb = layer.numblocks
        layer.numblocks = {k: tuple(1 for _ in v) for k, v in nb.items()}
        layer.__dict__.pop("_dims", None)
        return layer, None

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

    def mock(self) -> tuple[MaterializedLayer, Any | None]:
        mapping = self.mapping.copy()
        if not mapping:
            # no partitions at all
            return self, None
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
                return MaterializedLayer({(name, 0): task}), None
            return self, None

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
            return MaterializedLayer({(name, 0): task}), None

        # failed to cull during column opt
        return self, None


class AwkwardTreeReductionLayer(DataFrameTreeReduction):
    def mock(self) -> tuple[AwkwardTreeReductionLayer, Any | None]:
        return (
            AwkwardTreeReductionLayer(
                name=self.name,
                name_input=self.name_input,
                npartitions_input=1,
                concat_func=self.concat_func,
                tree_node_func=self.tree_node_func,
                finalize_func=self.finalize_func,
                split_every=self.split_every,
                tree_node_name=self.tree_node_name,
            ),
            None,
        )
