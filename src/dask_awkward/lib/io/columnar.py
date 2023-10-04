from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import awkward as ak
from awkward import Array as AwkwardArray
from awkward.forms.form import Form

from dask_awkward.lib.utils import (
    METADATA_ATTRIBUTES,
    buffer_keys_required_to_compute_shapes,
    form_with_unique_keys,
    parse_buffer_key,
    render_buffer_key,
    trace_form_structure,
)

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport

    from dask_awkward.lib.utils import FormStructure

log = logging.getLogger(__name__)


U = TypeVar("U")


class ImplementsColumnProjectionImpl(Protocol):
    @property
    def use_optimization(self) -> bool:
        ...

    def project_columns(self: U, columns: set[str]) -> U:
        ...

    @property
    def form(self) -> Form:
        ...

    @property
    def original_form(self) -> Form:
        ...

    @property
    def behavior(self) -> dict | None:
        ...


T = TypeVar("T", bound=ImplementsColumnProjectionImpl)


class ColumnProjectionMixin:
    def mock(self: ImplementsColumnProjectionImpl) -> AwkwardArray:
        return ak.typetracer.typetracer_from_form(self.form, behavior=self.behavior)

    def prepare_for_projection(
        self: ImplementsColumnProjectionImpl,
    ) -> tuple[AwkwardArray, TypeTracerReport, FormStructure]:
        form = form_with_unique_keys(self.form, "@")

        # Build typetracer and associated report object
        (meta, report) = ak.typetracer.typetracer_with_report(
            form,
            highlevel=True,
            behavior=self.behavior,
            buffer_key=render_buffer_key,
        )

        return (
            cast(AwkwardArray, meta),
            report,
            trace_form_structure(form, buffer_key=render_buffer_key),
        )

    def project(
        self: T,
        report: TypeTracerReport,
        state: FormStructure,
    ) -> T:
        if not self.use_optimization:
            return self

        assert self.original_form is None

        ## Read from stash
        # Form hierarchy information
        form_key_to_parent_key = state["form_key_to_parent_key"]
        # Buffer hierarchy information
        form_key_to_buffer_keys = state["form_key_to_buffer_keys"]
        # Parquet hierarchy information
        form_key_to_path = state["form_key_to_path"]
        # Parquet hierarchy information
        path_to_child_paths = state["path_to_child_paths"]
        # Inverse parentage
        form_key_to_child_keys: dict[str, list[str]] = {}
        for child_key, parent_key in form_key_to_parent_key.items():
            form_key_to_child_keys.setdefault(parent_key, []).append(child_key)

        # Require the data of metadata buffers above shape-only requests
        data_buffers = {
            *report.data_touched,
            *buffer_keys_required_to_compute_shapes(
                parse_buffer_key,
                report.shape_touched,
                form_key_to_parent_key,
                form_key_to_buffer_keys,
            ),
        }

        # We can't read buffers directly, but if we encounter a metadata
        # buffer, then we should be able to pick any child.
        paths = set()
        wildcard_paths = set()
        for buffer_key in data_buffers:
            form_key, attribute = parse_buffer_key(buffer_key)
            path = form_key_to_path[form_key]
            if attribute in METADATA_ATTRIBUTES:
                wildcard_paths.add(path)
            else:
                paths.add(path)

        # Select the most appropriate column for each wildcard
        for path in wildcard_paths:
            child_paths = path_to_child_paths[path]

            # This is a leaf! Therefore, we read this column
            if not child_paths:
                paths.add(path)
            else:
                for child_path in child_paths:
                    if child_path in paths:
                        break
                # Didn't find anyone else trying to read these offset or buffers
                # So take the first child
                else:
                    paths.add(child_paths[0])

        return self.project_columns({".".join(p) for p in paths if p})
