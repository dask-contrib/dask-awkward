from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import awkward as ak
from awkward import Array as AwkwardArray
from awkward.forms import Form

from dask_awkward.layers.layers import (
    BackendT,
    ImplementsIOFunction,
    ImplementsNecessaryColumns,
)
from dask_awkward.lib.utils import METADATA_ATTRIBUTES, FormStructure

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport

log = logging.getLogger(__name__)

T = TypeVar("T")


class ImplementsColumnProjectionMixin(ImplementsNecessaryColumns, Protocol):
    @property
    def form(self) -> Form: ...

    @property
    def behavior(self) -> dict | None: ...

    @property
    def attrs(self) -> dict | None: ...

    def project_columns(self: T, columns: frozenset[str]) -> T: ...

    def __call__(self, *args, **kwargs): ...


S = TypeVar("S", bound=ImplementsColumnProjectionMixin)


class ColumnProjectionMixin(ImplementsNecessaryColumns[FormStructure]):
    """A mixin to add column-centric buffer projection to an IO function.

    Classes that inherit from this mixin are assumed to be able to read at the
    granularity of _fields_ in a form. As such, the buffer projection is performed
    such that the smallest possible number of fields (columns) are read, even
    when only metadata buffers are required.
    """

    def mock_empty(self: S, backend: BackendT = "cpu") -> AwkwardArray:
        # used by failure report generation
        return cast(
            AwkwardArray,
            ak.to_backend(
                self.form.length_zero_array(highlevel=False, behavior=self.behavior),
                backend,
                highlevel=True,
            ),
        )
