from __future__ import annotations

import logging

import awkward as ak

from dask_awkward.layers.layers import BackendT

log = logging.getLogger(__name__)


class ColumnProjectionMixin:
    """A mixin to add column-centric buffer projection to an IO function.

    Classes that inherit from this mixin are assumed to be able to read at the
    granularity of _fields_ in a form. As such, the buffer projection is performed
    such that the smallest possible number of fields (columns) are read, even
    when only metadata buffers are required.
    """

    def project(self, *args, **kwargs):
        # default implementation does nothing
        return self

    def mock_empty(self, backend: BackendT = "cpu"):
        # used by failure report generation
        return (
            ak.to_backend(
                self.form.length_zero_array(highlevel=False, behavior=self.behavior),
                backend,
                highlevel=True,
            ),
        )
