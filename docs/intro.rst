Introduction
------------

.. note::

   This introduction assumes that you have some familiarity with
   `Dask`_.

The Dask project provides "collections" which behave as parallelized
and/or distributed versions of the core PyData data types:

- dask.array provides a NumPy like interface for creating task graphs
- dask.dataframe provides a Pandas like interface for creating task
  graphs
- dask.bag provides a functional interface for creating task graphs
- dask.delayed provides an interface for custom task graphs

With dask-awkward, we aim to provide an additional interface:

- dask-awkward provides and Awkward-Array_\-like interface for creating
  task graphs.

We accomplish this by creating a new collection type:
``dask_awkward``'s :py:class:`~dask_awkward.core.Array` class, which
is a partitioned representation of a concrete awkward Array.

Imagine a dataset of multiple line delimited JSON files (data.00.json,
data.01.json, and so on). Loading that data, and selecting a subset of
the dataset based on the total number of entries in a nested attribute
of the data can be done with both ``awkward`` and ``dask-awkward``
with the same programming style; on the left we operate eagerly with
``awkward`` and on the right we operate lazily with ``dask-awkward``
(notice the ``compute()`` method call):

.. code-block:: python

   import awkward._v2 as ak                import dask_awkward as dak
   x = ak.from_json("data.00.json")        x = dak.from_json("data.*.json")
   x = x[ak.num(x.foo) > 2]                x = x[dak.num(x.foo).compute()

.. note::

   dask-awkward depends on the in-development version 2 of awkward;
   which exists in the ``awkward._v2`` namespace.

For example usage of dask-awkward, `we have a demo repository
<https://github.com/douglasdavis/dask-awkward-demo>`__.

.. _Awkward-Array: https://awkward-array.org/
.. _Dask: https://dask.org/
