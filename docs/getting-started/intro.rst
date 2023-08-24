Introduction
------------

.. note::

   This introduction assumes that you have some familiarity with
   `Dask`_ and `Awkward-Array`_.

The Dask project provides **collections** which behave as parallelized
and/or distributed versions of the core PyData data types:

- dask.array provides a NumPy like interface for creating task graphs
  operating on chunked NumPy ndarrays.
- dask.dataframe provides a Pandas like interface for creating task
  graphs operating on partitioned Pandas DataFrames and Series
- dask.bag provides a functional interface for creating task graphs
  operating on Python iterables.
- dask.delayed provides an interface for custom task graphs.

With dask-awkward, we aim to provide an additional interface:

- dask-awkward provides an Awkward-Array_\-like interface for creating
  task graphs operating on partitioned awkward Arrays.

We accomplish this by creating a new collection type:
``dask_awkward``'s :py:class:`~dask_awkward.core.Array` class, which
is a partitioned representation of a concrete Awkward Array.

Imagine a dataset of multiple, line delimited JSON files
(data.00.json, data.01.json, and so on). Loading that data and
selecting a subset of the dataset based on the total number of entries
in some nested attribute of the data can be done with both ``awkward``
and ``dask-awkward`` with the same programming style; on the left we
operate eagerly with ``awkward`` (and on a single file only) and on
the right we operate lazily with ``dask-awkward`` on multiple files,
notice the use of wildcard syntax ("*").

.. grid:: 2

    .. grid-item-card::  Awkward Array

        .. code-block:: python

           from pathlib import Path
           import awkward as ak

           file = Path("data.00.json")
           x = ak.from_json(file, line_delimited=True)
           x = x[ak.num(x.foo) > 2]

    .. grid-item-card::  Dask

        .. code-block:: python

           import dask_awkward as dak

           # dask-awkward only supports line-delimited=True
           x = dak.from_json("data.*.json")
           x = x[dak.num(x.foo) > 2]

           # With Dask we have to ask for the result with compute
           x = x.compute()

On the left (the eager version) the ``from_json`` call will
immediately begin to read data from disk and decode the JSON.
Sequentially after that, the selection step will execute.

On the right (the lazy version) the ``from_json`` call will *stage*
the reading of each detected JSON file (task graph creation), the next
line will then stage the selection (extending the task graph). Dask
will execute the JSON reading and decoding of each file in parallel,
and when each reading task is done, the selection tasks will follow.
Dask will schedule the tasks itself (and it will attempt to optimize
its work).

For example usage of dask-awkward, `we have a demo repository
<https://github.com/douglasdavis/dask-awkward-demo>`__.

.. _Awkward-Array: https://awkward-array.org/
.. _Dask: https://dask.org/


.. raw:: html

    <script>
        window.goatcounter = {
            path: function(p) { return location.host + p }
        }
    </script>
    <script data-goatcounter="https://distdatacats.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
