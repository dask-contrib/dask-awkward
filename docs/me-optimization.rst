Optimization
------------

When you ask Dask to compute a collection (with the ``compute`` method
on the collection, the :py:func:`dask.compute` function, etc.), Dask
will (by default) optimize the task graph before beginning to execute
the graph (the ``optimize_graph`` argument exists to toggle this
behavior, but setting this to ``False`` is really meant for
debugging). Core Dask has a number of optimizations implemented that
we benefit from downstream in dask-awkward. You can read more about
Dask optimization in general :doc:`at this section of the Dask docs
<dask:optimize>`.

Necessary Columns
^^^^^^^^^^^^^^^^^

We have one dask-awkward specific optimization that targets efficient
data access from disk. We call it the "necessary columns"
optimization. This optimization will execute the task graph *without
operating on real data*. The data-less executation of the graph helps
determine which parts of a dataset sitting on disk are actually
required to read in order to successfully complete the compute.

Let's look at a simple example dataset: an awkward array with two top
level fields (``foo`` and ``bar``), with one field having two
subfields (``bar.x`` and ``bar.y``). Imagine this dataset is going to
be read off disk in Parquet format. In this format we'll have a column
of integers for ``foo``, a column of integers for ``bar.x`` and a
column of floats for ``bar.y``.

.. code:: js

   [
     {"foo": 5, "bar": {"x": [-1, -2], "y": -2.2}},
     {"foo": 6, "bar": {"x": [-3], "y":  3.3}},
     {"foo": 7, "bar": {"x": [-5, -6, -7], "y": -4.4}},
     {"foo": 8, "bar": {"x": [8, 9, 10, 11, 12], "y":  5.5}},
     ...
   ]

If our task graph is of the form:

.. code:: python

   >>> ds = dak.from_parquet("/path/to/data")
   >>> result = ds.bar.x / ds.foo

We have five layers in the graph:

1. Reading parquet from the path ``/path/to/data``
2. ``getattr`` to access the field ``bar``
3. ``getattr`` to access the field ``x`` from ``bar``
4. ``getattr`` to access the field ``foo``
5. Array division

Notice that we never actually need the ``bar.y`` column of floats.
Upon calling ``result.computeI()``, step (1) in our list above
(reading parquet) will be updated such that the parquet read will only
grab ``foo`` and ``bar.x``.

You can see which columns are determined to be necessary by calling
:func:`dask_awkward.necessary_columns` on the collection of interest
(it returns a mapping that pairs an input layer with the list of
necessary columns):

.. code:: python

   >>> dak.necessary_columns(result)
   {"some-layer-name": ["foo", "bar.x"]}

The optimization is performed by relying on upstream Awkward-Array
typetracers. **It is possible for this optimization to fail.** The
default configuration is such that a warning will be thrown if the
optimization fails. If you'd instead like to silence the warning or
raise an exception, the configuration parameter can be adjusted. Here
are the options for the ``awkward.optimization.on-fail`` configuration
parameter:

- ``pass``: fail silently; the optimization is skipped (can reduce
  performance by reading unncessary data from disk).
- ``raise``: fail by raising an exception: this will stop the process
  at compute time.
- ``warn`` (the default): fail with a warning but let the compute
  continue without the necessary columns optimization (can reduce
  performance by reading unncessary data from disk).
