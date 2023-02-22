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
data access from disk. We call it the "necessary columns" optimization.
