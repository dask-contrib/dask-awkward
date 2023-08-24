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

dask-awkward Optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two optimizations implemented in the dask-awkward code. One
is the ``layer-chains`` optimization that fuses adjacent task graph
layers together (if they are compatible with each other). This is a
relatively simple optimization that just simplifies the task graph.
The other optimization is the ``columns`` (or "necessary columns")
optimization; which is a bit more technical and described in a
follow-up section.

One can configure which optimizations to run at compute-time; read
more optimization. More information can be found in the
:ref:`configuration section
<how-to/configuration:Optimization specific table>` of the docs.


Necessary Columns
^^^^^^^^^^^^^^^^^

We have one dask-awkward specific optimization that targets efficient
data access from disk. We call it the "necessary columns"
optimization. This optimization will execute the task graph *without
operating on real data*. The data-less execution of the graph helps
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

.. code:: pycon

   >>> ds = dak.from_parquet("/path/to/data")
   >>> result = ds.bar.x / ds.foo

We have five layers in the graph:

1. Reading parquet from the path ``/path/to/data``
2. Access the field ``foo``
3. Access the field ``bar``
4. Access the field ``x`` from ``bar``
5. Array division

We can see this at the REPL by inspecting the ``.dask`` property of
the collection:

.. code-block::

   >>> result.dask
   HighLevelGraph with 5 layers.
     <dask.highlevelgraph.HighLevelGraph object at 0x134a4fc10>
      0. read-parquet-f4e4296edcc1309191080cae9018ab4c
      1. foo-791f3e559c4061a8c9df2e87a0524069
      2. bar-edf7073f1aab48e986099f7c67e81be9
      3. x-47d0bdfde8d53e07444a58204428ff2f
      4. divide-b85b7c773695128b08311b3a75b0002b

Notice that we never actually need the ``bar.y`` column of floats.
Upon calling ``result.compute()``, step (1) in our list of layers
above (reading parquet) will be updated such that the parquet read
will only grab ``foo`` and ``bar.x``.

.. note::

   This is done by replacing the *original* input layer with a new
   layer instance that will pass in the named argument
   ``columns=["foo", "bar.x"]`` to the concrete awkward
   :py:func:`ak.from_parquet` function at compute time.

You can see which columns are determined to be necessary by calling
:func:`dask_awkward.necessary_columns` on the collection of interest
(it returns a mapping that pairs an input layer with the list of
necessary columns):

.. code:: pycon

   >>> dak.necessary_columns(result)
   {"some-layer-name": ["foo", "bar.x"]}

The optimization is performed by relying on upstream Awkward-Array
typetracers. **It is possible for this optimization to fail.** The
default configuration is such that a warning will be thrown if the
optimization fails. If you'd instead like to silence the warning or
raise an exception, the configuration parameter can be adjusted. Here
are the options for the ``awkward.optimization.on-fail`` configuration
parameter:

- ``"pass"``: fail silently; the optimization is skipped (can reduce
  performance by reading unncessary data from disk).
- ``"raise"``: fail by raising an exception: this will stop the process
  at compute time.
- ``"warn"`` (the default): fail with a warning but let the compute
  continue without the necessary columns optimization (can reduce
  performance by reading unnecessary data from disk).

One can also use the ``columns=`` argument (with
:func:`~dask_awkward.from_parquet`, for example) to manually define
which columns should be read from disk. The
:func:`~dask_awkward.necessary_columns` function can be used to
determine how one should use the ``columns=`` argument. Using our
above example, we write

.. code:: pycon

   >>> ds = dak.from_parquet("/path/to/data", columns=["bar.x", "foo"])
   >>> result = ds.bar.x / ds.foo
   >>> with dask.config.set({"awkward.optimization.enabled": False}):
   ...     result.compute()
   ...

With this code we can save a little bit of overhead by not running the
necessary columns optimization after already defining, by hand, the
minimal set (one should be sure about what is needed with this
workflow).


.. raw:: html

    <script>
        window.goatcounter = {
            path: function(p) { return location.host + p }
        }
    </script>
    <script data-goatcounter="https://distdatacats.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
