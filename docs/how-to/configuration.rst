Configuration
-------------

Core Dask has :doc:`detailed documentation <dask:configuration>`
describing how configuration works in Dask. This page exists to cover
configuration options specific to dask-awkward.

The file which defines the default configuration parameters is in the
dask-awkward repository at the path ``src/dask_awkward/awkward.yaml``.
The file attempts to be self documenting.


Top level table
^^^^^^^^^^^^^^^

These top level parameters are configuration under the ``awkward``
namespace in Dask configuration format. For example, they can be set
with the form:

.. code-block:: python

   with dask.config.set({"awkward.<option>": value}):
       ...

- ``raise-failed-meta`` (default: ``False``): If this option is set to
  ``True``, then an exception will be raised if dask-awkward fails to
  automatically determine the metadata of a new collection as task
  graphs are built.
- ``compute-unknown-meta`` (default: ``True``): In the event that
  dask-awkward cannot determine the metadata for a collection, when
  this option is ``True`` we will compute the first partition of the
  collection to determine the metadata. This obviously triggers a
  compute and can take some time depending on the task graph.

Optimization specific table
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These optimization table parameters are configured under the
``awkward.optimization`` namespace in the Dask configuration format.
For example, they can be set with the form:


.. code-block:: python

   with dask.config.set({"awkward.optimization.<option>": value}):
       ...

- ``enabled`` (default: ``True``): Enable dask-awkward specific
  optimizations. More fine tuning can be handled with the ``which``
  option.
- ``which`` (default: ``[columns, layer-chains]``): Which of the
  optimizations to run. The default setting is to run all available
  optimizations. (if ``enabled`` is set to ``False`` this option is
  ignored).
- ``column-opt-formats`` (default: ``[parquet, json]``): Which input
  formats should use the column optimization.
- ``on-fail`` (default: ``warn``): When set to ``warn`` throw a
  warning of the optimization fails and continue without performing
  the optimization. If set to ``raise``, raise an exception at
  optimization time. If set to ``pass``, silently skip the
  optimization. More information can be found in the :ref:`necessary
  columns optimization <more/optimization:necessary columns>` section of
  the docs.

.. raw:: html

   <script data-goatcounter="https://dask-awkward.goatcounter.com/count"
           async src="//gc.zgo.at/count.js"></script>
