Data IO
-------

Creating dask-awkward collections typically begins with reading from
either local disk or cloud storage. There is built-in support for
datasets stored in Parquet or JSON format, along support for reading
text files with each line treated as an element of an array.

Take this code-block for example:

.. code:: pycon

   >>> import dask_awkward as dak
   >>> ds1 = dak.from_parquet("s3://path/to/dataset")
   >>> ds2 = dak.from_json("/path/to/json-files")
   >>> ds3 = dak.from_text("s3://some/text/*.txt")

In the Parquet and text examples we will read data from Amazon S3; in
the JSON example we're reading data from local disk. These collections
will partitioned on a per-file basis

Support for the ROOT file format is provided by the Uproot_ project.

It's also possible to instantiate dask-awkward
:class:`dask_awkward.Array` instances from other Dask collections
(like :class:`dask.array.Array`), or concrete objects like existing
awkward Array instances or Python lists.

.. _Uproot: https://github.com/scikit-hep/uproot5

See the :ref:`IO API docs<api/io:IO>` page for more information on the
possible ways to instantiate a new dask-awkward Array.

.. raw:: html

   <script data-goatcounter="https://dask-awkward.goatcounter.com/count"
           async src="//gc.zgo.at/count.js"></script>
