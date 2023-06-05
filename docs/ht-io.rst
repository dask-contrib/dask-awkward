Data IO
-------

Creating dask-awkward collections typically begins with reading from
either local disk or cloud storage. There is built-in support for
datasets stored in Parquet or JSON format.

Take this code-block for example:

.. code:: python

   >>> import dask_awkward as dak
   >>> ds1 = dak.from_parquet("s3://path/to/dataset")
   >>> ds2 = dak.from_json("/path/to/json-files/*.json")

Both the :py:func:`~dask_awkward.from_parquet` and
:func:`~dask_awkward.from_json` calls will create new
:class:`dask_awkward.Array` instances. In the Parquet example we will
read data from Amazon S3; in the JSON example we're reading data from
local disk (notice the wildcard syntax: all JSON files in that
directory will be discovered, and each file will become a partition in
the collection).

Support for the ROOT file format is provided by the Uproot_ project.

It's also possible to instantiate dask-awkward
:class:`dask_awkward.Array` instances from other Dask collections
(like :class:`dask.array.Array`), or concrete objects like existing
awkward Array instances or Python lists.

.. _Uproot: https://github.com/scikit-hep/uproot5

See the :ref:`IO API docs<api-io:IO>` page for more information on the
possible ways to instantiate a new dask-awkward Array.
