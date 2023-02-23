Data IO
-------

Creating dask-awkward collections typically begins with reading from
either local disk or cloud storage. There is built-in support for
datasets stored in JSON or Parquet format.

Take this code-block for example:

.. code:: python

   >>> import dask_awkward as dak
   >>> ds1 = dak.from_parquet("s3://path/to/dataset")
   >>> ds2 = dak.from_json("/path/to/json-files/*.json")

Both the :py:func:`~dask_awkward.from_parquet` and
:func:`~dask_awkward.from_json` calls will create new
:class:`~dask_awkward.Array` instances. In the Parquet example we will
read data from Amazon S3; in the JSON example we're reading data form
local disk (notice the wildcard syntax: all JSON files in that
directory will be discovered, and each file will become a partition in
the collection).
