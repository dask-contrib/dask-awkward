FAQ
---

**I just saw "UserWarning: The necessary columns optimization failed"
what does this mean?**

    When computing a dask-awkward collection that is reading from
    disk, the necessary columns optimization tries to discover which
    parts of a file on disk are necessary for the compute. The
    optimization then rewrites the data input layer in the graph such
    that you only read the minimum set of columns from the files on
    disk. If we are unable to determine the necessary columns, this
    optimization is simply skipped and the warning (that you've seen)
    is thrown.

    Consequences of this optimization being skipped include: an
    increase in the memory usage of your compute (because more data
    will be read from disk) and/or an increase in runtime (because it
    takes time to read the data that you actually do not need).

    You can read more about the optimization and how to configure it
    (by either silencing the warning or raise the exception) at
    :doc:`this section of the docs <optimization>`. Please open an
    issue on the GitHub issue tracker if you think you've found a
    failure in the optimization that should be fixed.

**I just saw "UserWarning: metadata could not be determined; a compute
on the first partition will occur." what does that mean?**

    When dask-awkward stages a new computation it runs the operation
    on a typetracer array to generate metadata for the new collection.
    If this is not possible, that is, awkward array itself (independent
    of Dask) was not able to execute the operation on the existing
    metadata, then the first partition will be computed to determine the
    new metadata.

    You can bypass the automatic metadata determination using Dask's
    `configuration manager <daskconfig_>`__. The configuration parameter
    is called ``awkward.compute-unknown-meta``. The default setting is
    ``True``. In code you can do something like this:

    .. code-block:: pycon

       with dask.config.set({"awkward.compute-unknown-meta": False}):
           # ... your code

    Or you can modify the configuration with environment variables or a
    YAML file. See Dask's documentation linked above.


.. _daskconfig: https://docs.dask.org/en/stable/configuration.html

.. raw:: html

   <script data-goatcounter="https://dask-awkward.goatcounter.com/count"
           async src="//gc.zgo.at/count.js"></script>
