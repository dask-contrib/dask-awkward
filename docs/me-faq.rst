FAQ
---

- **Q: I just saw "UserWarning: metadata could not be determined; a
  compute on the first partition will occur." what does that mean?**

  A: When dask-awkward stages a new computation it runs the operation
  on a typetracer array to generate metadata for the new collection.
  If this is not possible, that is, awkward array itself (independent
  of Dask) was not able to execute the operation on the existing
  metadata, then the first partition will be computed to determine the
  new metadata.

  You can bypass the automatic metadata determination using Dask's
  `configuration manager <daskconfig_>`__. The configuration parameter
  is called ``awkward.compute-unknown-meta``. The default setting is
  ``True``. In code you can do something like this:

  .. code-block:: python

     with dask.config.set({"awkward.compute-unknown-meta": False}):
         # ... your code

  Or you can modify the configuration with environment variables or a
  YAML file. See Dask's documentation linked above.


.. _daskconfig: https://docs.dask.org/en/stable/configuration.html
