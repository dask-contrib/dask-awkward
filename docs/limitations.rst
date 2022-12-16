Limitations
-----------

The goal of dask-awkward is to allow for the creation of lazily
evaluated computations on partitioned awkward arrays. We do this by
mimicking the awkward array API. There are a few categories of
operations under this umbrella:

1. **Things that are implemented**: this is the best case, a part of
   the awkward API that is readily available: For example,
   ``dak.sum(a, axis=1)``.

2. **Things that are possible but we haven't implemented**: An operation
   like ``dak.std(a, axis=0)``. Right now this operation will raise a
   ``DaskAwkwardNotImplemented`` exception. If you run into something
   like this please feel free to open up an issue on the GitHub issue
   tracker to start a discussion and get feature request process
   started.

3. **Things that just won't be implemented**: There will be parts of
   the dask-awkward API that we simply cannot implement for lazy
   computation!
