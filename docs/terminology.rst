Terminology
-----------

The most common class in both dask-awkward and awkward-array is the
``Array`` class. In dask-awkward the ``Array`` class provides a Dask
collection representing a partitioned and lazily computed version of
the awkward-array ``Array`` class. To help mitigate confusion between
dask-awkward and awkward-array, we try to maintain a clear distinction
between objects in both projects.

First, dask-awkward adopts ``dak`` as the standard import alias:

.. code-block:: python

   import dask_awkward as dak

We will always follow the standard import alias for awkward as well:

.. code-block:: python

   import awkward as ak

With the imports in mind, we will never have an unqualified ``Array``
object in the documentation, all instances will be either
``dak.Array`` or ``ak.Array``. We also discourage importing the
objects from the namespaces:

.. code-block:: python

   # don't do this!
   from dask_awkward import Array
   # or this!
   from awkward import Array

In written descriptions, we'll sometimes refer to the awkward-array
``Array`` object as a "materialized array", or an "eager array". The
dask-awkward object will be referred to as a "lazy array" or an "array
collection."

Finally, the result of calling the ``compute()`` method on a
``dak.Array`` object will almost always result in an ``ak.Array``
object.
