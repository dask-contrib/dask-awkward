Behaviors [experimental]
------------------------

.. warning::

   Awkward-Array behaviors are currently experimental in dask-awkward!

Awkward-Array's behaviors_ feature provides a powerful mechanism for
attaching methods and properties to Awkward Array. There is support in
dask-awkward for using behaviors, but the feature is currently
experimental and still in design development; there is no guarantee of
API stability at this time! WIth that caveat, it's still possible to
use behaviors now.

Here's a brief example (we suggest reading the :doc:`upstream
documentation <awkward:reference/ak.behavior>` for more information on the
topic).

.. code:: python

   import awkward as ak
   import dask_awkward as dak

   behavior: dict = {}


   @ak.mixin_class(behavior)
   class Point:
       def distance(self, other):
           return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


   points1 = ak.Array(
       [
           [{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}, {"x": 3, "y": 3.3}],
           [],
           [{"x": 4.0, "y": 4.4}, {"x": 5.0, "y": 5.5}],
           [{"x": 6.0, "y": 6.6}],
           [{"x": 7.0, "y": 7.7}, {"x": 8.0, "y": 8.8}, {"x": 9, "y": 9.9}],
       ]
   )

   points2 = ak.Array(
       [
           [{"x": 0.9, "y": 1.0}, {"x": 2.0, "y": 2.2}, {"x": 2.9, "y": 3.0}],
           [],
           [{"x": 3.9, "y": 4.0}, {"x": 5.0, "y": 5.5}],
           [{"x": 5.9, "y": 6.0}],
           [{"x": 6.9, "y": 7.0}, {"x": 8.0, "y": 8.8}, {"x": 8.9, "y": 9.0}],
       ]
   )

   array1 = dak.from_awkward(points1, npartitions=2)
   array2 = dak.from_awkward(points2, npartitions=2)

   array1 = dak.with_name(array1, name="Point", behavior=behavior)
   array2 = dak.with_name(array2, name="Point", behavior=behavior)

   distance = array1.distance(array2)
   result = distance.compute()

.. _behaviors: https://awkward-array.org/doc/main/reference/ak.behavior.html

.. raw:: html

   <script data-goatcounter="https://dask-awkward.goatcounter.com/count"
           async src="//gc.zgo.at/count.js"></script>
