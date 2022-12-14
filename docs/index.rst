.. dask-awkward documentation master file, created by
   sphinx-quickstart on Fri Sep 24 13:43:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dask-awkward
============

*Connecting Dask and Awkward Array.*

The dask-awkward project implements a native Dask collection for
representing partitioned Awkward arrays. Read more about Awkward
arrays at `their website <awkwardarray_>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install.rst
   intro.rst
   terminology.rst

.. toctree::
   :maxdepth: 1
   :caption: API

   api_array.rst
   api_io.rst
   api_reducers.rst
   api_structure.rst

.. _awkwardarray: https://awkward-array.org/
