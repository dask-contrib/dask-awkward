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


Table of Contents
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   gs-install.rst
   gs-intro.rst
   gs-terminology.rst
   gs-limitations.rst

.. toctree::
   :maxdepth: 1
   :caption: How Tos

   ht-configuration.rst
   ht-io.rst
   ht-behaviors.rst

.. toctree::
   :maxdepth: 1
   :caption: Deeper Explanation

   me-optimization.rst
   me-faq.rst

.. toctree::
   :maxdepth: 1
   :caption: API

   api-collections.rst
   api-inspect.rst
   api-io.rst
   api-reducers.rst
   api-structure.rst

.. toctree::
   :maxdepth: 1
   :caption: Development

   dev-contributing.rst
   dev-releasing.rst

------------

Support for this work was provided by NSF grant `OAC-2103945 <nsf_>`_.

.. _awkwardarray: https://awkward-array.org/
.. _nsf: https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103945
