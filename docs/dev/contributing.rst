Contributing
============

We're very happy recieve contributions to dask-awkward!

Reporting bugs
--------------

To report bugs or any general issues please use the `GitHub repository
issue tracker <issuetracker_>`_.

Adding code
-----------

To develop in the dask-awkward codebase, fork the repsitory, install
dask-awkward with optional dependencies, and create a new branch:

.. code-block::

   $ git clone <url to your fork>
   $ cd dask-awkward
   $ git remote add upstream https://github.com/dask-contrib/dask-awkward
   $ pip install -e ".[test,docs]"
   $ git checkout -b name-your-branch upstream/mean

Make your changes and be sure to add a test. Run ``pytest`` in the
dask-awkward repository with:

.. code-block::

   $ pytest

Commit your changes, push your branch to your fork, and open a Pull
Request. We suggest that you install `pre-commit <precommit_>`_ to run
some checks locally when creating new commits.

Typing
------

We include a pre-commit hook that runs ``mypy`` for static type
checking. Code added to dask-awkward is *not required* to be typed,
but the pre-commit check does enforce correctness when type hints are
present.

Adding documentation
--------------------

Documentation is generated with Sphinx. All files are in the ``docs/``
directory. When necessary, please also include an addition to the
documentation. To generate the documentation run

.. code-block::

   $ make html

Inside of the ``docs/`` directory to see how the documentation will be
rendered.

.. _issuetracker: https://github.com/dask-contrib/dask-awkward/issues
.. _precommit: https://pre-commit.com/

.. raw:: html

   <script data-goatcounter="https://dask-awkward.goatcounter.com/count"
           async src="//gc.zgo.at/count.js"></script>
