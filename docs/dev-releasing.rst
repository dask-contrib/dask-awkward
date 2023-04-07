Releasing
=========

Tagging a version
-----------------

We use calendar versioning (CalVer) with the format: ``YYYY.MM.X``
where ``X`` is incremented as needed depending on how many release
have already occurred in the same year and month. For example, if the
most recent release is the first release from March of 2023 it would
be ``2023.3.0``, the next release (no matter what day in that month)
would be ``2023.3.1``.

Check the latest tag with git (or just visit the GitHub repository
tags list):

.. code-block::

   $ git fetch --all --tags
   $ git describe --tags $(git rev-list --tags --max-count=1)
   2023.3.0

Create a new tag that follows our CalVer convention (using
``2023.3.0`` example above, we write the next tag accordingly):


.. code-block::

   $ git tag -a -m "2023.3.1" 2023.3.1

Push the tag to GitHub (assuming ``origin`` points to the
``dask-contrib/dask-awkward`` remote):

.. code-block::

   $ git push origin 2023.3.1

Making the release
------------------

To make a release of ``dask-awkward`` we just need the ``build`` and
``twine`` packages:

.. code-block::

   $ pip install build twine

The build-system that we use (``hatch`` with ``hatch-vcs``) will
automatically set a version based on the latest tag in the repository;
after making the tag we just need to generate the source distribution
and wheel, this is handled by the ``build`` package:

.. code-block::

   $ python -m build

Now a new ``dist/`` directory will appear, which contains the files
(continuing to use our example version `2023.3.1``):

.. code-block::

   dask_awkward-2023.3.1.tar.gz
   dask_awkward-2023.3.1-py3-none-any.whl

Now we just upload these files to PyPI with ``twine``:

.. code-block::

   $ twine upload dist/dask_awkward-2023.3.1*

The GitHub ``regro-cf-autotick-bot`` account will automatically create
a pull request to release a new version on ``conda-forge``.
