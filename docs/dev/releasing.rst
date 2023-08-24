Releasing
=========

Tagging a version
-----------------

We use calendar versioning (CalVer) with the format: ``YYYY.MM.X``
where ``X`` is incremented depending on how many releases have already
occurred in the same year and month. For example, if the most recent
release is the first release from March of 2023 it would be
``2023.3.0``, the next release (on any day in that month) would be
``2023.3.1``.

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

Push the tag to GitHub (assuming ``upstream`` points to the
``dask-contrib/dask-awkward`` remote):

.. code-block::

   $ git push upstream 2023.3.1

PyPI upload
-----------

We use a GitHub action to upload releases to PyPI when tags are pushed
to GitHub. It should take just a few minutes after pushing the tag.

Create GitHub Release
---------------------

Once the release is tagged and both the source and wheel distributions
are on PyPI, visit the GitHub repository releases_ page and click on
``Draft new release``. Create a new release with the same name as the
new tag. Click on the option to automatically generate release notes
(from the commits that are part of the release). Give those notes a
read and polish the automatically generated text as necessary.

.. _releases: https://github.com/dask-contrib/dask-awkward/releases

Merge conda-forge version update
--------------------------------

The conda-forge ``regro-cf-autotick-bot`` will create a PR after
detecting the new release on PyPI. You should be able to just merge
the PR when CI passes.


.. raw:: html

    <script>
        window.goatcounter = {
            path: function(p) { return location.host + p }
        }
    </script>
    <script data-goatcounter="https://distdatacats.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
