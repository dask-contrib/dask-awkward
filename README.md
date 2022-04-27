dask-awkward
============

> Connecting [awkward-array](https://awkward-array.org) and
[Dask](https://dask.org/).

[![Tests](https://github.com/ContinuumIO/dask-awkward/actions/workflows/pypi-tests.yml/badge.svg)](https://github.com/ContinuumIO/dask-awkward/actions/workflows/pypi-tests.yml)
[![Tests](https://github.com/ContinuumIO/dask-awkward/actions/workflows/conda-tests.yml/badge.svg)](https://github.com/ContinuumIO/dask-awkward/actions/workflows/conda-tests.yml)
[![Documentation Status](https://readthedocs.org/projects/dask-awkward/badge/?version=latest)](https://dask-awkward.readthedocs.io/en/latest/?badge=latest)
![stability-alpha](https://img.shields.io/badge/stability-alpha-blue.svg)

**This is alpha software** without any guaranteed API stability.

Installing
----------

If you are interested in experimenting with `dask-awkward` we
recommend installing from the `main` branch:

```
pip install git+https://github.com/ContinuumIO/dask-awkward@main
```

The [awkward-array](https://github.com/scikit-hep/awkward-1.0) project
is working on version 2 of the core `awkward` Python library; `awkard`
version 1.8 includes many components of what will be version 2, but in
the `awkward._v2` module. Installing `dask-awkward` requires
`awkward>=1.8.0`.

It's recommended to have a bleeding edge version of `awkward`
installed with:

```
pip install git+https://github.com/scikit-hep/awkward-1.0@main
```

Since `dask-awkward` is designed for `awkward` version 2, the
concrete, high-level array object (`ak.Array`) that we use is actually
the `awkward._v2.Array` object.
