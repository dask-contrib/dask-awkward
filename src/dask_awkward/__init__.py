from dask_awkward import config  # isort:skip; load awkward config


from dask_awkward.version import __version__


def __getattr__(value):
    import dask_awkward.lib

    return getattr(dask_awkward.lib, value)


original = dir()


def __dir__():
    import dask_awkward.lib  # pragma: no cover

    return original + dir(dask_awkward.lib)  # pragma: no cover
