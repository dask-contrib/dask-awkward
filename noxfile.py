import nox


@nox.session
def cov(session):
    session.install(".[test]")
    session.run("pytest", "--cov=dask_awkward", "--cov-report=html")
