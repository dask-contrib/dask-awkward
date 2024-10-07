# Contributing to dask-awkward

Install the development environment (once) with

```bash
uv venv
uv pip install -e '.[io,docs,test]'
pre-commit install
```

Source development environment with

```bash
source .venv/bin/activate
```

Run tests with

```bash
pytest .
```

Build documentation with

```bash
cd docs && make html
```
