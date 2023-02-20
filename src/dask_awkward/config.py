import os

import dask.config
import yaml

config = dask.config.config

fn = os.path.join(os.path.dirname(__file__), "awkward.yaml")
with open(fn) as f:
    defaults = yaml.safe_load(f)

dask.config.update_defaults(defaults)

allowed_imports = dask.config.get("distributed.scheduler.allowed-imports", default=[])
allowed_imports += ["dask", "distributed", "dask_awkward"]
dask.config.set({"distributed.scheduler.allowed-imports": list(set(allowed_imports))})
