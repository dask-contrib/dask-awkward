import os

import dask.config
import yaml

config = dask.config.config

fn = os.path.join(os.path.dirname(__file__), "awkward.yaml")
with open(fn) as f:
    defaults = yaml.safe_load(f)

dask.config.update_defaults(defaults)
