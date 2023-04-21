from __future__ import annotations

from pathlib import Path

import dask
import pytest

import dask_awkward as dak


def test_necessary_columns(
    daa: dak.Array, tmpdir_factory: pytest.TempdirFactory
) -> None:
    z = daa.points.x + daa.points.y
    for k, v in dak.necessary_columns(z).items():
        assert set(v) == {"points.x", "points.y"}

    w = dak.to_parquet(
        daa.points.x, str(Path(tmpdir_factory.mktemp("pq")) / "out"), compute=False
    )
    for k, v in dak.necessary_columns(w).items():
        assert set(v) == {"points.x"}

    q = {"z": z, "w": w}
    for k, v in dak.necessary_columns(q).items():
        assert set(v) == {"points.x", "points.y"}


def test_visualize_works(daa):
    query = daa.points.x

    with dask.config.set({"awkward.optimization.on-fail": "raise"}):
        dict(list(query.dask.layers.values())[1])
        dask.compute(query, optimize_graph=True)


def test_basic_root_works(daa):
    pytest.importorskip("hist")
    pytest.importorskip("uproot")
    import hist.dask as hda
    import uproot

    events = uproot.dask(
        {
            "https://github.com/CoffeaTeam/coffea/blob/master/"
            "tests/samples/nano_dy.root?raw=true": "Events"
        },
        steps_per_file=3,
    )

    q1_hist = (
        hda.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events.MET_pt)
    )

    dask.compute(q1_hist)
