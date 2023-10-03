from __future__ import annotations

import dask
import pytest

import dask_awkward as dak


def test_visualize_works(daa):
    query = daa.points.x

    with dask.config.set({"awkward.optimization.on-fail": "raise"}):
        dict(list(query.dask.layers.values())[1])
        dask.compute(query, optimize_graph=True)


def test_basic_root_works():
    pytest.importorskip("hist")
    pytest.importorskip("uproot")
    import hist.dask as hda
    import uproot

    events = uproot.dask(
        {"/tmp/tmp.zODEvn19Lm/nano_dy.root": "Events"},
        steps_per_file=3,
    )

    q1_hist = (
        hda.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events.MET_pt)
    )

    dask.compute(q1_hist)


def test_sample(daa):
    with pytest.raises(ValueError):
        dak.sample(daa)
    with pytest.raises(ValueError):
        dak.sample(daa, 1, 1)

    out = dak.sample(daa, factor=2)
    assert out.npartitions == daa.npartitions
    assert out.compute().tolist()[:5] == daa.compute()[[0, 2, 4, 5, 7]].tolist()

    out = dak.sample(daa, probability=0.5)
    assert out.npartitions == daa.npartitions
    arr = out.compute()
    assert 1 <= len(arr) <= 14
    assert all(a in daa.compute().tolist() for a in arr.tolist())
