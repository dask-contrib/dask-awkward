from __future__ import annotations

from pathlib import Path

import dask
import pytest

import dask_awkward as dak

test_uproot_path = Path(__file__).parent / "test-uproot"


def test_report_necessary_buffers(
    daa: dak.Array, tmpdir_factory: pytest.TempdirFactory
) -> None:
    z = daa.points.x + daa.points.y
    for k, v in dak.report_necessary_buffers(z).items():
        assert v == (
            frozenset(
                {
                    repr(("@", "points")) + "-offsets",
                    repr(("@", "points", None, "y")) + "-data",
                    repr(("@", "points", None, "x")) + "-data",
                }
            ),
            frozenset(),
        )

    w = dak.to_parquet(
        daa.points.x, str(Path(tmpdir_factory.mktemp("pq")) / "out"), compute=False
    )
    for k, v in dak.report_necessary_buffers(w).items():
        assert v == (
            # frozenset({"@.points-offsets", "@.points.content.x-data"}),
            frozenset(
                {
                    repr(("@", "points")) + "-offsets",
                    repr(("@", "points", None, "x")) + "-data",
                }
            ),
            frozenset(),
        )

    q = {"z": z, "w": w}
    for k, v in dak.report_necessary_buffers(q).items():
        assert v == (
            frozenset(
                {
                    repr(("@", "points")) + "-offsets",
                    repr(("@", "points", None, "y")) + "-data",
                    repr(("@", "points", None, "x")) + "-data",
                }
            ),
            frozenset(),
        )

    z = dak.zeros_like(daa.points.x)
    for k, v in dak.report_necessary_buffers(z).items():
        assert v == (
            frozenset({repr(("@", "points")) + "-offsets"}),
            frozenset({repr(("@", "points", None, "x")) + "-data"}),
        )


def test_report_necessary_columns(daa: dak.Array) -> None:
    result = dak.min(daa.points.x, axis=1)
    rep = dak.report_necessary_columns(result)
    for k, v in rep.items():
        assert v is not None
        assert sorted(["points.x"]) == sorted(v)

    result = dak.zeros_like(daa.points.y)
    rep = dak.report_necessary_columns(result)
    for k, v in rep.items():
        assert v is not None
        points, coord = list(v)[0].split(".")
        assert points == "points"
        assert coord in ["x", "y"]


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
        {test_uproot_path / "nano_dy.root": "Events"},
        steps_per_file=3,
    )

    q1_hist = (
        hda.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events.MET_pt)
    )

    columns = list(dak.necessary_columns(q1_hist).values())[0]
    assert columns == frozenset({"MET_pt"})
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
