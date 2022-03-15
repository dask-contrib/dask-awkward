import dask_awkward as dak


def test_force_by_lines_meta(line_delim_records_file) -> None:
    daa1 = dak.from_json(
        [line_delim_records_file] * 5,
        derive_meta_kwargs={"force_by_lines": True},
    )
    daa2 = dak.from_json([line_delim_records_file, line_delim_records_file])
    assert daa1._meta is not None
    assert daa2._meta is not None
    f1 = daa1._meta.layout.form
    f2 = daa2._meta.layout.form
    assert f1 == f2
