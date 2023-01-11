from dask.sizeof import sizeof


@sizeof.register_lazy("awkward")
def register_awkward():
    import awkward as ak

    @sizeof.register(ak.Array)
    def sizeof_ak_array(array):
        return array.nbytes
