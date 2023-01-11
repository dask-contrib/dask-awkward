def ak_array_sizeof_plugin(sizeof):
    import awkward as ak

    @sizeof.register(ak.Array)
    def sizeof_ak_array(array):
        return array.nbytes
