def register(sizeof):
    @sizeof.register_lazy("awkward")
    def lazy_sizeof_ak_array():
        import awkward as ak

        @sizeof.register(ak.Array)
        def sizeof_ak_array(array):
            return array.nbytes
