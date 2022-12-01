try:
    from awkward._typetracer import MaybeNone, OneOf, UnknownScalar
except ModuleNotFoundError:
    from awkward._nplikes.typetracer import (
        MaybeNone,
        OneOf,
        unknown_scalar,
        is_unknown_scalar,
    )
else:

    def is_unknown_scalar(obj) -> bool:
        return isinstance(obj, UnknownScalar)

    def unknown_scalar(dtype):
        return UnknownScalar(dtype)


__all__ = ("MaybeNone", "OneOf", "is_unknown_scalar", "unknown_scalar")
