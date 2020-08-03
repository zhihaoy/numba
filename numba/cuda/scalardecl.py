from numba.core import types, errors
from numba.core.typing.templates import (AttributeTemplate, bound_function,
                                         signature, Registry)

registry = Registry()
infer_attr = registry.register_attr


@infer_attr
class FloatAttribute(AttributeTemplate):
    key = types.Float

    @bound_function('scalar.view')
    def resolve_view(self, n, args, kws):
        assert not kws
        ty, = args
        R = ty.dtype
        if R.is_precise() and n.bitwidth == R.bitwidth:
            return signature(R, types.Any)
        else:
            raise errors.TypingError(
                "Float can only be viewed as integers of the same size")


@infer_attr
class IntegerAttribute(AttributeTemplate):
    key = types.Integer

    @bound_function('scalar.view')
    def resolve_view(self, n, args, kws):
        assert not kws
        ty, = args
        R = ty.dtype
        if not R.is_precise() and n.bitwidth == R.bitwidth:
            return signature(R, types.Any)
        else:
            raise errors.TypingError(
                "Integer can only be viewed as floating-point types "
                "of the same size")
