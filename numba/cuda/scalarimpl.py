import numpy as np
from numba.core import types
from numba.core.imputils import Registry

registry = Registry()
lower = registry.lower


def view_implement(nvname):
    def core(context, builder, sig, args):
        ty, _ = sig.args
        R = context.get_value_type(sig.return_type)
        T = context.get_value_type(ty)
        fnty = Type.function(R, [T])
        fn = builder.module.get_or_insert_function(fnty, name=nvname)
        return builder.call(fn, (args[0],))

    return core


for ty, nvname in ((types.float32, '__nv_float_as_int'),
                   (types.float64, '__nv_double_as_longlong'),
                   (types.int32, '__nv_int_as_float'),
                   (types.int64, '__nv_longlong_as_double')):
    lower('scalar.view', ty, types.DTypeSpec)(view_implement(nvname))
