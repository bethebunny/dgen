"""Ch5: Affine IR text serialization."""

from toy_python.dialects.affine_ops import (
    AffineModule,
    AffineFuncOp,
    AnyAffineOp,
    AllocOp,
    DeallocOp,
    AffineLoadOp,
    AffineStoreOp,
    AffineForOp,
    ArithConstantOp,
    IndexConstantOp,
    ArithMulFOp,
    ArithAddFOp,
    AffinePrintOp,
    AffineReturnOp,
    format_float,
)


def print_affine_module(m: AffineModule) -> str:
    s = ""
    for i, func in enumerate(m.functions):
        if i > 0:
            s += "\n"
        s += print_affine_func(func)
    return s


def print_affine_func(f: AffineFuncOp) -> str:
    s = f"%{f.name} = function ():\n"
    for op in f.body.ops:
        s += _print_op(op, indent=4)
    return s


def _print_op(op: AnyAffineOp, indent: int) -> str:
    pad = " " * indent
    if isinstance(op, AffineForOp):
        result = f"{pad}AffineFor %{op.var_name} = {op.lo} to {op.hi}:\n"
        for child_op in op.body:
            result += _print_op(child_op, indent + 4)
        return result
    if isinstance(op, AllocOp):
        return f"{pad}%{op.result} = Alloc<{'x'.join(str(d) for d in op.shape)}>()\n"
    if isinstance(op, DeallocOp):
        return f"{pad}Dealloc(%{op.input})\n"
    if isinstance(op, AffineLoadOp):
        return f"{pad}%{op.result} = AffineLoad %{op.memref}[{', '.join(op.indices)}]\n"
    if isinstance(op, AffineStoreOp):
        return (
            f"{pad}AffineStore %{op.value}, %{op.memref}[{', '.join(op.indices)}]\n"
        )
    if isinstance(op, ArithConstantOp):
        return f"{pad}%{op.result} = ArithConstant({format_float(op.value)})\n"
    if isinstance(op, IndexConstantOp):
        return f"{pad}%{op.result} = IndexConstant({op.value})\n"
    if isinstance(op, ArithMulFOp):
        return f"{pad}%{op.result} = MulF(%{op.lhs}, %{op.rhs})\n"
    if isinstance(op, ArithAddFOp):
        return f"{pad}%{op.result} = AddF(%{op.lhs}, %{op.rhs})\n"
    if isinstance(op, AffinePrintOp):
        return f"{pad}PrintMemRef(%{op.input}\n"
    if isinstance(op, AffineReturnOp):
        if op.value is not None:
            return f"{pad}return %{op.value}\n"
        return f"{pad}return\n"
    return f"{pad}<unknown affine op>\n"
