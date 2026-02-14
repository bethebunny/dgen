"""IR text serialization for the Toy dialect."""

from toy_python.dialects.toy_ops import (
    Module,
    FuncOp,
    AnyToyOp,
    ConstantOp,
    TransposeOp,
    ReshapeOp,
    MulOp,
    AddOp,
    GenericCallOp,
    PrintOp,
    ReturnOp,
    type_to_string,
    format_shape,
    format_float_list,
)


def print_module(m: Module) -> str:
    s = "from toy use *\n"
    for func in m.functions:
        s += "\n"
        s += print_func(func)
    return s


def print_func(f: FuncOp) -> str:
    s = "%" + f.name + " = function ("
    for i, arg in enumerate(f.body.args):
        if i > 0:
            s += ", "
        s += "%" + arg.name + ": " + type_to_string(arg.type)
    s += ")"
    if f.func_type.result is not None:
        s += " -> " + type_to_string(f.func_type.result)
    s += ":\n"
    for op in f.body.ops:
        s += "    " + print_op(op) + "\n"
    return s


def print_op(op: AnyToyOp) -> str:
    if isinstance(op, ConstantOp):
        return (
            f"%{op.result} = Constant({format_shape(op.shape)} "
            f"{format_float_list(op.value)}) : {type_to_string(op.type)}"
        )
    if isinstance(op, TransposeOp):
        return f"%{op.result} = Transpose(%{op.input}) : {type_to_string(op.type)}"
    if isinstance(op, ReshapeOp):
        return f"%{op.result} = Reshape(%{op.input}) : {type_to_string(op.type)}"
    if isinstance(op, MulOp):
        return f"%{op.result} = Mul(%{op.lhs}, %{op.rhs}) : {type_to_string(op.type)}"
    if isinstance(op, AddOp):
        return f"%{op.result} = Add(%{op.lhs}, %{op.rhs}) : {type_to_string(op.type)}"
    if isinstance(op, GenericCallOp):
        args_str = ", ".join(f"%{a}" for a in op.args)
        return (
            f"%{op.result} = GenericCall @{op.callee}({args_str}) : "
            f"{type_to_string(op.type)}"
        )
    if isinstance(op, PrintOp):
        return f"Print(%{op.input})"
    if isinstance(op, ReturnOp):
        if op.value is not None:
            return f"return %{op.value}"
        return "return"
    return "<unknown op>"
