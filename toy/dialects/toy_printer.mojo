"""IR text serialization for the Toy dialect."""

from toy.dialects.toy_ops import (
    Module, FuncOp, Block, ToyValue, AnyToyOp, AnyToyType,
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
    UnrankedTensorType, RankedTensorType, FunctionType,
    type_to_string,
)
from collections import Optional


fn print_module(m: Module) -> String:
    var s = String("from toy use *\n")
    for i in range(len(m.functions)):
        s += "\n"
        s += print_func(m.functions[i])
    return s


fn print_func(f: FuncOp) -> String:
    var s = String("%" + f.name + " = function (")
    # Parameters
    for i in range(len(f.body.args)):
        if i > 0:
            s += ", "
        s += "%" + f.body.args[i].name + ": " + type_to_string(f.body.args[i].type)
    s += ")"
    # Return type
    if f.func_type.result:
        s += " -> " + type_to_string(f.func_type.result.value())
    s += ":\n"
    # Body
    for i in range(len(f.body.ops)):
        s += "    " + print_op(f.body.ops[i]) + "\n"
    return s


fn print_op(op: AnyToyOp) -> String:
    if op.isa[ConstantOp](): return String(op[ConstantOp])
    if op.isa[TransposeOp](): return String(op[TransposeOp])
    if op.isa[ReshapeOp](): return String(op[ReshapeOp])
    if op.isa[MulOp](): return String(op[MulOp])
    if op.isa[AddOp](): return String(op[AddOp])
    if op.isa[GenericCallOp](): return String(op[GenericCallOp])
    if op.isa[PrintOp](): return String(op[PrintOp])
    if op.isa[ReturnOp](): return String(op[ReturnOp])
    return "<unknown op>"
