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
    var s = String("")
    if op.isa[ConstantOp]():
        op[ConstantOp].write_asm(s)
        return s
    if op.isa[TransposeOp]():
        op[TransposeOp].write_asm(s)
        return s
    if op.isa[ReshapeOp]():
        op[ReshapeOp].write_asm(s)
        return s
    if op.isa[MulOp]():
        op[MulOp].write_asm(s)
        return s
    if op.isa[AddOp]():
        op[AddOp].write_asm(s)
        return s
    if op.isa[GenericCallOp]():
        op[GenericCallOp].write_asm(s)
        return s
    if op.isa[PrintOp]():
        op[PrintOp].write_asm(s)
        return s
    if op.isa[ReturnOp]():
        op[ReturnOp].write_asm(s)
        return s
    return "<unknown op>"
