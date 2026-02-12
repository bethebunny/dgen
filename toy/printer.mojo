"""IR text serialization for the Toy dialect."""

from toy.toy import (
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
    if op.isa[ConstantOp]():
        return print_constant(op[ConstantOp])
    if op.isa[TransposeOp]():
        return print_transpose(op[TransposeOp])
    if op.isa[ReshapeOp]():
        return print_reshape(op[ReshapeOp])
    if op.isa[MulOp]():
        return print_mul(op[MulOp])
    if op.isa[AddOp]():
        return print_add(op[AddOp])
    if op.isa[GenericCallOp]():
        return print_generic_call(op[GenericCallOp])
    if op.isa[PrintOp]():
        return print_print(op[PrintOp])
    if op.isa[ReturnOp]():
        return print_return(op[ReturnOp])
    return "<unknown op>"


fn format_shape(shape: List[Int]) -> String:
    var s = String("<")
    for i in range(len(shape)):
        if i > 0:
            s += "x"
        s += String(shape[i])
    s += ">"
    return s


fn format_float_list(values: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(values)):
        if i > 0:
            s += ", "
        # Format float: if it's a whole number, show as X.0
        var v = values[i]
        var iv = Int(v)
        if Float64(iv) == v:
            s += String(iv) + ".0"
        else:
            s += String(v)
    s += "]"
    return s


fn print_constant(op: ConstantOp) -> String:
    return "%" + op.result + " = Constant(" + format_shape(op.shape) + " " + format_float_list(op.value) + ") : " + type_to_string(op.type)


fn print_transpose(op: TransposeOp) -> String:
    return "%" + op.result + " = Transpose(%" + op.input + ") : " + type_to_string(op.type)


fn print_reshape(op: ReshapeOp) -> String:
    return "%" + op.result + " = Reshape(%" + op.input + ") : " + type_to_string(op.type)


fn print_mul(op: MulOp) -> String:
    return "%" + op.result + " = Mul(%" + op.lhs + ", %" + op.rhs + ") : " + type_to_string(op.type)


fn print_add(op: AddOp) -> String:
    return "%" + op.result + " = Add(%" + op.lhs + ", %" + op.rhs + ") : " + type_to_string(op.type)


fn print_generic_call(op: GenericCallOp) -> String:
    var s = String("%" + op.result + " = GenericCall @" + op.callee + "(")
    for i in range(len(op.args)):
        if i > 0:
            s += ", "
        s += "%" + op.args[i]
    s += ") : " + type_to_string(op.type)
    return s


fn print_print(op: PrintOp) -> String:
    return "Print(%" + op.input + ")"


fn print_return(op: ReturnOp) -> String:
    if op.value:
        return "return %" + op.value.value()
    return "return"
