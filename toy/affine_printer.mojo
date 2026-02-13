"""Ch5: Affine IR text serialization."""

from toy.affine import (
    AffineModule, AffineFuncOp, AffineBlock, AffineValue,
    AnyAffineOp, AnyAffineType,
    AllocOp, DeallocOp, AffineLoadOp, AffineStoreOp,
    AffineForOp, ArithConstantOp, IndexConstantOp,
    ArithMulFOp, ArithAddFOp, AffinePrintOp, AffineReturnOp,
    MemRefType, IndexType, F64Type,
)
from collections import Optional


fn print_affine_module(m: AffineModule) -> String:
    var s = String("")
    for i in range(len(m.functions)):
        if i > 0:
            s += "\n"
        s += print_affine_func(m.functions[i])
    return s


fn print_affine_func(f: AffineFuncOp) -> String:
    var s = String("%" + f.name + " = function ():\n")
    for i in range(len(f.body.ops)):
        s += _print_op(f.body.ops[i], indent=4)
    return s


fn _print_op(op: AnyAffineOp, indent: Int) -> String:
    var pad = String("")
    for _ in range(indent):
        pad += " "
    if op.isa[AllocOp]():
        return pad + _print_alloc(op[AllocOp]) + "\n"
    if op.isa[DeallocOp]():
        return pad + "Dealloc(%" + String(op[DeallocOp].input) + ")\n"
    if op.isa[AffineLoadOp]():
        return pad + _print_load(op[AffineLoadOp]) + "\n"
    if op.isa[AffineStoreOp]():
        return pad + _print_store(op[AffineStoreOp]) + "\n"
    if op.isa[AffineForOp]():
        return _print_for(op[AffineForOp], indent)
    if op.isa[ArithConstantOp]():
        return pad + _print_arith_const(op[ArithConstantOp]) + "\n"
    if op.isa[IndexConstantOp]():
        return pad + "%" + String(op[IndexConstantOp].result) + " = IndexConstant(" + String(op[IndexConstantOp].value) + ")\n"
    if op.isa[ArithMulFOp]():
        return pad + "%" + String(op[ArithMulFOp].result) + " = MulF(%" + String(op[ArithMulFOp].lhs) + ", %" + String(op[ArithMulFOp].rhs) + ")\n"
    if op.isa[ArithAddFOp]():
        return pad + "%" + String(op[ArithAddFOp].result) + " = AddF(%" + String(op[ArithAddFOp].lhs) + ", %" + String(op[ArithAddFOp].rhs) + ")\n"
    if op.isa[AffinePrintOp]():
        return pad + "PrintMemRef(%" + String(op[AffinePrintOp].input) + ")\n"
    if op.isa[AffineReturnOp]():
        if op[AffineReturnOp].value:
            return pad + "return %" + String(op[AffineReturnOp].value.value()) + "\n"
        return pad + "return\n"
    return pad + "<unknown affine op>\n"


fn _print_alloc(op: AllocOp) -> String:
    var s = String("%" + op.result + " = Alloc<")
    for i in range(len(op.shape)):
        if i > 0:
            s += "x"
        s += String(op.shape[i])
    s += ">()"
    return s


fn _print_load(op: AffineLoadOp) -> String:
    var s = String("%" + op.result + " = AffineLoad %" + op.memref + "[")
    for i in range(len(op.indices)):
        if i > 0:
            s += ", "
        s += "%" + op.indices[i]
    s += "]"
    return s


fn _print_store(op: AffineStoreOp) -> String:
    var s = String("AffineStore %" + op.value + ", %" + op.memref + "[")
    for i in range(len(op.indices)):
        if i > 0:
            s += ", "
        s += "%" + op.indices[i]
    s += "]"
    return s


fn _print_for(op: AffineForOp, indent: Int) -> String:
    var pad = String("")
    for _ in range(indent):
        pad += " "
    var s = pad + "AffineFor %" + String(op.var_name) + " = " + String(op.lo) + " to " + String(op.hi) + ":\n"
    for i in range(len(op.body)):
        s += _print_op(op.body[i], indent + 4)
    return s


fn _print_arith_const(op: ArithConstantOp) -> String:
    var v = op.value
    var iv = Int(v)
    if Float64(iv) == v:
        return "%" + op.result + " = ArithConstant(" + String(iv) + ".0)"
    return "%" + op.result + " = ArithConstant(" + String(v) + ")"
