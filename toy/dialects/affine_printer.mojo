"""Ch5: Affine IR text serialization."""

from toy.dialects.affine_ops import (
    AffineModule,
    AffineFuncOp,
    AffineBlock,
    AffineValue,
    AnyAffineOp,
    AnyAffineType,
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
    MemRefType,
    IndexType,
    F64Type,
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
    var s = String("")
    if op.isa[AffineForOp]():
        var for_op = op[AffineForOp].copy()
        for_op.write_asm(s)
        var result = pad + s + "\n"
        for i in range(len(for_op.body)):
            result += _print_op(for_op.body[i], indent + 4)
        return result
    if op.isa[AllocOp]():
        op[AllocOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[DeallocOp]():
        op[DeallocOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[AffineLoadOp]():
        op[AffineLoadOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[AffineStoreOp]():
        op[AffineStoreOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[ArithConstantOp]():
        op[ArithConstantOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[IndexConstantOp]():
        op[IndexConstantOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[ArithMulFOp]():
        op[ArithMulFOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[ArithAddFOp]():
        op[ArithAddFOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[AffinePrintOp]():
        op[AffinePrintOp].write_asm(s)
        return pad + s + "\n"
    if op.isa[AffineReturnOp]():
        op[AffineReturnOp].write_asm(s)
        return pad + s + "\n"
    return pad + "<unknown affine op>\n"
