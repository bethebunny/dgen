"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from collections import Optional, Dict

from toy.dialects.toy_ops import (
    Module, FuncOp, Block, ToyValue, AnyToyOp, AnyToyType,
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
    UnrankedTensorType, RankedTensorType, FunctionType,
    type_to_string,
)


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

fn get_result_name(op: AnyToyOp) -> Optional[String]:
    if op.isa[ConstantOp]():
        return String(op[ConstantOp].result)
    if op.isa[TransposeOp]():
        return String(op[TransposeOp].result)
    if op.isa[ReshapeOp]():
        return String(op[ReshapeOp].result)
    if op.isa[MulOp]():
        return String(op[MulOp].result)
    if op.isa[AddOp]():
        return String(op[AddOp].result)
    if op.isa[GenericCallOp]():
        return String(op[GenericCallOp].result)
    return Optional[String]()


fn get_operands(op: AnyToyOp) -> List[String]:
    var result = List[String]()
    if op.isa[TransposeOp]():
        result.append(String(op[TransposeOp].input))
    elif op.isa[ReshapeOp]():
        result.append(String(op[ReshapeOp].input))
    elif op.isa[MulOp]():
        result.append(String(op[MulOp].lhs))
        result.append(String(op[MulOp].rhs))
    elif op.isa[AddOp]():
        result.append(String(op[AddOp].lhs))
        result.append(String(op[AddOp].rhs))
    elif op.isa[GenericCallOp]():
        for i in range(len(op[GenericCallOp].args)):
            result.append(String(op[GenericCallOp].args[i]))
    elif op.isa[PrintOp]():
        result.append(String(op[PrintOp].input))
    elif op.isa[ReturnOp]():
        if op[ReturnOp].value:
            result.append(String(op[ReturnOp].value.value()))
    return result^


fn collect_uses(ops: List[AnyToyOp]) raises -> Dict[String, Int]:
    var counts = Dict[String, Int]()
    for i in range(len(ops)):
        var operands = get_operands(ops[i])
        for j in range(len(operands)):
            var name = operands[j]
            if name in counts:
                counts[name] = counts[name] + 1
            else:
                counts[name] = 1
    return counts^


fn find_def(ops: List[AnyToyOp], name: String) -> Optional[Int]:
    for i in range(len(ops)):
        var r = get_result_name(ops[i])
        if r and r.value() == name:
            return i
    return Optional[Int]()


fn rewrite_uses(mut ops: List[AnyToyOp], old_name: String, new_name: String):
    for i in range(len(ops)):
        if ops[i].isa[TransposeOp]():
            if String(ops[i][TransposeOp].input) == old_name:
                var t_result = String(ops[i][TransposeOp].result)
                var t_type = ops[i][TransposeOp].type.copy()
                ops[i] = AnyToyOp(TransposeOp(
                    result=t_result^, input=String(new_name), type=t_type^,
                ))
        elif ops[i].isa[ReshapeOp]():
            if String(ops[i][ReshapeOp].input) == old_name:
                var r_result = String(ops[i][ReshapeOp].result)
                var r_type = ops[i][ReshapeOp].type.copy()
                ops[i] = AnyToyOp(ReshapeOp(
                    result=r_result^, input=String(new_name), type=r_type^,
                ))
        elif ops[i].isa[MulOp]():
            var m_lhs = String(ops[i][MulOp].lhs)
            var m_rhs = String(ops[i][MulOp].rhs)
            if m_lhs == old_name or m_rhs == old_name:
                var new_lhs = String(new_name) if m_lhs == old_name else m_lhs
                var new_rhs = String(new_name) if m_rhs == old_name else m_rhs
                ops[i] = AnyToyOp(MulOp(
                    result=String(ops[i][MulOp].result),
                    lhs=new_lhs^, rhs=new_rhs^,
                    type=ops[i][MulOp].type.copy(),
                ))
        elif ops[i].isa[AddOp]():
            var a_lhs = String(ops[i][AddOp].lhs)
            var a_rhs = String(ops[i][AddOp].rhs)
            if a_lhs == old_name or a_rhs == old_name:
                var new_lhs = String(new_name) if a_lhs == old_name else a_lhs
                var new_rhs = String(new_name) if a_rhs == old_name else a_rhs
                ops[i] = AnyToyOp(AddOp(
                    result=String(ops[i][AddOp].result),
                    lhs=new_lhs^, rhs=new_rhs^,
                    type=ops[i][AddOp].type.copy(),
                ))
        elif ops[i].isa[GenericCallOp]():
            var changed = False
            var new_args = List[String]()
            for j in range(len(ops[i][GenericCallOp].args)):
                if String(ops[i][GenericCallOp].args[j]) == old_name:
                    new_args.append(String(new_name))
                    changed = True
                else:
                    new_args.append(String(ops[i][GenericCallOp].args[j]))
            if changed:
                ops[i] = AnyToyOp(GenericCallOp(
                    result=String(ops[i][GenericCallOp].result),
                    callee=String(ops[i][GenericCallOp].callee),
                    args=new_args^,
                    type=ops[i][GenericCallOp].type.copy(),
                ))
        elif ops[i].isa[PrintOp]():
            if String(ops[i][PrintOp].input) == old_name:
                ops[i] = AnyToyOp(PrintOp(input=String(new_name)))
        elif ops[i].isa[ReturnOp]():
            if ops[i][ReturnOp].value:
                if String(ops[i][ReturnOp].value.value()) == old_name:
                    ops[i] = AnyToyOp(ReturnOp(value=String(new_name)))


# ===----------------------------------------------------------------------=== #
# Transforms
# ===----------------------------------------------------------------------=== #

fn eliminate_transpose(mut func: FuncOp):
    var to_remove = List[Int]()
    for i in range(len(func.body.ops)):
        if not func.body.ops[i].isa[TransposeOp]():
            continue
        var outer_input = String(func.body.ops[i][TransposeOp].input)
        var outer_result = String(func.body.ops[i][TransposeOp].result)
        var def_idx = find_def(func.body.ops, outer_input)
        if not def_idx:
            continue
        if not func.body.ops[def_idx.value()].isa[TransposeOp]():
            continue
        var inner_input = String(func.body.ops[def_idx.value()][TransposeOp].input)
        rewrite_uses(func.body.ops, outer_result, inner_input)
        to_remove.append(i)

    _remove_indices(func.body.ops, to_remove)


fn fold_constants(mut func: FuncOp):
    for i in range(len(func.body.ops)):
        if not func.body.ops[i].isa[ReshapeOp]():
            continue
        var reshape_input = String(func.body.ops[i][ReshapeOp].input)
        var reshape_result = String(func.body.ops[i][ReshapeOp].result)
        var def_idx = find_def(func.body.ops, reshape_input)
        if not def_idx:
            continue
        if not func.body.ops[def_idx.value()].isa[ConstantOp]():
            continue
        if not func.body.ops[i][ReshapeOp].type.isa[RankedTensorType]():
            continue
        var target_shape = func.body.ops[i][ReshapeOp].type[RankedTensorType].shape.copy()
        # Skip same-shape folds (simplify_reshape handles those)
        if func.body.ops[def_idx.value()][ConstantOp].type.isa[RankedTensorType]():
            var src_shape = func.body.ops[def_idx.value()][ConstantOp].type[RankedTensorType].shape.copy()
            if _shapes_equal(target_shape, src_shape):
                continue
        var cst_values = func.body.ops[def_idx.value()][ConstantOp].value.copy()
        func.body.ops[i] = AnyToyOp(ConstantOp(
            result=reshape_result,
            value=cst_values^,
            shape=target_shape.copy(),
            type=AnyToyType(RankedTensorType(shape=target_shape^)),
        ))


fn simplify_reshape(mut func: FuncOp):
    var to_remove = List[Int]()
    for i in range(len(func.body.ops)):
        if not func.body.ops[i].isa[ReshapeOp]():
            continue
        var reshape_input = String(func.body.ops[i][ReshapeOp].input)
        var reshape_result = String(func.body.ops[i][ReshapeOp].result)
        var def_idx = find_def(func.body.ops, reshape_input)
        if not def_idx:
            continue

        # Reshape of constant with matching shape -> remove
        if func.body.ops[def_idx.value()].isa[ConstantOp]():
            if func.body.ops[i][ReshapeOp].type.isa[RankedTensorType]() and func.body.ops[def_idx.value()][ConstantOp].type.isa[RankedTensorType]():
                var target = func.body.ops[i][ReshapeOp].type[RankedTensorType].shape.copy()
                var src = func.body.ops[def_idx.value()][ConstantOp].type[RankedTensorType].shape.copy()
                if _shapes_equal(target, src):
                    rewrite_uses(func.body.ops, reshape_result, reshape_input)
                    to_remove.append(i)
                    continue

        # Reshape of reshape -> collapse
        if func.body.ops[def_idx.value()].isa[ReshapeOp]():
            var inner_input = String(func.body.ops[def_idx.value()][ReshapeOp].input)
            var reshape_type = func.body.ops[i][ReshapeOp].type.copy()
            func.body.ops[i] = AnyToyOp(ReshapeOp(
                result=reshape_result,
                input=inner_input,
                type=reshape_type^,
            ))

    _remove_indices(func.body.ops, to_remove)


fn eliminate_dead_code(mut func: FuncOp) raises:
    var changed = True
    while changed:
        changed = False
        var uses = collect_uses(func.body.ops)
        var to_remove = List[Int]()
        for i in range(len(func.body.ops)):
            if func.body.ops[i].isa[PrintOp]() or func.body.ops[i].isa[ReturnOp]():
                continue
            var name = get_result_name(func.body.ops[i])
            if not name:
                continue
            if name.value() not in uses:
                to_remove.append(i)
                changed = True
        _remove_indices(func.body.ops, to_remove)


fn _shapes_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


fn _remove_indices(mut ops: List[AnyToyOp], indices: List[Int]):
    if len(indices) == 0:
        return
    for k in range(len(indices) - 1, -1, -1):
        var idx = indices[k]
        for j in range(idx, len(ops) - 1):
            ops[j] = ops[j + 1]
        _ = ops.pop()


# ===----------------------------------------------------------------------=== #
# Pipeline
# ===----------------------------------------------------------------------=== #

fn optimize(m: Module) raises -> Module:
    var functions = List[FuncOp]()
    for i in range(len(m.functions)):
        var func = m.functions[i].copy()
        eliminate_transpose(func)
        fold_constants(func)
        simplify_reshape(func)
        eliminate_dead_code(func)
        functions.append(func^)
    return Module(functions=functions^)
