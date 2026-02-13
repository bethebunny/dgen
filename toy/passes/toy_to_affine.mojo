"""Ch5: Toy IR to Affine IR lowering."""

from collections import Optional, Dict

from toy.dialects.toy_ops import (
    Module, FuncOp, Block, ToyValue, AnyToyOp, AnyToyType,
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
    UnrankedTensorType, RankedTensorType,
)
from toy.dialects.affine_ops import (
    AffineModule, AffineFuncOp, AffineBlock, AffineValue,
    AnyAffineOp, AnyAffineType,
    AllocOp, DeallocOp, AffineLoadOp, AffineStoreOp,
    AffineForOp, ArithConstantOp, IndexConstantOp,
    ArithMulFOp, ArithAddFOp, AffinePrintOp, AffineReturnOp,
    MemRefType, IndexType, F64Type,
)


struct ToyToAffineLowering(Movable):
    var counter: Int
    var ops: List[AnyAffineOp]
    var shape_map: Dict[String, List[Int]]
    var live_allocs: List[String]

    fn __init__(out self):
        self.counter = 0
        self.ops = List[AnyAffineOp]()
        self.shape_map = Dict[String, List[Int]]()
        self.live_allocs = List[String]()

    fn fresh(mut self) -> String:
        var name = String("a") + String(self.counter)
        self.counter += 1
        return name^

    fn lower_module(mut self, m: Module) raises -> AffineModule:
        var functions = List[AffineFuncOp]()
        for i in range(len(m.functions)):
            functions.append(self.lower_function(m.functions[i]))
        return AffineModule(functions=functions^)

    fn lower_function(mut self, f: FuncOp) raises -> AffineFuncOp:
        self.counter = 0
        self.ops = List[AnyAffineOp]()
        self.shape_map = Dict[String, List[Int]]()
        self.live_allocs = List[String]()

        for i in range(len(f.body.ops)):
            self.lower_op(f.body.ops[i])

        var ops = self.ops^
        self.ops = List[AnyAffineOp]()
        return AffineFuncOp(
            name=String(f.name),
            body=AffineBlock(args=List[AffineValue](), ops=ops^),
        )

    fn lower_op(mut self, op: AnyToyOp) raises:
        if op.isa[ConstantOp]():
            self._lower_constant(op[ConstantOp])
        elif op.isa[TransposeOp]():
            self._lower_transpose(op[TransposeOp])
        elif op.isa[MulOp]():
            self._lower_mul(op[MulOp])
        elif op.isa[AddOp]():
            self._lower_add(op[AddOp])
        elif op.isa[ReshapeOp]():
            self._lower_reshape(op[ReshapeOp])
        elif op.isa[PrintOp]():
            self._lower_print(op[PrintOp])
        elif op.isa[ReturnOp]():
            self._lower_return(op[ReturnOp])

    fn _lower_constant(mut self, op: ConstantOp) raises:
        var shape = op.shape.copy()
        var result_name = String(op.result)
        var alloc_name = self.fresh()

        # Alloc
        self.ops.append(AnyAffineOp(AllocOp(
            result=String(alloc_name), shape=shape.copy(),
        )))
        self.live_allocs.append(String(alloc_name))

        # Nested for loops to store each element
        var values = op.value.copy()
        if len(shape) == 1:
            # 1D: single loop
            var ivar = self.fresh()
            var body = List[AnyAffineOp]()
            for idx in range(Int(shape[0])):
                var cst_name = self.fresh()
                body.append(AnyAffineOp(ArithConstantOp(
                    result=String(cst_name), value=values[idx],
                )))
                var idx_name = self.fresh()
                body.append(AnyAffineOp(IndexConstantOp(
                    result=String(idx_name), value=idx,
                )))
                body.append(AnyAffineOp(AffineStoreOp(
                    value=String(cst_name),
                    memref=String(alloc_name),
                    indices=_make_list1(idx_name),
                )))
            self.ops.append(AnyAffineOp(AffineForOp(
                var_name=String(ivar), lo=0, hi=Int(shape[0]),
                body=body^,
            )))
        elif len(shape) == 2:
            # 2D: nested loops
            var ivar = self.fresh()
            var jvar = self.fresh()
            var outer_body = List[AnyAffineOp]()
            var inner_body = List[AnyAffineOp]()
            var rows = Int(shape[0])
            var cols = Int(shape[1])
            for r in range(rows):
                for c in range(cols):
                    var flat = r * cols + c
                    var cst_name = self.fresh()
                    inner_body.append(AnyAffineOp(ArithConstantOp(
                        result=String(cst_name), value=values[flat],
                    )))
                    var ri_name = self.fresh()
                    inner_body.append(AnyAffineOp(IndexConstantOp(
                        result=String(ri_name), value=r,
                    )))
                    var ci_name = self.fresh()
                    inner_body.append(AnyAffineOp(IndexConstantOp(
                        result=String(ci_name), value=c,
                    )))
                    inner_body.append(AnyAffineOp(AffineStoreOp(
                        value=String(cst_name),
                        memref=String(alloc_name),
                        indices=_make_list2(ri_name, ci_name),
                    )))
            outer_body.append(AnyAffineOp(AffineForOp(
                var_name=String(jvar), lo=0, hi=cols,
                body=inner_body^,
            )))
            self.ops.append(AnyAffineOp(AffineForOp(
                var_name=String(ivar), lo=0, hi=rows,
                body=outer_body^,
            )))

        self.shape_map[result_name] = shape^
        # Map result name to alloc name for later use
        self.shape_map[String("__alloc__") + result_name] = _make_list_from_string(alloc_name)

    fn _lower_transpose(mut self, op: TransposeOp) raises:
        var input_name = String(op.input)
        var result_name = String(op.result)
        var in_shape = self.shape_map[input_name].copy()
        if len(in_shape) != 2:
            raise Error("Transpose only supports 2D tensors")

        var rows = Int(in_shape[0])
        var cols = Int(in_shape[1])
        var out_shape = List[Int]()
        out_shape.append(cols)
        out_shape.append(rows)

        var alloc_name = self.fresh()
        self.ops.append(AnyAffineOp(AllocOp(
            result=String(alloc_name), shape=out_shape.copy(),
        )))
        self.live_allocs.append(String(alloc_name))

        var in_alloc = _get_alloc_name(self.shape_map, input_name)

        # Nested for: load [i,j] from input, store [j,i] to output
        var ivar = self.fresh()
        var jvar = self.fresh()
        var inner_body = List[AnyAffineOp]()

        var load_name = self.fresh()
        inner_body.append(AnyAffineOp(AffineLoadOp(
            result=String(load_name),
            memref=String(in_alloc),
            indices=_make_list2(String(ivar), String(jvar)),
        )))
        inner_body.append(AnyAffineOp(AffineStoreOp(
            value=String(load_name),
            memref=String(alloc_name),
            indices=_make_list2(String(jvar), String(ivar)),
        )))

        var outer_body = List[AnyAffineOp]()
        outer_body.append(AnyAffineOp(AffineForOp(
            var_name=String(jvar), lo=0, hi=cols,
            body=inner_body^,
        )))
        self.ops.append(AnyAffineOp(AffineForOp(
            var_name=String(ivar), lo=0, hi=rows,
            body=outer_body^,
        )))

        self.shape_map[result_name] = out_shape^
        self.shape_map[String("__alloc__") + result_name] = _make_list_from_string(alloc_name)

    fn _lower_mul(mut self, op: MulOp) raises:
        self._lower_binop(String(op.result), String(op.lhs), String(op.rhs), is_mul=True)

    fn _lower_add(mut self, op: AddOp) raises:
        self._lower_binop(String(op.result), String(op.lhs), String(op.rhs), is_mul=False)

    fn _lower_binop(mut self, result_name: String, lhs_name: String, rhs_name: String, is_mul: Bool) raises:
        var shape = self.shape_map[lhs_name].copy()
        var alloc_name = self.fresh()
        self.ops.append(AnyAffineOp(AllocOp(
            result=String(alloc_name), shape=shape.copy(),
        )))
        self.live_allocs.append(String(alloc_name))

        var lhs_alloc = _get_alloc_name(self.shape_map, lhs_name)
        var rhs_alloc = _get_alloc_name(self.shape_map, rhs_name)

        if len(shape) == 1:
            var ivar = self.fresh()
            var body = List[AnyAffineOp]()
            var lv = self.fresh()
            body.append(AnyAffineOp(AffineLoadOp(
                result=String(lv), memref=String(lhs_alloc),
                indices=_make_list1(String(ivar)),
            )))
            var rv = self.fresh()
            body.append(AnyAffineOp(AffineLoadOp(
                result=String(rv), memref=String(rhs_alloc),
                indices=_make_list1(String(ivar)),
            )))
            var res = self.fresh()
            if is_mul:
                body.append(AnyAffineOp(ArithMulFOp(result=String(res), lhs=String(lv), rhs=String(rv))))
            else:
                body.append(AnyAffineOp(ArithAddFOp(result=String(res), lhs=String(lv), rhs=String(rv))))
            body.append(AnyAffineOp(AffineStoreOp(
                value=String(res), memref=String(alloc_name),
                indices=_make_list1(String(ivar)),
            )))
            self.ops.append(AnyAffineOp(AffineForOp(
                var_name=String(ivar), lo=0, hi=Int(shape[0]),
                body=body^,
            )))
        elif len(shape) == 2:
            var ivar = self.fresh()
            var jvar = self.fresh()
            var inner_body = List[AnyAffineOp]()
            var lv = self.fresh()
            inner_body.append(AnyAffineOp(AffineLoadOp(
                result=String(lv), memref=String(lhs_alloc),
                indices=_make_list2(String(ivar), String(jvar)),
            )))
            var rv = self.fresh()
            inner_body.append(AnyAffineOp(AffineLoadOp(
                result=String(rv), memref=String(rhs_alloc),
                indices=_make_list2(String(ivar), String(jvar)),
            )))
            var res = self.fresh()
            if is_mul:
                inner_body.append(AnyAffineOp(ArithMulFOp(result=String(res), lhs=String(lv), rhs=String(rv))))
            else:
                inner_body.append(AnyAffineOp(ArithAddFOp(result=String(res), lhs=String(lv), rhs=String(rv))))
            inner_body.append(AnyAffineOp(AffineStoreOp(
                value=String(res), memref=String(alloc_name),
                indices=_make_list2(String(ivar), String(jvar)),
            )))
            var outer_body = List[AnyAffineOp]()
            outer_body.append(AnyAffineOp(AffineForOp(
                var_name=String(jvar), lo=0, hi=Int(shape[1]),
                body=inner_body^,
            )))
            self.ops.append(AnyAffineOp(AffineForOp(
                var_name=String(ivar), lo=0, hi=Int(shape[0]),
                body=outer_body^,
            )))

        self.shape_map[result_name] = shape^
        self.shape_map[String("__alloc__") + result_name] = _make_list_from_string(alloc_name)

    fn _lower_reshape(mut self, op: ReshapeOp) raises:
        var result_name = String(op.result)
        var input_name = String(op.input)
        # Reshape is a no-op: just update the shape map
        if op.type.isa[RankedTensorType]():
            self.shape_map[result_name] = op.type[RankedTensorType].shape.copy()
        else:
            self.shape_map[result_name] = self.shape_map[input_name].copy()
        # Point to the same alloc
        var alloc = _get_alloc_name(self.shape_map, input_name)
        self.shape_map[String("__alloc__") + result_name] = _make_list_from_string(alloc)

    fn _lower_print(mut self, op: PrintOp) raises:
        var input_name = String(op.input)
        var alloc = _get_alloc_name(self.shape_map, input_name)
        self.ops.append(AnyAffineOp(AffinePrintOp(input=String(alloc))))

    fn _lower_return(mut self, op: ReturnOp) raises:
        # Dealloc all live allocs
        for i in range(len(self.live_allocs)):
            self.ops.append(AnyAffineOp(DeallocOp(input=String(self.live_allocs[i]))))
        if op.value:
            self.ops.append(AnyAffineOp(AffineReturnOp(value=String(op.value.value()))))
        else:
            self.ops.append(AnyAffineOp(AffineReturnOp(value=Optional[String]())))


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

fn _get_alloc_name(shape_map: Dict[String, List[Int]], name: String) raises -> String:
    var key = String("__alloc__") + name
    if key in shape_map:
        var lst = shape_map[key].copy()
        # We encode alloc name as a single-element list with the char codes
        return _string_from_list(lst)
    # Fallback: the name is itself an alloc (e.g., function args)
    return name


fn _make_list_from_string(s: String) -> List[Int]:
    """Encode a string as a list of char codes for storage in shape_map."""
    var result = List[Int]()
    for i in range(len(s)):
        result.append(Int(s.as_bytes()[i]))
    return result^


fn _string_from_list(lst: List[Int]) -> String:
    """Decode a list of char codes back to a string."""
    var result = String("")
    for i in range(len(lst)):
        result += chr(lst[i])
    return result^


fn _make_list1(var s: String) -> List[String]:
    var result = List[String]()
    result.append(s^)
    return result^


fn _make_list2(var s1: String, var s2: String) -> List[String]:
    var result = List[String]()
    result.append(s1^)
    result.append(s2^)
    return result^


fn lower_to_affine(m: Module) raises -> AffineModule:
    var lowering = ToyToAffineLowering()
    return lowering.lower_module(m)
