"""Ch5: Toy IR to Affine IR lowering."""

from __future__ import annotations

from toy_python.dialects.toy import (
    Module,
    FuncOp,
    AnyToyOp,
    ConstantOp,
    TransposeOp,
    ReshapeOp,
    MulOp,
    AddOp,
    PrintOp,
    ReturnOp,
    RankedTensorType,
)
from toy_python.dialects.affine import (
    AffineModule,
    AffineFuncOp,
    AffineBlock,
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
)


class ToyToAffineLowering:
    def __init__(self):
        self.counter = 0
        self.ops: list[AnyAffineOp] = []
        self.shape_map: dict[str, list[int]] = {}
        self.alloc_map: dict[str, str] = {}
        self.live_allocs: list[str] = []

    def fresh(self) -> str:
        name = f"a{self.counter}"
        self.counter += 1
        return name

    def lower_module(self, m: Module) -> AffineModule:
        functions = [self.lower_function(f) for f in m.functions]
        return AffineModule(functions=functions)

    def lower_function(self, f: FuncOp) -> AffineFuncOp:
        self.counter = 0
        self.ops = []
        self.shape_map = {}
        self.alloc_map = {}
        self.live_allocs = []

        for op in f.body.ops:
            self.lower_op(op)

        ops = self.ops
        self.ops = []
        return AffineFuncOp(
            name=f.name,
            body=AffineBlock(args=[], ops=ops),
        )

    def lower_op(self, op: AnyToyOp):
        if isinstance(op, ConstantOp):
            self._lower_constant(op)
        elif isinstance(op, TransposeOp):
            self._lower_transpose(op)
        elif isinstance(op, MulOp):
            self._lower_binop(op.result, op.lhs, op.rhs, is_mul=True)
        elif isinstance(op, AddOp):
            self._lower_binop(op.result, op.lhs, op.rhs, is_mul=False)
        elif isinstance(op, ReshapeOp):
            self._lower_reshape(op)
        elif isinstance(op, PrintOp):
            self._lower_print(op)
        elif isinstance(op, ReturnOp):
            self._lower_return(op)

    def _lower_constant(self, op: ConstantOp):
        shape = list(op.shape)
        result_name = op.result
        alloc_name = self.fresh()

        # Alloc
        self.ops.append(AllocOp(result=alloc_name, shape=list(shape)))
        self.live_allocs.append(alloc_name)

        # Nested for loops to store each element
        values = op.value
        if len(shape) == 1:
            # 1D: single loop
            ivar = self.fresh()
            body: list[AnyAffineOp] = []
            for idx in range(shape[0]):
                cst_name = self.fresh()
                body.append(ArithConstantOp(result=cst_name, value=values[idx]))
                idx_name = self.fresh()
                body.append(IndexConstantOp(result=idx_name, value=idx))
                body.append(
                    AffineStoreOp(
                        value=cst_name, memref=alloc_name, indices=[idx_name]
                    )
                )
            self.ops.append(
                AffineForOp(var_name=ivar, lo=0, hi=shape[0], body=body)
            )
        elif len(shape) == 2:
            # 2D: nested loops
            ivar = self.fresh()
            jvar = self.fresh()
            rows, cols = shape[0], shape[1]
            inner_body: list[AnyAffineOp] = []
            for r in range(rows):
                for c in range(cols):
                    flat = r * cols + c
                    cst_name = self.fresh()
                    inner_body.append(
                        ArithConstantOp(result=cst_name, value=values[flat])
                    )
                    ri_name = self.fresh()
                    inner_body.append(
                        IndexConstantOp(result=ri_name, value=r)
                    )
                    ci_name = self.fresh()
                    inner_body.append(
                        IndexConstantOp(result=ci_name, value=c)
                    )
                    inner_body.append(
                        AffineStoreOp(
                            value=cst_name,
                            memref=alloc_name,
                            indices=[ri_name, ci_name],
                        )
                    )
            outer_body: list[AnyAffineOp] = [
                AffineForOp(var_name=jvar, lo=0, hi=cols, body=inner_body),
            ]
            self.ops.append(
                AffineForOp(var_name=ivar, lo=0, hi=rows, body=outer_body)
            )

        self.shape_map[result_name] = shape
        self.alloc_map[result_name] = alloc_name

    def _lower_transpose(self, op: TransposeOp):
        input_name = op.input
        result_name = op.result
        in_shape = self.shape_map[input_name]
        if len(in_shape) != 2:
            raise RuntimeError("Transpose only supports 2D tensors")

        rows, cols = in_shape[0], in_shape[1]
        out_shape = [cols, rows]

        alloc_name = self.fresh()
        self.ops.append(AllocOp(result=alloc_name, shape=list(out_shape)))
        self.live_allocs.append(alloc_name)

        in_alloc = self.alloc_map.get(input_name, input_name)

        # Nested for: load [i,j] from input, store [j,i] to output
        ivar = self.fresh()
        jvar = self.fresh()
        inner_body: list[AnyAffineOp] = []

        load_name = self.fresh()
        inner_body.append(
            AffineLoadOp(
                result=load_name, memref=in_alloc, indices=[ivar, jvar]
            )
        )
        inner_body.append(
            AffineStoreOp(
                value=load_name, memref=alloc_name, indices=[jvar, ivar]
            )
        )

        outer_body: list[AnyAffineOp] = [
            AffineForOp(var_name=jvar, lo=0, hi=cols, body=inner_body),
        ]
        self.ops.append(
            AffineForOp(var_name=ivar, lo=0, hi=rows, body=outer_body)
        )

        self.shape_map[result_name] = out_shape
        self.alloc_map[result_name] = alloc_name

    def _lower_binop(
        self,
        result_name: str,
        lhs_name: str,
        rhs_name: str,
        is_mul: bool,
    ):
        shape = list(self.shape_map[lhs_name])
        alloc_name = self.fresh()
        self.ops.append(AllocOp(result=alloc_name, shape=list(shape)))
        self.live_allocs.append(alloc_name)

        lhs_alloc = self.alloc_map.get(lhs_name, lhs_name)
        rhs_alloc = self.alloc_map.get(rhs_name, rhs_name)

        if len(shape) == 1:
            ivar = self.fresh()
            body: list[AnyAffineOp] = []
            lv = self.fresh()
            body.append(
                AffineLoadOp(result=lv, memref=lhs_alloc, indices=[ivar])
            )
            rv = self.fresh()
            body.append(
                AffineLoadOp(result=rv, memref=rhs_alloc, indices=[ivar])
            )
            res = self.fresh()
            if is_mul:
                body.append(ArithMulFOp(result=res, lhs=lv, rhs=rv))
            else:
                body.append(ArithAddFOp(result=res, lhs=lv, rhs=rv))
            body.append(
                AffineStoreOp(value=res, memref=alloc_name, indices=[ivar])
            )
            self.ops.append(
                AffineForOp(var_name=ivar, lo=0, hi=shape[0], body=body)
            )
        elif len(shape) == 2:
            ivar = self.fresh()
            jvar = self.fresh()
            inner_body: list[AnyAffineOp] = []
            lv = self.fresh()
            inner_body.append(
                AffineLoadOp(
                    result=lv, memref=lhs_alloc, indices=[ivar, jvar]
                )
            )
            rv = self.fresh()
            inner_body.append(
                AffineLoadOp(
                    result=rv, memref=rhs_alloc, indices=[ivar, jvar]
                )
            )
            res = self.fresh()
            if is_mul:
                inner_body.append(ArithMulFOp(result=res, lhs=lv, rhs=rv))
            else:
                inner_body.append(ArithAddFOp(result=res, lhs=lv, rhs=rv))
            inner_body.append(
                AffineStoreOp(
                    value=res, memref=alloc_name, indices=[ivar, jvar]
                )
            )
            outer_body: list[AnyAffineOp] = [
                AffineForOp(
                    var_name=jvar, lo=0, hi=shape[1], body=inner_body
                ),
            ]
            self.ops.append(
                AffineForOp(var_name=ivar, lo=0, hi=shape[0], body=outer_body)
            )

        self.shape_map[result_name] = shape
        self.alloc_map[result_name] = alloc_name

    def _lower_reshape(self, op: ReshapeOp):
        result_name = op.result
        input_name = op.input
        # Reshape is a no-op: just update the shape map
        if isinstance(op.type, RankedTensorType):
            self.shape_map[result_name] = list(op.type.shape)
        else:
            self.shape_map[result_name] = list(self.shape_map[input_name])
        # Point to the same alloc
        self.alloc_map[result_name] = self.alloc_map.get(
            input_name, input_name
        )

    def _lower_print(self, op: PrintOp):
        input_name = op.input
        alloc = self.alloc_map.get(input_name, input_name)
        self.ops.append(AffinePrintOp(input=alloc))

    def _lower_return(self, op: ReturnOp):
        # Dealloc all live allocs
        for alloc_name in self.live_allocs:
            self.ops.append(DeallocOp(input=alloc_name))
        self.ops.append(AffineReturnOp(value=op.value))


def lower_to_affine(m: Module) -> AffineModule:
    lowering = ToyToAffineLowering()
    return lowering.lower_module(m)
