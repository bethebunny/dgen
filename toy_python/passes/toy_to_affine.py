"""Ch5: Toy IR to Affine IR lowering."""

from __future__ import annotations

from toy_python.dialects import affine, builtin, toy


class ToyToAffineLowering:
    def __init__(self):
        self.ops: list[builtin.Op] = []
        self.shape_map: dict[builtin.Value, list[int]] = {}
        self.alloc_map: dict[builtin.Value, builtin.Value] = {}
        self.live_allocs: list[builtin.Value] = []

    def lower_module(self, m: builtin.Module) -> builtin.Module:
        functions = [self.lower_function(f) for f in m.functions]
        return builtin.Module(functions=functions)

    def lower_function(self, f: builtin.FuncOp) -> builtin.FuncOp:
        self.ops = []
        self.shape_map = {}
        self.alloc_map = {}
        self.live_allocs = []

        for op in f.body.ops:
            self.lower_op(op)

        ops = self.ops
        self.ops = []
        return builtin.FuncOp(
            name=f.name,
            body=builtin.Block(ops=ops),
            func_type=builtin.Function(result=builtin.Nil()),
        )

    def lower_op(self, op: builtin.Op):
        if isinstance(op, toy.ConstantOp):
            self._lower_constant(op)
        elif isinstance(op, toy.TransposeOp):
            self._lower_transpose(op)
        elif isinstance(op, toy.MulOp):
            self._lower_binop(op, op.lhs, op.rhs, is_mul=True)
        elif isinstance(op, toy.AddOp):
            self._lower_binop(op, op.lhs, op.rhs, is_mul=False)
        elif isinstance(op, toy.ReshapeOp):
            self._lower_reshape(op)
        elif isinstance(op, toy.PrintOp):
            self._lower_print(op)
        elif isinstance(op, builtin.ReturnOp):
            self._lower_return(op)

    def _lower_constant(self, op: toy.ConstantOp):
        shape = list(op.shape)

        alloc_op = affine.AllocOp(shape=list(shape))
        self.ops.append(alloc_op)
        self.live_allocs.append(alloc_op)

        values = op.value
        if len(shape) == 1:
            for i, v in enumerate(values):
                cst = builtin.ConstantOp(value=v, type=builtin.F64Type())
                self.ops.append(cst)
                idx = builtin.ConstantOp(value=i, type=builtin.IndexType())
                self.ops.append(idx)
                self.ops.append(affine.StoreOp(value=cst, memref=alloc_op, indices=[idx]))
        elif len(shape) == 2:
            rows, cols = shape[0], shape[1]
            for r in range(rows):
                for c in range(cols):
                    flat = r * cols + c
                    cst = builtin.ConstantOp(value=values[flat], type=builtin.F64Type())
                    self.ops.append(cst)
                    ri = builtin.ConstantOp(value=r, type=builtin.IndexType())
                    self.ops.append(ri)
                    ci = builtin.ConstantOp(value=c, type=builtin.IndexType())
                    self.ops.append(ci)
                    self.ops.append(affine.StoreOp(value=cst, memref=alloc_op, indices=[ri, ci]))

        self.shape_map[op] = shape
        self.alloc_map[op] = alloc_op

    def _lower_transpose(self, op: toy.TransposeOp):
        input_val = op.input
        in_shape = self.shape_map[input_val]
        if len(in_shape) != 2:
            raise RuntimeError("Transpose only supports 2D tensors")

        rows, cols = in_shape[0], in_shape[1]
        out_shape = [cols, rows]

        alloc_op = affine.AllocOp(shape=list(out_shape))
        self.ops.append(alloc_op)
        self.live_allocs.append(alloc_op)

        in_alloc = self.alloc_map.get(input_val, input_val)

        # Nested for: load [i,j] from input, store [j,i] to output
        ivar = builtin.BlockArg(type=affine.IndexType())
        jvar = builtin.BlockArg(type=affine.IndexType())
        inner_body: list[builtin.Op] = []

        load_op = affine.LoadOp(memref=in_alloc, indices=[ivar, jvar])
        inner_body.append(load_op)
        inner_body.append(
            affine.StoreOp(
                value=load_op, memref=alloc_op, indices=[jvar, ivar]
            )
        )

        outer_body: list[builtin.Op] = [
            affine.ForOp(lo=0, hi=cols, body=builtin.Block(ops=inner_body, args=[jvar])),
        ]
        self.ops.append(
            affine.ForOp(lo=0, hi=rows, body=builtin.Block(ops=outer_body, args=[ivar]))
        )

        self.shape_map[op] = out_shape
        self.alloc_map[op] = alloc_op

    def _lower_binop(
        self,
        result_op: builtin.Op,
        lhs_val: builtin.Value,
        rhs_val: builtin.Value,
        is_mul: bool,
    ):
        shape = list(self.shape_map[lhs_val])
        alloc_op = affine.AllocOp(shape=list(shape))
        self.ops.append(alloc_op)
        self.live_allocs.append(alloc_op)

        lhs_alloc = self.alloc_map.get(lhs_val, lhs_val)
        rhs_alloc = self.alloc_map.get(rhs_val, rhs_val)

        if len(shape) == 1:
            ivar = builtin.BlockArg(type=affine.IndexType())
            body: list[builtin.Op] = []
            lv = affine.LoadOp(memref=lhs_alloc, indices=[ivar])
            body.append(lv)
            rv = affine.LoadOp(memref=rhs_alloc, indices=[ivar])
            body.append(rv)
            if is_mul:
                res = affine.ArithMulFOp(lhs=lv, rhs=rv)
            else:
                res = affine.ArithAddFOp(lhs=lv, rhs=rv)
            body.append(res)
            body.append(
                affine.StoreOp(value=res, memref=alloc_op, indices=[ivar])
            )
            self.ops.append(
                affine.ForOp(lo=0, hi=shape[0], body=builtin.Block(ops=body, args=[ivar]))
            )
        elif len(shape) == 2:
            ivar = builtin.BlockArg(type=affine.IndexType())
            jvar = builtin.BlockArg(type=affine.IndexType())
            inner_body: list[builtin.Op] = []
            lv = affine.LoadOp(
                memref=lhs_alloc, indices=[ivar, jvar]
            )
            inner_body.append(lv)
            rv = affine.LoadOp(
                memref=rhs_alloc, indices=[ivar, jvar]
            )
            inner_body.append(rv)
            if is_mul:
                res = affine.ArithMulFOp(lhs=lv, rhs=rv)
            else:
                res = affine.ArithAddFOp(lhs=lv, rhs=rv)
            inner_body.append(res)
            inner_body.append(
                affine.StoreOp(
                    value=res, memref=alloc_op, indices=[ivar, jvar]
                )
            )
            outer_body: list[builtin.Op] = [
                affine.ForOp(
                    lo=0, hi=shape[1], body=builtin.Block(ops=inner_body, args=[jvar])
                ),
            ]
            self.ops.append(
                affine.ForOp(
                    lo=0, hi=shape[0], body=builtin.Block(ops=outer_body, args=[ivar])
                )
            )

        self.shape_map[result_op] = shape
        self.alloc_map[result_op] = alloc_op

    def _lower_reshape(self, op: toy.ReshapeOp):
        input_val = op.input
        # Reshape is a no-op: just update the shape map
        if isinstance(op.type, toy.RankedTensorType):
            self.shape_map[op] = list(op.type.shape)
        else:
            self.shape_map[op] = list(self.shape_map[input_val])
        # Point to the same alloc
        self.alloc_map[op] = self.alloc_map.get(input_val, input_val)

    def _lower_print(self, op: toy.PrintOp):
        input_val = op.input
        alloc = self.alloc_map.get(input_val, input_val)
        self.ops.append(affine.PrintOp(input=alloc))

    def _lower_return(self, op: builtin.ReturnOp):
        # Dealloc all live allocs
        for alloc_val in self.live_allocs:
            self.ops.append(affine.DeallocOp(input=alloc_val))
        self.ops.append(builtin.ReturnOp(value=op.value))


def lower_to_affine(m: builtin.Module) -> builtin.Module:
    lowering = ToyToAffineLowering()
    return lowering.lower_module(m)
