"""Ch5: Toy IR to Affine IR lowering."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin
from dgen.dialects.builtin import FunctionOp, Index, List, Nil
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import affine, shape_constant, toy


def _make_index_list(
    values: Sequence[dgen.Value],
) -> tuple[list[dgen.Op], dgen.Value]:
    """Create a PackOp for an index list.

    Returns (ops_to_emit, pack_value).
    """
    list_type = List(element_type=Index())
    pack_op = PackOp(values=list(values), type=list_type)
    return [pack_op], pack_op


class ToyToAffine(Pass):
    allow_unregistered_ops = True  # ConstantOp, ChainOp, AddIndexOp pass through

    def __init__(self) -> None:
        self.live_allocs: list[dgen.Value] = []

    def run(self, module: Module) -> Module:
        functions = [self._lower_function(f) for f in module.functions]
        return Module(functions=functions)

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self.live_allocs = []
        self._run_block(f.body)
        current_val = f.body.result
        for alloc_val in self.live_allocs:
            underlying = alloc_val
            while isinstance(underlying, builtin.ChainOp):
                underlying = underlying.lhs
            dealloc = affine.DeallocOp(input=underlying)
            if isinstance(current_val, Nil):
                current_val = dealloc
            else:
                current_val = builtin.ChainOp(
                    lhs=current_val, rhs=dealloc, type=current_val.type
                )
        f.body.result = current_val
        return FunctionOp(name=f.name, body=f.body, result=f.result)

    def _make_alloc(self, shape: dgen.Value) -> affine.AllocOp:
        return affine.AllocOp(shape=shape, type=affine.MemRef(shape=shape))

    def _nested_for_op(
        self,
        shape: Sequence[int],
        body_fn: Callable[[Sequence[dgen.Value]], list[dgen.Op]],
    ) -> affine.ForOp:
        """Build nested ForOps for each dimension. Returns the outermost ForOp."""
        ivars: list[dgen.Value] = [BlockArgument(type=builtin.Index()) for _ in shape]
        ops = list(body_fn(ivars))
        for dim, var in reversed(list(zip(shape, ivars))):
            ops = [
                affine.ForOp(
                    lo=builtin.Index().constant(0),
                    hi=builtin.Index().constant(dim),
                    body=dgen.Block(result=ops[-1], args=[var]),
                )
            ]
        return ops[0]

    @lowering_for(ConstantOp)
    def lower_constant(self, op: ConstantOp, rewriter: Rewriter) -> bool:
        if not isinstance(op.type, toy.Tensor):
            return False
        new_const = ConstantOp(value=op.value, type=op.type)
        rewriter.replace_uses(op, new_const)
        self.live_allocs.append(new_const)
        return True

    @lowering_for(toy.TransposeOp)
    def lower_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.type, toy.Tensor)
        assert isinstance(op.input.type, toy.Tensor)
        in_shape = op.input.type.shape.__constant__.to_json()
        in_alloc = op.input

        alloc_op = self._make_alloc(op.type.shape)
        self.live_allocs.append(alloc_op)

        def body(ivars: Sequence[dgen.Value]) -> list[dgen.Op]:
            load_idx_ops, load_idx = _make_index_list(list(ivars))
            store_idx_ops, store_idx = _make_index_list(list(reversed(ivars)))
            load = affine.LoadOp(memref=in_alloc, indices=load_idx)
            store = affine.StoreOp(value=load, memref=alloc_op, indices=store_idx)
            return load_idx_ops + [load] + store_idx_ops + [store]

        for_op = self._nested_for_op(in_shape, body)
        # Chain in_alloc before for_op so walk_ops visits it first (ordering fix).
        inner_chain = builtin.ChainOp(lhs=in_alloc, rhs=for_op, type=alloc_op.type)
        chain = builtin.ChainOp(lhs=alloc_op, rhs=inner_chain, type=alloc_op.type)
        rewriter.replace_uses(op, chain)
        return True

    @lowering_for(toy.MulOp)
    def lower_mul(self, op: toy.MulOp, rewriter: Rewriter) -> bool:
        return self._lower_binop(op, op.lhs, op.rhs, affine.MulFOp, rewriter)

    @lowering_for(toy.AddOp)
    def lower_add(self, op: toy.AddOp, rewriter: Rewriter) -> bool:
        return self._lower_binop(op, op.lhs, op.rhs, affine.AddFOp, rewriter)

    def _lower_binop(
        self,
        op: dgen.Op,
        lhs_val: dgen.Value,
        rhs_val: dgen.Value,
        binop_cls: type,
        rewriter: Rewriter,
    ) -> bool:
        assert isinstance(lhs_val.type, (toy.Tensor, affine.MemRef))
        shape = lhs_val.type.shape.__constant__.to_json()
        assert isinstance(shape, list)
        alloc_op = self._make_alloc(lhs_val.type.shape)
        self.live_allocs.append(alloc_op)
        lhs_alloc = lhs_val
        rhs_alloc = rhs_val

        def body(ivars: Sequence[dgen.Value]) -> list[dgen.Op]:
            idx_ops, idx = _make_index_list(list(ivars))
            lv = affine.LoadOp(memref=lhs_alloc, indices=idx)
            rv = affine.LoadOp(memref=rhs_alloc, indices=idx)
            res = binop_cls(lhs=lv, rhs=rv)
            store = affine.StoreOp(value=res, memref=alloc_op, indices=idx)
            return idx_ops + [lv, rv, res, store]

        for_op = self._nested_for_op(shape, body)
        # Chain lhs/rhs before for_op so walk_ops visits inputs first (ordering fix).
        inner = builtin.ChainOp(lhs=rhs_alloc, rhs=for_op, type=alloc_op.type)
        inner2 = builtin.ChainOp(lhs=lhs_alloc, rhs=inner, type=alloc_op.type)
        chain = builtin.ChainOp(lhs=alloc_op, rhs=inner2, type=alloc_op.type)
        rewriter.replace_uses(op, chain)
        return True

    @lowering_for(toy.ReshapeOp)
    def lower_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, op.input)
        return True

    @lowering_for(toy.PrintOp)
    def lower_print(self, op: toy.PrintOp, rewriter: Rewriter) -> bool:
        # op.input was already updated by replace_uses if from a previous handler.
        # Keep any ChainOp reference intact so the for-loop stays graph-reachable.
        print_op = affine.PrintMemrefOp(input=op.input)
        rewriter.replace_uses(op, print_op)
        return True

    @lowering_for(toy.DimSizeOp)
    def lower_dim_size(self, op: toy.DimSizeOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.input.type, (toy.Tensor, affine.MemRef))
        shape = op.input.type.shape.__constant__.to_json()
        assert isinstance(shape, list)
        axis = op.axis.__constant__.to_json()
        assert isinstance(axis, int)
        const = ConstantOp(value=shape[axis], type=Index())
        rewriter.replace_uses(op, const)
        return True

    @lowering_for(toy.TileOp)
    def lower_tile(self, op: toy.TileOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.count, ConstantOp)
        count = op.count.__constant__.to_json()
        assert isinstance(count, int)
        assert isinstance(op.input.type, toy.Tensor)
        input_shape = op.input.type.shape.__constant__.to_json()
        output_shape: list[int] = [count] + input_shape
        in_alloc = op.input

        # Unwrap ChainOp to get the underlying MemRef
        underlying = in_alloc
        while isinstance(underlying, builtin.ChainOp):
            underlying = underlying.lhs

        alloc_op = self._make_alloc(shape_constant(output_shape))
        self.live_allocs.append(alloc_op)

        def body(ivars: Sequence[dgen.Value]) -> list[dgen.Op]:
            iv = list(ivars)
            load_idx_ops, load_idx = _make_index_list(iv[1:])
            store_idx_ops, store_idx = _make_index_list(iv)
            load = affine.LoadOp(memref=underlying, indices=load_idx)
            store = affine.StoreOp(value=load, memref=alloc_op, indices=store_idx)
            return load_idx_ops + [load] + store_idx_ops + [store]

        for_op = self._nested_for_op(output_shape, body)
        # Chain in_alloc before for_op so walk_ops visits it first (ordering fix).
        inner_chain = builtin.ChainOp(lhs=in_alloc, rhs=for_op, type=alloc_op.type)
        chain = builtin.ChainOp(lhs=alloc_op, rhs=inner_chain, type=alloc_op.type)
        rewriter.replace_uses(op, chain)
        return True

    @lowering_for(toy.ConcatOp)
    def lower_concat(self, op: toy.ConcatOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.lhs.type, (toy.Tensor, affine.MemRef))
        assert isinstance(op.rhs.type, (toy.Tensor, affine.MemRef))
        lhs_shape = op.lhs.type.shape.__constant__.to_json()
        rhs_shape = op.rhs.type.shape.__constant__.to_json()
        assert isinstance(lhs_shape, list)
        assert isinstance(rhs_shape, list)
        axis = op.axis.__constant__.to_json()
        assert isinstance(axis, int)

        output_shape = list(lhs_shape)
        output_shape[axis] = lhs_shape[axis] + rhs_shape[axis]

        lhs_alloc = op.lhs
        rhs_alloc = op.rhs
        # Unwrap ChainOps
        underlying_lhs = lhs_alloc
        while isinstance(underlying_lhs, builtin.ChainOp):
            underlying_lhs = underlying_lhs.lhs
        underlying_rhs = rhs_alloc
        while isinstance(underlying_rhs, builtin.ChainOp):
            underlying_rhs = underlying_rhs.lhs

        alloc_op = self._make_alloc(shape_constant(output_shape))
        self.live_allocs.append(alloc_op)

        def lhs_body(ivars: Sequence[dgen.Value]) -> list[dgen.Op]:
            idx_ops, idx = _make_index_list(list(ivars))
            load = affine.LoadOp(memref=underlying_lhs, indices=idx)
            store = affine.StoreOp(value=load, memref=alloc_op, indices=idx)
            return idx_ops + [load, store]

        lhs_for = self._nested_for_op(lhs_shape, lhs_body)

        offset_const = ConstantOp(value=lhs_shape[axis], type=builtin.Index())

        def rhs_body(ivars: Sequence[dgen.Value]) -> list[dgen.Op]:
            load_idx_ops, load_idx = _make_index_list(list(ivars))
            load = affine.LoadOp(memref=underlying_rhs, indices=load_idx)
            offset_idx = builtin.AddIndexOp(lhs=ivars[axis], rhs=offset_const)
            out_indices: list[dgen.Value] = list(ivars)
            out_indices[axis] = offset_idx
            store_idx_ops, store_idx = _make_index_list(out_indices)
            store = affine.StoreOp(value=load, memref=alloc_op, indices=store_idx)
            return load_idx_ops + [load, offset_idx] + store_idx_ops + [store]

        rhs_for = self._nested_for_op(rhs_shape, rhs_body)

        # Chain inputs before each ForOp so walk_ops visits them first (ordering fix).
        inner_lhs = builtin.ChainOp(lhs=lhs_alloc, rhs=lhs_for, type=alloc_op.type)
        inner_rhs = builtin.ChainOp(lhs=rhs_alloc, rhs=rhs_for, type=alloc_op.type)
        chain = builtin.ChainOp(
            lhs=builtin.ChainOp(lhs=alloc_op, rhs=inner_lhs, type=alloc_op.type),
            rhs=inner_rhs,
            type=alloc_op.type,
        )
        rewriter.replace_uses(op, chain)
        return True

    @lowering_for(toy.NonzeroCountOp)
    def lower_nonzero_count(self, op: toy.NonzeroCountOp, rewriter: Rewriter) -> bool:
        new_op = toy.NonzeroCountOp(input=op.input)
        rewriter.replace_uses(op, new_op)
        return True



def lower_to_affine(m: Module) -> Module:
    return ToyToAffine().run(m)
