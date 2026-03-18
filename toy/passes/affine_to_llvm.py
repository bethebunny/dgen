"""Ch6: Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from math import prod

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, Nil, String
from dgen.graph import build_result, chain_body, group_into_blocks, placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass
from toy.dialects import affine, toy

_PTR_TYPE = llvm.Ptr()
_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=Nil()))


def _extract_list_elements(
    list_val: dgen.Value,
    value_map: dict[dgen.Value, dgen.Value],
) -> list[dgen.Value]:
    """Extract elements from a PackOp, mapping values through value_map."""
    assert isinstance(list_val, PackOp)
    return [value_map.get(v, v) for v in list_val.values]


def _make_pack(values: list[dgen.Value]) -> PackOp:
    """Create a PackOp wrapping the given values."""
    if not values:
        return _EMPTY_PACK
    return PackOp(values=values, type=builtin.List(element_type=values[0].type))


class AffineToLLVMLowering(Pass):
    def __init__(self) -> None:
        self.loop_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}  # affine -> llvm
        self.alloc_shapes: dict[dgen.Value, list[int]] = {}  # llvm alloca -> shape
        self.alloc_sizes: dict[dgen.Value, int] = {}  # llvm alloca -> total size
        self._seen: set[dgen.Value] = set()  # ops already processed

    def run(self, m: Module) -> Module:
        functions = [self._lower_function(f) for f in m.functions]
        return Module(functions=functions)

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self.loop_counter = 0
        self.value_map = {}
        self.alloc_shapes = {}
        self.alloc_sizes = {}
        self._seen: set[dgen.Value] = set()
        # Register block args (function parameters)
        prologue: list[dgen.Op] = []
        for arg in f.body.args:
            if isinstance(arg.type, toy.Tensor):
                # arg is a Span struct ptr; load the data ptr from it
                load_op = llvm.LoadOp(ptr=arg, type=_PTR_TYPE)
                prologue.append(load_op)
                self.value_map[arg] = load_op
                shape = arg.type.shape.__constant__.to_json()
                self.alloc_shapes[load_op] = shape
                self.alloc_sizes[load_op] = prod(shape)
            else:
                self.value_map[arg] = arg

        # Generate ops (header/body labels are built directly in _lower_for;
        # exit labels are yielded as boundary markers for subsequent ops)
        flat_ops: list[dgen.Op] = list(prologue)
        for op in f.body.ops:
            flat_ops.extend(self.lower_op(op))

        # Group at exit-label boundaries (header/body labels are already built)
        return_val = self.value_map.get(f.body.result, f.body.result)
        entry_ops, label_groups = group_into_blocks(flat_ops)
        for label_op, body_ops in label_groups:
            label_op.body = dgen.Block(
                result=build_result(return_val, body_ops),
                args=label_op.body.args,
            )

        new_result = build_result(return_val, entry_ops)
        return FunctionOp(
            name=f.name,
            body=dgen.Block(result=new_result, args=f.body.args),
            result=f.result,
        )

    def _map(self, old: dgen.Value) -> dgen.Value:
        """Resolve an affine Value to its LLVM counterpart."""
        return self.value_map.get(old, old)

    def lower_op(self, op: dgen.Op) -> Iterator[dgen.Op]:
        if op in self._seen:
            return
        self._seen.add(op)
        if isinstance(op, affine.AllocOp):
            yield from self._lower_alloc(op)
        elif isinstance(op, affine.DeallocOp):
            pass  # Stack alloc, no free needed
        elif isinstance(op, affine.LoadOp):
            yield from self._lower_load(op)
        elif isinstance(op, affine.StoreOp):
            yield from self._lower_store(op)
        elif isinstance(op, affine.ForOp):
            yield from self._lower_for(op)
        elif isinstance(op, ConstantOp):
            new_op = ConstantOp(value=op.value, type=op.type)
            yield new_op
            if isinstance(op.type, toy.Tensor):
                # new_op is a Span struct ptr; load the data ptr from it
                load_op = llvm.LoadOp(ptr=new_op, type=_PTR_TYPE)
                yield load_op
                self.value_map[op] = load_op
                shape = op.type.shape.__constant__.to_json()
                self.alloc_shapes[load_op] = shape
                self.alloc_sizes[load_op] = prod(shape)
            else:
                self.value_map[op] = new_op
        elif isinstance(op, affine.MulFOp):
            llvm_op = llvm.FmulOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, affine.AddFOp):
            llvm_op = llvm.FaddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, affine.PrintMemrefOp):
            yield from self._lower_print(op)
        elif isinstance(op, builtin.ChainOp):
            # rhs side-effects were already emitted individually; resolve to lhs.
            new_lhs = self._map(op.lhs)
            self.value_map[op] = new_lhs
            if new_lhs in self.alloc_shapes:
                self.alloc_shapes[op] = self.alloc_shapes[new_lhs]
                self.alloc_sizes[op] = self.alloc_sizes[new_lhs]
        elif isinstance(op, builtin.AddIndexOp):
            llvm_op = llvm.AddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, toy.NonzeroCountOp):
            yield from self._lower_nonzero_count(op)

    def _lower_alloc(self, op: affine.AllocOp) -> Iterator[dgen.Op]:
        assert isinstance(op.type, affine.MemRef)
        shape = op.type.shape.__constant__.to_json()  # MemRef, not Tensor
        total = prod(shape)
        alloca_op = llvm.AllocaOp(elem_count=builtin.Index().constant(total))
        yield alloca_op
        self.value_map[op] = alloca_op
        self.alloc_shapes[alloca_op] = shape
        self.alloc_sizes[alloca_op] = total

    def _lower_load(self, op: affine.LoadOp) -> Iterator[dgen.Op]:
        memref_val = self._map(op.memref)
        index_vals = _extract_list_elements(op.indices, self.value_map)
        linear = yield from self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        yield ptr_op
        load_op = llvm.LoadOp(ptr=ptr_op)
        yield load_op
        self.value_map[op] = load_op

    def _lower_store(self, op: affine.StoreOp) -> Iterator[dgen.Op]:
        memref_val = self._map(op.memref)
        index_vals = _extract_list_elements(op.indices, self.value_map)
        linear = yield from self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        yield ptr_op
        yield llvm.StoreOp(value=self._map(op.value), ptr=ptr_op)

    def _linearize_indices(
        self, memref: dgen.Value, indices: list[dgen.Value]
    ) -> Generator[dgen.Op, None, dgen.Value]:
        if len(indices) == 1:
            return indices[0]

        shape = self.alloc_shapes[memref]

        result_val = None
        for i, idx_val in enumerate(indices):
            stride = prod(shape[i + 1 :])

            if stride == 1:
                if result_val is None:
                    result_val = idx_val
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=idx_val)
                    yield add_op
                    result_val = add_op
            else:
                stride_op = ConstantOp(value=stride, type=builtin.Index())
                yield stride_op
                mul_op = llvm.MulOp(lhs=idx_val, rhs=stride_op)
                yield mul_op
                if result_val is None:
                    result_val = mul_op
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=mul_op)
                    yield add_op
                    result_val = add_op

        assert result_val is not None
        return result_val

    def _lower_for(self, op: affine.ForOp) -> Iterator[dgen.Op]:
        loop_id = self.loop_counter
        self.loop_counter += 1

        header_label = f"loop_header{loop_id}"
        body_label = f"loop_body{loop_id}"
        exit_label = f"loop_exit{loop_id}"

        # lo/hi may be ConstantOp objects shared across multiple ForOps (e.g. two
        # loops both bound by the same constant node in multiply_transpose).  If
        # _map already has a mapping the constant was emitted by an earlier loop;
        # reuse it.  Otherwise emit a fresh ConstantOp and record the mapping so
        # subsequent loops can reuse it.
        init_op = self._map(op.lo)
        if init_op is op.lo:
            init_op = ConstantOp(
                value=op.lo.__constant__.to_json(), type=builtin.Index()
            )
            yield init_op
            self.value_map[op.lo] = init_op
            self._seen.add(op.lo)

        # Collect alloca pointers that need to be threaded through the loop.
        # Filter out ChainOps — they're transparent aliases for the actual alloca.
        alloca_entries: list[dgen.Value] = [
            v for v in self.alloc_shapes if not isinstance(v, builtin.ChainOp)
        ]

        # --- Block args for header: loop_var + alloca ptrs ---
        header_loop_var = BlockArgument(name=f"i{loop_id}", type=builtin.Index())
        header_alloca_args = [BlockArgument(type=_PTR_TYPE) for _ in alloca_entries]
        header_args = [header_loop_var] + header_alloca_args

        # --- Block args for body: loop_var + alloca ptrs ---
        body_loop_var = BlockArgument(name=f"j{loop_id}", type=builtin.Index())
        body_alloca_args = [BlockArgument(type=_PTR_TYPE) for _ in alloca_entries]
        body_args = [body_loop_var] + body_alloca_args

        # --- Build header block body ---
        # Map the affine loop variable to the header's block arg
        self.value_map[op.body.args[0]] = header_loop_var

        header_ops: list[dgen.Op] = []
        # Same deduplication as lo above (see comment there).
        hi_op = self._map(op.hi)
        if hi_op is op.hi:
            hi_op = ConstantOp(value=op.hi.__constant__.to_json(), type=builtin.Index())
            header_ops.append(hi_op)
            self.value_map[op.hi] = hi_op
            self._seen.add(op.hi)
        cmp_op = llvm.IcmpOp(
            pred=String().constant("slt"), lhs=header_loop_var, rhs=hi_op
        )
        header_ops.append(cmp_op)

        # Exit label (needs to exist before cond_br references it)
        exit_label_op = llvm.LabelOp(
            name=exit_label,
            body=placeholder_block(),
        )

        # Body label (forward-declared, body filled below)
        body_label_op = llvm.LabelOp(
            name=body_label,
            body=dgen.Block(result=dgen.Value(type=Nil()), args=body_args),
        )

        # cond_br passes header's loop_var + alloca args to body
        cond_true_pack = _make_pack([header_loop_var] + header_alloca_args)
        cond_br = llvm.CondBrOp(
            cond=cmp_op,
            true_target=body_label_op,
            false_target=exit_label_op,
            true_args=cond_true_pack,
            false_args=_EMPTY_PACK,
        )
        header_ops.append(cond_br)

        # Header label — all header ops reachable from cond_br via operands
        header_label_op = llvm.LabelOp(
            name=header_label,
            body=dgen.Block(result=chain_body(header_ops), args=header_args),
        )

        # --- Build body block ---
        # Save all mutable state so we can restore after the loop body.
        saved_value_map = dict(self.value_map)
        saved_alloc_shapes = dict(self.alloc_shapes)
        saved_alloc_sizes = dict(self.alloc_sizes)

        # Remap: inside body, loop var resolves to body's block arg.
        self.value_map[op.body.args[0]] = body_loop_var

        # For each alloca entry, redirect all value_map entries that currently
        # point to it so they point to the corresponding body block arg instead.
        # This ensures _map(affine_op) returns the body block arg, not the
        # entry-block alloca (critical for nested loops).
        for entry_alloca, body_arg in zip(alloca_entries, body_alloca_args):
            # Register shape/size metadata on the body block arg
            self.alloc_shapes[body_arg] = self.alloc_shapes[entry_alloca]
            self.alloc_sizes[body_arg] = self.alloc_sizes[entry_alloca]
            # Remove the original so inner loops only see body block args
            del self.alloc_shapes[entry_alloca]
            del self.alloc_sizes[entry_alloca]
            # Redirect any value_map entry that resolves to this alloca
            for key, val in list(self.value_map.items()):
                if val is entry_alloca:
                    self.value_map[key] = body_arg

        # Lower body ops into a local list
        body_flat: list[dgen.Op] = []
        for child_op in op.body.ops:
            body_flat.extend(self.lower_op(child_op))

        # Increment and branch back to header
        one_op = ConstantOp(value=1, type=builtin.Index())
        body_flat.append(one_op)
        next_op = llvm.AddOp(lhs=body_loop_var, rhs=one_op)
        body_flat.append(next_op)

        # Back-edge: pass next loop var + body's alloca args back to header
        back_pack = _make_pack([next_op] + body_alloca_args)
        back_br = llvm.BrOp(target=header_label_op, args=back_pack)
        body_flat.append(back_br)

        # Group at exit-label boundaries (inner loops yield exit labels)
        body_ops, body_label_groups = group_into_blocks(body_flat)
        for lbl, lbl_body_ops in body_label_groups:
            lbl.body = dgen.Block(
                result=chain_body(lbl_body_ops),
                args=lbl.body.args,
            )

        # Fill body label's block with chained body ops
        body_label_op.body = dgen.Block(
            result=chain_body(body_ops),
            args=body_args,
        )

        # Restore value_map so code after the loop references entry-block allocas
        self.value_map = saved_value_map
        self.alloc_shapes = saved_alloc_shapes
        self.alloc_sizes = saved_alloc_sizes

        # Yield ops for the enclosing block:
        # branch to header, then the exit label as a boundary marker
        entry_pack = _make_pack([init_op] + alloca_entries)
        yield llvm.BrOp(target=header_label_op, args=entry_pack)
        yield exit_label_op

    def _lower_nonzero_count(self, op: toy.NonzeroCountOp) -> Iterator[dgen.Op]:
        """Unrolled nonzero_count: count non-zero elements in a tensor."""
        input_val = self._map(op.input)
        assert isinstance(op.input.type, toy.Tensor)
        total = prod(op.input.type.shape.__constant__.to_json())

        zero_f = ConstantOp(value=0.0, type=builtin.F64())
        yield zero_f
        acc = ConstantOp(value=0, type=builtin.Index())
        yield acc

        for i in range(total):
            idx = ConstantOp(value=i, type=builtin.Index())
            yield idx
            gep = llvm.GepOp(base=input_val, index=idx)
            yield gep
            elem = llvm.LoadOp(ptr=gep)
            yield elem
            cmp = llvm.FcmpOp(pred=String().constant("one"), lhs=elem, rhs=zero_f)
            yield cmp
            ext = llvm.ZextOp(input=cmp)
            yield ext
            new_acc = llvm.AddOp(lhs=acc, rhs=ext)
            yield new_acc
            acc = new_acc

        self.value_map[op] = acc

    def _lower_print(self, op: affine.PrintMemrefOp) -> Iterator[dgen.Op]:
        input_val = self._map(op.input)
        size = self.alloc_sizes[input_val]
        size_op = ConstantOp(value=size, type=builtin.Index())
        yield size_op
        pack = PackOp(
            values=[input_val, size_op], type=builtin.List(element_type=input_val.type)
        )
        yield pack
        call_op = llvm.CallOp(
            callee=String().constant("print_memref"),
            args=pack,
        )
        yield call_op
        self.value_map[op] = call_op


def lower_to_llvm(m: Module) -> Module:
    return AffineToLLVMLowering().run(m)
