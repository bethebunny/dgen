"""Ch6: Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from math import prod

import dgen
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, Nil, String
from dgen.graph import chain_body, group_into_blocks, placeholder_block
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


class AffineToLLVMLowering(Pass):
    def __init__(self) -> None:
        self.loop_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}  # affine -> llvm
        self.alloc_shapes: dict[dgen.Value, list[int]] = {}  # llvm alloca -> shape
        self.alloc_sizes: dict[dgen.Value, int] = {}  # llvm alloca -> total size
        self.current_label: dgen.Value = dgen.Value(name="entry", type=llvm.Label())
        self._seen: set[dgen.Value] = set()  # ops already processed

    def run(self, m: Module) -> Module:
        functions = [self._lower_function(f) for f in m.functions]
        return Module(functions=functions)

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self.loop_counter = 0
        self.value_map = {}
        self.alloc_shapes = {}
        self.alloc_sizes = {}
        self.current_label = dgen.Value(name="entry", type=llvm.Label())
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

        # Phase 1: Generate ops linearly (labels as boundary markers)
        flat_ops = list(prologue)
        for op in f.body.ops:
            flat_ops.extend(self.lower_op(op))

        # Phase 2: Group into basic blocks and build label body blocks
        entry_ops, label_groups = group_into_blocks(flat_ops)
        for label_op, body_ops in label_groups:
            label_op.body = dgen.Block(result=chain_body(body_ops), ops=body_ops)

        func_ops: list[dgen.Op] = entry_ops + [lg[0] for lg in label_groups]
        return FunctionOp(
            name=f.name,
            body=dgen.Block(ops=func_ops, args=f.body.args),
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
        elif isinstance(op, builtin.ReturnOp):
            if isinstance(op.value, Nil):
                yield builtin.ReturnOp()
            else:
                yield builtin.ReturnOp(value=self._map(op.value))

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
        # TODO: this deduplication is fragile — revisit once ForOp bounds are
        # modelled as proper SSA Values so _seen / value_map handle them uniformly.
        init_op = self._map(op.lo)
        if init_op is op.lo:
            init_op = ConstantOp(
                value=op.lo.__constant__.to_json(), type=builtin.Index()
            )
            yield init_op
            self.value_map[op.lo] = init_op
            self._seen.add(op.lo)
        header_label_op = llvm.LabelOp(name=header_label, body=placeholder_block())
        body_label_op = llvm.LabelOp(name=body_label, body=placeholder_block())
        exit_label_op = llvm.LabelOp(name=exit_label, body=placeholder_block())

        yield llvm.BrOp(target=header_label_op, args=_EMPTY_PACK)

        # Header label
        yield header_label_op
        self.current_label = header_label_op

        # TODO(closed-blocks): replace phi with block args on header label
        # For now, use a placeholder Value for the loop variable
        loop_var_placeholder = dgen.Value(name=f"i{loop_id}", type=builtin.Index())
        self.value_map[op.body.args[0]] = loop_var_placeholder

        # Same deduplication as lo above (see comment there).
        hi_op = self._map(op.hi)
        if hi_op is op.hi:
            hi_op = ConstantOp(value=op.hi.__constant__.to_json(), type=builtin.Index())
            yield hi_op
            self.value_map[op.hi] = hi_op
            self._seen.add(op.hi)
        cmp_op = llvm.IcmpOp(pred=String().constant("slt"), lhs=loop_var_placeholder, rhs=hi_op)
        yield cmp_op
        yield llvm.CondBrOp(
            cond=cmp_op,
            true_target=body_label_op,
            false_target=exit_label_op,
            true_args=_EMPTY_PACK,
            false_args=_EMPTY_PACK,
        )

        # Body label
        yield body_label_op
        self.current_label = body_label_op

        # Lower body ops
        for child_op in op.body.ops:
            yield from self.lower_op(child_op)

        # Increment and branch back
        one_op = ConstantOp(value=1, type=builtin.Index())
        yield one_op
        next_op = llvm.AddOp(lhs=loop_var_placeholder, rhs=one_op)
        yield next_op
        yield llvm.BrOp(target=header_label_op, args=_EMPTY_PACK)

        # Exit label
        yield exit_label_op
        self.current_label = exit_label_op

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
