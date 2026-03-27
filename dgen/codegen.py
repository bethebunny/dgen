"""Emit valid LLVM IR text from LLModule and JIT-compile via llvmlite."""

from __future__ import annotations

import _ctypes
import ctypes
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import llvmlite.binding as llvmlite

import dgen
from dgen import Type
from dgen.asm.formatting import SlotTracker, format_float
from dgen.dialects import builtin, function, goto, llvm
from dgen.module import ConstantOp, Module, PackOp, string_value
from dgen.layout import Layout
from dgen.compiler import Compiler, IdentityPass
from dgen.passes.builtin_to_llvm import BuiltinToLLVMLowering
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.type import Constant, Memory, Value

# ---------------------------------------------------------------------------
# Layout → LLVM / ctypes mapping
# ---------------------------------------------------------------------------

_FMT_LLVM = {"q": "i64", "d": "double"}
_FMT_CTYPE = {"q": ctypes.c_int64, "d": ctypes.c_double}


def _llvm_type(layout: Layout) -> str:
    """Derive LLVM IR type string from a layout."""
    if not layout.register_passable:
        return "ptr"
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_LLVM.get(fmt, "ptr")


def _ctype(layout: Layout) -> type[ctypes._CData]:
    """Derive ctypes type from a layout."""
    if not layout.register_passable:
        return ctypes.c_void_p
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_CTYPE.get(fmt, ctypes.c_void_p)


# ---------------------------------------------------------------------------
# Runtime: print_memref callback
# ---------------------------------------------------------------------------


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)
def _print_memref(ptr: int, size: int) -> None:
    arr = (ctypes.c_double * size).from_address(ptr)
    print(", ".join(f"{arr[i]:g}" for i in range(size)))


# ---------------------------------------------------------------------------
# LLVM initialization
# ---------------------------------------------------------------------------

_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        llvmlite.initialize_native_target()
        llvmlite.initialize_native_asmprinter()
        llvmlite.add_symbol(
            "print_memref", ctypes.cast(_print_memref, ctypes.c_void_p).value
        )
        _initialized = True


# ---------------------------------------------------------------------------
# IR emission
# ---------------------------------------------------------------------------


def emit_llvm_ir(module: Module, *, externs: Sequence[str] = ()) -> tuple[str, list]:
    """Emit valid LLVM IR text that llvmlite can parse.

    Returns (ir_text, host_buffers) where host_buffers keeps Memory objects
    alive for the lifetime of the JIT.
    """
    host_buffers: list = []

    def _lines() -> Iterator[str]:
        yield "declare void @print_memref(ptr, i64)"
        yield "declare ptr @malloc(i64)"
        yield from externs
        yield ""
        for func in module.functions:
            yield from _emit_func(func, host_buffers)

    return "\n".join(_lines()), host_buffers


def _result_type_str(ty: Value[dgen.TypeType]) -> str | None:
    """Derive LLVM IR type string from an op's type, or None for void ops."""
    if isinstance(ty, builtin.Nil):
        return None
    resolved = dgen.type.type_constant(ty)
    if isinstance(resolved, llvm.Int):
        return f"i{resolved.bits.__constant__.to_json()}"
    return _llvm_type(resolved.__layout__)


def _emit_func(f: function.FunctionOp, host_buffers: list) -> Iterator[str]:
    """Emit a function as LLVM IR by recursively walking block.ops."""
    tracker = SlotTracker()
    constants: dict[dgen.Value, str] = {}
    types: dict[dgen.Value, str] = {}
    param_to_label: dict[dgen.Value, goto.LabelOp] = {}
    predecessors: dict[int, list[tuple[dgen.Value, list[dgen.Value]]]] = {}
    entry_sentinel = dgen.Value(name="entry", type=goto.Label())

    def unpack(val: Value) -> list[Value]:
        return val.values if isinstance(val, PackOp) else [val]

    def resolve_target(target: dgen.Value) -> dgen.Value:
        if isinstance(target, goto.LabelOp):
            return target
        if target in param_to_label:
            return param_to_label[target]
        # BlockParameter like %exit — used directly as a label name
        return target

    def typed_ref(val: dgen.Value) -> str:
        return (
            constants[val]
            if val in constants
            else f"{types.get(val, 'i64')} %{tracker.track_name(val)}"
        )

    def bare_ref(val: dgen.Value) -> str:
        return (
            constants[val].split(" ", 1)[1]
            if val in constants
            else f"%{tracker.track_name(val)}"
        )

    def _register(val: dgen.Value) -> None:
        """Register a value's SSA name, type, and constants. Idempotent."""
        if val in types or val in constants:
            return
        tracker.track_name(val)
        if isinstance(val, ConstantOp):
            mem = val.__constant__
            if not mem.layout.register_passable:
                host_buffers.append(mem)
                constants[val] = f"ptr inttoptr (i64 {mem.address} to ptr)"
                types[val] = "ptr"
            else:
                lt = _llvm_type(mem.layout)
                raw = mem.unpack()[0]
                constants[val] = (
                    f"{lt} {format_float(raw) if isinstance(raw, float) else raw}"
                )
                types[val] = lt
        elif isinstance(val, builtin.ChainOp):
            _register(val.lhs)
            types[val] = types.get(val.lhs, "i64")
            constants[val] = (
                constants[val.lhs]
                if val.lhs in constants
                else f"{types.get(val.lhs, 'i64')} %{tracker.track_name(val.lhs)}"
            )
        elif isinstance(val, dgen.Op):
            rt = _result_type_str(val.type)
            if rt is not None:
                types[val] = rt
            for _, operand in val.operands:
                if isinstance(operand, Constant):
                    _register(operand)

    def _register_block_args(block: dgen.Block) -> None:
        for arg in block.args:
            tracker.track_name(arg)
            types[arg] = _llvm_type(dgen.type.type_constant(arg.type).__layout__)

    # --- Predecessor map (needed before emission for phi nodes) ---

    def _scan_preds(block: dgen.Block, src: dgen.Value) -> None:
        for op in block.ops:
            if isinstance(op, goto.LabelOp):
                # Only map %self (back-edge target) to the label.
                # %exit parameters are emitted as separate labels by the codegen.
                for param in op.body.parameters:
                    if param.name == "self":
                        param_to_label[param] = op
                _scan_preds(op.body, op)
            elif isinstance(op, goto.BranchOp):
                predecessors.setdefault(id(resolve_target(op.target)), []).append(
                    (src, unpack(op.arguments))
                )
            elif isinstance(op, goto.ConditionalBranchOp):
                for t, a in [
                    (op.true_target, op.true_arguments),
                    (op.false_target, op.false_arguments),
                ]:
                    predecessors.setdefault(id(resolve_target(t)), []).append(
                        (src, unpack(a))
                    )

    _scan_preds(f.body, entry_sentinel)

    # --- Emit ---

    _register_block_args(f.body)
    llvm_ret = (
        "void"
        if isinstance(f.result, builtin.Nil)
        else _llvm_type(dgen.type.type_constant(f.result).__layout__)
    )
    params = ", ".join(
        f"{types.get(a, 'i64')} %{tracker.track_name(a)}" for a in f.body.args
    )
    yield f"define {llvm_ret} @{tracker.track_name(f)}({params}) {{"
    yield "entry:"

    def _emit_op(op: dgen.Op) -> Iterator[str]:
        """Emit a single non-label op as LLVM IR."""
        name = tracker.track_name(op)
        if isinstance(op, goto.BranchOp):
            yield f"  br label %{tracker.track_name(resolve_target(op.target))}"
        elif isinstance(op, goto.ConditionalBranchOp):
            yield f"  br i1 %{tracker.track_name(op.condition)}, label %{tracker.track_name(resolve_target(op.true_target))}, label %{tracker.track_name(resolve_target(op.false_target))}"
        elif isinstance(op, (ConstantOp, PackOp, builtin.ChainOp)):
            pass
        elif isinstance(op, llvm.CallOp):
            a = ", ".join(typed_ref(v) for v in unpack(op.args))
            callee = string_value(op.callee)
            yield (
                f"  call void @{callee}({a})"
                if isinstance(op.type, builtin.Nil)
                else f"  %{name} = call {types[op]} @{callee}({a})"
            )
        elif isinstance(op, llvm.StoreOp):
            yield f"  store {typed_ref(op.value)}, {typed_ref(op.ptr)}"
        elif isinstance(op, llvm.AllocaOp):
            yield f"  %{name} = alloca double, i64 {op.elem_count.__constant__.to_json()}"
        elif isinstance(op, llvm.GepOp):
            yield f"  %{name} = getelementptr double, ptr {bare_ref(op.base)}, {typed_ref(op.index)}"
        elif isinstance(op, llvm.LoadOp):
            yield f"  %{name} = load {_llvm_type(dgen.type.type_constant(op.type).__layout__)}, {typed_ref(op.ptr)}"
        elif isinstance(op, llvm.ZextOp):
            yield f"  %{name} = zext i1 {bare_ref(op.input)} to i64"
        elif isinstance(op, (llvm.FaddOp, llvm.FmulOp)):
            yield f"  %{name} = {'fadd' if isinstance(op, llvm.FaddOp) else 'fmul'} double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, (llvm.AddOp, llvm.SubOp, llvm.MulOp)):
            instr = {"AddOp": "add", "SubOp": "sub", "MulOp": "mul"}[type(op).__name__]
            yield f"  %{name} = {instr} i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.IcmpOp):
            yield f"  %{name} = icmp {string_value(op.pred)} i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.FcmpOp):
            yield f"  %{name} = fcmp {string_value(op.pred)} double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"

    def _emit(block: dgen.Block) -> Iterator[str]:
        # Split block ops into: pre-branch ops, labels, and post-branch ops.
        # Pre-branch ops are the current block's instructions.
        # Labels start new basic blocks (emitted after the terminator).
        # Post-branch ops are the continuation (emitted after the exit label).
        labels: list[goto.LabelOp] = []
        pre_branch: list[dgen.Op] = []
        post_branch: list[dgen.Op] = []
        saw_branch = False
        for op in block.ops:
            _register(op)
            if isinstance(op, goto.LabelOp):
                _register_block_args(op.body)
                for param in op.body.parameters:
                    tracker.track_name(param)
                labels.append(op)
            elif saw_branch:
                post_branch.append(op)
            else:
                pre_branch.append(op)
                if isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp)):
                    saw_branch = True

        # Emit pre-branch ops (current block's instructions + terminator).
        terminated = False
        for op in pre_branch:
            yield from _emit_op(op)
            if isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp)):
                terminated = True
        if not terminated:
            yield (
                "  ret void"
                if llvm_ret == "void"
                else f"  ret {typed_ref(f.body.result)}"
            )
        # Labels (sub-blocks) after the current block's terminator.
        for label_op in labels:
            yield f"{tracker.track_name(label_op)}:"
            for i, arg in enumerate(label_op.body.args):
                parts = [
                    f"[ {bare_ref(a[i])}, %{tracker.track_name(s)} ]"
                    for s, a in predecessors.get(id(label_op), [])
                    if i < len(a)
                ]
                if parts:
                    yield f"  %{tracker.track_name(arg)} = phi {types.get(arg, 'i64')} {', '.join(parts)}"
            yield from _emit(label_op.body)
            for param in label_op.body.parameters:
                if param.name and param.name.startswith("exit"):
                    yield f"{tracker.track_name(param)}:"
        # Post-branch continuation (fall-through after exit label).
        for op in post_branch:
            yield from _emit_op(op)

    yield from _emit(f.body)
    yield "}"


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------


@dataclass
class Executable:
    """Compiled LLVM IR ready for JIT execution."""

    ir: str
    input_types: list[Type]
    result_type: Type
    main_name: str
    host_refs: list = field(default_factory=list)

    @property
    def ctype(self) -> type[_ctypes.CFuncPtr]:
        """ctypes function pointer type for this executable."""
        param_ctypes = [_ctype(t.__layout__) for t in self.input_types]
        result_ctype = _ctype(self.result_type.__layout__)
        return ctypes.CFUNCTYPE(result_ctype, *param_ctypes)

    def run(self, *args: Memory | object) -> Memory:
        """JIT and execute, returning the function's result as a Memory.

        Args can be Memory objects or raw Python values (str, int, float).
        Raw values are converted to Memory via Memory.from_value.
        """
        memories: list[Memory] = [
            arg if isinstance(arg, Memory) else Memory.from_value(ty, arg)
            for arg, ty in zip(args, self.input_types)
        ]

        engine = _jit_engine(self)
        func_ptr = engine.get_function_address(self.main_name)
        cfunc = self.ctype(func_ptr)
        ctypes_args = [
            m.unpack()[0] if t.__layout__.register_passable else m.address
            for m, t in zip(memories, self.input_types)
        ]
        result = cfunc(*ctypes_args)
        if self.result_type.__layout__.register_passable:
            return Memory.from_value(self.result_type, result)
        return Memory.from_raw(self.result_type, result)


def compile(module: Module, *, externs: Sequence[str] = ()) -> Executable:
    """Emit LLVM IR and bundle with execution metadata."""
    _dummy = Compiler([], IdentityPass())
    module = BuiltinToLLVMLowering().run(module, _dummy)
    module = AlgebraToLLVM().run(module, _dummy)
    ir, host_buffers = emit_llvm_ir(module, externs=externs)
    main = module.functions[0]
    assert main.name is not None
    result = dgen.type.type_constant(main.result)
    return Executable(
        ir=ir,
        input_types=[dgen.type.type_constant(arg.type) for arg in main.body.args],
        main_name=main.name,
        result_type=result,
        host_refs=host_buffers,
    )


def _jit_engine(exe: Executable) -> Any:  # noqa: ANN401
    """Parse, verify, and create MCJIT engine from an Executable."""
    _ensure_initialized()
    mod = llvmlite.parse_assembly(exe.ir)
    mod.verify()
    target = llvmlite.Target.from_default_triple()
    tm = target.create_target_machine()
    return llvmlite.create_mcjit_compiler(mod, tm)


# ---------------------------------------------------------------------------
# LLVMCodegen exit pass
# ---------------------------------------------------------------------------


class LLVMCodegen:
    """Exit pass: lower builtin ops to LLVM, emit IR, bundle as Executable."""

    def run(self, module: Module) -> Executable:
        return compile(module)
