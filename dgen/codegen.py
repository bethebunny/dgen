"""Emit valid LLVM IR text from LLModule and JIT-compile via llvmlite."""

from __future__ import annotations

import _ctypes
import ctypes
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import dgen
from dgen import Type
from dgen.asm.formatting import SlotTracker, format_float
from dgen.dialects import builtin, llvm
from dgen.module import ConstantOp, Module, PackOp, string_value
from dgen.layout import Layout
from dgen.passes.builtin_to_llvm import lower_builtin_to_llvm
from dgen.type import Constant, Memory, Value

# ---------------------------------------------------------------------------
# struct.format → LLVM / ctypes mapping
# ---------------------------------------------------------------------------

_FMT_LLVM = {"q": "i64", "d": "double"}
_FMT_CTYPE = {"q": ctypes.c_int64, "d": ctypes.c_double}


def _llvm_type(layout: Layout) -> str:
    """Derive LLVM type from a layout's struct format."""
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_LLVM.get(fmt, "ptr")


def _ctype(layout: Layout) -> type[ctypes._CData]:
    """Derive ctypes type from a layout's struct format."""
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
        import llvmlite.binding as _llvm

        _llvm.initialize_native_target()
        _llvm.initialize_native_asmprinter()
        _llvm.add_symbol(
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
    lines: list[str] = ["declare void @print_memref(ptr, i64)"]
    lines.extend(externs)
    lines.append("")
    for func in module.functions:
        lines.extend(_emit_func(func, host_buffers))
    return "\n".join(lines), host_buffers


def _result_type_str(ty: Value[dgen.TypeType]) -> str | None:
    """Derive LLVM IR type string from an op's type, or None for void ops."""
    if isinstance(ty, builtin.Nil):
        return None
    resolved = dgen.type.type_constant(ty)
    if isinstance(resolved, llvm.Int):
        return f"i{resolved.bits.__constant__.to_json()}"
    return _llvm_type(resolved.__layout__)


def _emit_func(f: builtin.FunctionOp, host_buffers: list) -> list[str]:
    # Build a SlotTracker so all ops get stable names
    tracker = SlotTracker()
    # Register block args (function parameters) first
    for arg in f.body.args:
        tracker.track_name(arg)

    # Separate entry ops from label ops
    entry_ops: list[dgen.Op] = []
    label_ops: list[llvm.LabelOp] = []
    for op in f.body.ops:
        if isinstance(op, llvm.LabelOp):
            label_ops.append(op)
        else:
            entry_ops.append(op)

    # Collect all ops (entry + label bodies) for registration
    all_ops: list[dgen.Op] = list(entry_ops)
    for label_op in label_ops:
        all_ops.extend(label_op.body.ops)

    tracker.register(all_ops)
    # Also register label ops themselves (they're not in all_ops since they're structural)
    for label_op in label_ops:
        tracker.track_name(label_op)

    # Pre-scan: build constants and types maps
    constants: dict[dgen.Value, str] = {}
    types: dict[dgen.Value, str] = {}

    def _register_constant(c: Constant) -> None:
        if c in constants:
            return
        mem = c.__constant__
        layout = mem.layout
        if _ctype(layout) is ctypes.c_void_p:
            # Pointer-passed layout: emit struct address (Span: [data_ptr, len])
            host_buffers.append(mem)
            constants[c] = f"ptr inttoptr (i64 {mem.address} to ptr)"
            types[c] = "ptr"
        else:
            lt = _llvm_type(layout)
            raw = mem.unpack()[0]
            val_str = format_float(raw) if isinstance(raw, float) else str(raw)
            constants[c] = f"{lt} {val_str}"
            types[c] = lt

    # Register block arg types
    for arg in f.body.args:
        types[arg] = _llvm_type(dgen.type.type_constant(arg.type).__layout__)

    for op in all_ops:
        if isinstance(op, ConstantOp):
            _register_constant(op)
        elif isinstance(op, PackOp):
            continue
        elif isinstance(op, builtin.ChainOp):
            # Chain is transparent: alias to lhs at runtime
            types[op] = types.get(op.lhs, "i64")
        elif isinstance(op, llvm.PhiOp):
            types[op] = types.get(op.a, "i64")
        else:
            if (rt := _result_type_str(op.type)) is not None:
                types[op] = rt
            # Register inline Constant operands (e.g. literal args)
            for operand_name, _ in op.__operands__:
                operand = getattr(op, operand_name)
                if isinstance(operand, Constant):
                    _register_constant(operand)

    # Resolve chain aliases: ChainOp is transparent, maps to its lhs
    for op in all_ops:
        if isinstance(op, builtin.ChainOp):
            if op.lhs in constants:
                constants[op] = constants[op.lhs]
            else:
                # Point to same slot as lhs
                lhs_name = tracker.track_name(op.lhs)
                lhs_ty = types.get(op.lhs, "i64")
                constants[op] = f"{lhs_ty} %{lhs_name}"

    def typed_ref(val: dgen.Value) -> str:
        """'type value' — e.g. 'double 1.0' or 'ptr %v3'."""
        if val in constants:
            return constants[val]
        return f"{types.get(val, 'i64')} %{tracker.track_name(val)}"

    def bare_ref(val: dgen.Value) -> str:
        """Just the value — e.g. '1.0' or '%v3'."""
        if val in constants:
            return constants[val].split(" ", 1)[1]
        return f"%{tracker.track_name(val)}"

    def emit_op(op: dgen.Op, lines: list[str]) -> None:
        """Emit a single op as LLVM IR."""
        if isinstance(op, (ConstantOp, PackOp, builtin.ChainOp)):
            return

        name = tracker.track_name(op)

        if isinstance(op, llvm.AllocaOp):
            lines.append(
                f"  %{name} = alloca double, i64 {op.elem_count.__constant__.to_json()}"
            )
        elif isinstance(op, llvm.GepOp):
            lines.append(
                f"  %{name} = getelementptr double, ptr {bare_ref(op.base)},"
                f" {typed_ref(op.index)}"
            )
        elif isinstance(op, llvm.LoadOp):
            lt = _llvm_type(dgen.type.type_constant(op.type).__layout__)
            lines.append(f"  %{name} = load {lt}, {typed_ref(op.ptr)}")
        elif isinstance(op, llvm.StoreOp):
            lines.append(f"  store {typed_ref(op.value)}, {typed_ref(op.ptr)}")
        elif isinstance(op, llvm.FaddOp):
            lines.append(
                f"  %{name} = fadd double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.FmulOp):
            lines.append(
                f"  %{name} = fmul double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.AddOp):
            lines.append(f"  %{name} = add i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.SubOp):
            lines.append(f"  %{name} = sub i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.MulOp):
            lines.append(f"  %{name} = mul i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.FcmpOp):
            lines.append(
                f"  %{name} = fcmp {string_value(op.pred)} double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.ZextOp):
            lines.append(f"  %{name} = zext i1 {bare_ref(op.input)} to i64")
        elif isinstance(op, llvm.IcmpOp):
            lines.append(
                f"  %{name} = icmp {string_value(op.pred)} i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.BrOp):
            lines.append(f"  br label %{tracker.track_name(op.target)}")
        elif isinstance(op, llvm.CondBrOp):
            lines.append(
                f"  br i1 %{tracker.track_name(op.cond)}, label %{tracker.track_name(op.true_target)}, label %{tracker.track_name(op.false_target)}"
            )
        elif isinstance(op, llvm.PhiOp):
            ty = types.get(op, "i64")
            lines.append(
                f"  %{name} = phi {ty} "
                f"[ {bare_ref(op.a)}, %{tracker.track_name(op.label_a)} ], "
                f"[ {bare_ref(op.b)}, %{tracker.track_name(op.label_b)} ]"
            )
        elif isinstance(op, llvm.CallOp):
            callee = string_value(op.callee)
            call_args = op.args.values if isinstance(op.args, PackOp) else [op.args]
            a = ", ".join(typed_ref(arg) for arg in call_args)
            if isinstance(op.type, builtin.Nil):
                lines.append(f"  call void @{callee}({a})")
            else:
                ret_ty = types[op]
                lines.append(f"  %{name} = call {ret_ty} @{callee}({a})")
        elif isinstance(op, builtin.ReturnOp):
            if llvm_ret == "void":
                lines.append("  ret void")
            else:
                lines.append(f"  ret {typed_ref(op.value)}")

    assert f.name is not None
    func_name = tracker.track_name(f)
    # Derive LLVM return type from function signature
    result_type = f.result
    if isinstance(result_type, builtin.Nil):
        llvm_ret = "void"
    else:
        llvm_ret = _llvm_type(dgen.type.type_constant(result_type).__layout__)
    # Build parameter list from block args
    params = []
    for arg in f.body.args:
        name = tracker.track_name(arg)
        ty = types.get(arg, "i64")
        params.append(f"{ty} %{name}")
    param_str = ", ".join(params)
    lines = [f"define {llvm_ret} @{func_name}({param_str}) {{", "entry:"]

    # Emit entry block ops
    for op in entry_ops:
        emit_op(op, lines)

    # Emit each label's body
    for label_op in label_ops:
        label_name = tracker.track_name(label_op)
        lines.append(f"{label_name}:")
        for body_op in label_op.body.ops:
            emit_op(body_op, lines)

    lines.append("}")
    return lines


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

    def run(self, *args: Memory | object) -> object:
        """JIT and execute, returning the function's result.

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
        param_ctypes = [_ctype(t.__layout__) for t in self.input_types]
        ctypes_args = [
            m.address if ct is ctypes.c_void_p else m.unpack()[0]
            for m, ct, t in zip(memories, param_ctypes, self.input_types)
        ]
        return cfunc(*ctypes_args)


def compile(module: Module, *, externs: Sequence[str] = ()) -> Executable:
    """Emit LLVM IR and bundle with execution metadata."""
    module = lower_builtin_to_llvm(module)
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
    import llvmlite.binding as _llvm

    _ensure_initialized()
    mod = _llvm.parse_assembly(exe.ir)
    mod.verify()
    target = _llvm.Target.from_default_triple()
    tm = target.create_target_machine()
    return _llvm.create_mcjit_compiler(mod, tm)


# ---------------------------------------------------------------------------
# LLVMCodegen exit pass
# ---------------------------------------------------------------------------


class LLVMCodegen:
    """Exit pass: lower builtin ops to LLVM, emit IR, bundle as Executable."""

    def run(self, module: Module) -> Executable:
        return compile(module)
