"""Tests for type values as first-class SSA citizens."""

from copy import deepcopy

import pytest

from dgen import Block, asm
from dgen.type import format_value
from dgen.asm.parser import ASMParser, parse_module, value_expression
from dgen.testing import assert_ir_equivalent
from dgen.block import BlockArgument
from dgen.codegen import Executable, LLVMCodegen, compile as compile_module
from dgen.compiler import Compiler, IdentityPass
from dgen import layout
from dgen.dialects import algebra, builtin, llvm
from dgen.dialects.index import Index
from dgen.dialects.function import Function, FunctionOp
from dgen.layout import TypeValue
from dgen.module import ConstantOp, Module, pack
from dgen.passes.pass_ import Pass
from dgen.type import Memory, TypeType, type_constant
from dgen.testing import strip_prefix
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from dgen.passes.memory_to_llvm import MemoryToLLVM

_compiler = Compiler([], IdentityPass())


def lower_to_llvm(m: Module) -> Module:
    m = ControlFlowToGoto().run(m, _compiler)
    m = NDBufferToMemory().run(m, _compiler)
    return MemoryToLLVM().run(m, _compiler)


def test_parse_dict_literal():
    """value_expression handles {key: value, ...} and returns a Python dict."""
    parser = ASMParser('{"tag": "index.Index"}')
    result = value_expression(parser)
    assert result == {"tag": "index.Index"}


def test_format_dict_literal():
    """format_value handles dicts."""
    result = format_value({"tag": "index.Index"})
    assert result == '{"tag": "index.Index"}'


def test_typetype_constant_asm_roundtrip():
    """TypeType constant with dict literal round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_ssa_ref_as_op_type():
    """SSA ref in type position: %x's type is the SSA value %t."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %x : %t = 42
    """)
    module = parse_module(ir)
    ops = module.functions[0].body.ops
    t_op = ops[0]  # %t = TypeType constant
    x_op = ops[1]  # %x : %t = 42
    # %x's type is the SSA value %t (a resolved ConstantOp)
    assert x_op.type is t_op
    assert x_op.ready


def test_ssa_ref_as_op_type_roundtrip():
    """SSA ref in type position round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %x : %t = 42
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_typetype_memory_roundtrip():
    """TypeType constant serializes to dict and round-trips through Memory."""
    idx = Index()
    mem = idx.__constant__
    data = mem.to_json()
    assert data == {"tag": "index.Index"}
    # Round-trip: dict → Memory → dict
    mem2 = Memory.from_json(idx.type, data)
    assert mem2.to_json() == data


def test_parameterized_typetype_constant_roundtrip():
    """TypeType for Array<index.Index, 4> round-trips with nested params in dict."""
    arr_ty = builtin.Array(
        element_type=Index(),
        n=Index().constant(4),
    )
    mem = arr_ty.__constant__
    data = mem.to_json()
    assert data == {
        "tag": "builtin.Array",
        "element_type": {"tag": "index.Index"},
        "n": 4,
    }

    # ASM round-trip with the parameterized TypeType
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "builtin.Array", "element_type": {"tag": "index.Index"}, "n": 4}
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_array_with_ssa_dimension():
    """Array<index.Index, %n> — SSA value as type parameter, round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %n : index.Index = 4
        |     %arr : Array<index.Index, %n> = [1, 2, 3, 4]
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    # Verify the Array type's `n` param is the SSA value %n
    ops = module.functions[0].body.ops
    n_op = ops[0]
    arr_op = ops[1]
    assert isinstance(arr_op.type, builtin.Array)
    assert arr_op.type.n is n_op


def test_array_with_ssa_element_type():
    """Array<%t, 4> — SSA type value as element_type param, round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %arr : Array<%t, 4> = [1, 2, 3, 4]
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    # Verify the Array type's element_type param is the SSA value %t
    ops = module.functions[0].body.ops
    t_op = ops[0]
    arr_op = ops[1]
    assert isinstance(arr_op.type, builtin.Array)
    assert arr_op.type.element_type is t_op


def test_array_with_ssa_element_type_layout():
    """Array<%t, 4> — type_constant resolves the element type for layout computation."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %arr : Array<%t, 4> = [1, 2, 3, 4]
    """)
    module = parse_module(ir)
    ops = module.functions[0].body.ops
    arr_op = ops[1]
    assert isinstance(arr_op.type, builtin.Array)
    # element_type is an SSA ref but the type is ready (ConstantOp is resolved)
    assert arr_op.type.ready
    assert arr_op.ready
    # type_constant resolves the element_type to compute the layout
    arr_layout = arr_op.type.__layout__
    assert arr_layout == layout.Array(layout.Int(), 4)


def test_pointer_with_ssa_pointee():
    """Pointer<%t> — SSA type value as pointee param, round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %p : Pointer<%t> = 0
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    ops = module.functions[0].body.ops
    t_op = ops[0]
    p_op = ops[1]
    assert isinstance(p_op.type, builtin.Pointer)
    assert p_op.type.pointee is t_op
    assert p_op.type.ready
    assert p_op.ready
    assert p_op.type.__layout__ == layout.Pointer(layout.Int())


def test_type_value_jit_identity():
    """TypeType value survives JIT identity function (passed as pointer)."""
    idx = Index()
    mem = idx.__constant__

    # Build: main(t: TypeType<index.Index>) -> TypeType<index.Index> { return t }
    arg = BlockArgument(name="t", type=idx.type)
    func = FunctionOp(
        name="main",
        body=Block(result=arg, args=[arg]),
        result_type=idx.type,
        type=Function(arguments=pack([arg.type]), result_type=idx.type),
    )
    exe = compile_module(Module(ops=[func]))
    result = exe.run(mem)
    assert result.to_json() == mem.to_json()


def test_type_constant_jit_return():
    """JIT function returns a TypeType constant, read back via Memory.from_raw."""
    idx = Index()
    const = ConstantOp(value=idx.__constant__.to_json(), type=idx.type)
    func = FunctionOp(
        name="main",
        body=Block(result=const, args=[]),
        result_type=idx.type,
        type=Function(arguments=pack(), result_type=idx.type),
    )
    exe = compile_module(Module(ops=[func]))
    result = exe.run()
    assert result.to_json() == {"tag": "index.Index"}


def test_span_with_ssa_element_type():
    """Span<%t> — SSA type value as pointee param, round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %xs : Span<%t> = [1, 2, 3]
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    ops = module.functions[0].body.ops
    t_op = ops[0]
    xs_op = ops[1]
    assert isinstance(xs_op.type, builtin.Span)
    assert xs_op.type.pointee is t_op
    assert xs_op.type.ready
    assert xs_op.type.__layout__ == layout.Span(layout.Int())


def test_fat_pointer_with_ssa_pointee():
    """Span<%t> — SSA type value as pointee param, round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "number.Float64"}
        |     %p : Span<%t> = [0.0, 0.0]
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    ops = module.functions[0].body.ops
    t_op = ops[0]
    p_op = ops[1]
    assert isinstance(p_op.type, builtin.Span)
    assert p_op.type.pointee is t_op
    assert p_op.type.ready
    assert p_op.type.__layout__ == layout.Span(layout.Float64())


def test_function_with_ssa_result_type():
    """function<%t> — SSA type value as result param, round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %f : function.Function<[index.Index], %t> = function.function<%t>() body(%x: index.Index):
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    ops = module.functions[0].body.ops
    t_op = next(op for op in ops if isinstance(op, ConstantOp))
    f_op = next(op for op in ops if isinstance(op, FunctionOp))
    assert f_op.result_type is t_op


def test_block_argument_constant_raises_type_error():
    """BlockArgument.__constant__ raises TypeError — it's not a constant."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %main : function.Function<[Type, index.Index], index.Index> = function.function<index.Index>() body(%t: Type, %x: index.Index):
        |     %y : %t = algebra.add(%x, %x)
    """)
    module = parse_module(ir)
    ops = module.functions[0].body.ops
    add_op = ops[0]

    # op.type is a BlockArgument — not a constant
    assert isinstance(add_op.type, BlockArgument)
    with pytest.raises(TypeError):
        add_op.type.__constant__

    # type_constant also fails on a BlockArgument
    with pytest.raises(TypeError):
        type_constant(add_op.type)


def test_type_constant_resolves_ssa_constant():
    """type_constant resolves a ConstantOp TypeType value to a concrete Type."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %x : %t = 42
    """)
    module = parse_module(ir)
    t_op = module.functions[0].body.ops[0]
    resolved = type_constant(t_op)
    assert isinstance(resolved, Index)


def test_compile_with_ssa_function_result():
    """compile() resolves FunctionOp.result_type when it's a ConstantOp type ref."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %f : function.Function<[index.Index], %t> = function.function<%t>() body(%x: index.Index):
    """)
    module = parse_module(ir)
    inner_func = next(
        op for op in module.functions[0].body.ops if isinstance(op, FunctionOp)
    )
    # result is a ConstantOp (SSA ref), not a concrete Type
    assert isinstance(inner_func.result_type, ConstantOp)


def test_compile_function_with_ssa_typed_block_arg():
    """compile() handles block args whose type is a ConstantOp (SSA ref).

    If a function's block arg has type = ConstantOp (not a concrete Type),
    codegen must resolve it via type_constant before accessing __layout__.
    """
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %f : function.Function<[%t], ()> = function.function<Nil>() body(%x: %t):
    """)
    module = parse_module(ir)
    inner_func = next(
        op for op in module.functions[0].body.ops if isinstance(op, FunctionOp)
    )
    # The block arg %x has type = ConstantOp (SSA ref %t), not a concrete Type
    x_arg = inner_func.body.args[0]
    assert isinstance(x_arg.type, ConstantOp)
    # compile() must resolve this via type_constant (not crash on __layout__)
    exe = compile_module(Module(ops=[inner_func]))
    exe.run(42)


def test_compile_constant_with_ssa_type():
    """compile() handles ConstantOp whose type is an SSA ref (ConstantOp).

    %t : Type = {"tag": "index.Index"}
    %x : %t = 42

    codegen must resolve %x's type via type_constant to get __layout__.
    """
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %x : %t = 42
    """)
    module = parse_module(ir)
    x_op = module.functions[0].body.ops[1]
    assert isinstance(x_op, ConstantOp)
    assert isinstance(x_op.type, ConstantOp)  # type is SSA ref, not Type
    # compile must handle this
    exe = compile_module(module)
    exe.run()


def test_compile_input_types_resolved_from_ssa():
    """compile() resolves input_types via type_constant for ConstantOp-typed args.

    Executable.run() uses input_types to convert Python values to Memory,
    so they must be concrete Types, not ConstantOps.
    """
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %f : function.Function<[%t], %t> = function.function<%t>() body(%x: %t):
    """)
    module = parse_module(ir)
    inner_func = next(
        op for op in module.functions[0].body.ops if isinstance(op, FunctionOp)
    )
    exe = compile_module(Module(ops=[inner_func]))
    # input_types must be concrete Types for Memory.from_value to work
    assert all(isinstance(t, builtin.Index) for t in exe.input_types)


def test_staging_resolves_block_arg_type():
    """Staging resolves a BlockArgument TypeType at runtime, not compile time.

    main(%t: Type, %x: index.Index) -> index.Index:
        %y : %t = index.add(%x, %x)

    %t is a function parameter — it can't be resolved at compile time
    (BlockArgument.__constant__ raises TypeError). The staging system
    compiles a callback-based thunk, then resolves %t from the runtime
    arg when called.
    """
    lower_calls: int = 0

    class CountingLowerToLLVM(Pass):
        allow_unregistered_ops = True

        def run(self, m: Module, compiler: Compiler) -> Module:
            nonlocal lower_calls
            lower_calls += 1
            return lower_to_llvm(m)

    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %main : function.Function<[Type, index.Index], index.Index> = function.function<index.Index>() body(%t: Type, %x: index.Index):
        |     %y : %t = algebra.add(%x, %x)
    """)
    module = parse_module(ir)

    # Verify the op's type is a BlockArgument (not resolvable at compile time)
    add_op = module.functions[0].body.ops[0]
    assert isinstance(add_op.type, BlockArgument)

    # Compile produces a callback-based thunk — lower is NOT called yet
    compiler: Compiler[Executable] = Compiler(
        passes=[CountingLowerToLLVM()], exit=LLVMCodegen()
    )
    exe = compiler.compile(module)
    compile_time_calls = lower_calls

    # Each run() triggers runtime JIT — lower is called for each invocation
    assert exe.run({"tag": "index.Index"}, 21).to_json() == 42
    calls_after_first_run = lower_calls
    assert calls_after_first_run > compile_time_calls

    assert exe.run({"tag": "index.Index"}, 100).to_json() == 200
    assert lower_calls > calls_after_first_run


def test_typetype_layout_with_block_arg_is_fixed():
    """TypeType.__layout__ returns TypeValue regardless of concrete.

    TypeValue is a fixed-size pointer layout — it doesn't need to resolve
    the concrete type at layout time. Resolution happens at read time via
    the self-describing tag.
    """
    tt = TypeType()
    assert isinstance(tt.__layout__, TypeValue)
    assert tt.__layout__.byte_size == 8


def test_parse_typetype_block_arg_constant_materializes():
    """ConstantOp with TypeType<%arr_ty> can materialize memory.

    TypeValue is a fixed-size pointer layout, so memory can be materialized
    even when the declared type references a BlockArgument — the self-describing
    tag in the value dict is sufficient for the TypeValue layout to serialize.
    """
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[Type], ()> = function.function<Nil>() body(%arr_ty: Type):
        |     %tt : Type = {"tag": "builtin.Array", "element_type": {"tag": "index.Index"}, "n": 4}
    """)
    module = parse_module(ir)
    tt_op = module.functions[0].body.ops[0]
    assert isinstance(tt_op, ConstantOp)
    assert isinstance(tt_op.value, dict)
    # Memory materializes fine — TypeValue layout is self-describing
    mem = tt_op.memory
    assert mem.to_json() == {
        "tag": "builtin.Array",
        "element_type": {"tag": "index.Index"},
        "n": 4,
    }


def test_staging_with_ssa_result_type():
    """Staging resolves func.result_type when it's a ConstantOp SSA ref.

    main() -> Nil:
        %t : Type = {"tag": "index.Index"}
        %f : function.Function<[Type, index.Index], %t> = function.function<%t>() body(%rt: Type, %x: index.Index):
            %y : %rt = index.add(%x, %x)

    The inner function %f has result = %t (ConstantOp), and its block arg
    %rt is a TypeType<index.Index>. The staging system must resolve both %t and %rt.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %f : function.Function<[Type, index.Index], %t> = function.function<%t>() body(%rt: Type, %x: index.Index):
        |         %y : %rt = algebra.add(%x, %x)
    """)
    module = parse_module(ir)
    inner_func = next(
        op for op in module.functions[0].body.ops if isinstance(op, FunctionOp)
    )
    assert isinstance(inner_func.result_type, ConstantOp)

    class LowerToLLVMPass(Pass):
        allow_unregistered_ops = True

        def run(self, m: Module, compiler: Compiler) -> Module:
            return lower_to_llvm(m)

    compiler: Compiler[Executable] = Compiler(
        passes=[LowerToLLVMPass()], exit=LLVMCodegen()
    )
    exe = compiler.compile(Module(ops=[inner_func]))
    assert exe.run({"tag": "index.Index"}, 21).to_json() == 42


def test_staging_resolves_type_value():
    """Staging resolves a TypeType function param used as op type.

    main(%t: Type, %x: index.Index) -> index.Index:
        %y : %t = index.add(%x, %x)

    The staging system resolves %t to Index, then codegen proceeds normally.
    """

    class LowerAlgebraAdd(Pass):
        allow_unregistered_ops = True

        def run(self, m: Module, compiler: Compiler) -> Module:
            """Lower algebra.add to llvm.add."""
            m = deepcopy(m)
            for func in m.functions:
                for i, op in enumerate(func.body.ops):
                    if isinstance(op, algebra.AddOp):
                        new_op = llvm.AddOp(
                            name=op.name, lhs=op.left, rhs=op.right, type=op.type
                        )
                        func.body.ops[i] = new_op
                        # Patch references
                        for other in func.body.ops:
                            for field_name, _ in other.__operands__:
                                val = getattr(other, field_name)
                                if val is op:
                                    setattr(other, field_name, new_op)
            return m

    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %main : function.Function<[Type, index.Index], index.Index> = function.function<index.Index>() body(%t: Type, %x: index.Index):
        |     %y : %t = algebra.add(%x, %x)
    """)
    module = parse_module(ir)

    compiler: Compiler[Executable] = Compiler(
        passes=[LowerAlgebraAdd()], exit=LLVMCodegen()
    )
    exe = compiler.compile(module)
    assert exe.run({"tag": "index.Index"}, 21).to_json() == 42
