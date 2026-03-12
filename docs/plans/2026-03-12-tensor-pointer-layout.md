# Plan: Change toy.Tensor layout to Pointer<dtype>, remove method support

## Context

The `.dgen` generator has a partially-implemented method mini-language (parser + codegen)
that is fragile and not well-exercised. The only real usage is:
- `affine.Shape.num_elements()` — never called from any pass code
- `toy.Tensor.unpack_shape()` — used extensively by the passes, but its implementation
  (`self.shape.__constant__.to_json()`) is trivial Python

Additionally, `toy.Tensor.__layout__` is currently monkey-patched at runtime in
`toy/dialects/__init__.py` using `layout.Array(layout.Float64(), prod(dims))` because
the generator couldn't express `Array<dtype, shape.num_elements()>` (method call in
data field args). The simpler, correct layout is `Pointer<dtype>` — a raw pointer to the
element type. This removes the shape-dependency from the layout, which is appropriate
since layout describes memory representation, not semantics.

Goal: switch `toy.Tensor` to `Pointer<dtype>` layout (expressible in `.dgen`), then
remove the method parsing/generation machinery entirely.

## Files to Modify

### .dgen source files
- `toy/dialects/toy.dgen` — remove `method unpack_shape`, remove TODO comment, add `data: Pointer<dtype>`
- `toy/dialects/affine.dgen` — remove `method num_elements` from `Shape`

### Generated files (regenerate after .dgen changes)
- `toy/dialects/toy.py` — regenerate: gains `__layout__` property, loses `unpack_shape` method
- `toy/dialects/affine.py` — regenerate: loses `num_elements` method

### Runtime helper file
- `toy/dialects/__init__.py` — remove `_tensor_layout` monkey-patch; add `unpack_shape` monkey-patch

### Generator: AST
- `dgen/gen/ast.py` — remove method-related nodes: `Expr`, `NameExpr`, `LiteralExpr`,
  `AttrExpr`, `BinOpExpr`, `CallExpr`, `Assignment`, `ReturnStmt`, `ForStmt`, `IfStmt`,
  `Statement`, `MethodDecl`; remove `methods: list[MethodDecl]` field from `TypeDecl`

### Generator: Parser
- `dgen/gen/parser.py` — remove `method ...` branch from `_parse_type_body()` (lines 175–207);
  remove functions: `_parse_expr`, `_parse_postfix`, `_find_binop`, `_find_matching_reverse`,
  `_find_assignment_eq`, `_parse_method_body` (and all method-AST imports)

### Generator: Python codegen
- `dgen/gen/python.py` — remove `_emit_expr`, `_emit_stmts`, the `for method in td.methods`
  loop (lines 327–330); remove method-related AST imports

### Tests
- `test/test_gen_parser.py` — remove 6 `test_parse_method_*` tests; clean up method-AST imports
- `test/test_gen_python.py` — remove 3 `test_generate_method_*` tests; clean up method-AST imports
- `test/test_gen_ast.py` — remove `test_method_decl` and `test_type_with_methods`; clean up imports

## Step-by-step

1. **Edit `toy/dialects/toy.dgen`**:
   - Remove lines 5–9 (TODO comment + `method unpack_shape` block)
   - Add `    data: Pointer<dtype>` as the data field for `Tensor`

2. **Edit `toy/dialects/affine.dgen`**:
   - Remove lines 6–10 (`method num_elements` block from `Shape`)

3. **Regenerate `toy/dialects/affine.py`**:
   ```
   python -m dgen.gen toy/dialects/affine.dgen > toy/dialects/affine.py
   ```

4. **Regenerate `toy/dialects/toy.py`**:
   ```
   python -m dgen.gen toy/dialects/toy.dgen -I affine=toy.dialects.affine > toy/dialects/toy.py
   ```
   The generated `Tensor` class will now have a `__layout__` property returning
   `layout.Pointer(dgen.type.type_constant(self.dtype).__layout__)` and no `unpack_shape`.

5. **Edit `toy/dialects/__init__.py`**:
   - Remove `_tensor_layout` property and `Tensor.__layout__ = _tensor_layout` line (layout now generated)
   - Remove `from math import prod` import if unused after this removal
   - Remove `Tensor` import if no longer needed (nothing left referencing it)

6. **Edit `dgen/gen/ast.py`**:
   - Remove the "Expression AST" section (lines 80–124): `Expr`, `NameExpr`, `LiteralExpr`,
     `AttrExpr`, `BinOpExpr`, `CallExpr`
   - Remove the "Statement AST" section (lines 127–166): `Assignment`, `ReturnStmt`,
     `ForStmt`, `IfStmt`, `Statement`
   - Remove `MethodDecl` (lines 169–176)
   - Remove `methods: list[MethodDecl] = field(default_factory=list)` from `TypeDecl`

7. **Edit `dgen/gen/parser.py`**:
   - Remove method-AST imports (all `Expr`, statement, `MethodDecl` imports)
   - Remove `method ...` branch in `_parse_type_body()` (lines 175–207)
   - Update `_parse_type_body` return signature: remove `methods` from return tuple;
     update callers
   - Remove functions: `_parse_expr` (~35 lines), `_parse_postfix` (~30 lines),
     `_find_binop` (~35 lines), `_find_matching_reverse` (~15 lines),
     `_find_assignment_eq` (~20 lines), `_parse_method_body` (~60 lines)

8. **Edit `dgen/gen/python.py`**:
   - Remove method-related imports: `Assignment`, `AttrExpr`, `BinOpExpr`, `CallExpr`,
     `Expr`, `ForStmt`, `IfStmt`, `LiteralExpr`, `NameExpr`, `ReturnStmt`, `Statement`
   - Remove `_emit_expr` function (~15 lines)
   - Remove `_emit_stmts` function (~15 lines)
   - Remove the `for method in td.methods:` loop block (lines 327–330)

9. **Inline `unpack_shape()` calls in the pass files**:
   - `toy/passes/shape_inference.py` (6 calls): replace `x.unpack_shape()` → `x.shape.__constant__.to_json()`
   - `toy/passes/toy_to_affine.py` (6 calls): same replacement
   - `toy/passes/affine_to_llvm.py` (3 calls): same replacement
   Note: `shape.__constant__.to_json()` returns `list[int]` from the `Array<Index, rank>` layout.

10. **Edit test files** to remove method-related tests and imports:
   - `test/test_gen_parser.py`: remove 6 `test_parse_method_*` functions and the
     `ReturnStmt`, `LiteralExpr`, `Assignment`, `ForStmt`, `BinOpExpr`, `AttrExpr`,
     `CallExpr`, `IfStmt`, `NameExpr` imports
   - `test/test_gen_python.py`: remove 3 `test_generate_method_*` functions and related
     imports (`MethodDecl`, `ReturnStmt`, `LiteralExpr`, `Assignment`, `ForStmt`,
     `AttrExpr`, `BinOpExpr`, `NameExpr`)
   - `test/test_gen_ast.py`: remove `test_method_decl`, `test_type_with_methods`, and
     `MethodDecl` import

11. **Run `ruff format && ruff check --fix && ty check`** to clean up

12. **Run `pytest . -q`** to verify all tests pass (minus the ~11 removed method tests)

13. **Commit and push** to `claude/refactor-tensor-pointer-layout-vznpz`

## Expected generated Tensor layout

After the change, `toy.py` Tensor will have:
```python
@property
def __layout__(self) -> layout.Layout:
    return layout.Pointer(dgen.type.type_constant(self.dtype).__layout__)
```

This replaces the monkey-patched `layout.Array(layout.Float64(), prod(dims))`.

## Verification

- `pytest . -q` — all remaining tests pass
- `python -m toy.cli toy/test/testdata/constant.toy` — end-to-end pipeline still works
- Confirm `toy/dialects/toy.py` and `toy/dialects/affine.py` have no `def num_elements`
  or `def unpack_shape` in them
- Confirm `dgen/gen/parser.py` has no `_parse_method_body`, `MethodDecl`, or `method ` parsing
