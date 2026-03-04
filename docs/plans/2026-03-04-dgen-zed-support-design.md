# Zed Editor Support for `.dgen` Files — Design

## Goal

Syntax highlighting and go-to-definition for `.dgen` dialect specification files in the Zed editor.

## Architecture

Three components, layered:

1. **Tree-sitter grammar** (`editors/tree-sitter-dgen/`) — parses `.dgen` files into a concrete syntax tree. Required by Zed for all syntax-aware features.

2. **Zed extension** (`editors/zed-dgen/`) — maps the tree-sitter CST to highlight colors, bracket matching, outline symbols, and auto-indentation. No Rust/WASM needed (query files only).

3. **Python LSP** (`editors/dgen-lsp/`) — provides go-to-definition by parsing all workspace `.dgen` files, building a symbol index, and resolving cross-file references via import declarations. Uses `pygls` and reuses `dgen/gen/parser.py`.

## The `.dgen` Language Surface

Keywords: `from`, `import`, `trait`, `type`, `op`, `block`
Comments: `#` line comments
Punctuation: `<>`, `()`, `:`, `,`, `->`, `=`
Type references: `Name`, `Name<args>` (nestable)
Indented bodies after `:` on type/op declarations

## Tree-sitter Grammar Design

Node types for `grammar.js`:

- `source_file` — top-level, contains declarations
- `import_declaration` — `from <module> import <name>, ...`
- `trait_declaration` — `trait <Name>`
- `type_declaration` — `type <Name>[<params>][: <body>]`
- `op_declaration` — `op <name>[<params>](<operands>) -> <return_type>[: <body>]`
- `type_body` / `op_body` — indented blocks
- `data_field` — `name: TypeExpr`
- `block_declaration` — `block <name>`
- `param_list` / `operand_list` — comma-separated declarations
- `param` / `operand` — `name: Type [= default]`
- `type_ref` — `Name` or `Name<type_ref, ...>`
- `comment` — `# ...`
- `identifier` / `type_identifier` — names

## Zed Extension Files

- `extension.toml` — points to tree-sitter-dgen grammar (local `file://` path for dev)
- `languages/dgen/config.toml` — `.dgen` suffix, `#` comments, 4-space indent
- `languages/dgen/highlights.scm` — keyword, type, comment, operator, punctuation captures
- `languages/dgen/outline.scm` — type/op/trait names appear in symbol outline
- `languages/dgen/brackets.scm` — `<>`, `()`
- `languages/dgen/indents.scm` — indent after `:` on type/op declarations

## LSP Design

A `pygls`-based Python LSP server providing:

- **textDocument/definition** — resolve type references to their defining `.dgen` file and line
- **Workspace indexing** — on startup and file change, parse all `.dgen` files, build `{name: (file, line)}` map
- **Import resolution** — follow `from module import Name` using a configurable module-to-file mapping (`.dgen-lsp.json` or CLI flags)

Configured in Zed `settings.json` as a custom language server (no WASM glue needed).

## File Layout

```
editors/
  tree-sitter-dgen/
    grammar.js          # Tree-sitter grammar definition
    package.json        # Node project for tree-sitter CLI
    src/                # Generated parser (C code, committed)
  zed-dgen/
    extension.toml
    languages/dgen/
      config.toml
      highlights.scm
      outline.scm
      brackets.scm
      indents.scm
  dgen-lsp/
    __main__.py         # pygls LSP server entry point
```
