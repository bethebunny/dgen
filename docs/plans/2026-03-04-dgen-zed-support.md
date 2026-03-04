# Zed `.dgen` Editor Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Syntax highlighting and go-to-definition for `.dgen` dialect specification files in Zed.

**Architecture:** Tree-sitter grammar (no external scanner — `.dgen` is simple enough that keyword-terminated bodies work with whitespace in extras), Zed extension with highlight/outline queries, and a Python LSP server using `pygls` that reuses the existing `dgen/gen/parser.py` for go-to-definition.

**Tech Stack:** tree-sitter (grammar.js + C parser), Zed extension (TOML + Scheme queries), Python + pygls (LSP)

**Prerequisites already verified:**
- `tree-sitter` CLI 0.25.5 installed at `~/.cargo/bin/tree-sitter`
- Node 22.14.0 installed at `~/.local/share/nodeenv/bin/node` (needed by `tree-sitter generate`)
- `cargo` available, `uv` available for Python packages

**Important:** Add `~/.local/share/nodeenv/bin` to `PATH` before running `tree-sitter generate`.

---

### Task 1: Create tree-sitter-dgen project skeleton

**Files:**
- Create: `editors/tree-sitter-dgen/grammar.js`
- Create: `editors/tree-sitter-dgen/package.json`
- Create: `editors/tree-sitter-dgen/tree-sitter.json`

**Step 1: Create directory and package.json**

```bash
mkdir -p editors/tree-sitter-dgen
```

Write `editors/tree-sitter-dgen/package.json`:
```json
{
  "name": "tree-sitter-dgen",
  "version": "0.1.0",
  "description": "Tree-sitter grammar for DGEN dialect specification files",
  "main": "grammar.js",
  "tree-sitter": [
    {
      "scope": "source.dgen",
      "file-types": ["dgen"],
      "highlights": "queries/highlights.scm"
    }
  ]
}
```

**Step 2: Create tree-sitter.json**

Write `editors/tree-sitter-dgen/tree-sitter.json`:
```json
{
  "grammars": [
    {
      "name": "dgen",
      "scope": "source.dgen",
      "path": ".",
      "file-types": ["dgen"]
    }
  ]
}
```

**Step 3: Write grammar.js**

Write `editors/tree-sitter-dgen/grammar.js`:
```javascript
/// <reference types="tree-sitter-cli/dsl" />

module.exports = grammar({
  name: "dgen",

  extras: $ => [/\s/, $.comment],

  word: $ => $.identifier,

  rules: {
    source_file: $ => repeat($._declaration),

    _declaration: $ => choice(
      $.import_declaration,
      $.trait_declaration,
      $.type_declaration,
      $.op_declaration,
    ),

    // from module import Name1, Name2
    import_declaration: $ => seq(
      'from',
      field('module', $.identifier),
      'import',
      field('name', commaSep1($.identifier)),
    ),

    // trait HasSingleBlock
    trait_declaration: $ => seq(
      'trait',
      field('name', $.identifier),
    ),

    // type Name<params>:
    //     field: TypeRef
    type_declaration: $ => seq(
      'type',
      field('name', $.identifier),
      optional(field('type_parameters', $.type_parameters)),
      optional(seq(':', repeat($.data_field))),
    ),

    data_field: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type_ref),
    ),

    // op name<params>(operands) -> ReturnType:
    //     block body
    op_declaration: $ => seq(
      'op',
      field('name', $.identifier),
      optional(field('type_parameters', $.type_parameters)),
      '(',
      optional(field('operands', $.operand_list)),
      ')',
      '->',
      field('return_type', $._type_ref),
      optional(seq(':', repeat($.block_declaration))),
    ),

    block_declaration: $ => seq(
      'block',
      field('name', $.identifier),
    ),

    // <name: Type, name: Type = default>
    type_parameters: $ => seq(
      '<',
      commaSep1($.parameter),
      '>',
    ),

    parameter: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type_ref),
      optional(seq('=', field('default', $.identifier))),
    ),

    operand_list: $ => commaSep1($.operand),

    operand: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type_ref),
      optional(seq('=', field('default', $.identifier))),
    ),

    // Type references: Name or Name<T, U>
    _type_ref: $ => choice(
      $.generic_type,
      $.type_identifier,
    ),

    generic_type: $ => prec(1, seq(
      field('name', $.type_identifier),
      '<',
      commaSep1(field('argument', $._type_ref)),
      '>',
    )),

    type_identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    comment: $ => token(seq('#', /.*/)),
  },
});

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}
```

**Design note:** `type_identifier` and `identifier` have the same regex but are separate rules so highlights.scm can distinguish type references from other identifiers by node type. The `word` declaration uses `identifier` for keyword extraction, ensuring `from`, `import`, `trait`, `type`, `op`, `block` are recognized as keywords and never parsed as identifiers. `_type_ref` uses `type_identifier` so all type references get the `@type` highlight.

**Potential issue:** Since `identifier` and `type_identifier` have the same regex, tree-sitter may have a conflict. If `tree-sitter generate` fails with a conflict error, merge them into one `identifier` rule and use only contextual queries in highlights.scm to distinguish types. The fix would be:
- Change `_type_ref` to use `$.identifier` instead of `$.type_identifier`
- Remove `type_identifier` rule
- In highlights.scm, use field-based queries to highlight type positions

**Step 4: Generate the parser**

```bash
export PATH="$HOME/.local/share/nodeenv/bin:$PATH"
cd editors/tree-sitter-dgen
tree-sitter generate
```

Expected: creates `src/parser.c`, `src/tree_sitter/parser.h`, `src/grammar.json`, `src/node-types.json`.

**Step 5: Test with all .dgen files**

```bash
cd editors/tree-sitter-dgen
tree-sitter parse ../../dgen/dialects/builtin.dgen
tree-sitter parse ../../toy/dialects/toy.dgen
tree-sitter parse ../../toy/dialects/affine.dgen
```

Expected: each file produces a syntax tree with NO `(ERROR)` nodes. If there are errors, fix the grammar and re-generate.

**Step 6: Commit**

```bash
jj new -m "feat: add tree-sitter grammar for .dgen files"
```

---

### Task 2: Create Zed extension with syntax highlighting

**Files:**
- Create: `editors/zed-dgen/extension.toml`
- Create: `editors/zed-dgen/languages/dgen/config.toml`
- Create: `editors/zed-dgen/languages/dgen/highlights.scm`
- Create: `editors/zed-dgen/languages/dgen/outline.scm`
- Create: `editors/zed-dgen/languages/dgen/brackets.scm`
- Create: `editors/zed-dgen/languages/dgen/indents.scm`

**Step 1: Create extension.toml**

Write `editors/zed-dgen/extension.toml`:
```toml
id = "dgen"
name = "DGEN"
version = "0.0.1"
schema_version = 1
authors = ["Stef"]
description = "DGEN dialect specification language support"

[grammars.dgen]
repository = "file:///home/stef/dgen/editors/tree-sitter-dgen"
rev = ""
```

**Note:** The `rev` field may need to be the HEAD commit sha of the repo, or may be ignored for `file://` paths. If Zed complains, try setting `rev` to the current `jj log --no-graph -r @ -T 'commit_id'` value, or remove the field entirely.

**Step 2: Create language config**

Write `editors/zed-dgen/languages/dgen/config.toml`:
```toml
name = "DGEN"
grammar = "dgen"
path_suffixes = ["dgen"]
line_comments = ["# "]
tab_size = 4
```

**Step 3: Write highlights.scm**

Write `editors/zed-dgen/languages/dgen/highlights.scm`:
```scheme
; Keywords
["from" "import" "trait" "type" "op" "block"] @keyword

; Arrow operator
"->" @operator

; Assignment
"=" @operator

; Delimiters
[":" ","] @punctuation.delimiter

; Brackets
["(" ")"] @punctuation.bracket
["<" ">"] @punctuation.bracket

; Comments
(comment) @comment

; Declaration names
(trait_declaration name: (identifier) @type)
(type_declaration name: (identifier) @type)
(op_declaration name: (identifier) @function)

; Import module
(import_declaration module: (identifier) @module)

; Imported names (these are types/traits)
(import_declaration name: (identifier) @type)

; Data field names
(data_field name: (identifier) @property)

; Block declaration names
(block_declaration name: (identifier) @variable)

; Parameter and operand names
(parameter name: (identifier) @variable.parameter)
(operand name: (identifier) @variable.parameter)

; Default values
(parameter default: (identifier) @constant)
(operand default: (identifier) @constant)

; Type references
(type_identifier) @type

; Generic type name
(generic_type name: (type_identifier) @type)
```

**Note:** The exact capture names depend on what the grammar produces. If `type_identifier` was merged with `identifier` (see Task 1 fallback), replace `(type_identifier) @type` with context-based queries like `(parameter type: (identifier) @type)` etc.

**Step 4: Write outline.scm**

Write `editors/zed-dgen/languages/dgen/outline.scm`:
```scheme
(trait_declaration
  name: (identifier) @name) @item

(type_declaration
  name: (identifier) @name) @item

(op_declaration
  name: (identifier) @name) @item
```

**Step 5: Write brackets.scm**

Write `editors/zed-dgen/languages/dgen/brackets.scm`:
```scheme
("(" @open ")" @close)
("<" @open ">" @close)
```

**Step 6: Write indents.scm**

Write `editors/zed-dgen/languages/dgen/indents.scm`:
```scheme
(type_declaration ":" @indent)
(op_declaration ":" @indent)
```

**Step 7: Install dev extension in Zed**

Open Zed, run command palette: `zed: install dev extension`, select the `editors/zed-dgen/` directory.

**Step 8: Test highlighting**

Open each `.dgen` file in Zed and verify:
- Keywords (`from`, `import`, `trait`, `type`, `op`, `block`) are highlighted
- Type names in declarations are highlighted as types
- Op names are highlighted as functions
- Comments are grayed out
- Type references (`FatPointer<Byte>`, `Array<Index, rank>`) are highlighted as types
- Parameters and operands have distinct colors

**Step 9: Test outline**

Press `Cmd+Shift+O` (or equivalent) in a `.dgen` file. Verify that type, op, and trait names appear in the outline panel.

**Step 10: Commit**

```bash
jj new -m "feat: add Zed extension for .dgen syntax highlighting"
```

---

### Task 3: Write Python LSP server for go-to-definition

**Files:**
- Create: `editors/dgen-lsp/__main__.py`
- Create: `editors/dgen-lsp/run.sh`

**Step 1: Install pygls**

```bash
uv pip install pygls
```

Verify:
```bash
python -c "import pygls; print(pygls.__version__)"
```

**Step 2: Write the LSP server**

Write `editors/dgen-lsp/__main__.py`:
```python
"""DGEN Language Server — go-to-definition for .dgen dialect files."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from lsprotocol import types as lsp
from pygls.server import LanguageServer

from dgen.gen.ast import DgenFile, TypeDecl, OpDecl, TraitDecl
from dgen.gen import parser as dgen_parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol index
# ---------------------------------------------------------------------------

SymbolTable = dict[str, lsp.Location]


def _index_file(path: Path) -> tuple[str, dict[str, lsp.Location]]:
    """Parse a .dgen file and return (module_name, {symbol: Location})."""
    module_name = path.stem
    text = path.read_text()
    lines = text.splitlines()
    symbols: dict[str, lsp.Location] = {}
    uri = path.as_uri()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("trait "):
            name = stripped.split()[1]
            symbols[name] = lsp.Location(
                uri=uri,
                range=lsp.Range(
                    start=lsp.Position(line=i, character=line.index(name)),
                    end=lsp.Position(line=i, character=line.index(name) + len(name)),
                ),
            )
        elif stripped.startswith("type "):
            rest = stripped[5:]
            name = re.split(r"[<:\s]", rest)[0]
            col = line.index(name, 5)
            symbols[name] = lsp.Location(
                uri=uri,
                range=lsp.Range(
                    start=lsp.Position(line=i, character=col),
                    end=lsp.Position(line=i, character=col + len(name)),
                ),
            )
        elif stripped.startswith("op "):
            rest = stripped[3:]
            name = re.split(r"[<(\s]", rest)[0]
            col = line.index(name, 3)
            symbols[name] = lsp.Location(
                uri=uri,
                range=lsp.Range(
                    start=lsp.Position(line=i, character=col),
                    end=lsp.Position(line=i, character=col + len(name)),
                ),
            )

    return module_name, symbols


def _index_workspace(roots: list[Path]) -> tuple[dict[str, Path], SymbolTable]:
    """Index all .dgen files under workspace roots.

    Returns:
        modules: {module_name: file_path}
        symbols: {symbol_name: Location}
    """
    modules: dict[str, Path] = {}
    symbols: SymbolTable = {}

    for root in roots:
        for dgen_file in root.rglob("*.dgen"):
            module_name, file_symbols = _index_file(dgen_file)
            modules[module_name] = dgen_file
            symbols.update(file_symbols)

    return modules, symbols


# ---------------------------------------------------------------------------
# Language server
# ---------------------------------------------------------------------------

server = LanguageServer("dgen-lsp", "v0.1")


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
    """Re-index on file open."""
    _reindex()


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
    """Re-index on file save."""
    _reindex()


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def definition(params: lsp.DefinitionParams) -> lsp.Location | None:
    """Go-to-definition: resolve the word at cursor to its definition."""
    doc = server.workspace.get_text_document(params.text_document.uri)
    word = _word_at_position(doc.source, params.position)
    if not word:
        return None

    symbols: SymbolTable = getattr(server, "_symbols", {})
    return symbols.get(word)


def _word_at_position(source: str, pos: lsp.Position) -> str | None:
    """Extract the identifier at the given position."""
    lines = source.splitlines()
    if pos.line >= len(lines):
        return None
    line = lines[pos.line]
    if pos.character >= len(line):
        return None

    # Find word boundaries
    start = pos.character
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
        start -= 1
    end = pos.character
    while end < len(line) and (line[end].isalnum() or line[end] == '_'):
        end += 1

    word = line[start:end]
    return word if word else None


def _reindex() -> None:
    """Re-index all .dgen files in the workspace."""
    roots = []
    for folder in server.workspace.folders.values():
        path = Path(folder.uri.removeprefix("file://"))
        roots.append(path)
    if not roots:
        # Fallback: use cwd
        roots = [Path.cwd()]

    modules, symbols = _index_workspace(roots)
    server._symbols = symbols  # type: ignore[attr-defined]
    server._modules = modules  # type: ignore[attr-defined]
    logger.info("Indexed %d symbols from %d modules", len(symbols), len(modules))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server.start_io()
```

**Step 3: Create run script**

Write `editors/dgen-lsp/run.sh`:
```bash
#!/usr/bin/env bash
# Launch the DGEN LSP server.
# Assumes it's run from the dgen repo root (or PYTHONPATH is set).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
exec python -m editors.dgen-lsp "$@"
```

Wait — Python module names can't have hyphens. Let's use a direct script path instead:

```bash
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
exec python "$SCRIPT_DIR/__main__.py" "$@"
```

Make executable:
```bash
chmod +x editors/dgen-lsp/run.sh
```

**Step 4: Test the LSP manually**

```bash
cd /home/stef/dgen
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"rootUri":"file:///home/stef/dgen"}}' | python editors/dgen-lsp/__main__.py 2>/dev/null | head -1
```

Expected: a JSON-RPC response with server capabilities (or at least no crash).

**Step 5: Commit**

```bash
jj new -m "feat: add Python LSP server for .dgen go-to-definition"
```

---

### Task 4: Configure LSP in Zed and test end-to-end

**Files:**
- Modify: `~/.config/zed/settings.json` (user's Zed settings)

**Step 1: Add LSP configuration to Zed settings**

Add to `~/.config/zed/settings.json`:
```json
{
  "lsp": {
    "dgen-lsp": {
      "binary": {
        "path": "/home/stef/dgen/editors/dgen-lsp/run.sh",
        "arguments": []
      }
    }
  },
  "languages": {
    "DGEN": {
      "language_servers": ["dgen-lsp"]
    }
  }
}
```

**Note:** Merge these keys into existing settings (don't overwrite the file). If Zed doesn't recognize the custom LSP, check:
1. The extension is installed (language "DGEN" must be registered first)
2. The `run.sh` script is executable
3. The PYTHONPATH includes the repo root

**Step 2: Restart Zed**

Close and reopen Zed (or run `zed: reload extensions`).

**Step 3: Test go-to-definition**

1. Open `toy/dialects/toy.dgen` in Zed
2. Place cursor on `Shape` in `type Tensor<shape: Shape, ...>`
3. Press `Cmd+Click` or `F12` (go-to-definition)
4. Expected: jumps to `affine.dgen` line 3 where `type Shape<rank: Index>:` is defined

5. Place cursor on `Index` in `from builtin import Index, ...`
6. Go-to-definition
7. Expected: jumps to `builtin.dgen` line 3 where `type index:` is defined

**Note:** The `Index` name in imports maps to the type name `index` in builtin.dgen. The LSP indexes the raw `.dgen` names (lowercase `index`), while imports use the Python-friendly names (`Index`). If go-to-definition doesn't resolve imported names, we may need to add a name normalization step to the LSP (lowercase the first letter of the lookup). This can be addressed as a follow-up.

**Step 4: Test outline navigation**

1. Open any `.dgen` file in Zed
2. Press `Cmd+Shift+O` to open the symbol outline
3. Verify all type, op, and trait declarations appear
4. Click one to navigate to it

**Step 5: Commit**

```bash
jj new -m "docs: add Zed LSP configuration instructions"
```

---

## Troubleshooting

**Grammar conflicts:** If `tree-sitter generate` reports conflicts between `identifier` and `type_identifier` (same regex), merge them into one `identifier` rule and use field-based queries in highlights.scm instead.

**Zed doesn't load the grammar:** For `file://` repository paths, Zed may need the generated `src/` directory committed. Make sure `editors/tree-sitter-dgen/src/parser.c` exists.

**LSP doesn't start:** Check `zed: open log` for errors. Common issues: `run.sh` not executable, `pygls` not installed, PYTHONPATH not set correctly.

**Import names don't resolve:** The `.dgen` codegen uses capitalized names in `from builtin import Index` for the Python type `index`. The LSP indexes raw names. A name-normalization mapping may be needed.
