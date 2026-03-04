# Editor Support for `.dgen` Files

## Zed Setup

### 1. Install the Zed extension (syntax highlighting + outline)

In Zed, open the command palette and run `zed: install dev extension`.
Select the `editors/zed-dgen/` directory.

This gives you:
- Syntax highlighting (keywords, types, ops, comments, parameters)
- Symbol outline (`Cmd+Shift+O`)
- Bracket matching (`<>`, `()`)
- Auto-indentation after `:` on type/op declarations

### 2. Configure the LSP (go-to-definition)

Add to your `~/.config/zed/settings.json`:

```json
{
  "lsp": {
    "dgen-lsp": {
      "binary": {
        "path": "dgen/editors/dgen-lsp/run.sh",
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

Restart Zed after changing settings.

### 3. Verify

1. Open any `.dgen` file — keywords, types, and comments should be highlighted
2. `Cmd+Shift+O` — type/op/trait names appear in the outline
3. `Cmd+Click` on a type reference (e.g., `Shape`) — jumps to its definition

## Dependencies

- `pygls` (Python LSP framework): `uv pip install pygls`
- `tree-sitter` CLI: `cargo install tree-sitter-cli` (only needed if regenerating the grammar)
- Node.js: only needed if regenerating the grammar (`tree-sitter generate` requires node)

## Regenerating the grammar

If you modify `editors/tree-sitter-dgen/grammar.js`:

```bash
export PATH="$HOME/.local/share/nodeenv/bin:$PATH"
cd editors/tree-sitter-dgen
tree-sitter generate
```

Then commit, push to GitHub, and update the `rev` in `editors/zed-dgen/extension.toml`
to the new commit SHA. Then reinstall the dev extension in Zed.
