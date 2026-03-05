"""DGEN Language Server — go-to-definition for .dgen dialect files."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

logger = logging.getLogger(__name__)

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
    """Index all .dgen files under workspace roots."""
    modules: dict[str, Path] = {}
    symbols: SymbolTable = {}
    for root in roots:
        for dgen_file in root.rglob("*.dgen"):
            module_name, file_symbols = _index_file(dgen_file)
            modules[module_name] = dgen_file
            symbols.update(file_symbols)
    return modules, symbols


server = LanguageServer("dgen-lsp", "v0.1")


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
    _reindex()


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
    _reindex()


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def definition(params: lsp.DefinitionParams) -> lsp.Location | None:
    doc = server.workspace.get_text_document(params.text_document.uri)
    word = _word_at_position(doc.source, params.position)
    if not word:
        return None
    symbols: SymbolTable = getattr(server, "_symbols", {})
    return symbols.get(word)


def _word_at_position(source: str, pos: lsp.Position) -> str | None:
    lines = source.splitlines()
    if pos.line >= len(lines):
        return None
    line = lines[pos.line]
    if pos.character >= len(line):
        return None
    start = pos.character
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
        start -= 1
    end = pos.character
    while end < len(line) and (line[end].isalnum() or line[end] == "_"):
        end += 1
    word = line[start:end]
    return word if word else None


def _reindex() -> None:
    roots = []
    for folder in server.workspace.folders.values():
        path = Path(folder.uri.removeprefix("file://"))
        roots.append(path)
    if not roots:
        roots = [Path.cwd()]
    modules, symbols = _index_workspace(roots)
    server._symbols = symbols  # type: ignore[attr-defined]
    server._modules = modules  # type: ignore[attr-defined]
    logger.info("Indexed %d symbols from %d modules", len(symbols), len(modules))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server.start_io()
