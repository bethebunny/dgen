"""Toy source language tokenizer."""

from dataclasses import dataclass


@dataclass
class Token:
    kind: str
    text: str

    def __str__(self) -> str:
        return f"{self.kind}({self.text})"


class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def peek_char(self) -> str:
        if self.at_end():
            return ""
        return self.text[self.pos]

    def advance_char(self) -> str:
        c = self.text[self.pos]
        self.pos += 1
        return c

    def skip_whitespace_and_comments(self) -> None:
        while not self.at_end():
            c = self.peek_char()
            if c in " \t\n\r":
                self.pos += 1
            elif c == "#":
                while not self.at_end() and self.peek_char() != "\n":
                    self.pos += 1
            else:
                break

    def next_token(self) -> Token:
        self.skip_whitespace_and_comments()
        if self.at_end():
            return Token(kind="EOF", text="")

        c = self.peek_char()

        # Single-character tokens
        if c in "(){}[]":
            self.advance_char()
            return Token(kind=c, text=c)
        if c in "<>,;=*+":
            self.advance_char()
            return Token(kind=c, text=c)

        # Numbers: [0-9]+ (. [0-9]+)?
        if c.isdigit():
            return self._lex_number()

        # Identifiers and keywords
        if c.isalpha() or c == "_":
            return self._lex_identifier()

        # Unknown character
        self.advance_char()
        return Token(kind="UNKNOWN", text=c)

    def _lex_number(self) -> Token:
        start = self.pos
        while not self.at_end() and self.text[self.pos].isdigit():
            self.pos += 1
        if not self.at_end() and self.text[self.pos] == ".":
            self.pos += 1
            while not self.at_end() and self.text[self.pos].isdigit():
                self.pos += 1
        text = self.text[start : self.pos]
        return Token(kind="NUMBER", text=text)

    def _lex_identifier(self) -> Token:
        start = self.pos
        while not self.at_end():
            c = self.text[self.pos]
            if c.isalnum() or c == "_":
                self.pos += 1
            else:
                break
        text = self.text[start : self.pos]
        if text in ("def", "var", "return"):
            return Token(kind=text, text=text)
        return Token(kind="IDENT", text=text)

    def peek(self) -> Token:
        """Peek at the next token without consuming it."""
        saved_pos = self.pos
        tok = self.next_token()
        self.pos = saved_pos
        return tok
