"""Toy source language tokenizer."""

from collections import Optional


@fieldwise_init
struct Token(Copyable, Movable):
    var kind: String
    var text: String

    fn __str__(self) -> String:
        return self.kind + "(" + self.text + ")"


struct Lexer(Movable):
    var text: String
    var pos: Int

    fn __init__(out self, text: String):
        self.text = text
        self.pos = 0

    fn at_end(self) -> Bool:
        return self.pos >= len(self.text)

    fn peek_char(self) -> String:
        if self.at_end():
            return ""
        return String(self.text[byte=self.pos])

    fn advance_char(mut self) -> String:
        var c = String(self.text[byte=self.pos])
        self.pos += 1
        return c

    fn skip_whitespace_and_comments(mut self):
        while not self.at_end():
            var c = self.peek_char()
            if c == " " or c == "\t" or c == "\n" or c == "\r":
                self.pos += 1
            elif c == "#":
                # Skip comment to end of line
                while not self.at_end() and self.peek_char() != "\n":
                    self.pos += 1
            else:
                break

    fn next_token(mut self) -> Token:
        self.skip_whitespace_and_comments()
        if self.at_end():
            return Token(kind="EOF", text="")

        var c = self.peek_char()

        # Single-character tokens
        if c == "(" or c == ")" or c == "{" or c == "}" or c == "[" or c == "]":
            _ = self.advance_char()
            return Token(kind=c, text=c)
        if c == "<" or c == ">" or c == "," or c == ";" or c == "=" or c == "*" or c == "+":
            _ = self.advance_char()
            return Token(kind=c, text=c)

        # Numbers: [0-9]+ (. [0-9]+)?
        if _is_digit(c):
            return self._lex_number()

        # Identifiers and keywords
        if _is_alpha(c) or c == "_":
            return self._lex_identifier()

        # Unknown character
        _ = self.advance_char()
        return Token(kind="UNKNOWN", text=c)

    fn _lex_number(mut self) -> Token:
        var start = self.pos
        while not self.at_end() and _is_digit(self.peek_char()):
            self.pos += 1
        if not self.at_end() and self.peek_char() == ".":
            self.pos += 1
            while not self.at_end() and _is_digit(self.peek_char()):
                self.pos += 1
        var text = String(self.text[start:self.pos])
        return Token(kind="NUMBER", text=text^)

    fn _lex_identifier(mut self) -> Token:
        var start = self.pos
        while not self.at_end():
            var c = self.peek_char()
            if _is_alpha(c) or _is_digit(c) or c == "_":
                self.pos += 1
            else:
                break
        var text = String(self.text[start:self.pos])
        # Check for keywords
        if text == "def" or text == "var" or text == "return":
            return Token(kind=text, text=text)
        return Token(kind="IDENT", text=text^)

    fn peek(mut self) -> Token:
        """Peek at the next token without consuming it."""
        var saved_pos = self.pos
        var tok = self.next_token()
        self.pos = saved_pos
        return tok^


fn _is_alpha(c: String) -> Bool:
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z")


fn _is_digit(c: String) -> Bool:
    return c >= "0" and c <= "9"
