"""Manim scene: visualize ToyOptimize transpose elimination.

Render with:
    manim -ql toy/viz/lowering_scene.py TransposeEliminationScene

Flags:
  -q  low quality (faster)
  -l  480p
  For high quality use -qh
"""

from __future__ import annotations

import re

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    MarkupText,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    Write,
)

from dgen import asm
from dgen.compiler import Compiler, IdentityPass
from toy.test.helpers import strip_prefix
from toy.viz.trace import Event, ExamineEvent, MatchEvent, TracingToyOptimize

# ── Palette ───────────────────────────────────────────────────────────────────

BG = "#1e1e2e"
IR_DIM = "#45475a"
IR_EXAMINE = "#f9e2af"
IR_DEAD = "#f38ba8"
TITLE_COLOR = "#cba6f7"
STATUS_OK = "#a6e3a1"
STATUS_INFO = "#89b4fa"
STATUS_DEAD = "#f38ba8"

# ── Typography ────────────────────────────────────────────────────────────────

MONO = "Liberation Mono"
TITLE_FS = 26
IR_FS = 16
STATUS_FS = 18

# ── Syntax-highlighting token patterns (priority order) ───────────────────────

_TOKENS: list[tuple[str, str]] = [
    ("ssa", r"%\w+"),
    ("float_lit", r"\b\d+\.\d+\b"),
    ("dialect_qual", r"\b\w+\.\w+"),
    ("type_name", r"\b(?:Nil|F64|F32|Index|Bool|I\d+|U\d+)\b"),
    ("int_lit", r"\b\d+\b"),
    ("kw", r"\b(?:import|function|chain)\b"),
    ("punct", r"[=:,]"),
    ("bracket", r"[<>()\[\]{}]"),
    ("word", r"\w+"),
    ("ws", r"\s+"),
    ("other", r"."),
]

_TOKEN_COLOR: dict[str, str | None] = {
    "ssa": "#89dceb",  # cyan   — SSA values
    "float_lit": "#a6e3a1",  # green  — float literals
    "dialect_qual": "#f9e2af",  # yellow — dialect.name tokens
    "type_name": "#cba6f7",  # purple — built-in type names
    "int_lit": "#fab387",  # peach  — integers
    "kw": "#cba6f7",  # purple — keywords
    "punct": "#6c7086",  # dim    — = :
    "bracket": "#6c7086",  # dim    — < > ( )
    "word": "#cdd6f4",  # normal — unclassified words
    "ws": None,  # no color (pass through)
    "other": "#6c7086",  # dim    — everything else
}

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKENS))


def _pango_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hl_markup(line: str) -> str:
    """Convert a plain ASM line to Pango markup with syntax colouring."""
    parts: list[str] = []
    for m in _TOKEN_RE.finditer(line):
        kind = m.lastgroup or "other"
        raw = m.group()
        escaped = _pango_escape(raw)
        color = _TOKEN_COLOR.get(kind)
        if color:
            parts.append(f'<span foreground="{color}">{escaped}</span>')
        else:
            parts.append(escaped)
    return "".join(parts)


# ── IR display helpers ────────────────────────────────────────────────────────


def _op_name_from_line(line: str) -> str | None:
    """'%2 : ...' → '2', else None."""
    s = line.strip()
    if s.startswith("%") and " : " in s:
        return s[1 : s.index(" : ")]
    return None


def _ir_mob(line: str) -> MarkupText:
    return MarkupText(_hl_markup(line), font=MONO, font_size=IR_FS)


def _build_ir_group(lines: list[str]) -> tuple[VGroup, dict[str, MarkupText]]:
    """Build a VGroup of highlighted IR lines, plus a name→mob lookup."""
    name_to_mob: dict[str, MarkupText] = {}
    mobs: list[MarkupText] = []
    for line in lines:
        mob = _ir_mob(line)
        name = _op_name_from_line(line)
        if name:
            name_to_mob[name] = mob
        mobs.append(mob)
    grp = VGroup(*mobs).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    return grp, name_to_mob


def _status(msg: str, color: str = STATUS_INFO) -> Text:
    return Text(msg, font_size=STATUS_FS, color=color).to_edge(DOWN, buff=0.55)


def _parse_uses(lines: list[str]) -> dict[str, list[str]]:
    """Return {op_name: [operand_names]} parsed from ASM lines."""
    result: dict[str, list[str]] = {}
    for line in lines:
        name = _op_name_from_line(line)
        if name and "=" in line:
            rhs = line[line.index("=") + 1 :]
            result[name] = re.findall(r"%(\w+)", rhs)
    return result


# ── Scene ─────────────────────────────────────────────────────────────────────


class TransposeEliminationScene(Scene):
    """Animate ToyOptimize eliminating a transpose(transpose(x)) pair."""

    def construct(self) -> None:
        self.camera.background_color = BG  # type: ignore[attr-defined]

        # ── 1. Build trace ────────────────────────────────────────────────────
        ir_text = strip_prefix("""
            | import toy
            |
            | %main : Nil = function<Nil>() ():
            |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            |     %1 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%0)
            |     %2 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.transpose(%1)
            |     %3 : Nil = toy.print(%2)
        """)
        module = asm.parse(ir_text)
        tracer = TracingToyOptimize()
        tracer.run(module, Compiler([], IdentityPass()))
        events: list[Event] = tracer.events

        # ── 2. Title ──────────────────────────────────────────────────────────
        title = Text(
            "ToyOptimize: Transpose Elimination",
            font_size=TITLE_FS,
            color=TITLE_COLOR,
        ).to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=0.8)

        # ── 3. Header (static, dimmed) + initial body ─────────────────────────
        #
        # Use Text (not MarkupText) for header: consistent bounding-box LEFT
        # edges across different string lengths.
        #
        import_mob = Text("import toy", font=MONO, font_size=IR_FS, color=IR_DIM)
        funcdef_mob = Text(
            "%main : Nil = function<Nil>() ():",
            font=MONO,
            font_size=IR_FS,
            color=IR_DIM,
        )
        funcdef_mob.next_to(import_mob, DOWN, buff=0.42, aligned_edge=LEFT)
        header_group = VGroup(import_mob, funcdef_mob)

        body_lines = tracer.initial_asm_lines[:]
        body_group, name_to_mob = _build_ir_group(body_lines)

        INDENT = 0.6
        body_group.next_to(funcdef_mob, DOWN, buff=0.18, aligned_edge=LEFT)
        body_group.shift(RIGHT * INDENT)

        full_ir = VGroup(header_group, body_group)
        full_ir.center().shift(UP * 0.4)

        self.play(FadeIn(full_ir), run_time=0.8)
        self.wait(0.3)

        # Status label
        status_mob = _status("")
        self.add(status_mob)

        def set_status(msg: str, color: str = STATUS_INFO) -> None:
            self.play(Transform(status_mob, _status(msg, color)), run_time=0.3)

        # ── 4. Event loop ─────────────────────────────────────────────────────
        active_rect: SurroundingRectangle | None = None
        current_lines = body_lines[:]

        def clear_rect() -> None:
            nonlocal active_rect
            if active_rect is not None:
                self.play(FadeOut(active_rect), run_time=0.2)
                active_rect = None

        def highlight_line(op_name: str, color: str) -> SurroundingRectangle | None:
            mob = name_to_mob.get(op_name)
            if mob is None:
                return None
            rect = SurroundingRectangle(mob, color=color, buff=0.07, corner_radius=0.06)
            self.play(Create(rect), run_time=0.3)
            return rect

        def rebuild_body(lines: list[str]) -> None:
            nonlocal body_group, name_to_mob
            new_body, new_n2m = _build_ir_group(lines)
            new_body.align_to(body_group, UP + LEFT)
            self.play(
                Transform(body_group, new_body),
                run_time=0.5,
            )
            body_group = new_body
            name_to_mob = new_n2m

        for event in events:
            if isinstance(event, ExamineEvent):
                if event.asm_lines != current_lines:
                    rebuild_body(event.asm_lines)
                    current_lines = event.asm_lines[:]
                clear_rect()
                active_rect = highlight_line(event.op_name, IR_EXAMINE)
                set_status(
                    f"Examining  %{event.op_name}  ({event.op_type})",
                    STATUS_INFO,
                )
                self.wait(0.9)

            elif isinstance(event, MatchEvent):
                clear_rect()
                body_group, name_to_mob = self._inline_replace(
                    body_group=body_group,
                    name_to_mob=name_to_mob,
                    before_lines=current_lines,
                    after_lines=event.asm_lines_after,
                    old_name=event.old_name,
                    status_mob=status_mob,
                )
                current_lines = event.asm_lines_after[:]

        # ── 5. Final ──────────────────────────────────────────────────────────
        clear_rect()
        set_status("Done — 2 ops eliminated.", STATUS_OK)
        self.wait(2.5)

    # ── In-place replacement animation ───────────────────────────────────────

    def _inline_replace(
        self,
        body_group: VGroup,
        name_to_mob: dict[str, MarkupText],
        before_lines: list[str],
        after_lines: list[str],
        old_name: str,
        status_mob: Text,
    ) -> tuple[VGroup, dict[str, MarkupText]]:
        """
        Animate replace_uses(old → new) entirely in the IR view — no
        separate graph panel.

          1. Red highlight boxes appear around the dead ops.
          2. Dead ops fade out; surviving ops slide up to fill the gap
             (the changed consumer line morphs to its new text in place).
          3. Return surviving mobs as the new body state.
        """
        # ── Parse what's dead ─────────────────────────────────────────────────
        uses = _parse_uses(before_lines)
        before_order = [n for n in (_op_name_from_line(ln) for ln in before_lines) if n]
        after_order = [n for n in (_op_name_from_line(ln) for ln in after_lines) if n]
        after_name_set = set(after_order)
        dead_order = [n for n in before_order if n not in after_name_set]

        consumer_name = next(
            n
            for n, deps in uses.items()
            if old_name in deps and n in after_name_set
        )

        # ── Step 1: Highlight dead lines ──────────────────────────────────────
        dead_mobs = [name_to_mob[n] for n in dead_order if n in name_to_mob]
        dead_rects = [
            SurroundingRectangle(
                m, color=IR_DEAD, buff=0.07, corner_radius=0.06, stroke_width=1.5
            )
            for m in dead_mobs
        ]
        self.play(
            *[Create(r) for r in dead_rects],
            Transform(
                status_mob,
                _status(
                    f"Eliminating  %{' + %'.join(dead_order)}",
                    STATUS_DEAD,
                ),
            ),
            run_time=0.45,
        )
        self.wait(0.7)

        # ── Step 2: Build final layout, animate in-place ──────────────────────
        #
        # new_body has after_lines arranged from the same top-left anchor as
        # body_group.  Transform(old_mob, new_mob) morphs each survivor's
        # content and position simultaneously — so the consumer line slides up
        # and its %old_name reference visibly becomes %new_name.
        #
        new_body, new_n2m = _build_ir_group(after_lines)
        new_body.align_to(body_group, UP + LEFT)

        anims: list = [FadeOut(r) for r in dead_rects]
        anims += [FadeOut(name_to_mob[n]) for n in dead_order if n in name_to_mob]
        for name in after_order:
            old_m = name_to_mob.get(name)
            new_m = new_n2m.get(name)
            if old_m is not None and new_m is not None:
                anims.append(Transform(old_m, new_m))

        self.play(
            *anims,
            Transform(
                status_mob,
                _status(
                    f"replace_uses(%{old_name}, operand of %{consumer_name})",
                    STATUS_OK,
                ),
            ),
            run_time=1.1,
        )
        self.wait(0.5)

        # ── Step 3: Return surviving mobs as new body state ───────────────────
        #
        # Transform mutates the src mob in-place; src is on screen.
        # Build result_body from those (now-transformed) old mobs.
        #
        for name in dead_order:
            if name in name_to_mob:
                self.remove(name_to_mob[name])

        result_body = VGroup(*[name_to_mob[n] for n in after_order if n in name_to_mob])
        result_n2m = {n: name_to_mob[n] for n in after_order if n in name_to_mob}
        return result_body, result_n2m
