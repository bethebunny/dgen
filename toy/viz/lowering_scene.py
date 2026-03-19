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


def _hl_markup_highlight_operand(line: str, operand: str, color: str) -> str:
    """Like _hl_markup but overrides the color of %operand on the RHS."""
    eq_idx = line.find("=")
    if eq_idx == -1:
        return _hl_markup(line)
    lhs_markup = _hl_markup(line[: eq_idx + 1])
    rhs = line[eq_idx + 1 :]
    ssa_target = f"%{operand}"
    parts: list[str] = []
    for m in _TOKEN_RE.finditer(rhs):
        kind = m.lastgroup or "other"
        raw = m.group()
        escaped = _pango_escape(raw)
        if kind == "ssa" and raw == ssa_target:
            parts.append(
                f'<span foreground="{color}" font_weight="bold">{escaped}</span>'
            )
        else:
            col = _TOKEN_COLOR.get(kind)
            if col:
                parts.append(f'<span foreground="{col}">{escaped}</span>')
            else:
                parts.append(escaped)
    return lhs_markup + "".join(parts)


def _ir_mob(line: str) -> MarkupText:
    return MarkupText(_hl_markup(line), font=MONO, font_size=IR_FS)


def _ir_mob_hl_operand(line: str, operand: str, color: str) -> MarkupText:
    return MarkupText(
        _hl_markup_highlight_operand(line, operand, color), font=MONO, font_size=IR_FS
    )


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
        Animate replace_uses(old → new) entirely in the IR view.

          A. Draw boxes around all %old_name occurrences (def line + operand tokens).
          B. Draw a differently-colored box around the %new_name definition.
          C. Morph consumer text (%old → %new) and operand box color (orange → cyan).
          D. Fade all SSA boxes; red boxes on dead ops; slide survivors up.
        """
        # ── Parse ─────────────────────────────────────────────────────────────
        before_uses = _parse_uses(before_lines)
        after_uses = _parse_uses(after_lines)
        before_order = [n for n in (_op_name_from_line(ln) for ln in before_lines) if n]
        after_order = [n for n in (_op_name_from_line(ln) for ln in after_lines) if n]
        after_name_set = set(after_order)
        dead_order = [n for n in before_order if n not in after_name_set]

        after_line_map = {n: ln for ln in after_lines if (n := _op_name_from_line(ln))}

        # Surviving ops that used old_name as an operand
        consumer_names = [
            n
            for n, deps in before_uses.items()
            if old_name in deps and n in after_name_set
        ]

        # new_name: what replaced old_name in the first consumer
        first_consumer = consumer_names[0]
        after_refs = after_uses.get(first_consumer, [])
        before_refs = before_uses.get(first_consumer, [])
        new_name_candidates = [r for r in after_refs if r not in before_refs]
        new_name = new_name_candidates[0] if new_name_candidates else after_refs[0]

        ORANGE = "#f9e2af"
        CYAN = "#89dceb"

        def _line_box(name: str, color: str) -> SurroundingRectangle | None:
            m = name_to_mob.get(name)
            if m is None:
                return None
            return SurroundingRectangle(
                m, color=color, buff=0.07, corner_radius=0.05, stroke_width=2
            )

        def _token_box(
            mob: MarkupText, token: str, color: str
        ) -> SurroundingRectangle | None:
            # mob.text has spaces stripped; mob.chars has one submob per non-space char.
            # Search after '=' so we find operand occurrences, not the definition.
            clean = mob.text
            eq_idx = clean.find("=")
            search_from = eq_idx + 1 if eq_idx != -1 else 0
            # token also has spaces stripped when searching
            clean_token = token.replace(" ", "")
            idx = clean.find(clean_token, search_from)
            if idx == -1:
                return None
            glyph_list = mob.chars.submobjects[idx : idx + len(clean_token)]
            if not glyph_list:
                return None
            return SurroundingRectangle(
                VGroup(*glyph_list),
                color=color,
                buff=0.04,
                corner_radius=0.04,
                stroke_width=2,
            )

        # ── Step A: Draw boxes around all %old_name occurrences ───────────────
        old_def_box = _line_box(old_name, ORANGE)
        consumer_use_boxes: dict[str, SurroundingRectangle] = {}
        for cn in consumer_names:
            box = _token_box(name_to_mob[cn], f"%{old_name}", ORANGE)
            if box is not None:
                consumer_use_boxes[cn] = box

        self.play(
            *([Create(old_def_box)] if old_def_box else []),
            *[Create(b) for b in consumer_use_boxes.values()],
            Transform(
                status_mob,
                _status(f"replace_uses(%{old_name}  →  %{new_name})", STATUS_INFO),
            ),
            run_time=0.5,
        )
        self.wait(0.4)

        # ── Step B: Draw box around %new_name definition ──────────────────────
        new_def_box = _line_box(new_name, CYAN)
        if new_def_box is not None:
            self.play(Create(new_def_box), run_time=0.4)
        self.wait(0.3)

        # ── Step C: Morph consumer text + operand box orange → cyan ───────────
        morph_anims: list = []
        for cn in consumer_names:
            final_mob = _ir_mob(after_line_map[cn])
            final_mob.move_to(name_to_mob[cn])
            morph_anims.append(Transform(name_to_mob[cn], final_mob))
        for box in consumer_use_boxes.values():
            morph_anims.append(Transform(box, box.copy().set_color(CYAN)))
        self.play(*morph_anims, run_time=0.6)
        self.wait(0.5)

        # ── Step D: Fade SSA boxes → red dead boxes + slide survivors up ──────
        dead_mobs = [name_to_mob[n] for n in dead_order if n in name_to_mob]
        dead_rects = [
            SurroundingRectangle(
                m, color=IR_DEAD, buff=0.07, corner_radius=0.06, stroke_width=1.5
            )
            for m in dead_mobs
        ]
        self.play(
            *([FadeOut(old_def_box)] if old_def_box else []),
            *([FadeOut(new_def_box)] if new_def_box else []),
            *[FadeOut(b) for b in consumer_use_boxes.values()],
            *[Create(r) for r in dead_rects],
            Transform(
                status_mob,
                _status(f"Eliminating  %{' + %'.join(dead_order)}", STATUS_DEAD),
            ),
            run_time=0.5,
        )
        self.wait(0.6)

        new_body, new_n2m = _build_ir_group(after_lines)
        new_body.align_to(body_group, UP + LEFT)

        slide_anims: list = [FadeOut(r) for r in dead_rects]
        slide_anims += [FadeOut(name_to_mob[n]) for n in dead_order if n in name_to_mob]
        for name in after_order:
            old_m = name_to_mob.get(name)
            new_m = new_n2m.get(name)
            if old_m is not None and new_m is not None:
                slide_anims.append(Transform(old_m, new_m))
        self.play(*slide_anims, run_time=0.8)
        self.wait(0.5)

        # ── Return surviving mobs ─────────────────────────────────────────────
        for name in dead_order:
            if name in name_to_mob:
                self.remove(name_to_mob[name])

        result_body = VGroup(*[name_to_mob[n] for n in after_order if n in name_to_mob])
        result_n2m = {n: name_to_mob[n] for n in after_order if n in name_to_mob}
        return result_body, result_n2m
