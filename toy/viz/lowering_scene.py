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

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    FadeIn,
    FadeOut,
    MarkupText,
    RoundedRectangle,
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
IR_NORMAL = "#cdd6f4"
IR_DIM = "#45475a"
IR_EXAMINE = "#f9e2af"
IR_DEAD = "#f38ba8"
IR_MATCH = "#a6e3a1"
TITLE_COLOR = "#cba6f7"
STATUS_OK = "#a6e3a1"
STATUS_INFO = "#89b4fa"
STATUS_DEAD = "#f38ba8"

# Arrow colours
ARROW_OLD = "#f38ba8"
ARROW_NEW = "#a6e3a1"
ARROW_NEUTRAL = "#6c7086"

# ── Typography ────────────────────────────────────────────────────────────────

MONO = "Liberation Mono"
TITLE_FS = 26
IR_FS = 16
PANEL_FS = 13  # smaller font for the replacement graph panel
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


def _ir_mob(line: str, font_size: int = IR_FS) -> MarkupText:
    return MarkupText(_hl_markup(line), font=MONO, font_size=font_size)


def _build_ir_group(
    lines: list[str],
    font_size: int = IR_FS,
) -> tuple[VGroup, dict[str, MarkupText]]:
    """Build a VGroup of highlighted IR lines, plus a name→mob lookup."""
    name_to_mob: dict[str, MarkupText] = {}
    mobs: list[MarkupText] = []
    for line in lines:
        mob = _ir_mob(line, font_size)
        name = _op_name_from_line(line)
        if name:
            name_to_mob[name] = mob
        mobs.append(mob)
    grp = VGroup(*mobs).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    return grp, name_to_mob


def _status(msg: str, color: str = STATUS_INFO) -> Text:
    return Text(msg, font_size=STATUS_FS, color=color).to_edge(DOWN, buff=0.55)


# ── Graph helpers (used by _graph_replace) ────────────────────────────────────


def _parse_uses(lines: list[str]) -> dict[str, list[str]]:
    """Return {op_name: [operand_names]} parsed from ASM lines."""
    result: dict[str, list[str]] = {}
    for line in lines:
        name = _op_name_from_line(line)
        if name and "=" in line:
            rhs = line[line.index("=") + 1 :]
            result[name] = re.findall(r"%(\w+)", rhs)
    return result


def _topo_sort(names: set[str], uses: dict[str, list[str]]) -> list[str]:
    """Return names in topological order (producers before consumers)."""
    result: list[str] = []
    visited: set[str] = set()

    def visit(n: str) -> None:
        if n in visited or n not in names:
            return
        visited.add(n)
        for dep in uses.get(n, []):
            visit(dep)
        result.append(n)

    for n in sorted(names):
        visit(n)
    return result


def _line_box(mob: MarkupText, color: str, buff: float = 0.12) -> RoundedRectangle:
    """A thin rounded box behind a text line."""
    w = mob.width + buff * 2
    h = mob.height + buff * 2
    box = RoundedRectangle(
        width=w,
        height=h,
        corner_radius=0.06,
        fill_color=color,
        fill_opacity=0.12,
        stroke_color=color,
        stroke_width=1.5,
    )
    box.move_to(mob)
    return box


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

        # Collect op_type from examine events
        op_types: dict[str, str] = {
            ev.op_name: ev.op_type for ev in events if isinstance(ev, ExamineEvent)
        }

        # ── 2. Title ──────────────────────────────────────────────────────────
        title = Text(
            "ToyOptimize: Transpose Elimination",
            font_size=TITLE_FS,
            color=TITLE_COLOR,
        ).to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=1.0)
        self.wait(0.2)

        # ── 3. Header (static, dimmed) + initial body ─────────────────────────
        #
        # Use Text (not MarkupText) for header so bounding-box LEFT edges are
        # consistent across different string lengths — avoiding the glyph-metrics
        # drift that causes MarkupText to appear misaligned.
        #
        import_mob = Text("import toy", font=MONO, font_size=IR_FS, color=IR_DIM)
        funcdef_mob = Text(
            "%main : Nil = function<Nil>() ():",
            font=MONO,
            font_size=IR_FS,
            color=IR_DIM,
        )
        # Position funcdef below import with a blank-line gap, LEFT-aligned
        funcdef_mob.next_to(import_mob, DOWN, buff=0.42, aligned_edge=LEFT)
        header_group = VGroup(import_mob, funcdef_mob)

        body_lines = tracer.initial_asm_lines[:]
        body_group, name_to_mob = _build_ir_group(body_lines)

        # Place body below funcdef, indented to the right by INDENT units
        INDENT = 0.6
        body_group.next_to(funcdef_mob, DOWN, buff=0.18, aligned_edge=LEFT)
        body_group.shift(RIGHT * INDENT)

        full_ir = VGroup(header_group, body_group)
        full_ir.center().shift(DOWN * 0.3)

        self.play(FadeIn(full_ir), run_time=0.9)
        self.wait(0.4)

        # Status label
        status_mob = _status("")
        self.add(status_mob)

        def set_status(msg: str, color: str = STATUS_INFO) -> None:
            self.play(Transform(status_mob, _status(msg, color)), run_time=0.35)

        # ── 4. Event loop ─────────────────────────────────────────────────────
        active_rect: SurroundingRectangle | None = None
        current_lines = body_lines[:]

        def clear_rect() -> None:
            nonlocal active_rect
            if active_rect is not None:
                self.play(FadeOut(active_rect), run_time=0.25)
                active_rect = None

        def highlight_line(op_name: str, color: str) -> SurroundingRectangle | None:
            mob = name_to_mob.get(op_name)
            if mob is None:
                return None
            rect = SurroundingRectangle(mob, color=color, buff=0.07, corner_radius=0.06)
            self.play(Create(rect), run_time=0.35)
            return rect

        def rebuild_body(lines: list[str], animate: bool = True) -> None:
            nonlocal body_group, name_to_mob
            new_body, new_n2m = _build_ir_group(lines)
            new_body.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            new_body.align_to(body_group, UP + LEFT)
            if animate:
                self.play(FadeOut(body_group), run_time=0.3)
                self.remove(body_group)
                self.add(new_body)
                self.play(FadeIn(new_body), run_time=0.3)
            else:
                self.remove(body_group)
                self.add(new_body)
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
                self.wait(0.7)

            elif isinstance(event, MatchEvent):
                clear_rect()
                body_group, name_to_mob = self._graph_replace(
                    body_group=body_group,
                    header_group=header_group,
                    name_to_mob=name_to_mob,
                    before_lines=current_lines,
                    after_lines=event.asm_lines_after,
                    old_name=event.old_name,
                    new_name=event.new_name,
                    op_types=op_types,
                    status_mob=status_mob,
                )
                current_lines = event.asm_lines_after[:]

        # ── 5. Final ──────────────────────────────────────────────────────────
        clear_rect()
        set_status("Done.  2 ops eliminated.", STATUS_OK)
        self.wait(2.5)

    # ── Replacement animation ─────────────────────────────────────────────────

    def _graph_replace(
        self,
        body_group: VGroup,
        header_group: VGroup,
        name_to_mob: dict[str, MarkupText],
        before_lines: list[str],
        after_lines: list[str],
        old_name: str,
        new_name: str,
        op_types: dict[str, str],
        status_mob: Text,
    ) -> tuple[VGroup, dict[str, MarkupText]]:
        """
        Animate replace_uses(old → new) using the actual ASM lines as nodes.

        Steps:
          1. Fade out body → show replacement graph (ASM-line nodes, arrows).
          2. Eliminated lines in a red box (left); producer + consumer on right.
          3. Arrows show old chain (red) and new direct edge (green).
          4. Return to updated IR text.
        """
        # ── Parse structure ───────────────────────────────────────────────────
        uses = _parse_uses(before_lines)
        before_order = [n for n in (_op_name_from_line(ln) for ln in before_lines) if n]
        after_name_set = {
            n for n in (_op_name_from_line(ln) for ln in after_lines) if n
        }
        dead_set = {n for n in before_order if n not in after_name_set}
        dead_order = [n for n in before_order if n in dead_set]

        consumer_name = next(
            n for n, deps in uses.items() if old_name in deps and n not in dead_set
        )

        # ── Build panel mobs (ASM lines at PANEL_FS) ──────────────────────────
        #
        # Diamond layout:
        #
        #          [producer]          (x=0, top)
        #         /           \
        #   [dead chain]   (bypass)    (x=-2.5, middle)
        #         \           /
        #          [consumer]          (x=0, bottom)
        #
        # Panel mobs are capped at MAX_MOB_W units wide to stay on screen.
        #
        X_CENTER = 0.0
        X_DEAD = -3.0
        Y_PROD = 2.3
        Y_DEAD_TOP = 0.55
        Y_DEAD_STEP = -1.3
        Y_CONS = -2.3
        MAX_MOB_W = 5.8

        def _find_line(name: str, lines: list[str]) -> str:
            return next(ln for ln in lines if _op_name_from_line(ln) == name)

        def _panel_mob(line: str, x: float, y: float) -> MarkupText:
            mob = _ir_mob(line, PANEL_FS)
            if mob.width > MAX_MOB_W:
                mob.scale(MAX_MOB_W / mob.width)
            mob.move_to(np.array([x, y, 0.0]))
            return mob

        # Producer mob — centered at top
        prod_line = _find_line(new_name, before_lines)
        prod_mob = _panel_mob(prod_line, X_CENTER, Y_PROD)

        # Dead mobs — left column
        dead_mobs: list[MarkupText] = []
        for i, name in enumerate(dead_order):
            line = _find_line(name, before_lines)
            mob = _panel_mob(line, X_DEAD, Y_DEAD_TOP + i * Y_DEAD_STEP)
            dead_mobs.append(mob)

        # Consumer mob — centered at bottom (old version, references %old_name)
        cons_old_line = _find_line(consumer_name, before_lines)
        cons_old_mob = _panel_mob(cons_old_line, X_CENTER, Y_CONS)

        # Consumer mob — new version (references %new_name directly)
        cons_new_line = _find_line(consumer_name, after_lines)
        cons_new_mob = _panel_mob(cons_new_line, X_CENTER, Y_CONS)

        # ── Fade out body, dim header, show graph ─────────────────────────────
        self.play(
            FadeOut(body_group),
            header_group.animate.set_opacity(0.12),
            run_time=0.45,
        )
        self.remove(body_group)

        all_graph = VGroup(prod_mob, cons_old_mob, *dead_mobs)
        self.play(FadeIn(all_graph), run_time=0.5)
        self.wait(0.2)

        # ── Red bounding box around eliminated lines ───────────────────────────
        dead_vg = VGroup(*dead_mobs)
        dead_box = SurroundingRectangle(
            dead_vg, color=IR_DEAD, buff=0.2, corner_radius=0.1, stroke_width=2.0
        )
        dead_label = Text("eliminated", font_size=13, color=IR_DEAD)
        dead_label.next_to(dead_box, UP, buff=0.1)

        self.play(
            Create(dead_box),
            FadeIn(dead_label),
            Transform(
                status_mob,
                _status(
                    f"Subgraph  %{dead_order[0]} → %{dead_order[-1]}  eliminated",
                    STATUS_DEAD,
                ),
            ),
            run_time=0.5,
        )
        self.wait(0.5)

        # ── Use-def arrows (old chain, red) ───────────────────────────────────
        #
        # Arrows show SSA data flow: producer → dead[0] → … → dead[-1] → consumer
        # The arrows leave the bottom-left of the producer, flow through the
        # dead chain on the left, and re-enter the top-left of the consumer.
        #
        def _diag_arrow(
            src: MarkupText, dst: MarkupText, color: str
        ) -> Arrow:
            return Arrow(
                src.get_bottom(),
                dst.get_top(),
                buff=0.1,
                color=color,
                stroke_width=2.5,
                max_tip_length_to_length_ratio=0.15,
            )

        def _vert_arrow(
            src: MarkupText, dst: MarkupText, color: str
        ) -> Arrow:
            return Arrow(
                src.get_bottom(),
                dst.get_top(),
                buff=0.08,
                color=color,
                stroke_width=2.5,
                max_tip_length_to_length_ratio=0.15,
            )

        # prod → dead[0]  (diagonal, top-center → top-left)
        arr_prod_dead = _diag_arrow(prod_mob, dead_mobs[0], ARROW_OLD)
        # dead chain (vertical, within left column)
        dead_chain = [
            _vert_arrow(dead_mobs[i], dead_mobs[i + 1], ARROW_OLD)
            for i in range(len(dead_mobs) - 1)
        ]
        # dead[-1] → consumer  (diagonal, bottom-left → bottom-center)
        arr_dead_cons = _diag_arrow(dead_mobs[-1], cons_old_mob, ARROW_OLD)

        old_arrows = VGroup(arr_prod_dead, *dead_chain, arr_dead_cons)
        self.play(Create(old_arrows), run_time=0.6)
        self.wait(0.6)

        # ── New direct connection (bypass, green) ─────────────────────────────
        #
        # Show the new consumer line below the old one, draw a direct green
        # arrow on the right side (bypassing the dead chain).
        #
        cons_new_mob.shift(DOWN * 0.85)  # place just below old consumer
        bypass_label = Text(
            f"direct: %{new_name} → %{consumer_name}",
            font_size=13,
            color=ARROW_NEW,
        )
        bypass_label.next_to(cons_new_mob, UP, buff=0.08)

        # Bypass arrow: prod → new consumer, offset to the right so it's distinct
        bypass_start = np.array(
            [prod_mob.get_right()[0] + 0.3, prod_mob.get_center()[1], 0.0]
        )
        bypass_end = np.array(
            [cons_new_mob.get_right()[0] + 0.3, cons_new_mob.get_center()[1], 0.0]
        )
        arr_bypass = Arrow(
            bypass_start,
            bypass_end,
            buff=0.0,
            color=ARROW_NEW,
            stroke_width=3.0,
            max_tip_length_to_length_ratio=0.12,
        )

        new_cons_box = _line_box(cons_new_mob, IR_MATCH, buff=0.12)

        self.play(
            FadeIn(cons_new_mob),
            Create(new_cons_box),
            FadeIn(bypass_label),
            Transform(
                status_mob,
                _status(
                    f"New edge: %{new_name} feeds %{consumer_name} directly",
                    STATUS_OK,
                ),
            ),
            run_time=0.5,
        )
        self.play(Create(arr_bypass), run_time=0.5)
        self.wait(0.8)

        # ── Collapse dead subgraph ────────────────────────────────────────────
        self.play(
            FadeOut(dead_box),
            FadeOut(dead_label),
            FadeOut(dead_vg),
            FadeOut(old_arrows),
            run_time=0.6,
        )
        self.wait(0.3)

        # ── Restore text view ─────────────────────────────────────────────────
        remaining = VGroup(
            prod_mob, cons_old_mob, cons_new_mob, new_cons_box, bypass_label, arr_bypass
        )
        new_body, new_n2m = _build_ir_group(after_lines)
        new_body.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        new_body.align_to(body_group, UP + LEFT)

        self.play(
            FadeOut(remaining),
            header_group.animate.set_opacity(1.0),
            run_time=0.4,
        )
        self.play(FadeIn(new_body), run_time=0.45)
        return new_body, new_n2m
