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

# Graph node colours
NODE_BG = "#313244"
NODE_DEAD_BG = "#4a1428"
NODE_DEAD_FG = "#f38ba8"
NODE_NEW_BG = "#1a3d28"
NODE_NEW_FG = "#a6e3a1"

# Arrow colours
ARROW_NEUTRAL = "#6c7086"
ARROW_OLD = "#f38ba8"
ARROW_NEW = "#a6e3a1"

# ── Typography ────────────────────────────────────────────────────────────────

MONO = "Liberation Mono"
TITLE_FS = 26
IR_FS = 15
STATUS_FS = 18
NODE_FS = 13

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
    "ssa": "#89dceb",           # cyan   — SSA values
    "float_lit": "#a6e3a1",     # green  — float literals
    "dialect_qual": "#f9e2af",  # yellow — dialect.name tokens
    "type_name": "#cba6f7",     # purple — built-in type names
    "int_lit": "#fab387",       # peach  — integers
    "kw": "#cba6f7",            # purple — keywords
    "punct": "#6c7086",         # dim    — = :
    "bracket": "#6c7086",       # dim    — < > ( )
    "word": "#cdd6f4",          # normal — unclassified words
    "ws": None,                 # no color (pass through)
    "other": "#6c7086",         # dim    — everything else
}

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKENS)
)


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


def _build_ir_group(
    lines: list[str],
) -> tuple[VGroup, dict[str, MarkupText]]:
    """Build a VGroup of highlighted IR lines, plus a name→mob lookup."""
    name_to_mob: dict[str, MarkupText] = {}
    mobs: list[MarkupText] = []
    for line in lines:
        mob = _ir_mob(line)
        name = _op_name_from_line(line)
        if name:
            name_to_mob[name] = mob
        mobs.append(mob)
    grp = VGroup(*mobs).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
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


def _ir_node(
    op_name: str,
    short_type: str,
    fill: str = NODE_BG,
    fg: str = IR_NORMAL,
) -> VGroup:
    box = RoundedRectangle(
        width=3.2,
        height=0.75,
        corner_radius=0.1,
        fill_color=fill,
        fill_opacity=1.0,
        stroke_color="#6c7086",
        stroke_width=1.5,
    )
    label = MarkupText(
        f'<span foreground="#89dceb">%{op_name}</span>'
        f'  <span foreground="{fg}">{short_type}</span>',
        font=MONO,
        font_size=NODE_FS,
    )
    label.move_to(box)
    return VGroup(box, label)


def _graph_arrow(
    src: VGroup, dst: VGroup, color: str = ARROW_NEUTRAL
) -> Arrow:
    return Arrow(
        src.get_bottom(),
        dst.get_top(),
        buff=0.08,
        color=color,
        stroke_width=2.5,
        max_tip_length_to_length_ratio=0.15,
    )


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
            ev.op_name: ev.op_type
            for ev in events
            if isinstance(ev, ExamineEvent)
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
        header_lines = ["import toy", "", "%main : Nil = function<Nil>() ():"]
        header_group = VGroup(
            *[
                MarkupText(_hl_markup(ln), font=MONO, font_size=IR_FS, color=IR_DIM)
                if ln
                else Text("", font_size=IR_FS)
                for ln in header_lines
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        body_lines = tracer.initial_asm_lines[:]
        body_group, name_to_mob = _build_ir_group(body_lines)
        body_group.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        for mob in body_group:
            mob.shift(RIGHT * 0.5)

        full_ir = VGroup(header_group, body_group).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2
        )
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

        def highlight_line(
            op_name: str, color: str
        ) -> SurroundingRectangle | None:
            mob = name_to_mob.get(op_name)
            if mob is None:
                return None
            rect = SurroundingRectangle(mob, color=color, buff=0.07, corner_radius=0.06)
            self.play(Create(rect), run_time=0.35)
            return rect

        def rebuild_body(lines: list[str], animate: bool = True) -> None:
            nonlocal body_group, name_to_mob
            new_body, new_n2m = _build_ir_group(lines)
            new_body.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            for mob in new_body:
                mob.shift(RIGHT * 0.5)
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

    # ── Graph replacement animation ───────────────────────────────────────────

    def _graph_replace(
        self,
        body_group: VGroup,
        header_group: VGroup,
        before_lines: list[str],
        after_lines: list[str],
        old_name: str,
        new_name: str,
        op_types: dict[str, str],
        status_mob: Text,
    ) -> tuple[VGroup, dict[str, MarkupText]]:
        """
        Visualise replace_uses(old, new) as a graph animation:
          1. Fade out text → show use-def graph.
          2. Highlight eliminated subgraph (red).
          3. Draw bypass arrow on the right (green) — the new direct edge.
          4. Slide bypass left into centre while old nodes collapse.
          5. Return to text with updated IR.
        """
        # ── Parse structure ───────────────────────────────────────────────────
        uses = _parse_uses(before_lines)
        after_name_set = {
            _op_name_from_line(ln) for ln in after_lines if _op_name_from_line(ln)
        }
        dead_names = {n for n in uses if n not in after_name_set}
        consumer_name = next(
            n
            for n, deps in uses.items()
            if old_name in deps and n not in dead_names
        )
        dead_order = _topo_sort(dead_names, uses)  # e.g. ["1", "2"]

        def stype(name: str) -> str:
            return op_types.get(name, "?").split(".")[-1]

        # ── Node positions ────────────────────────────────────────────────────
        X_DEAD = -1.6
        X_BYPASS = 2.8       # bypass arrow is drawn here, then slides left
        Y_PROD = 2.0
        Y_CONS = -2.0
        Y_DEAD_TOP = 0.55
        Y_DEAD_STEP = -1.5

        n_prod = _ir_node(new_name, stype(new_name), NODE_BG, IR_NORMAL)
        n_prod.move_to(np.array([0.0, Y_PROD, 0.0]))

        n_cons = _ir_node(consumer_name, stype(consumer_name), NODE_BG, IR_NORMAL)
        n_cons.move_to(np.array([0.0, Y_CONS, 0.0]))

        dead_nodes: list[VGroup] = []
        for i, name in enumerate(dead_order):
            node = _ir_node(name, stype(name), NODE_DEAD_BG, NODE_DEAD_FG)
            node.move_to(np.array([X_DEAD, Y_DEAD_TOP + i * Y_DEAD_STEP, 0.0]))
            dead_nodes.append(node)

        # ── Edges (old path) ──────────────────────────────────────────────────
        arr_in = _graph_arrow(n_prod, dead_nodes[0], ARROW_OLD)
        dead_arrows = [
            _graph_arrow(dead_nodes[i], dead_nodes[i + 1], ARROW_OLD)
            for i in range(len(dead_nodes) - 1)
        ]
        arr_out = _graph_arrow(dead_nodes[-1], n_cons, ARROW_OLD)

        old_graph = VGroup(
            n_prod, n_cons, *dead_nodes, arr_in, *dead_arrows, arr_out
        )

        # ── Fade out text, dim header, show graph ─────────────────────────────
        self.play(
            FadeOut(body_group),
            header_group.animate.set_opacity(0.15),
            run_time=0.45,
        )
        self.remove(body_group)
        self.play(FadeIn(old_graph), run_time=0.6)
        self.wait(0.3)

        # ── Label eliminated subgraph ─────────────────────────────────────────
        dead_vg = VGroup(*dead_nodes)
        dead_rect = SurroundingRectangle(
            dead_vg,
            color=IR_DEAD,
            buff=0.22,
            corner_radius=0.12,
            stroke_width=2.0,
        )
        elim_label = Text("eliminated", font_size=12, color=IR_DEAD)
        elim_label.next_to(dead_rect, UP, buff=0.12)

        self.play(Create(dead_rect), FadeIn(elim_label), run_time=0.5)
        self.play(
            Transform(
                status_mob,
                _status(
                    f"Subgraph  %{dead_order[0]} → %{dead_order[-1]}  will be removed",
                    STATUS_DEAD,
                ),
            ),
            run_time=0.35,
        )
        self.wait(0.8)

        # ── Draw bypass arrow (new direct edge) on the right ──────────────────
        #
        # The arrow starts at the same y-levels as n_prod/n_cons but offset
        # to x = X_BYPASS.  Sliding it LEFT by X_BYPASS brings it exactly on
        # the centre line between n_prod and n_cons.
        bypass_start = np.array([X_BYPASS, n_prod.get_bottom()[1], 0.0])
        bypass_end = np.array([X_BYPASS, n_cons.get_top()[1], 0.0])

        bypass_arrow = Arrow(
            bypass_start,
            bypass_end,
            color=ARROW_NEW,
            stroke_width=3.5,
            max_tip_length_to_length_ratio=0.12,
        )
        bypass_tag = Text(
            f"direct:  %{new_name} → %{consumer_name}",
            font_size=12,
            color=ARROW_NEW,
        )
        bypass_tag.next_to(bypass_arrow, RIGHT, buff=0.15)
        bypass_grp = VGroup(bypass_arrow, bypass_tag)

        self.play(
            FadeIn(bypass_grp),
            Transform(
                status_mob,
                _status(
                    f"New edge: %{new_name} feeds %{consumer_name} directly",
                    STATUS_OK,
                ),
            ),
            run_time=0.6,
        )
        self.wait(0.9)

        # ── Slide bypass left; collapse old subgraph ──────────────────────────
        self.play(
            Transform(
                status_mob,
                _status("Sliding into place — dead ops collapse", STATUS_OK),
            ),
            run_time=0.3,
        )
        self.play(
            bypass_grp.animate.shift(LEFT * X_BYPASS),
            FadeOut(dead_rect),
            FadeOut(elim_label),
            *[FadeOut(n) for n in dead_nodes],
            FadeOut(arr_in),
            *[FadeOut(a) for a in dead_arrows],
            FadeOut(arr_out),
            run_time=1.3,
        )
        self.wait(0.6)

        # ── Transition back to updated text ───────────────────────────────────
        new_body, new_n2m = _build_ir_group(after_lines)
        new_body.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        for mob in new_body:
            mob.shift(RIGHT * 0.5)
        # Reuse the original body_group position (object still holds its coords)
        new_body.align_to(body_group, UP + LEFT)

        remaining = VGroup(n_prod, n_cons, bypass_grp)
        self.play(
            FadeOut(remaining),
            header_group.animate.set_opacity(1.0),
            run_time=0.4,
        )
        self.play(FadeIn(new_body), run_time=0.45)
        return new_body, new_n2m
