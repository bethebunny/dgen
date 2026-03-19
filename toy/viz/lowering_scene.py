"""Manim scene: visualize ToyOptimize transpose elimination.

Render with:
    manim -pql toy/viz/lowering_scene.py TransposeEliminationScene

Flags:
  -p  preview after render
  -q  low quality (faster)
  -l  480p (low resolution)
  For high quality use -pqh or -pq4k
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
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
IR_DIM = "#585b70"
IR_EXAMINE = "#f9e2af"  # yellow  — op being examined
IR_MATCH = "#a6e3a1"  # green   — replacement target (survives)
IR_DEAD = "#f38ba8"  # red     — op about to be eliminated
TITLE_COLOR = "#cba6f7"  # purple
STATUS_OK = "#a6e3a1"
STATUS_INFO = "#89b4fa"  # blue

MONO = "Monospace"
TITLE_FS = 26
IR_FS = 14
STATUS_FS = 19

# ── Helpers ───────────────────────────────────────────────────────────────────


def _op_name_from_line(line: str) -> str | None:
    """Extract the SSA name from a line like '%2 : ...' → '2'.

    Returns None for lines that aren't op definitions.
    """
    s = line.strip()
    if s.startswith("%") and " : " in s:
        return s[1 : s.index(" : ")]
    return None


def _ir_line(text: str, color: str = IR_NORMAL) -> Text:
    return Text(text, font=MONO, font_size=IR_FS, color=color)


def _build_ir_group(
    lines: list[str],
    colors: dict[str, str] | None = None,
) -> VGroup:
    """Build a VGroup of monospace Text objects, one per ASM line."""
    if colors is None:
        colors = {}
    texts: list[Text] = []
    for line in lines:
        name = _op_name_from_line(line)
        color = colors.get(name, IR_NORMAL) if name else IR_NORMAL
        texts.append(_ir_line(line, color))
    grp = VGroup(*texts).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    return grp


def _status(msg: str, color: str = STATUS_INFO, fs: int = STATUS_FS) -> Text:
    return Text(msg, font_size=fs, color=color).to_edge(DOWN, buff=0.55)


# ── Scene ─────────────────────────────────────────────────────────────────────


class TransposeEliminationScene(Scene):
    """Animate how ToyOptimize eliminates a transpose(transpose(x)) pair."""

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
        self.play(Write(title), run_time=1.0)
        self.wait(0.2)

        # ── 3. Function header (static, dimmed) ───────────────────────────────
        header_lines = [
            "import toy",
            "",
            "%main : Nil = function<Nil>() ():",
        ]
        header_group = VGroup(*[_ir_line(ln, IR_DIM) for ln in header_lines]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.18
        )

        # ── 4. Initial IR body ─────────────────────────────────────────────────
        body_lines = tracer.initial_asm_lines[:]
        body_group = _build_ir_group(body_lines)
        # Indent body under header
        body_group.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        for mob in body_group:
            mob.shift(RIGHT * 0.5)

        full_ir = VGroup(header_group, body_group).arrange(
            DOWN, aligned_edge=LEFT, buff=0.18
        )
        full_ir.center().shift(DOWN * 0.2)

        self.play(FadeIn(full_ir), run_time=0.9)
        self.wait(0.4)

        # Status label (reused across animations via Transform)
        status_mob = _status("")
        self.add(status_mob)

        def set_status(msg: str, color: str = STATUS_INFO) -> None:
            new_s = _status(msg, color)
            self.play(Transform(status_mob, new_s), run_time=0.35)

        # ── 5. Animate events ─────────────────────────────────────────────────
        # body_group tracks the VGroup currently on screen for the IR body.
        active_rect: SurroundingRectangle | None = None

        def clear_rect() -> None:
            nonlocal active_rect
            if active_rect is not None:
                self.play(FadeOut(active_rect), run_time=0.25)
                active_rect = None

        def highlight_line(op_name: str, color: str) -> SurroundingRectangle | None:
            """Draw a colored rectangle around the line for op_name."""
            for mob in body_group:
                assert isinstance(mob, Text)
                name = _op_name_from_line(mob.original_text)
                if name == op_name:
                    rect = SurroundingRectangle(
                        mob, color=color, buff=0.06, corner_radius=0.05
                    )
                    self.play(Create(rect), run_time=0.35)
                    return rect
            return None

        def rebuild_body(
            lines: list[str],
            colors: dict[str, str] | None = None,
            animate: bool = True,
        ) -> None:
            """Replace the body_group mobject with a new one for `lines`."""
            nonlocal body_group

            new_body = _build_ir_group(lines, colors)
            new_body.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            for mob in new_body:
                mob.shift(RIGHT * 0.5)

            # Anchor top-left of new body to the same position as the old body.
            # Avoids repositioning the already-displayed header_group.
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

        # Track the current displayed lines (mirrors what's on screen)
        current_lines = body_lines[:]

        for event in events:
            if isinstance(event, ExamineEvent):
                # Sync display with event's asm_lines (catches post-replacement state)
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

                # Show matched op (being eliminated) in red, replacement in green
                old = event.old_name
                new = event.new_name
                match_colors = {old: IR_DEAD, new: IR_MATCH}
                rebuild_body(current_lines, match_colors)

                set_status(
                    f"Pattern matched!  replace_uses(%{old}, %{new})",
                    STATUS_OK,
                )
                self.wait(1.0)

                # Update to post-replacement ASM (dead ops disappear)
                rebuild_body(event.asm_lines_after)
                current_lines = event.asm_lines_after[:]
                self.wait(0.5)

        # ── 6. Final ──────────────────────────────────────────────────────────
        clear_rect()
        set_status("Done.  2 ops eliminated.", STATUS_OK)
        self.wait(2.5)
