"""
manim_stats/scenes/demo_hypothesis.py
=======================================
HypothesisDemo — A complete, cinematic hypothesis testing demonstration
scene for Manim statistics animations.

Story arc (7 acts)
------------------
Act 0  The courtroom analogy
       Scales-of-justice icon built from Manim primitives.
       "Innocent until proven guilty" ↔ H₀ until evidence rejects it.
       The four concepts: H₀, H₁, α, p-value introduced visually.

Act 1  One-sample Z-test
       Problem: "Is the mean reaction time > 300 ms?"
       Data dots scatter on a number line, mean extracted.
       Step-by-step calculation panel.
       Full HypothesisTest3D with BuildTest → DropStatistic →
       RevealPValue → RevealDecision.
       One-tailed vs two-tailed comparison.

Act 2  One-sample t-test
       Problem: "Has the drug changed systolic blood pressure?"
       Small sample (n=12), unknown σ → t-distribution.
       t-curve with heavier tails highlighted vs Normal.
       Full test sequence with BuildInfoPanel.

Act 3  Two-sample t-test
       Problem: "Do two teaching methods produce different scores?"
       Two dot strips side by side, means extracted, pooled SE computed.
       Welch t-test (unequal variances).
       Both one-tailed and two-tailed shown simultaneously.

Act 4  Chi-square goodness-of-fit
       Problem: "Is this die fair?"
       Bar chart of observed vs expected counts.
       Chi-square statistic built term by term: Σ(O-E)²/E
       Right-skewed chi² curve.
       Decision: suspicious die!

Act 5  Power analysis
       "What if our sample was larger?"
       TypeITypeII visualization with NarrowCurves animation.
       Power curve: plot of power vs n.
       Effect size arrow: larger δ → more power.

Act 6  Decision framework + closing
       Visual decision flowchart:
         State H₀/H₁ → Choose α → Collect data →
         Compute statistic → Compare to critical value →
         Decision → Interpret
       Common mistakes panel.
       Closing: the philosophy of hypothesis testing.

Scene uses
----------
  manim_stats.inference.hypothesis   — HypothesisTest3D, BuildTest,
    DropStatistic, RevealPValue, RevealDecision, BuildInfoPanel,
    SweepStatistic, ChangeAlpha, make_full_sequence, compare_tests
  manim_stats.inference.error_types  — TypeITypeII, NarrowCurves,
    RevealAll, FlashDecision

Dependencies
------------
  manim (CE or GL), numpy, scipy (optional)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from manim import (
    Scene,
    VGroup,
    Rectangle, RoundedRectangle, Square,
    Circle, Annulus, Dot, Polygon, Line, DashedLine, Arc,
    Arrow, DoubleArrow,
    Text, MathTex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut, GrowFromCenter,
    Create, Write, Uncreate,
    Indicate, Flash,
    Rotate, Transform, ReplacementTransform,
    ValueTracker,
    interpolate_color, color_to_rgb,
    BLACK, WHITE,
    GREY,  GREY_B,  GREY_C,
    RED,   RED_B,   RED_C,
    GREEN, GREEN_B, GREEN_C,
    BLUE,  BLUE_B,  BLUE_C,  BLUE_D,
    YELLOW, YELLOW_A,
    ORANGE, TEAL, TEAL_B,
    GOLD,  GOLD_B,  GOLD_C,
    PURPLE_A, PURPLE_B, MAROON,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    VMobject,
)

# ─────────────────────────────────────────────────────────────────────────────
# Import inference components
# ─────────────────────────────────────────────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from inference.hypothesis import (
        HypothesisTest3D, BuildTest, DropStatistic,
        RevealPValue, RevealDecision, BuildInfoPanel,
        SweepStatistic, ChangeAlpha, make_full_sequence, compare_tests,
    )
    from inference.error_types import (
        TypeITypeII, NarrowCurves, RevealAll, FlashDecision,
        BuildDecisionTable,
    )
    _INF_AVAILABLE = True
except ImportError:
    _INF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

P = {
    "bg":           "#080C12",
    "bg_panel":     "#0C1018",
    "panel_border": "#2A3848",
    "title":        "#D8F0FF",
    "subtitle":     "#607888",
    "section":      "#405060",

    # Hypothesis components
    "h0":           "#3A7AC8",
    "h1":           "#C84030",
    "alpha":        "#D4AF37",
    "pvalue":       "#8040C0",
    "reject":       "#D03030",
    "keep":         "#30A050",

    # Data dots
    "dot":          "#F0C040",
    "dot_mean":     "#FF6040",
    "dot_strip":    "#304050",

    # Calculation panel
    "calc_bg":      "#0E1420",
    "calc_border":  "#2A3848",
    "calc_formula": "#70B8F0",
    "calc_sub":     "#F0A050",
    "calc_result":  "#60E080",
    "calc_key":     "#6080A0",

    # Checklist
    "check_yes":    "#40C060",
    "check_no":     "#C04040",
    "check_box":    "#304050",

    # p-value gauge
    "gauge_bg":     "#101820",
    "gauge_fill":   "#8040C0",
    "gauge_border": "#2A3848",

    # Flowchart
    "flow_node":    "#1A2838",
    "flow_border":  "#3A5878",
    "flow_arrow":   "#506878",
    "flow_decision":"#1A2820",
    "flow_dec_brd": "#3A6850",

    # Court analogy
    "scales_metal": "#B8A060",
    "scales_dark":  "#6A5830",
    "scales_chain": "#D4B870",

    "chi2_bar_obs": "#E07030",
    "chi2_bar_exp": "#3070C0",
    "chi2_diff":    "#A04080",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section_label(text: str) -> Text:
    return Text(text, font_size=16, color=P["section"])


def _fade(*mobs, scene: Scene, rt: float = 0.50):
    scene.play(*[FadeOut(m) for m in mobs if m is not None], run_time=rt)


def _badge(text: str, color: str, fs: int = 17) -> VGroup:
    g   = VGroup()
    lbl = Text(text, font_size=fs, color=color)
    bg  = RoundedRectangle(
        width=lbl.width+0.24, height=lbl.height+0.16,
        corner_radius=0.07,
        fill_color=P["bg_panel"], fill_opacity=0.92,
        stroke_color=color, stroke_width=0.8,
    )
    bg.move_to(ORIGIN)
    lbl.move_to([0, 0, 0.001])
    g.add(bg, lbl)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# _ScalesIcon  —  scales of justice from primitives
# ─────────────────────────────────────────────────────────────────────────────

class _ScalesIcon(VGroup):
    """
    Scales of justice built from Manim primitives.

    Components:
      • Vertical central post (Rectangle)
      • Horizontal beam (Rectangle, slightly tapered)
      • Two chains (DashedLine arcs)
      • Two pans (Arc + horizontal line)
      • Base triangle (Polygon)
      • Central pivot circle

    Parameters
    ----------
    scale : overall size factor
    tilt  : beam tilt angle (radians) — positive tilts H₀ side down
    """

    def __init__(
        self,
        scale:    float = 1.0,
        tilt:     float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0     = 0.001
        col    = P["scales_metal"]
        dark   = P["scales_dark"]
        chain  = P["scales_chain"]

        # ── Base ──────────────────────────────────────────────────────
        base = Polygon(
            [-0.50*scale, -1.60*scale, z0],
            [ 0.50*scale, -1.60*scale, z0],
            [ 0.06*scale, -0.90*scale, z0],
            [-0.06*scale, -0.90*scale, z0],
            fill_color=dark, fill_opacity=1.0, stroke_width=0,
        )
        self.add(base)

        # ── Post ──────────────────────────────────────────────────────
        post = Rectangle(
            width=0.10*scale, height=1.90*scale,
            fill_color=col, fill_opacity=1.0, stroke_width=0,
        )
        post.move_to([0, -0.05*scale, z0+0.001])
        self.add(post)

        # ── Pivot circle ──────────────────────────────────────────────
        pivot = Circle(
            radius=0.10*scale,
            fill_color=col, fill_opacity=1.0, stroke_width=0,
        )
        pivot.move_to([0, 0.85*scale, z0+0.003])
        self.add(pivot)

        # ── Beam (tilted) ─────────────────────────────────────────────
        beam = Rectangle(
            width=2.60*scale, height=0.08*scale,
            fill_color=col, fill_opacity=1.0, stroke_width=0,
        )
        beam.move_to([0, 0.85*scale, z0+0.002])
        beam.rotate(tilt, about_point=[0, 0.85*scale, 0])
        self.add(beam)

        # ── Chains + pans ─────────────────────────────────────────────
        for side, label_str in [(-1, "H₀"), (1, "H₁")]:
            # Chain attachment point (end of beam)
            bx    = side * 1.30 * scale
            by_top = 0.85 * scale + side * np.sin(tilt) * 1.30 * scale
            by_pan = by_top - 0.70 * scale

            # Chain (dashed line)
            self.add(DashedLine(
                [bx, by_top, z0+0.003],
                [bx, by_pan, z0+0.003],
                stroke_color=chain, stroke_width=1.6,
                dash_length=0.06, dashed_ratio=0.6,
            ))

            # Pan
            pan_r = 0.42 * scale
            pan_arc = Arc(
                radius=pan_r, start_angle=PI, angle=PI,
                stroke_color=col, stroke_width=2.2,
            )
            pan_arc.move_arc_center_to([bx, by_pan, z0+0.003])
            pan_line = Line(
                [bx - pan_r, by_pan, z0+0.003],
                [bx + pan_r, by_pan, z0+0.003],
                stroke_color=col, stroke_width=2.2,
            )
            self.add(pan_arc, pan_line)

            # Label in pan
            try:
                pan_lbl = MathTex(label_str, font_size=int(20*scale),
                                  color=P["h0"] if "H₀" in label_str
                                  else P["h1"])
            except Exception:
                pan_lbl = Text(label_str, font_size=int(18*scale),
                               color=P["h0"] if "H₀" in label_str
                               else P["h1"])
            pan_lbl.move_to([bx, by_pan - 0.22*scale, z0+0.004])
            self.add(pan_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# _HypothesisSetupPanel  —  problem statement + hypotheses + formula
# ─────────────────────────────────────────────────────────────────────────────

class _HypothesisSetupPanel(VGroup):
    """
    Full test setup panel showing:
      • Problem statement (research question)
      • H₀ and H₁ hypotheses
      • Test statistic formula
      • Assumptions checklist

    Parameters
    ----------
    problem   : str  — research question text
    h0_str    : str  — null hypothesis (LaTeX)
    h1_str    : str  — alternative hypothesis (LaTeX)
    formula   : str  — test statistic formula (LaTeX)
    test_name : str
    assumptions : list of (text, bool) — assumption + whether it holds
    data_summary: list of (key, val) — n, x̄, s, etc.
    """

    def __init__(
        self,
        problem:     str,
        h0_str:      str,
        h1_str:      str,
        formula:     str,
        test_name:   str  = "Hypothesis Test",
        assumptions: list = None,
        data_summary: list = None,
        panel_w:     float = 9.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assumptions  = assumptions  or []
        data_summary = data_summary or []
        z0 = 0.001
        row_h = 0.46

        total_h = 1.20 + len(assumptions)*row_h*0.90 + len(data_summary)*row_h*0.90 + 0.80
        total_h = max(total_h, 3.80)

        # ── Background ────────────────────────────────────────────────
        bg = RoundedRectangle(
            width=panel_w, height=total_h,
            corner_radius=0.14,
            fill_color=P["bg_panel"], fill_opacity=0.94,
            stroke_color=P["panel_border"], stroke_width=0.9,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        # Left border strip (blue = H₀ side)
        strip = RoundedRectangle(
            width=0.09, height=total_h-0.06,
            corner_radius=0.045,
            fill_color=P["h0"], fill_opacity=0.80, stroke_width=0,
        )
        strip.move_to([-panel_w/2+0.065, 0, z0+0.001])
        self.add(strip)

        y = total_h/2 - 0.30

        # ── Test name ─────────────────────────────────────────────────
        self.add(Text(test_name, font_size=20, color=P["title"])
                 .move_to([0, y, z0+0.002]))
        y -= 0.48
        self.add(Line([-panel_w/2+0.14, y+0.12, z0+0.002],
                      [ panel_w/2-0.10, y+0.12, z0+0.002],
                      stroke_color=P["panel_border"], stroke_width=0.6))

        # ── Problem statement ─────────────────────────────────────────
        prob_lbl = Text("Research Question:", font_size=13,
                        color=P["subtitle"])
        prob_lbl.move_to([-panel_w/2+0.22+prob_lbl.width/2, y, z0+0.003])
        prob_val = Text(problem, font_size=13, color=P["title"])
        prob_val.move_to([panel_w/2-0.12-prob_val.width/2, y, z0+0.003])
        self.add(prob_lbl, prob_val)
        y -= 0.44

        # ── Hypotheses ────────────────────────────────────────────────
        for sym, hs, col in [
            ("H₀:", h0_str, P["h0"]),
            ("H₁:", h1_str, P["h1"]),
        ]:
            try:
                sym_m = MathTex(sym, font_size=20, color=col)
                hs_m  = MathTex(hs,  font_size=18, color=col)
            except Exception:
                sym_m = Text(sym, font_size=17, color=col)
                hs_m  = Text(hs,  font_size=15, color=col)
            sym_m.move_to([-panel_w/2+0.38, y, z0+0.003])
            hs_m.move_to([-panel_w/2+0.55+hs_m.width/2, y, z0+0.003])
            self.add(sym_m, hs_m)
            y -= row_h * 0.92

        y -= 0.10

        # ── Test statistic formula ────────────────────────────────────
        try:
            form_lbl = MathTex(formula, font_size=18,
                               color=P["calc_formula"])
        except Exception:
            form_lbl = Text(formula, font_size=15,
                            color=P["calc_formula"])
        form_lbl.move_to([0, y, z0+0.003])
        self.add(form_lbl)
        y -= 0.52

        # ── Data summary ──────────────────────────────────────────────
        if data_summary:
            self.add(Line([-panel_w/2+0.14, y+0.14, z0+0.002],
                          [ panel_w/2-0.10, y+0.14, z0+0.002],
                          stroke_color=P["panel_border"],
                          stroke_width=0.5))
            self.add(Text("Data Summary", font_size=13,
                          color=P["subtitle"])
                     .move_to([-panel_w/2+0.80, y-0.04, z0+0.003]))
            y -= 0.38
            for k, v in data_summary:
                km = Text(k, font_size=12, color=P["calc_key"])
                vm = Text(v, font_size=12, color=P["calc_sub"],
                          font="monospace")
                km.move_to([-panel_w/2+0.22+km.width/2, y, z0+0.003])
                vm.move_to([ panel_w/2-0.12-vm.width/2, y, z0+0.003])
                self.add(km, vm)
                y -= row_h * 0.82

        # ── Assumptions checklist ─────────────────────────────────────
        if assumptions:
            self.add(Line([-panel_w/2+0.14, y+0.12, z0+0.002],
                          [ panel_w/2-0.10, y+0.12, z0+0.002],
                          stroke_color=P["panel_border"],
                          stroke_width=0.5))
            self.add(Text("Assumptions", font_size=13,
                          color=P["subtitle"])
                     .move_to([-panel_w/2+0.70, y-0.04, z0+0.003]))
            y -= 0.36
            for assumption_text, holds in assumptions:
                col  = P["check_yes"] if holds else P["check_no"]
                mark = "✓" if holds else "✗"
                self.add(Text(mark, font_size=14, color=col)
                         .move_to([-panel_w/2+0.28, y, z0+0.003]))
                self.add(Text(assumption_text, font_size=12,
                              color=col)
                         .move_to([-panel_w/2+0.50+
                                    Text(assumption_text,
                                         font_size=12).width/2,
                                    y, z0+0.003]))
                y -= row_h * 0.80


# ─────────────────────────────────────────────────────────────────────────────
# _CalcPanel  —  step-by-step computation panel
# ─────────────────────────────────────────────────────────────────────────────

class _CalcPanel(VGroup):
    """
    Step-by-step calculation panel.

    Shows three lines:
      Line 1: symbolic formula         (blue)
      Line 2: numerical substitution   (orange)
      Line 3: result                   (green)

    Each line can be revealed separately for animation.

    Parameters
    ----------
    symbolic  : str  — LaTeX formula
    substituted: str — LaTeX with numbers
    result    : str  — LaTeX final value
    test_name : str
    """

    def __init__(
        self,
        symbolic:    str,
        substituted: str,
        result:      str,
        test_name:   str = "Calculation",
        panel_w:     float = 7.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        bg = RoundedRectangle(
            width=panel_w, height=2.60,
            corner_radius=0.12,
            fill_color=P["calc_bg"], fill_opacity=0.94,
            stroke_color=P["calc_border"], stroke_width=0.8,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        strip = RoundedRectangle(
            width=0.08, height=2.52,
            corner_radius=0.04,
            fill_color=P["calc_formula"],
            fill_opacity=0.70, stroke_width=0,
        )
        strip.move_to([-panel_w/2+0.06, 0, z0+0.001])
        self.add(strip)

        self.add(Text(test_name, font_size=16, color=P["title"])
                 .move_to([0, 1.02, z0+0.002]))
        self.add(Line([-panel_w/2+0.12, 0.72, z0+0.002],
                      [ panel_w/2-0.10, 0.72, z0+0.002],
                      stroke_color=P["calc_border"], stroke_width=0.6))

        self.line_symbolic = self._make_line(symbolic, P["calc_formula"],
                                             panel_w, 0.30, z0)
        self.line_subst    = self._make_line(substituted, P["calc_sub"],
                                             panel_w, -0.26, z0)
        self.line_result   = self._make_line(result, P["calc_result"],
                                             panel_w, -0.82, z0)
        self.add(self.line_symbolic, self.line_subst, self.line_result)

    @staticmethod
    def _make_line(tex_str: str, color: str,
                   panel_w: float, y: float, z0: float) -> VGroup:
        g = VGroup()
        try:
            mob = MathTex(tex_str, font_size=20, color=color)
        except Exception:
            mob = Text(tex_str, font_size=17, color=color)
        mob.move_to([0, y, z0+0.003])
        g.add(mob)
        return g


# ─────────────────────────────────────────────────────────────────────────────
# _PValueGauge  —  horizontal p-value meter
# ─────────────────────────────────────────────────────────────────────────────

class _PValueGauge(VGroup):
    """
    Horizontal gauge showing p-value on a [0, 1] scale with α marker.

    Layers:
      1. Background rail
      2. Filled region from 0 to p_value (purple)
      3. α line + label (gold vertical marker)
      4. p-value label badge
      5. "Reject zone" / "Fail zone" annotations
    """

    def __init__(
        self,
        p_value: float,
        alpha:   float = 0.05,
        width:   float = 5.0,
        height:  float = 0.38,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        # Background rail
        bg = Rectangle(
            width=width, height=height,
            fill_color=P["gauge_bg"], fill_opacity=1.0,
            stroke_color=P["gauge_border"], stroke_width=0.8,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        # Filled region 0→p_value
        fill_w = p_value * width
        if fill_w > 0.01:
            fill = Rectangle(
                width=fill_w, height=height-0.04,
                fill_color=P["gauge_fill"], fill_opacity=0.85,
                stroke_width=0,
            )
            fill.move_to([-width/2 + fill_w/2, 0, z0+0.001])
            self.add(fill)

        # α line
        alpha_x = alpha * width - width/2
        self.add(Line(
            [alpha_x, -height/2-0.06, z0+0.003],
            [alpha_x,  height/2+0.06, z0+0.003],
            stroke_color=P["alpha"], stroke_width=2.0,
        ))
        try:
            albl = MathTex(rf"\alpha = {alpha:.2f}", font_size=14,
                           color=P["alpha"])
        except Exception:
            albl = Text(f"α={alpha:.2f}", font_size=12,
                        color=P["alpha"])
        albl.move_to([alpha_x, height/2+0.28, z0+0.003])
        self.add(albl)

        # p-value badge
        p_str = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"
        p_x   = min(p_value * width - width/2, width/2 - 0.60)
        p_lbl = Text(p_str, font_size=14, color=P["pvalue"])
        self.add(p_lbl.move_to([p_x, -height/2-0.28, z0+0.003]))

        # Zone labels
        reject_col = P["reject"] if p_value < alpha else P["keep"]
        verdict    = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT"

        self.add(Text("Reject zone", font_size=11,
                      color=P["reject"])
                 .move_to([-width/2 + alpha*width/2,
                            -height/2-0.55, z0+0.003]))
        self.add(Text("Fail-to-reject zone", font_size=11,
                      color=P["keep"])
                 .move_to([alpha_x + (1-alpha)*width/2,
                            -height/2-0.55, z0+0.003]))

        # Verdict
        verdict_lbl = Text(verdict, font_size=18, color=reject_col)
        self.add(verdict_lbl.move_to([0, -height/2-0.90, z0+0.003]))

        # 0 and 1 labels
        self.add(Text("0", font_size=11, color=P["subtitle"])
                 .move_to([-width/2, -height/2-0.14, z0+0.002]))
        self.add(Text("1", font_size=11, color=P["subtitle"])
                 .move_to([ width/2, -height/2-0.14, z0+0.002]))
        self.add(Text("p-value", font_size=13, color=P["pvalue"])
                 .move_to([-width/2-0.52, 0, z0+0.002]))


# ─────────────────────────────────────────────────────────────────────────────
# _DataDotStrip  —  raw data visualisation on a number line
# ─────────────────────────────────────────────────────────────────────────────

class _DataDotStrip(VGroup):
    """
    Scatter dots on a number line for a data sample.

    Each dot falls from slightly above the line.
    Mean marker highlighted.

    Parameters
    ----------
    data      : array of observed values
    x_lo, x_hi: axis range
    label     : strip title
    color     : dot colour
    width     : strip width
    strip_y   : y position of the number line
    """

    def __init__(
        self,
        data:    np.ndarray,
        x_lo:    float,
        x_hi:    float,
        label:   str   = "",
        color:   str   = None,
        width:   float = 6.0,
        strip_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        col   = color or P["dot"]
        z0    = 0.001
        data  = np.asarray(data)

        def to_px(v):
            return (v - x_lo) / (x_hi - x_lo) * width - width/2

        # Axis
        self.add(Line([-width/2, strip_y, z0],
                      [ width/2, strip_y, z0],
                      stroke_color=P["dot_strip"], stroke_width=1.4))

        # Ticks
        n_ticks = 6
        for i in range(n_ticks+1):
            v  = x_lo + i*(x_hi-x_lo)/n_ticks
            xp = to_px(v)
            self.add(Line([xp, strip_y-0.07, z0+0.001],
                          [xp, strip_y+0.07, z0+0.001],
                          stroke_color=P["dot_strip"],
                          stroke_width=0.9))
            self.add(Text(f"{v:.0f}", font_size=10, color=P["subtitle"])
                     .move_to([xp, strip_y-0.24, z0+0.001]))

        # Jittered dots
        rng = np.random.default_rng(42)
        for val in data:
            val   = float(np.clip(val, x_lo, x_hi))
            xp    = to_px(val)
            y_jit = strip_y + rng.uniform(0.06, 0.22)
            d     = Dot(radius=0.07,
                        point=[xp, y_jit, z0+0.003],
                        color=col, fill_opacity=0.82)
            self.add(d)

        # Mean marker
        mean_val = float(np.mean(data))
        mean_px  = to_px(mean_val)
        mean_dot = Dot(radius=0.12,
                       point=[mean_px, strip_y+0.14, z0+0.005],
                       color=P["dot_mean"], fill_opacity=1.0)
        self.add(mean_dot)
        try:
            m_lbl = MathTex(rf"\bar{{x}} = {mean_val:.1f}",
                            font_size=15, color=P["dot_mean"])
        except Exception:
            m_lbl = Text(f"x̄ = {mean_val:.1f}", font_size=13,
                         color=P["dot_mean"])
        m_lbl.move_to([mean_px, strip_y+0.40, z0+0.005])
        self.add(m_lbl)

        # Label
        if label:
            self.add(Text(label, font_size=14, color=P["subtitle"])
                     .move_to([-width/2-0.55, strip_y+0.10, z0+0.002]))

        self._mean_px = mean_px
        self._strip_y = strip_y
        self._to_px   = to_px


# ─────────────────────────────────────────────────────────────────────────────
# _ChiSquareBarChart  —  observed vs expected comparison
# ─────────────────────────────────────────────────────────────────────────────

class _ChiSquareBarChart(VGroup):
    """
    Grouped bar chart comparing observed vs expected counts.
    Shows (O-E)²/E contribution above each pair.

    Parameters
    ----------
    categories : list of category names
    observed   : array of observed counts
    expected   : array of expected counts
    width, height, baseline_y : geometry
    """

    def __init__(
        self,
        categories: list,
        observed:   np.ndarray,
        expected:   np.ndarray,
        width:      float = 6.0,
        height:     float = 2.5,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0      = 0.001
        n       = len(categories)
        grp_w   = width / n
        bar_w   = grp_w * 0.38
        max_val = max(float(observed.max()), float(expected.max())) * 1.10

        def to_h(v):
            return (v / max_val) * height

        # Axis
        self.add(Line([-width/2, baseline_y, z0],
                      [ width/2, baseline_y, z0],
                      stroke_color=P["dot_strip"], stroke_width=1.3))

        chi2_total = 0.0
        for i, (cat, obs, exp) in enumerate(
            zip(categories, observed, expected)
        ):
            cx    = i * grp_w - width/2 + grp_w/2
            h_obs = to_h(float(obs))
            h_exp = to_h(float(exp))
            contrib = (obs-exp)**2 / max(exp, 1e-6)
            chi2_total += contrib

            # Observed bar (orange)
            self.add(Rectangle(
                width=bar_w, height=h_obs,
                fill_color=P["chi2_bar_obs"],
                fill_opacity=0.88, stroke_width=0,
            ).move_to([cx - bar_w*0.55, baseline_y+h_obs/2, z0+0.002]))

            # Expected bar (blue, outline only)
            self.add(Rectangle(
                width=bar_w, height=h_exp,
                fill_color=P["chi2_bar_exp"],
                fill_opacity=0.35,
                stroke_color=P["chi2_bar_exp"],
                stroke_width=1.5,
            ).move_to([cx + bar_w*0.55, baseline_y+h_exp/2, z0+0.002]))

            # Count labels
            self.add(Text(str(int(obs)), font_size=10,
                          color=P["chi2_bar_obs"])
                     .move_to([cx-bar_w*0.55,
                                baseline_y+h_obs+0.12, z0+0.003]))
            self.add(Text(str(int(exp)), font_size=10,
                          color=P["chi2_bar_exp"])
                     .move_to([cx+bar_w*0.55,
                                baseline_y+h_exp+0.12, z0+0.003]))

            # (O-E)²/E contribution
            diff_h = max(h_obs, h_exp) + 0.35
            self.add(Text(f"{contrib:.2f}", font_size=10,
                          color=P["chi2_diff"])
                     .move_to([cx, baseline_y+diff_h, z0+0.003]))

            # Category label
            self.add(Text(str(cat), font_size=11, color=P["subtitle"])
                     .move_to([cx, baseline_y-0.24, z0+0.001]))

        # χ² total label
        self.add(Text(f"χ² = {chi2_total:.3f}", font_size=16,
                      color=P["chi2_diff"])
                 .move_to([0, baseline_y+height+0.45, z0+0.003]))

        # Legend
        self.add(Dot(radius=0.06,
                     point=[-width/2+0.20, baseline_y+height+0.22, z0+0.003],
                     color=P["chi2_bar_obs"]))
        self.add(Text("Observed", font_size=12, color=P["chi2_bar_obs"])
                 .move_to([-width/2+0.70, baseline_y+height+0.22, z0+0.003]))
        self.add(Dot(radius=0.06,
                     point=[0.30, baseline_y+height+0.22, z0+0.003],
                     color=P["chi2_bar_exp"]))
        self.add(Text("Expected", font_size=12, color=P["chi2_bar_exp"])
                 .move_to([0.80, baseline_y+height+0.22, z0+0.003]))

        self._chi2 = chi2_total


# ─────────────────────────────────────────────────────────────────────────────
# _PowerCurve  —  power vs n plot
# ─────────────────────────────────────────────────────────────────────────────

class _PowerCurve(VGroup):
    """
    Plot of statistical power vs sample size n.

    Shows power curves for multiple effect sizes (δ).

    Parameters
    ----------
    sigma  : float — population std dev
    alpha  : float — significance level
    n_max  : int   — maximum n on x-axis
    deltas : list  — list of effect sizes to plot
    width, height, baseline_y : geometry
    """

    def __init__(
        self,
        sigma:      float = 1.0,
        alpha:      float = 0.05,
        n_max:      int   = 100,
        deltas:     list  = None,
        width:      float = 6.0,
        height:     float = 3.0,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        deltas = deltas or [0.5, 1.0, 1.5, 2.0]
        z0     = 0.001

        try:
            from scipy.stats import norm as _sn
            def _power(n, delta, sigma, alpha):
                se    = sigma / np.sqrt(n)
                z_a   = _sn.ppf(1 - alpha)
                z_obs = delta / se
                return float(_sn.cdf(z_obs - z_a))
        except ImportError:
            def _power(n, delta, sigma, alpha):
                se    = sigma / np.sqrt(n)
                z_a   = 1.645
                z_obs = delta / se
                return max(0.0, min(1.0,
                    0.5*(1 + (z_obs-z_a)/max(abs(z_obs-z_a)+0.1, 1))))

        ns = np.arange(2, n_max+1)

        def to_px_x(n):
            return (n - 2) / (n_max - 2) * width - width/2

        def to_px_y(p):
            return p * height + baseline_y

        # Background
        bg = RoundedRectangle(
            width=width+0.24, height=height+0.80,
            corner_radius=0.12,
            fill_color=P["bg_panel"], fill_opacity=0.92,
            stroke_color=P["panel_border"], stroke_width=0.7,
        )
        bg.move_to([0, baseline_y+height/2-0.10, z0])
        self.add(bg)

        # Axes
        self.add(Line([-width/2, baseline_y, z0+0.001],
                      [ width/2, baseline_y, z0+0.001],
                      stroke_color=P["dot_strip"], stroke_width=1.3))
        self.add(Line([-width/2, baseline_y, z0+0.001],
                      [-width/2, baseline_y+height, z0+0.001],
                      stroke_color=P["dot_strip"], stroke_width=1.3))

        # 80% power reference line
        y_80 = to_px_y(0.80)
        self.add(DashedLine(
            [-width/2, y_80, z0+0.002],
            [ width/2, y_80, z0+0.002],
            stroke_color=P["alpha"], stroke_width=1.2,
            dash_length=0.10, dashed_ratio=0.55,
        ))
        self.add(Text("80%", font_size=11, color=P["alpha"])
                 .move_to([-width/2-0.35, y_80, z0+0.002]))

        # Y ticks
        for pv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            yp = to_px_y(pv)
            self.add(Line([-width/2-0.07, yp, z0+0.001],
                          [-width/2+0.07, yp, z0+0.001],
                          stroke_color=P["dot_strip"],
                          stroke_width=0.8))
            self.add(Text(f"{pv:.0%}", font_size=10,
                          color=P["subtitle"])
                     .move_to([-width/2-0.35, yp, z0+0.001]))

        # X ticks
        for nv in [10, 20, 30, 50, 75, 100]:
            if nv <= n_max:
                xp = to_px_x(nv)
                self.add(Line([xp, baseline_y-0.06, z0+0.001],
                              [xp, baseline_y+0.06, z0+0.001],
                              stroke_color=P["dot_strip"],
                              stroke_width=0.8))
                self.add(Text(str(nv), font_size=10,
                              color=P["subtitle"])
                         .move_to([xp, baseline_y-0.25, z0+0.001]))

        # Power curves
        delta_colors = [P["h0"], P["calc_result"], P["alpha"], P["reject"]]
        for di, delta in enumerate(deltas):
            col   = delta_colors[di % len(delta_colors)]
            pwrs  = [_power(n, delta, sigma, alpha) for n in ns]
            step  = max(1, len(ns)//80)
            pts3d = [np.array([to_px_x(float(ns[i])),
                                to_px_y(float(pwrs[i])), z0+0.005])
                     for i in range(0, len(ns), step)]
            if len(pts3d) >= 2:
                sp = VMobject(stroke_color=col,
                              stroke_width=1.8, stroke_opacity=0.90)
                sp.set_points_smoothly(pts3d)
                self.add(sp)
            # End label
            self.add(Text(f"δ={delta:.1f}", font_size=11, color=col)
                     .move_to([width/2+0.32, to_px_y(pwrs[-1]),
                                z0+0.005]))

        # Title + axis labels
        self.add(Text("Power vs Sample Size", font_size=16,
                      color=P["title"])
                 .move_to([0, baseline_y+height+0.30, z0+0.003]))
        self.add(Text("n (sample size)", font_size=12, color=P["subtitle"])
                 .move_to([0, baseline_y-0.45, z0+0.002]))
        self.add(Text("Power  1−β", font_size=12, color=P["subtitle"])
                 .rotate(PI/2)
                 .move_to([-width/2-0.65, baseline_y+height/2,
                            z0+0.002]))


# ─────────────────────────────────────────────────────────────────────────────
# _DecisionFlowchart  —  hypothesis testing workflow
# ─────────────────────────────────────────────────────────────────────────────

class _DecisionFlowchart(VGroup):
    """
    Visual flowchart of the hypothesis testing decision process.

    Nodes (top to bottom):
      1. State H₀ and H₁
      2. Choose significance level α
      3. Collect data, compute test statistic
      4. Compute p-value
      5. Decision diamond: p < α?
         → Yes: Reject H₀
         → No:  Fail to Reject H₀
      6. Interpret in context

    Parameters
    ----------
    width : float — total flowchart width
    """

    def __init__(self, width: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        z0  = 0.001
        nw  = width * 0.72
        nh  = 0.44
        dy  = 0.78

        def _rect_node(text, y, col=P["flow_border"], fs=13):
            g  = VGroup()
            bg = RoundedRectangle(
                width=nw, height=nh, corner_radius=0.08,
                fill_color=P["flow_node"], fill_opacity=0.92,
                stroke_color=col, stroke_width=0.9,
            )
            bg.move_to([0, y, z0])
            lbl = Text(text, font_size=fs, color=P["title"])
            lbl.move_to([0, y, z0+0.002])
            g.add(bg, lbl)
            return g

        def _diamond(text, y):
            d  = 0.50
            g  = VGroup()
            bg = Polygon(
                [0,  d, z0], [ nw/2*0.9, 0, z0],
                [0, -d, z0], [-nw/2*0.9, 0, z0],
                fill_color=P["flow_decision"],
                fill_opacity=0.92,
                stroke_color=P["flow_dec_brd"],
                stroke_width=0.9,
            )
            bg.move_to([0, y, 0])
            lbl = Text(text, font_size=12, color=P["alpha"])
            lbl.move_to([0, y, z0+0.002])
            g.add(bg, lbl)
            return g

        def _arrow(y_start, y_end, x=0):
            return Arrow(
                start=[x, y_start, z0+0.003],
                end  =[x, y_end,   z0+0.003],
                stroke_color=P["flow_arrow"],
                stroke_width=1.4, tip_length=0.14, buff=0,
            )

        y = 2.60
        n1 = _rect_node("1.  State H₀ and H₁", y)
        y -= dy
        n2 = _rect_node("2.  Choose α (significance level)", y)
        y -= dy
        n3 = _rect_node("3.  Collect data, compute statistic", y)
        y -= dy
        n4 = _rect_node("4.  Compute p-value", y)
        y -= dy * 0.90
        d5 = _diamond("p < α ?", y)
        y -= dy * 0.90

        # Reject / Fail branches
        rej_node = _rect_node("Reject H₀", y,
                              col=P["reject"], fs=14)
        rej_node.move_to([-nw*0.72, y, 0])
        keep_node = _rect_node("Fail to Reject H₀", y,
                               col=P["keep"], fs=14)
        keep_node.move_to([ nw*0.72, y, 0])

        y -= dy * 0.85
        n6 = _rect_node("5.  Interpret in context", y)

        # Arrows
        prev_y = 2.60
        for node in [n1, n2, n3, n4]:
            self.add(node)
            curr_y = node.get_center()[1]
            if prev_y != curr_y:
                self.add(_arrow(prev_y - nh/2 - 0.02,
                                curr_y + nh/2 + 0.02))
            prev_y = curr_y

        # Arrow to diamond
        self.add(_arrow(prev_y - nh/2 - 0.02,
                        d5.get_center()[1] + 0.52))
        self.add(d5)

        # Side arrows to branches
        d5y  = d5.get_center()[1]
        bry  = rej_node.get_center()[1] + nh/2
        self.add(Arrow(
            start=[-nw/2*0.9, d5y, z0+0.003],
            end  =[-nw*0.72,  bry, z0+0.003],
            stroke_color=P["reject"], stroke_width=1.3,
            tip_length=0.13, buff=0,
        ))
        self.add(Text("Yes", font_size=11, color=P["reject"])
                 .move_to([-nw*0.55, d5y-0.18, z0+0.003]))

        self.add(Arrow(
            start=[ nw/2*0.9, d5y, z0+0.003],
            end  =[ nw*0.72,  bry, z0+0.003],
            stroke_color=P["keep"], stroke_width=1.3,
            tip_length=0.13, buff=0,
        ))
        self.add(Text("No", font_size=11, color=P["keep"])
                 .move_to([ nw*0.55, d5y-0.18, z0+0.003]))

        self.add(rej_node, keep_node)

        # Arrows from branches to interpretation
        for nx in [-nw*0.72, nw*0.72]:
            self.add(_arrow(
                bry - nh - 0.02,
                n6.get_center()[1] + nh/2 + 0.02,
                x=nx,
            ))
        self.add(n6)


# ─────────────────────────────────────────────────────────────────────────────
# The full demo scene
# ─────────────────────────────────────────────────────────────────────────────

class HypothesisDemo(Scene):
    """
    Complete hypothesis testing cinematic demonstration.

    Run with:
        manim -pql demo_hypothesis.py HypothesisDemo
    For high quality:
        manim -pqh demo_hypothesis.py HypothesisDemo
    """

    def construct(self):
        self._bg = Rectangle(width=16, height=9,
                              fill_color=P["bg"],
                              fill_opacity=1.0, stroke_width=0)
        self.add(self._bg)

        self._act0_courtroom()
        self._act1_z_test()
        self._act2_t_test()
        self._act3_two_sample_t()
        self._act4_chi_square()
        self._act5_power()
        self._act6_framework()

    # ─────────────────────────────────────────────────────────────────
    # Act 0 — Courtroom analogy
    # ─────────────────────────────────────────────────────────────────

    def _act0_courtroom(self):
        lbl = _section_label("Act 0 — The Logic of Hypothesis Testing")
        lbl.to_corner(LEFT+UP, buff=0.22)
        self.play(FadeIn(lbl, run_time=0.4))

        # Scales in balance (H₀ = innocent)
        scales_balanced = _ScalesIcon(scale=1.0, tilt=0.0)
        scales_balanced.scale(0.90).move_to([-1.0, 0.40, 0])
        self.play(FadeIn(scales_balanced, shift=UP*0.15,
                         rate_func=rate_functions.ease_out_back,
                         run_time=1.2))

        analogy = Text(
            "Innocent until proven guilty",
            font_size=26, color=P["h0"],
        )
        analogy.move_to([3.20, 1.20, 0])
        self.play(Write(analogy, run_time=1.0))

        arrow_to = Arrow(
            start=[2.30, 0.55, 0], end=[3.0, 0.55, 0],
            stroke_color=P["subtitle"], stroke_width=1.4,
            tip_length=0.14, buff=0,
        )
        ht_lbl = Text("H₀: no effect\n(default assumption)",
                      font_size=17, color=P["h0"])
        ht_lbl.move_to([4.30, 0.20, 0])
        self.play(Create(arrow_to, run_time=0.5),
                  FadeIn(ht_lbl, run_time=0.5))
        self.wait(0.5)

        # Evidence tips the scales
        scales_tipped = _ScalesIcon(scale=1.0, tilt=-0.18)
        scales_tipped.scale(0.90).move_to([-1.0, 0.40, 0])
        evidence_lbl = Text(
            "Evidence tips the scales…",
            font_size=22, color=P["subtitle"],
        )
        evidence_lbl.move_to([-1.0, -1.40, 0])
        self.play(
            ReplacementTransform(scales_balanced, scales_tipped,
                                 run_time=1.0),
            FadeIn(evidence_lbl, run_time=0.7),
        )
        self.wait(0.5)

        # The four concepts
        concepts = [
            ("H₀",      "Null hypothesis   (default)",           P["h0"]),
            ("H₁",      "Alternative hypothesis",                 P["h1"]),
            ("α",        "Significance level (false-positive rate)",P["alpha"]),
            ("p-value", "Probability of observed data | H₀ true",P["pvalue"]),
        ]
        concept_grp = VGroup()
        for i, (sym, desc, col) in enumerate(concepts):
            try:
                sm = MathTex(sym, font_size=22, color=col)
            except Exception:
                sm = Text(sym, font_size=20, color=col)
            dm = Text(desc, font_size=14, color=P["subtitle"])
            sm.move_to([-1.30, -2.10 + i*0.52, 0])
            dm.move_to([sm.get_right()[0]+0.18+dm.width/2,
                        -2.10 + i*0.52, 0])
            concept_grp.add(sm, dm)
        self.play(FadeIn(concept_grp, shift=UP*0.08, run_time=0.9))
        self.wait(1.0)

        _fade(scales_tipped, analogy, arrow_to, ht_lbl,
              evidence_lbl, concept_grp, lbl, scene=self)

    # ─────────────────────────────────────────────────────────────────
    # Act 1 — One-sample Z-test
    # ─────────────────────────────────────────────────────────────────

    def _act1_z_test(self):
        lbl = _section_label("Act 1 — One-Sample Z-Test")
        lbl.to_corner(LEFT+UP, buff=0.22)
        self.play(FadeIn(lbl, run_time=0.4))

        # Problem parameters
        mu0    = 300.0   # ms (null: mean reaction time)
        sigma  = 40.0    # ms (known)
        n      = 36
        x_bar  = 313.5
        alpha  = 0.05
        se     = sigma / np.sqrt(n)
        z_obs  = (x_bar - mu0) / se   # ≈ 2.025

        # Setup panel
        setup = _HypothesisSetupPanel(
            problem="Is mean reaction time > 300 ms?",
            h0_str=r"\mu = 300",
            h1_str=r"\mu > 300",
            formula=r"Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}",
            test_name="One-Sample Z-Test  (σ known)",
            assumptions=[
                ("Population σ known",     True),
                ("Simple random sample",   True),
                ("n ≥ 30 (or normal pop)", True),
            ],
            data_summary=[
                ("n =",    str(n)),
                ("x̄ =",   f"{x_bar:.1f} ms"),
                ("σ =",    f"{sigma:.1f} ms"),
                ("μ₀ =",   f"{mu0:.1f} ms"),
                ("SE =",   f"{se:.3f}"),
                ("α =",    f"{alpha:.2f}"),
            ],
            panel_w=8.0,
        )
        setup.scale(0.82).center().shift(UP*0.10)
        setup.set_opacity(0)
        self.play(FadeIn(setup, shift=UP*0.12, run_time=1.0))
        self.wait(0.6)

        # Data dot strip
        rng  = np.random.default_rng(7)
        data = rng.normal(x_bar, sigma, n)
        dots = _DataDotStrip(data, x_lo=180, x_hi=420,
                             label="Sample", color=P["dot"],
                             width=5.5, strip_y=0.0)
        dots.scale(0.78).move_to([0, -2.40, 0])
        dots.set_opacity(0)
        self.play(FadeIn(dots, shift=UP*0.10, run_time=0.7))
        self.wait(0.3)

        self.play(FadeOut(setup, dots, run_time=0.45))

        # Calculation panel
        calc = _CalcPanel(
            symbolic=r"Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}",
            substituted=(rf"Z = \frac{{{x_bar:.1f} - "
                         rf"{mu0:.1f}}}{{{sigma:.1f}/\sqrt{{{n}}}}}"),
            result=rf"Z_{{obs}} = {z_obs:.3f}",
            test_name="Z-Test Calculation",
            panel_w=7.2,
        )
        calc.scale(0.88).center()
        calc.set_opacity(0)
        self.play(FadeIn(calc, shift=UP*0.10, run_time=0.7))

        # Reveal calculation lines one by one
        calc.line_subst.set_opacity(0)
        calc.line_result.set_opacity(0)
        self.wait(0.3)
        self.play(FadeIn(calc.line_subst, run_time=0.5))
        self.wait(0.3)
        self.play(FadeIn(calc.line_result, run_time=0.5))
        self.wait(0.5)
        self.play(Indicate(calc.line_result,
                           color=P["calc_result"],
                           scale_factor=1.12, run_time=0.6))
        self.wait(0.4)
        self.play(FadeOut(calc, run_time=0.4))

        # Full hypothesis test visualisation
        if _INF_AVAILABLE:
            ht = HypothesisTest3D.z_test(
                x_bar=x_bar, mu0=mu0, sigma=sigma, n=n,
                alpha=alpha, tail="right",
                plot_width=8.5, plot_height=3.0,
                baseline_y=-1.6, show_panel=True,
            )
            ht.scale(0.82).center()
            self.add(ht)
            self.play(make_full_sequence(ht))
            self.wait(0.7)

            # p-value gauge below
            from scipy.stats import norm as _sn
            p_val = float(1 - _sn.cdf(z_obs))
            gauge = _PValueGauge(p_value=p_val, alpha=alpha,
                                 width=5.5, height=0.40)
            gauge.scale(0.85).move_to([0, -3.50, 0])
            self.play(FadeIn(gauge, shift=UP*0.08, run_time=0.8))
            self.wait(0.7)

            # One-tailed vs two-tailed comparison
            ht2 = HypothesisTest3D.z_test(
                x_bar=x_bar, mu0=mu0, sigma=sigma, n=n,
                alpha=alpha, tail="both",
                plot_width=7.0, plot_height=2.6,
                baseline_y=-1.4, show_panel=False,
            )
            two_lbl = Text("Two-tailed: p doubles",
                           font_size=18, color=P["pvalue"])
            two_lbl.move_to([0, 2.0, 0])
            self.play(
                FadeOut(ht, gauge, run_time=0.45),
                FadeIn(ht2, run_time=0.55),
                FadeIn(two_lbl, run_time=0.45),
            )
            self.play(DropStatistic(ht2, run_time=0.9))
            self.play(RevealPValue(ht2, run_time=0.8))
            self.wait(0.6)
            _fade(ht2, two_lbl, lbl, scene=self)
        else:
            _fade(lbl, scene=self)

    # ─────────────────────────────────────────────────────────────────
    # Act 2 — One-sample t-test
    # ─────────────────────────────────────────────────────────────────

    def _act2_t_test(self):
        lbl = _section_label("Act 2 — One-Sample t-Test")
        lbl.to_corner(LEFT+UP, buff=0.22)
        self.play(FadeIn(lbl, run_time=0.4))

        # Problem parameters
        mu0    = 120.0   # mmHg (null: no change)
        n      = 12
        x_bar  = 115.3
        s      = 8.2
        alpha  = 0.05
        se     = s / np.sqrt(n)
        t_obs  = (x_bar - mu0) / se
        df     = n - 1

        setup = _HypothesisSetupPanel(
            problem="Did drug lower systolic BP?",
            h0_str=r"\mu = 120",
            h1_str=r"\mu < 120",
            formula=r"t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}",
            test_name="One-Sample t-Test  (σ unknown, n=12)",
            assumptions=[
                ("Population σ unknown (use s)",  True),
                ("Simple random sample",           True),
                ("Population approximately normal",True),
            ],
            data_summary=[
                ("n =",  str(n)),
                ("x̄ =", f"{x_bar:.1f} mmHg"),
                ("s =",  f"{s:.2f} mmHg"),
                ("μ₀ =", f"{mu0:.1f} mmHg"),
                ("df =", str(df)),
                ("α =",  f"{alpha:.2f}"),
            ],
            panel_w=8.0,
        )
        setup.scale(0.80).center().shift(UP*0.10)
        setup.set_opacity(0)
        self.play(FadeIn(setup, shift=UP*0.10, run_time=0.9))
        self.wait(0.6)

        # Highlight: t vs Z difference
        diff_lbl = Text(
            "t-distribution: heavier tails than Normal (df=11)",
            font_size=19, color=P["h1"],
        )
        diff_lbl.move_to([0, -2.60, 0])
        self.play(Write(diff_lbl, run_time=0.9))
        self.wait(0.5)
        self.play(FadeOut(setup, diff_lbl, run_time=0.40))

        # Calculation
        calc = _CalcPanel(
            symbolic=r"t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}",
            substituted=(rf"t = \frac{{{x_bar:.1f} - {mu0:.1f}}}"
                         rf"{{{s:.2f}/\sqrt{{{n}}}}}"),
            result=rf"t_{{obs}} = {t_obs:.3f}\quad(df={df})",
            test_name="t-Test Calculation",
        )
        calc.scale(0.88).center()
        calc.set_opacity(0)
        self.play(FadeIn(calc, run_time=0.6))
        calc.line_subst.set_opacity(0)
        calc.line_result.set_opacity(0)
        self.play(FadeIn(calc.line_subst, run_time=0.45))
        self.play(FadeIn(calc.line_result, run_time=0.45))
        self.play(Indicate(calc.line_result, color=P["calc_result"],
                           scale_factor=1.10, run_time=0.55))
        self.wait(0.4)
        self.play(FadeOut(calc, run_time=0.4))

        if _INF_AVAILABLE:
            ht = HypothesisTest3D.one_sample_t(
                x_bar=x_bar, mu0=mu0, s=s, n=n,
                alpha=alpha, tail="left",
                plot_width=8.5, plot_height=3.0,
                baseline_y=-1.6, show_panel=True,
            )
            ht.scale(0.82).center()
            self.add(ht)
            self.play(make_full_sequence(ht))
            self.wait(0.8)

            # Change α to see effect
            self.play(ChangeAlpha(ht, new_alpha=0.01, run_time=1.5))
            self.wait(0.4)
            self.play(ChangeAlpha(ht, new_alpha=0.05, run_time=1.5))
            self.wait(0.6)
            _fade(ht, lbl, scene=self)
        else:
            _fade(lbl, scene=self)

    # ─────────────────────────────────────────────────────────────────
    # Act 3 — Two-sample t-test
    # ─────────────────────────────────────────────────────────────────

    def _act3_two_sample_t(self):
        lbl = _section_label("Act 3 — Two-Sample t-Test (Welch)")
        lbl.to_corner(LEFT+UP, buff=0.22)
        self.play(FadeIn(lbl, run_time=0.4))

        # Problem parameters
        n1, n2  = 20, 18
        x1, x2  = 76.4, 70.8
        s1, s2  = 8.5, 9.2
        alpha   = 0.05

        # Welch df
        se_sq   = s1**2/n1 + s2**2/n2
        se      = np.sqrt(se_sq)
        df_w    = int((se_sq**2) / (
            (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        ))
        t_obs   = (x1 - x2) / se

        setup = _HypothesisSetupPanel(
            problem="Different teaching methods → different scores?",
            h0_str=r"\mu_1 = \mu_2",
            h1_str=r"\mu_1 \neq \mu_2",
            formula=r"t = \frac{\bar{x}_1 - \bar{x}_2}"
                    r"{\sqrt{s_1^2/n_1 + s_2^2/n_2}}",
            test_name="Two-Sample Welch t-Test",
            assumptions=[
                ("Independent samples",            True),
                ("Both approx. normal (n>15)",     True),
                ("Unequal variances (Welch)",       True),
            ],
            data_summary=[
                ("Group 1: n₁, x̄₁, s₁", f"{n1},  {x1},  {s1}"),
                ("Group 2: n₂, x̄₂, s₂", f"{n2},  {x2},  {s2}"),
                ("SE =", f"{se:.3f}"),
                ("df (Welch) =", str(df_w)),
            ],
            panel_w=8.0,
        )
        setup.scale(0.80).center().shift(UP*0.10)
        setup.set_opacity(0)
        self.play(FadeIn(setup, run_time=0.9))
        self.wait(0.55)

        # Two dot strips side by side
        rng = np.random.default_rng(13)
        d1  = rng.normal(x1, s1, n1)
        d2  = rng.normal(x2, s2, n2)
        st1 = _DataDotStrip(d1, x_lo=45, x_hi=105,
                            label="Group 1", color=P["h0"],
                            width=4.2, strip_y=0.0)
        st2 = _DataDotStrip(d2, x_lo=45, x_hi=105,
                            label="Group 2", color=P["h1"],
                            width=4.2, strip_y=0.0)
        st1.scale(0.75).move_to([-3.0, -2.40, 0])
        st2.scale(0.75).move_to([ 3.0, -2.40, 0])
        st1.set_opacity(0)
        st2.set_opacity(0)
        self.play(
            FadeIn(st1, shift=UP*0.08, run_time=0.6),
            FadeIn(st2, shift=UP*0.08, run_time=0.6),
        )
        self.wait(0.4)
        self.play(FadeOut(setup, st1, st2, run_time=0.45))

        if _INF_AVAILABLE:
            # Side-by-side one-tailed vs two-tailed
            ht_two = HypothesisTest3D.two_sample_t(
                x1=x1, x2=x2, s1=s1, s2=s2, n1=n1, n2=n2,
                alpha=alpha, tail="both",
                plot_width=7.0, plot_height=2.5,
                baseline_y=-1.4, show_panel=True,
            )
            ht_two.scale(0.82).center()
            self.add(ht_two)
            self.play(make_full_sequence(ht_two))
            self.wait(0.8)
            _fade(ht_two, lbl, scene=self)
        else:
            _fade(lbl, scene=self)

    # ─────────────────────────────────────────────────────────────────
    # Act 4 — Chi-square goodness-of-fit
    # ─────────────────────────────────────────────────────────────────

    def _act4_chi_square(self):
        lbl = _section_label("Act 4 — Chi-Square Goodness-of-Fit Test")
        lbl.to_corner(LEFT+UP, buff=0.22)
        self.play(FadeIn(lbl, run_time=0.4))

        # Die fairness test
        observed = np.array([18, 22, 30, 14, 16, 20])
        expected = np.array([20, 20, 20, 20, 20, 20])
        alpha    = 0.05
        chi2_stat = float(np.sum((observed-expected)**2 / expected))
        cats     = [1, 2, 3, 4, 5, 6]

        setup = _HypothesisSetupPanel(
            problem="Is this die fair?  (120 rolls)",
            h0_str=r"p_i = 1/6 \text{ for all faces}",
            h1_str=r"\exists\, i : p_i \neq 1/6",
            formula=r"\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}",
            test_name="Chi-Square Goodness-of-Fit",
            assumptions=[
                ("Each expected count ≥ 5",         True),
                ("Observations independent",         True),
                ("Categorical data",                 True),
            ],
            data_summary=[
                ("Total rolls =", "120"),
                ("k categories =", "6"),
                ("df = k−1 =", "5"),
                ("α =", f"{alpha:.2f}"),
            ],
            panel_w=8.0,
        )
        setup.scale(0.80).center().shift(UP*0.15)
        setup.set_opacity(0)
        self.play(FadeIn(setup, run_time=0.9))
        self.wait(0.55)
        self.play(FadeOut(setup, run_time=0.40))

        # Bar chart O vs E
        bar_chart = _ChiSquareBarChart(
            categories=cats,
            observed=observed,
            expected=expected,
            width=5.8, height=2.4,
            baseline_y=0.0,
        )
        bar_chart.scale(0.88).center().shift(UP*0.20)
        bar_chart.set_opacity(0)
        self.play(FadeIn(bar_chart, shift=UP*0.10, run_time=0.90))

        # Build chi2 term by term label
        terms_str = "  +  ".join(
            f"({o}-{e})²/{e}" for o, e in zip(observed, expected)
        )
        try:
            chi2_lbl = MathTex(
                rf"\chi^2 = {chi2_stat:.3f}",
                font_size=22, color=P["chi2_diff"],
            )
        except Exception:
            chi2_lbl = Text(f"χ² = {chi2_stat:.3f}",
                            font_size=19, color=P["chi2_diff"])
        chi2_lbl.move_to([0, -2.10, 0])
        self.play(Write(chi2_lbl, run_time=0.8))
        self.wait(0.5)
        self.play(FadeOut(bar_chart, chi2_lbl, run_time=0.40))

        if _INF_AVAILABLE:
            ht = HypothesisTest3D.chi_square(
                observed=observed.tolist(),
                expected=expected.tolist(),
                alpha=alpha,
                plot_width=8.5, plot_height=3.0,
                baseline_y=-1.6, show_panel=True,
            )
            ht.scale(0.82).center()
            self.add(ht)
            self.play(make_full_sequence(ht))
            self.wait(0.8)
            _fade(ht, lbl, scene=self)
        else:
            _fade(lbl, scene=self)

    # ─────────────────────────────────────────────────────────────────
    # Act 5 — Power analysis
    # ─────────────────────────────────────────────────────────────────

    def _act5_power(self):
        lbl = _section_label("Act 5 — Power Analysis")
        lbl.to_corner(LEFT+UP, buff=0.22)
        self.play(FadeIn(lbl, run_time=0.4))

        intro = Text(
            "Power = P(reject H₀ | H₁ is true)  =  1 − β",
            font_size=22, color=P["keep"],
        )
        intro.move_to([0, 2.60, 0])
        self.play(Write(intro, run_time=1.0))

        # TypeITypeII + NarrowCurves
        if _INF_AVAILABLE:
            viz = TypeITypeII(
                mu0=0.0, mu1=2.0,
                sigma=1.0, n=1,
                alpha=0.05, two_tailed=False,
                plot_width=7.5, plot_height=2.8,
                show_table=True,
                show_effect_arrow=True,
                show_all_regions=False,
                baseline_y=-0.60,
            )
            viz.scale(0.75).center().shift(DOWN*0.20)
            self.add(viz)
            self.play(RevealAll(viz))
            self.wait(0.5)

            n_badge = _badge(f"n = 1", P["f_n"] if hasattr(P,"f_n")
                             else P["alpha"])
            n_badge.move_to([-5.50, 2.0, 0])
            self.play(FadeIn(n_badge, run_time=0.4))

            for new_n, nt in [(4, "n = 4"), (16, "n = 16"),
                               (36, "n = 36")]:
                self.play(NarrowCurves(viz, new_n=new_n, run_time=2.0))
                new_badge = _badge(nt, P["alpha"])
                new_badge.move_to(n_badge.get_center())
                self.play(ReplacementTransform(n_badge, new_badge,
                                               run_time=0.4))
                n_badge = new_badge
                self.wait(0.4)

            _fade(viz, n_badge, scene=self, rt=0.50)
        self.wait(0.2)

        # Power curve
        power_plot = _PowerCurve(
            sigma=1.0, alpha=0.05, n_max=80,
            deltas=[0.5, 1.0, 1.5, 2.0],
            width=6.0, height=3.0, baseline_y=-0.10,
        )
        power_plot.scale(0.88).center()
        power_plot.set_opacity(0)
        self.play(FadeIn(power_plot, shift=UP*0.12, run_time=1.1))

        key_msg = Text(
            "Larger n or larger δ  →  more power.",
            font_size=20, color=P["subtitle"],
        )
        key_msg.move_to([0, -2.95, 0])
        self.play(Write(key_msg, run_time=0.9))
        self.wait(0.8)

        tradeoff = Text(
            "Tension:  larger n = higher cost;  higher α = more Type I errors.",
            font_size=17, color=P["alpha"],
        )
        tradeoff.move_to([0, -3.60, 0])
        self.play(FadeIn(tradeoff, shift=UP*0.06, run_time=0.7))
        self.wait(0.9)

        _fade(power_plot, key_msg, tradeoff, intro, lbl, scene=self)

    # ─────────────────────────────────────────────────────────────────
    # Act 6 — Decision framework + closing
    # ─────────────────────────────────────────────────────────────────

    def _act6_framework(self):
        lbl = _section_label("Act 6 — Decision Framework & Common Pitfalls")
        lbl.to_corner(LEFT+UP, buff=0.22)
        self.play(FadeIn(lbl, run_time=0.4))

        # Flowchart on the left
        flow = _DecisionFlowchart(width=4.8)
        flow.scale(0.88).move_to([-3.50, 0.20, 0])
        flow.set_opacity(0)
        self.play(FadeIn(flow, shift=RIGHT*0.12, run_time=1.1))
        self.wait(0.5)

        # Common pitfalls panel on the right
        pitfalls = [
            ("✗", "p > 0.05 ≠ 'H₀ is true'",
             "Absence of evidence ≠ evidence of absence"),
            ("✗", "p-value ≠ P(H₀ is true)",
             "It's P(data | H₀), not P(H₀ | data)"),
            ("✗", "Statistical ≠ practical significance",
             "Large n can make tiny effects 'significant'"),
            ("✗", "Multiple testing inflates α",
             "Apply Bonferroni or FDR correction"),
            ("✓", "Report effect size + CI, not just p",
             "e.g. Cohen's d, odds ratio, CI"),
        ]
        pit_grp = VGroup()
        for i, (mark, title, desc) in enumerate(pitfalls):
            col  = P["reject"] if mark == "✗" else P["keep"]
            y_r  = 1.85 - i * 0.85
            mk   = Text(mark, font_size=18, color=col)
            mk.move_to([1.30, y_r, 0])
            tt   = Text(title, font_size=14, color=col)
            tt.move_to([1.58+tt.width/2, y_r+0.15, 0])
            dt   = Text(desc, font_size=11, color=P["subtitle"])
            dt.move_to([1.58+dt.width/2, y_r-0.18, 0])
            pit_grp.add(mk, tt, dt)

        pit_grp.set_opacity(0)
        self.play(FadeIn(pit_grp, shift=LEFT*0.10, run_time=0.9))
        self.wait(1.0)

        # Highlight flowchart steps
        for sub in flow.submobjects[:6]:
            self.play(Indicate(sub, scale_factor=1.06,
                               run_time=0.35))
            self.wait(0.06)
        self.wait(0.7)

        _fade(flow, pit_grp, lbl, scene=self)

        # Closing title card
        bg_end   = Rectangle(width=16, height=9,
                              fill_color=P["bg"],
                              fill_opacity=1.0, stroke_width=0)
        title_end = Text("Hypothesis Testing",
                         font_size=44, color=P["title"])
        try:
            sub_end = MathTex(
                r"p < \alpha \;\Rightarrow\; \text{Reject } H_0"
                r"\qquad\text{(but always consider context)}",
                font_size=26, color=P["subtitle"],
            )
        except Exception:
            sub_end = Text(
                "p < α  →  Reject H₀   (but always consider context)",
                font_size=22, color=P["subtitle"],
            )
        title_end.move_to([0,  0.55, 0])
        sub_end.move_to([0, -0.45, 0])

        self.play(
            FadeIn(bg_end, run_time=0.5),
            FadeIn(title_end, run_time=0.8),
        )
        self.play(FadeIn(sub_end,
                         rate_func=rate_functions.ease_out_back,
                         run_time=0.7))
        self.wait(2.0)
        self.play(FadeOut(VGroup(bg_end, title_end, sub_end)),
                  run_time=0.9)