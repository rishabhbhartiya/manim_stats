"""
manim_stats/scenes/demo_bayes.py
==================================
BayesDemo — A complete, cinematic Bayes' theorem demonstration scene.

Story arc (7 acts)
------------------
Act 0  Cold open
       A single urn emerges from darkness with a "?" above it.
       Title card: "Which urn did this ball come from?"

Act 1  Setup — two urns
       Urn A (3 red, 2 blue) and Urn B (1 red, 4 blue) appear side by side.
       Ball counts animate in.  "One urn was chosen at random."

Act 2  Prior
       Prior probability bar: P(A) = 0.5 │ P(B) = 0.5
       Animated split with labels and percentage badges.
       Formula badge: "P(A) = P(B) = 1/2 (equal prior)"

Act 3  Draw — a red ball emerges
       A red ball is drawn from an unseen urn (ShakeUrn + DrawBall).
       "We drew a RED ball.  Which urn is more likely?"

Act 4  Likelihood
       Likelihood table animates in:
           P(red | A) = 3/5 = 0.600
           P(red | B) = 1/5 = 0.200
       Each cell highlights as its value is spoken.
       Marginal: P(red) = P(red|A)·P(A) + P(red|B)·P(B) = 0.400

Act 5  Bayes formula
       Full formula types in one term at a time:
           P(A | red) = P(red | A) · P(A) / P(red)
                      = 0.600 × 0.500 / 0.400
                      = 0.750
       Each numeric substitution highlights in a matching colour.

Act 6  Posterior bar — first update
       Prior bar morphs into posterior: P(A|red)=0.75 │ P(B|red)=0.25
       Animated shift with counting number badges.

Act 7  Sequential updates — two more draws
       Draw 2: another red ball  → posterior shifts further toward A
       Draw 3: a blue ball       → posterior pulls back toward B
       After each draw:
         • Likelihood row highlights
         • Formula re-substitutes
         • Posterior bar morphs
         • Posterior history trail shows all three bars stacked

Act 8  Beta distribution posterior
       After the 3 draws (2 red, 1 blue from a Beta-Binomial model),
       a Beta(α, β) curve replaces the discrete bar — showing the
       continuous Bayesian posterior.

Act 9  Decision and summary
       "Urn A is 3× more likely than Urn B."
       Decision badge flashes on Urn A.
       Posterior bar glows.
       Summary panel: prior → likelihood → posterior, one line each.

Scene uses
----------
  manim_stats.props.urn    — Urn3D, Ball3D, DrawBall, ShakeUrn, FillUrn
  (all other components built inline for self-contained demo)

Dependencies
------------
  manim (CE or GL), numpy, scipy (optional)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from manim import (
    Scene, ThreeDScene,
    VGroup,
    Rectangle, RoundedRectangle, Square,
    Circle, Annulus, Polygon, Line, DashedLine,
    Arrow, DoubleArrow,
    Text, MathTex, Tex,
    Dot,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut, GrowFromCenter,
    Create, Write, Uncreate,
    Indicate, Flash,
    Rotate,
    Transform, ReplacementTransform,
    MoveToTarget,
    ValueTracker,
    interpolate_color, color_to_rgb,
    always_redraw,
    BLACK, WHITE,
    GREY,   GREY_A,   GREY_B,   GREY_C,   GREY_D,
    RED,    RED_A,    RED_B,    RED_C,    RED_D,
    GREEN,  GREEN_A,  GREEN_B,  GREEN_C,  GREEN_D,  GREEN_E,
    BLUE,   BLUE_A,   BLUE_B,   BLUE_C,   BLUE_D,   BLUE_E,
    YELLOW, YELLOW_A, YELLOW_E,
    ORANGE, TEAL, TEAL_A, TEAL_B, TEAL_C,
    GOLD,   GOLD_A,   GOLD_B,   GOLD_C,   GOLD_D,
    PURPLE_A, PURPLE_B, MAROON, PINK,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    ManimColor,
    VMobject,
    NumberLine,
    DEGREES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Optional urn import (graceful fallback to plain shapes if unavailable)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from props.urn import Urn3D, Ball3D, DrawBall, ShakeUrn, FillUrn
    from props.urn import get_packed_positions
    _URN_AVAILABLE = True
except ImportError:
    _URN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Scene palette
# ─────────────────────────────────────────────────────────────────────────────

P = {
    # Urn colours
    "urn_a":        "#C1440E",    # terracotta
    "urn_b":        "#1B3A8A",    # cobalt

    # Ball colours
    "red_ball":     "#E63946",
    "blue_ball":    "#457B9D",

    # Prior / posterior bars
    "bar_a":        "#E07030",    # warm orange for Urn A
    "bar_b":        "#3070C0",    # cool blue for Urn B
    "bar_bg":       "#0C1018",
    "bar_border":   "#2A3848",
    "bar_text":     "#E8F0F8",

    # Likelihood table
    "tbl_bg":       "#0E1420",
    "tbl_header":   "#1A2438",
    "tbl_red_row":  "#2A1010",
    "tbl_blue_row": "#101828",
    "tbl_border":   "#2A3848",
    "tbl_val_red":  "#FF9090",
    "tbl_val_blue": "#90C0FF",
    "tbl_val_marg": "#D4AF37",

    # Bayes formula
    "formula_bg":   "#080C14",
    "term_prior":   "#D4AF37",    # gold
    "term_like":    "#E07030",    # orange
    "term_marg":    "#70B060",    # green
    "term_post":    "#C060E0",    # purple
    "term_eq":      "#C8D8E8",

    # Decision badge
    "decision_bg":  "#1A2410",
    "decision_fg":  "#60E060",

    # Beta curve
    "beta_fill":    "#2A1040",
    "beta_ridge":   "#A060E0",
    "beta_shade":   "#6030A0",

    # Title cards
    "title_fg":     "#D8F0FF",
    "title_bg":     "#050810",
    "subtitle_fg":  "#8090A8",

    # Summary panel
    "summary_bg":   "#0C1018",
    "summary_border":"#3A4A5A",
    "summary_arrow":"#607080",

    # General
    "bg":           "#080C12",
    "glow":         "#FFD060",
    "axis":         "#3A4A5A",
}


# ─────────────────────────────────────────────────────────────────────────────
# Reusable scene-building helpers
# ─────────────────────────────────────────────────────────────────────────────

def _title_card(
    main:     str,
    subtitle: str = "",
    main_fs:  int = 44,
    sub_fs:   int = 22,
) -> VGroup:
    """Full-screen title card (dark bg, centred text)."""
    g   = VGroup()
    bg  = Rectangle(width=16, height=9,
                    fill_color=P["title_bg"],
                    fill_opacity=1.0, stroke_width=0)
    g.add(bg)
    ml  = Text(main, font_size=main_fs, color=P["title_fg"],
               font="sans-serif")
    ml.move_to([0, 0.35 if subtitle else 0, 0])
    g.add(ml)
    if subtitle:
        sl  = Text(subtitle, font_size=sub_fs,
                   color=P["subtitle_fg"])
        sl.move_to([0, -0.45, 0])
        g.add(sl)
    return g


def _section_label(text: str, color: str = None) -> Text:
    """Small act label in the corner."""
    return Text(text, font_size=17,
                color=color or P["subtitle_fg"])


# ─────────────────────────────────────────────────────────────────────────────
# UrnSchematic — lightweight urn + ball counts for 2-D scene
# ─────────────────────────────────────────────────────────────────────────────

class _UrnSchematic(VGroup):
    """
    A clean 2-D schematic urn (for use in a flat Scene rather than
    ThreeDScene), showing:
      • Trapezoid body with coloured glaze fill
      • Two handles (line arcs)
      • Interior dots representing balls
      • Label badge above (Urn A / Urn B)
      • Prior / posterior probability badge below

    Parameters
    ----------
    label        : str   — "Urn A" / "Urn B"
    n_red, n_blue: int   — ball counts
    fill_color   : str   — body colour
    prior        : float — prior probability (shown below)
    """

    def __init__(
        self,
        label:      str,
        n_red:      int,
        n_blue:     int,
        fill_color: str,
        prior:      float = 0.5,
        width:      float = 1.60,
        height:     float = 2.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._n_red   = n_red
        self._n_blue  = n_blue
        self._fill    = fill_color
        self._label   = label
        self._width   = width
        self._height  = height

        dark = interpolate_color(fill_color, BLACK, 0.45)
        lite = interpolate_color(fill_color, WHITE, 0.28)

        # ── Body (trapezoid) ──────────────────────────────────────────
        bw  = width
        tw  = width * 0.62     # top width (narrower neck)
        h   = height
        body = Polygon(
            [-bw/2,  -h/2, 0],
            [ bw/2,  -h/2, 0],
            [ tw/2,   h/2, 0],
            [-tw/2,   h/2, 0],
            fill_color=fill_color,
            fill_opacity=1.0,
            stroke_color=dark,
            stroke_width=1.8,
        )
        self.add(body)

        # Specular stripe
        spec = Polygon(
            [-tw/2 + tw*0.05,  h/2,       0.001],
            [-tw/2 + tw*0.28,  h/2,       0.001],
            [-bw/2 + bw*0.28, -h/2,       0.001],
            [-bw/2 + bw*0.05, -h/2,       0.001],
            fill_color=lite,
            fill_opacity=0.25, stroke_width=0,
        )
        self.add(spec)

        # Lip (top opening)
        lip = Rectangle(
            width=tw * 1.08, height=h * 0.055,
            fill_color=lite,
            fill_opacity=1.0, stroke_width=0,
        )
        lip.move_to([0, h/2 + lip.height/2, 0.001])
        self.add(lip)

        # Inner dark well
        well = Polygon(
            [-tw/2 + 0.05, h/2, 0.001],
            [ tw/2 - 0.05, h/2, 0.001],
            [ tw/2 - 0.05, h/2 - h*0.12, 0.001],
            [-tw/2 + 0.05, h/2 - h*0.12, 0.001],
            fill_color=interpolate_color(fill_color, BLACK, 0.70),
            fill_opacity=1.0, stroke_width=0,
        )
        self.add(well)

        # ── Handles ───────────────────────────────────────────────────
        for side in [-1, 1]:
            hx = side * bw / 2
            handle_pts = []
            n_pts = 10
            for i in range(n_pts):
                t   = i / (n_pts - 1)
                ang = PI * (0.30 + 0.40 * t) * side
                r   = bw * 0.28
                cx  = hx + side * r * 0.45
                cy  = -h * 0.05
                handle_pts.append(
                    [cx + r * np.cos(ang),
                     cy + r * np.sin(ang) * 0.80, 0]
                )
            for i in range(len(handle_pts)-1):
                self.add(Line(
                    handle_pts[i], handle_pts[i+1],
                    stroke_color=dark, stroke_width=3.5,
                ))

        # ── Interior balls (dots) ─────────────────────────────────────
        self._ball_dots = VGroup()
        total = n_red + n_blue
        cols  = ([P["red_ball"]] * n_red
                 + [P["blue_ball"]] * n_blue)
        np.random.seed(hash(label) % (2**31))
        np.random.shuffle(cols)
        rows_n  = max(1, int(np.ceil(np.sqrt(total * 0.8))))
        dot_r   = min(0.10, (bw * 0.70) / (2 * rows_n + 1))
        row_h   = (h * 0.68) / max(rows_n, 1)

        placed = 0
        for row in range(rows_n + 1):
            y_dot = -h/2 + dot_r + row * row_h * 0.85
            if y_dot > h/2 - dot_r * 2: break
            n_in_row = min(rows_n + 1, total - placed)
            row_w    = n_in_row * dot_r * 2.2
            for col in range(n_in_row):
                if placed >= total: break
                xd = -row_w/2 + col * dot_r * 2.2 + dot_r
                d  = Circle(
                    radius=dot_r,
                    fill_color=cols[placed],
                    fill_opacity=0.92,
                    stroke_color=interpolate_color(
                        cols[placed], BLACK, 0.25),
                    stroke_width=0.6,
                )
                d.move_to([xd, y_dot, 0.003])
                # Specular highlight
                sh = Circle(
                    radius=dot_r * 0.30,
                    fill_color=interpolate_color(
                        cols[placed], WHITE, 0.55),
                    fill_opacity=0.65, stroke_width=0,
                )
                sh.move_to([xd - dot_r*0.22,
                            y_dot + dot_r*0.25, 0.004])
                self._ball_dots.add(d, sh)
                placed += 1
        self.add(self._ball_dots)

        # ── Urn label ─────────────────────────────────────────────────
        lbl = Text(label, font_size=22,
                   color=interpolate_color(fill_color, WHITE, 0.60))
        lbl.move_to([0, h/2 + 0.42, 0.002])
        self.add(lbl)

        # Ball count badges
        for count, col, sign in [
            (n_red,  P["red_ball"],  -1),
            (n_blue, P["blue_ball"],  1),
        ]:
            ct = Text(f"●×{count}", font_size=14, color=col)
            ct.move_to([sign * tw/2 * 0.55, -h/2 - 0.28, 0.002])
            self.add(ct)

        # Prior badge
        self._prior_badge = self._make_prob_badge(prior, label)
        self._prior_badge.move_to([0, -h/2 - 0.68, 0.002])
        self.add(self._prior_badge)

    def _make_prob_badge(self, prob: float, urn: str) -> VGroup:
        g    = VGroup()
        col  = P["bar_a"] if "A" in urn else P["bar_b"]
        txt  = Text(f"P({urn[-1]}) = {prob:.3f}",
                    font_size=15, color=col)
        bg   = RoundedRectangle(
            width=txt.width + 0.22,
            height=txt.height + 0.14,
            corner_radius=0.06,
            fill_color=P["bar_bg"],
            fill_opacity=0.88,
            stroke_color=col,
            stroke_width=0.7,
        )
        bg.move_to(ORIGIN)
        txt.move_to([0, 0, 0.001])
        g.add(bg, txt)
        return g

    def update_prior_badge(self, new_prob: float) -> VGroup:
        """Replace the probability badge with a new value."""
        old_pos = self._prior_badge.get_center()
        new_badge = self._make_prob_badge(
            new_prob, self._label[-1]
        )
        new_badge.move_to(old_pos)
        if self._prior_badge in self.submobjects:
            self.remove(self._prior_badge)
        self._prior_badge = new_badge
        self.add(new_badge)
        return new_badge


# ─────────────────────────────────────────────────────────────────────────────
# Probability bar
# ─────────────────────────────────────────────────────────────────────────────

class _ProbBar(VGroup):
    """
    Horizontal split probability bar showing P(A) and P(B).

    Layers:
      1. Background rect
      2. Left segment (Urn A, P_a wide)
      3. Right segment (Urn B, P_b wide)
      4. Divider line
      5. Labels above each segment
      6. Percentage badges

    Parameters
    ----------
    p_a     : float   — P(Urn A)
    width   : float   — total bar width
    height  : float   — bar height
    label_a : str
    label_b : str
    """

    # Class-level colour constants so morphing is possible
    COLOR_A = P["bar_a"]
    COLOR_B = P["bar_b"]

    def __init__(
        self,
        p_a:     float,
        width:   float = 6.0,
        height:  float = 0.55,
        label_a: str   = "Urn A",
        label_b: str   = "Urn B",
        title:   str   = "Prior Probability",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._p_a    = p_a
        self._width  = width
        self._height = height

        p_b    = 1 - p_a
        w_a    = p_a * width
        w_b    = p_b * width
        z0     = 0.001

        # ── Background ────────────────────────────────────────────────
        bg = Rectangle(
            width=width + 0.06,
            height=height + 0.06,
            fill_color=P["bar_bg"],
            fill_opacity=1.0,
            stroke_color=P["bar_border"],
            stroke_width=0.8,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        # ── Segment A ─────────────────────────────────────────────────
        seg_a = Rectangle(
            width=max(w_a - 0.02, 0.01),
            height=height,
            fill_color=P["bar_a"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        seg_a.move_to([-width/2 + w_a/2, 0, z0])
        self.add(seg_a)

        # Specular highlight on A
        if w_a > 0.15:
            hl_a = Rectangle(
                width=min(w_a * 0.35, 0.60),
                height=height * 0.28,
                fill_color=interpolate_color(P["bar_a"], WHITE, 0.45),
                fill_opacity=0.35, stroke_width=0,
            )
            hl_a.move_to([-width/2 + w_a * 0.30,
                          height * 0.22, z0+0.001])
            self.add(hl_a)

        # ── Segment B ─────────────────────────────────────────────────
        seg_b = Rectangle(
            width=max(w_b - 0.02, 0.01),
            height=height,
            fill_color=P["bar_b"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        seg_b.move_to([width/2 - w_b/2, 0, z0])
        self.add(seg_b)

        if w_b > 0.15:
            hl_b = Rectangle(
                width=min(w_b * 0.35, 0.60),
                height=height * 0.28,
                fill_color=interpolate_color(P["bar_b"], WHITE, 0.45),
                fill_opacity=0.35, stroke_width=0,
            )
            hl_b.move_to([-width/2 + w_a + w_b * 0.30,
                          height * 0.22, z0+0.001])
            self.add(hl_b)

        # ── Divider ───────────────────────────────────────────────────
        div_x = -width/2 + w_a
        self.add(Line(
            [div_x, -height/2, z0+0.003],
            [div_x,  height/2, z0+0.003],
            stroke_color=WHITE,
            stroke_width=1.8,
        ))

        # ── Labels above ──────────────────────────────────────────────
        if w_a > 0.4:
            la = Text(f"{label_a}  {p_a:.1%}",
                      font_size=15, color=P["bar_text"])
            la.move_to([-width/2 + w_a/2, height/2 + 0.28, z0+0.002])
            self.add(la)
        if w_b > 0.4:
            lb = Text(f"{p_b:.1%}  {label_b}",
                      font_size=15, color=P["bar_text"])
            lb.move_to([width/2 - w_b/2, height/2 + 0.28, z0+0.002])
            self.add(lb)

        # ── Title ─────────────────────────────────────────────────────
        tl = Text(title, font_size=18, color=P["subtitle_fg"])
        tl.move_to([0, -height/2 - 0.32, z0+0.002])
        self.add(tl)


def _morph_prob_bar(
    scene:   Scene,
    old_bar: _ProbBar,
    new_p_a: float,
    run_time: float = 1.2,
    **kwargs,
) -> _ProbBar:
    """
    Create a new _ProbBar with updated probability and cross-fade to it.
    Returns the new bar (added to scene in place of old).
    """
    new_bar = _ProbBar(
        p_a=new_p_a,
        width=old_bar._width,
        height=old_bar._height,
        title="Posterior Probability",
    )
    new_bar.move_to(old_bar.get_center())
    scene.play(
        ReplacementTransform(old_bar, new_bar, run_time=run_time),
        **kwargs,
    )
    return new_bar


# ─────────────────────────────────────────────────────────────────────────────
# Likelihood table
# ─────────────────────────────────────────────────────────────────────────────

class _LikelihoodTable(VGroup):
    """
    Likelihood table showing P(obs | Urn A) and P(obs | Urn B).

    Structure:
      ┌─────────────┬──────────────┬──────────────┐
      │             │    Urn A     │    Urn B     │
      ├─────────────┼──────────────┼──────────────┤
      │ P(red  | ·) │  3/5 = 0.600 │  1/5 = 0.200 │
      │ P(blue | ·) │  2/5 = 0.400 │  4/5 = 0.800 │
      ├─────────────┼──────────────┼──────────────┤
      │ P(red) (marginal)  =  0.400              │
      └─────────────────────────────────────────────┘

    Parameters
    ----------
    p_red_a, p_red_b   : likelihoods
    p_a                : current prior P(A)
    """

    def __init__(
        self,
        p_red_a: float,
        p_red_b: float,
        p_a:     float,
        cell_w:  float = 1.65,
        cell_h:  float = 0.60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        p_b    = 1 - p_a
        p_blue_a = 1 - p_red_a
        p_blue_b = 1 - p_red_b
        p_red_marg  = p_red_a  * p_a + p_red_b  * p_b
        p_blue_marg = p_blue_a * p_a + p_blue_b * p_b

        row_w   = 1.80   # row header width
        n_cols  = 2
        total_w = row_w + n_cols * cell_w
        total_h = cell_h * 4    # header + 2 data rows + marginal

        z0 = 0.001

        # ── Background ────────────────────────────────────────────────
        bg = RoundedRectangle(
            width=total_w + 0.24,
            height=total_h + 0.24,
            corner_radius=0.10,
            fill_color=P["tbl_bg"],
            fill_opacity=0.95,
            stroke_color=P["tbl_border"],
            stroke_width=0.9,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        x_cols = [
            -total_w/2 + row_w/2,
            -total_w/2 + row_w + cell_w/2,
            -total_w/2 + row_w + cell_w * 1.5,
        ]
        y_header = total_h/2 - cell_h/2

        # ── Header row ────────────────────────────────────────────────
        for xi, txt, col in [
            (x_cols[0], "",       P["tbl_header"]),
            (x_cols[1], "Urn A",  P["tbl_header"]),
            (x_cols[2], "Urn B",  P["tbl_header"]),
        ]:
            self._add_cell(xi, y_header, cell_w if xi != x_cols[0] else row_w,
                           cell_h, col, txt, z0)

        # ── Data rows ─────────────────────────────────────────────────
        data_rows = [
            ("P(red | · )",
             f"3/5 = {p_red_a:.3f}",  P["tbl_val_red"],
             f"1/5 = {p_red_b:.3f}",  P["tbl_val_red"],
             P["tbl_red_row"]),
            ("P(blue | · )",
             f"2/5 = {p_blue_a:.3f}", P["tbl_val_blue"],
             f"4/5 = {p_blue_b:.3f}", P["tbl_val_blue"],
             P["tbl_blue_row"]),
        ]
        for ri, (row_lbl, va, ca, vb, cb, rbg) in enumerate(data_rows):
            yr = y_header - (ri + 1) * cell_h
            self._add_cell(x_cols[0], yr, row_w, cell_h,
                           P["tbl_header"], row_lbl, z0)
            self._add_cell(x_cols[1], yr, cell_w, cell_h, rbg, va, z0,
                           val_color=ca)
            self._add_cell(x_cols[2], yr, cell_w, cell_h, rbg, vb, z0,
                           val_color=cb)

        # ── Marginal row ──────────────────────────────────────────────
        y_marg = y_header - 3 * cell_h
        marg_str = (f"P(red) = P(red|A)·P(A) + P(red|B)·P(B)"
                    f" = {p_red_marg:.3f}")
        marg_lbl = Text(marg_str, font_size=13,
                        color=P["tbl_val_marg"])
        marg_bg  = Rectangle(
            width=total_w, height=cell_h,
            fill_color=interpolate_color(P["tbl_bg"], P["tbl_val_marg"], 0.08),
            fill_opacity=1.0, stroke_width=0,
        )
        marg_bg.move_to([0, y_marg, z0])
        marg_lbl.move_to([0, y_marg, z0+0.002])
        self.add(marg_bg, marg_lbl)

        # ── Grid lines ────────────────────────────────────────────────
        for xi in [-total_w/2 + row_w,
                   -total_w/2 + row_w + cell_w]:
            self.add(Line(
                [xi, -total_h/2, z0+0.002],
                [xi,  total_h/2, z0+0.002],
                stroke_color=P["tbl_border"], stroke_width=0.7,
            ))
        for yi in [y_header - cell_h*0.5,
                   y_header - cell_h*1.5,
                   y_header - cell_h*2.5]:
            self.add(Line(
                [-total_w/2, yi, z0+0.002],
                [ total_w/2, yi, z0+0.002],
                stroke_color=P["tbl_border"], stroke_width=0.7,
            ))

        self._p_red_marg = p_red_marg
        self._cell_h     = cell_h
        self._total_h    = total_h
        self._y_header   = y_header

    def _add_cell(
        self,
        x, y, w, h,
        fill_col, text_str, z,
        val_color=None,
    ):
        rect = Rectangle(width=w, height=h,
                         fill_color=fill_col, fill_opacity=1.0,
                         stroke_width=0)
        rect.move_to([x, y, z])
        self.add(rect)
        if text_str:
            t = Text(text_str, font_size=13,
                     color=val_color or P["bar_text"])
            t.move_to([x, y, z+0.002])
            self.add(t)


# ─────────────────────────────────────────────────────────────────────────────
# Bayes formula display
# ─────────────────────────────────────────────────────────────────────────────

class _BayesFormula(VGroup):
    """
    Multi-line Bayes formula with colour-coded terms and
    numerical substitution below.

    Structure:
      Line 1 (symbolic):
        P(A | red) = P(red | A) · P(A)  /  P(red)
                     ───────────────────────────────
      Line 2 (substituted):
        P(A | red) = {p_red_a:.3f} × {p_a:.3f}  /  {p_red_marg:.3f}
      Line 3 (result):
        P(A | red) = {posterior_a:.3f}

    Each term is a separate Text/MathTex so it can be highlighted
    independently.

    Parameters
    ----------
    p_a       : prior P(A)
    p_red_a   : P(red | A)
    p_red_b   : P(red | B)
    """

    def __init__(
        self,
        p_a:     float,
        p_red_a: float,
        p_red_b: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        p_b        = 1 - p_a
        p_red_marg = p_red_a * p_a + p_red_b * p_b
        posterior_a = (p_red_a * p_a) / p_red_marg
        posterior_b = 1 - posterior_a

        z0     = 0.001
        fs_sym = 22
        fs_num = 21

        # ── Background panel ──────────────────────────────────────────
        bg = RoundedRectangle(
            width=8.20, height=3.40,
            corner_radius=0.12,
            fill_color=P["formula_bg"],
            fill_opacity=0.95,
            stroke_color=P["tbl_border"],
            stroke_width=0.9,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        # Left border strip
        strip = RoundedRectangle(
            width=0.08, height=3.32,
            corner_radius=0.04,
            fill_color=P["term_post"],
            fill_opacity=0.85, stroke_width=0,
        )
        strip.move_to([-4.05, 0, z0+0.001])
        self.add(strip)

        # ── Title ─────────────────────────────────────────────────────
        title = Text("Bayes' Theorem — Numerical Update",
                     font_size=17, color=P["subtitle_fg"])
        title.move_to([0, 1.45, z0+0.002])
        self.add(title)

        # ── Symbolic line ─────────────────────────────────────────────
        # Build as VGroup of colour-coded pieces
        sym_line = VGroup()
        pieces_sym = [
            ("P(A | red)",   P["term_post"]),
            ("  =  ",        P["term_eq"]),
            ("P(red | A)",   P["term_like"]),
            ("  ·  ",        P["term_eq"]),
            ("P(A)",         P["term_prior"]),
            ("  /  ",        P["term_eq"]),
            ("P(red)",       P["term_marg"]),
        ]
        x_cur = -3.60
        for txt, col in pieces_sym:
            m = Text(txt, font_size=fs_sym, color=col)
            m.move_to([x_cur + m.width/2, 0.72, z0+0.003])
            sym_line.add(m)
            x_cur += m.width + 0.02
        self.sym_pieces = sym_line
        self.add(sym_line)

        # ── Numeric substitution line ─────────────────────────────────
        num_line = VGroup()
        pieces_num = [
            ("P(A | red)",         P["term_post"]),
            ("  =  ",              P["term_eq"]),
            (f"{p_red_a:.3f}",     P["term_like"]),
            ("  ×  ",              P["term_eq"]),
            (f"{p_a:.3f}",         P["term_prior"]),
            ("  /  ",              P["term_eq"]),
            (f"{p_red_marg:.3f}",  P["term_marg"]),
        ]
        x_cur = -3.60
        for txt, col in pieces_num:
            m = Text(txt, font_size=fs_num, color=col,
                     font="monospace")
            m.move_to([x_cur + m.width/2, 0.10, z0+0.003])
            num_line.add(m)
            x_cur += m.width + 0.02
        self.num_pieces = num_line
        self.add(num_line)

        # Divider
        self.add(Line(
            [-3.80, -0.22, z0+0.002],
            [ 3.80, -0.22, z0+0.002],
            stroke_color=P["tbl_border"], stroke_width=0.7,
        ))

        # ── Result line ───────────────────────────────────────────────
        result_pieces = [
            ("P(A | red)",             P["term_post"]),
            ("  =  ",                  P["term_eq"]),
            (f"{posterior_a:.4f}",     P["term_post"]),
            ("        ",               P["term_eq"]),
            ("P(B | red)",             P["term_b"] if hasattr(P,"term_b") else P["bar_b"]),
            ("  =  ",                  P["term_eq"]),
            (f"{posterior_b:.4f}",     P["bar_b"]),
        ]
        res_line = VGroup()
        x_cur    = -3.80
        for txt, col in result_pieces:
            m = Text(txt, font_size=fs_num + 2,
                     color=col,
                     font="monospace")
            m.move_to([x_cur + m.width/2, -0.62, z0+0.003])
            res_line.add(m)
            x_cur += m.width + 0.02
        self.result_pieces = res_line
        self.add(res_line)

        # Odds line
        odds = posterior_a / max(posterior_b, 1e-9)
        odds_str = f"Posterior odds:  P(A|red) / P(B|red)  =  {odds:.2f}×"
        odds_lbl = Text(odds_str, font_size=16,
                        color=P["term_marg"])
        odds_lbl.move_to([0, -1.20, z0+0.003])
        self.add(odds_lbl)

        self._posterior_a = posterior_a
        self._posterior_b = posterior_b


# ─────────────────────────────────────────────────────────────────────────────
# Beta distribution curve
# ─────────────────────────────────────────────────────────────────────────────

class _BetaCurve(VGroup):
    """
    Beta(α, β) posterior distribution curve.

    Shown as a filled polygon over a [0,1] axis,
    representing the posterior belief about P(red | chosen urn).

    Parameters
    ----------
    alpha_param : float  — Beta α (successes + 1)
    beta_param  : float  — Beta β (failures + 1)
    width       : float  — plot width
    height      : float  — max curve height
    """

    def __init__(
        self,
        alpha_param: float,
        beta_param:  float,
        width:       float = 5.0,
        height:      float = 2.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            from scipy.stats import beta as _beta_dist
            x_raw  = np.linspace(0.001, 0.999, 400)
            y_raw  = _beta_dist.pdf(x_raw, alpha_param, beta_param)
        except ImportError:
            # Manual Beta PDF (simplified)
            import math
            def beta_pdf(x, a, b):
                with np.errstate(all="ignore"):
                    v = (x**(a-1) * (1-x)**(b-1))
                    try:
                        norm = (math.gamma(a) * math.gamma(b)
                                / math.gamma(a + b))
                    except Exception:
                        norm = 1.0
                    return np.where((x>0)&(x<1), v/norm, 0.0)
            x_raw = np.linspace(0.001, 0.999, 400)
            y_raw = beta_pdf(x_raw, alpha_param, beta_param)

        y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
        peak  = float(y_raw.max()) if y_raw.max() > 1e-12 else 1.0
        scale = height / peak
        x_px  = (x_raw - 0.5) * width
        y_px  = y_raw * scale
        z0    = 0.001
        baseline = 0.0

        # ── Fill polygon ──────────────────────────────────────────────
        verts = (
            [[float(x_px[0]), baseline, 0]]
            + [[float(x), float(y), 0] for x,y in zip(x_px,y_px)]
            + [[float(x_px[-1]), baseline, 0]]
        )
        if len(verts) >= 3:
            fill = Polygon(*verts,
                           fill_color=P["beta_fill"],
                           fill_opacity=0.75, stroke_width=0)
            fill.shift([0, 0, z0])
            self.add(fill)

        # ── Ridge spine ───────────────────────────────────────────────
        step  = max(1, len(x_px)//80)
        pts3d = [np.array([float(x_px[i]), float(y_px[i]), z0+0.004])
                 for i in range(0, len(x_px), step) if y_px[i] > 0.01]
        if len(pts3d) >= 2:
            spine = VMobject(stroke_color=P["beta_ridge"],
                             stroke_width=2.2, stroke_opacity=0.92)
            spine.set_points_smoothly(pts3d)
            self.add(spine)

        # ── AO tail darkening ─────────────────────────────────────────
        ao_frac = 0.08
        for x_e, x_i in [(x_px[0], x_px[0]+width*ao_frac),
                          (x_px[-1], x_px[-1]-width*ao_frac)]:
            xl,xr = min(x_e,x_i), max(x_e,x_i)
            mask  = (x_px>=xl)&(x_px<=xr)
            xs,ys = x_px[mask], y_px[mask]
            if len(xs)>=2:
                av = ([[xl,baseline,0]]
                      +[[float(x),float(y),0] for x,y in zip(xs,ys)]
                      +[[xr,baseline,0]])
                self.add(Polygon(*av,
                                 fill_color=BLACK,
                                 fill_opacity=0.45, stroke_width=0))

        # ── Axis ──────────────────────────────────────────────────────
        ax = Line([-width/2-0.2, baseline, z0+0.001],
                  [ width/2+0.2, baseline, z0+0.001],
                  stroke_color=P["axis"], stroke_width=1.5)
        self.add(ax)

        # Ticks at 0, 0.2, 0.4, 0.6, 0.8, 1.0
        for rv in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            xp = (rv - 0.5) * width
            self.add(Line([xp, -0.07, z0+0.001],
                          [xp,  0.07, z0+0.001],
                          stroke_color=P["axis"], stroke_width=1.0))
            self.add(Text(f"{rv:.1f}", font_size=12,
                          color=P["subtitle_fg"]).move_to(
                              [xp, -0.28, z0+0.001]))

        # ── Labels ────────────────────────────────────────────────────
        mode = (alpha_param - 1) / max(alpha_param + beta_param - 2, 1e-6)
        mode = float(np.clip(mode, 0.05, 0.95))
        peak_px = (mode - 0.5) * width
        peak_py = float(y_px[np.argmin(np.abs(x_raw - mode))])

        try:
            title_lbl = MathTex(
                rf"\text{{Beta}}(\alpha={alpha_param:.1f},\;"
                rf"\beta={beta_param:.1f})",
                font_size=22, color=P["beta_ridge"],
            )
        except Exception:
            title_lbl = Text(
                f"Beta(α={alpha_param:.1f}, β={beta_param:.1f})",
                font_size=18, color=P["beta_ridge"],
            )
        title_lbl.move_to([0, peak_py + 0.45, z0+0.005])
        self.add(title_lbl)

        x_label = Text("θ = P(red ball from chosen urn)",
                       font_size=14, color=P["subtitle_fg"])
        x_label.move_to([0, baseline - 0.55, z0+0.002])
        self.add(x_label)


# ─────────────────────────────────────────────────────────────────────────────
# Posterior history trail
# ─────────────────────────────────────────────────────────────────────────────

class _PosteriorTrail(VGroup):
    """
    A vertical stack of small probability bars showing the evolution
    of the posterior after each draw.

    Each row: draw index, observation icon, posterior bar.
    """

    def __init__(
        self,
        history: list[tuple[str, float]],   # [(obs_label, p_a), ...]
        bar_w:   float = 3.2,
        bar_h:   float = 0.30,
        spacing: float = 0.48,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        for i, (obs_lbl, p_a) in enumerate(history):
            y_row = -i * spacing
            p_b   = 1 - p_a

            # Row label
            lbl = Text(f"Draw {i+1}: {obs_lbl}",
                       font_size=13, color=P["subtitle_fg"])
            lbl.move_to([-bar_w/2 - 0.88, y_row, z0+0.002])
            self.add(lbl)

            # Mini prob bar
            w_a = p_a * bar_w
            w_b = p_b * bar_w

            bg = Rectangle(width=bar_w+0.04, height=bar_h+0.04,
                           fill_color=P["bar_bg"],
                           fill_opacity=1.0, stroke_width=0)
            bg.move_to([0, y_row, z0])
            self.add(bg)

            if w_a > 0.01:
                sa = Rectangle(width=w_a-0.01, height=bar_h,
                                fill_color=P["bar_a"],
                                fill_opacity=0.90, stroke_width=0)
                sa.move_to([-bar_w/2 + w_a/2, y_row, z0+0.001])
                self.add(sa)
            if w_b > 0.01:
                sb = Rectangle(width=w_b-0.01, height=bar_h,
                                fill_color=P["bar_b"],
                                fill_opacity=0.90, stroke_width=0)
                sb.move_to([bar_w/2 - w_b/2, y_row, z0+0.001])
                self.add(sb)

            # Probability labels
            ta = Text(f"A:{p_a:.2f}", font_size=11, color=P["bar_text"])
            tb = Text(f"B:{p_b:.2f}", font_size=11, color=P["bar_text"])
            ta.move_to([-bar_w/2 + w_a/2, y_row, z0+0.003])
            tb.move_to([bar_w/2 - w_b/2, y_row, z0+0.003])
            self.add(ta, tb)


# ─────────────────────────────────────────────────────────────────────────────
# Summary panel
# ─────────────────────────────────────────────────────────────────────────────

class _SummaryPanel(VGroup):
    """
    Final summary: Prior → Likelihood → Posterior, one row each,
    with colour-coded term labels and a compact visual.
    """

    def __init__(
        self,
        p_a_prior:     float,
        p_red_a:       float,
        p_red_b:       float,
        p_a_posterior: float,
        panel_w: float = 7.50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        p_b_prior = 1 - p_a_prior
        p_red_marg = p_red_a*p_a_prior + p_red_b*p_b_prior
        p_b_post   = 1 - p_a_posterior
        z0 = 0.001

        rows = [
            ("Prior",      f"P(A) = {p_a_prior:.3f}",
             f"P(B) = {p_b_prior:.3f}",
             P["term_prior"]),
            ("Likelihood", f"P(red|A) = {p_red_a:.3f}",
             f"P(red|B) = {p_red_b:.3f}",
             P["term_like"]),
            ("Marginal",   f"P(red) = {p_red_marg:.3f}",
             "",
             P["term_marg"]),
            ("Posterior",  f"P(A|red) = {p_a_posterior:.3f}",
             f"P(B|red) = {p_b_post:.3f}",
             P["term_post"]),
        ]
        row_h   = 0.62
        total_h = len(rows) * row_h + 0.60

        bg = RoundedRectangle(
            width=panel_w+0.24, height=total_h+0.24,
            corner_radius=0.12,
            fill_color=P["summary_bg"],
            fill_opacity=0.95,
            stroke_color=P["summary_border"],
            stroke_width=0.9,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        title = Text("Bayesian Update — Summary",
                     font_size=20, color=P["title_fg"])
        title.move_to([0, total_h/2 - 0.30, z0+0.002])
        self.add(title)

        self.add(Line(
            [-panel_w/2+0.10, total_h/2-0.55, z0+0.002],
            [ panel_w/2-0.10, total_h/2-0.55, z0+0.002],
            stroke_color=P["summary_border"], stroke_width=0.7,
        ))

        for i, (term, val_a, val_b, col) in enumerate(rows):
            yr = total_h/2 - 0.78 - i * row_h

            # Left border dot
            dot = Circle(radius=0.06,
                         fill_color=col,
                         fill_opacity=1.0, stroke_width=0)
            dot.move_to([-panel_w/2 + 0.22, yr, z0+0.003])
            self.add(dot)

            term_lbl = Text(term, font_size=15, color=col)
            term_lbl.move_to([-panel_w/2 + 0.55 + term_lbl.width/2,
                               yr, z0+0.003])
            self.add(term_lbl)

            val_a_lbl = Text(val_a, font_size=14,
                             color=P["bar_a"],
                             font="monospace")
            val_a_lbl.move_to([0.50, yr, z0+0.003])
            self.add(val_a_lbl)

            if val_b:
                val_b_lbl = Text(val_b, font_size=14,
                                 color=P["bar_b"],
                                 font="monospace")
                val_b_lbl.move_to([panel_w/2 - 0.10 - val_b_lbl.width/2,
                                   yr, z0+0.003])
                self.add(val_b_lbl)

            # Arrow to next row
            if i < len(rows) - 1:
                arrow = Arrow(
                    start=[panel_w/2 - 0.85, yr - 0.18, z0+0.002],
                    end  =[panel_w/2 - 0.85, yr - row_h + 0.18,
                           z0+0.002],
                    stroke_color=P["summary_arrow"],
                    stroke_width=1.2,
                    tip_length=0.12, buff=0,
                )
                self.add(arrow)


# ─────────────────────────────────────────────────────────────────────────────
# Drawn ball indicator
# ─────────────────────────────────────────────────────────────────────────────

def _drawn_ball_display(
    color:  str,
    label:  str = "RED",
    radius: float = 0.45,
) -> VGroup:
    """
    Large ball display for "we drew a {color} ball" reveal.
    Layers: shadow, body, specular, label.
    """
    g        = VGroup()
    dark_col = interpolate_color(color, BLACK, 0.55)
    lite_col = interpolate_color(color, WHITE, 0.50)

    shadow = Circle(radius=radius*1.08,
                    fill_color=dark_col,
                    fill_opacity=0.45, stroke_width=0)
    shadow.shift([radius*0.06, -radius*0.06, -0.001])
    g.add(shadow)

    body = Circle(radius=radius,
                  fill_color=color,
                  fill_opacity=1.0, stroke_width=0)
    g.add(body)

    spec = Circle(radius=radius*0.32,
                  fill_color=lite_col,
                  fill_opacity=0.72, stroke_width=0)
    spec.scale([1.35, 0.75, 1])
    spec.move_to([-radius*0.26, radius*0.28, 0.003])
    g.add(spec)

    rim  = Annulus(inner_radius=radius*0.28,
                   outer_radius=radius*0.40,
                   fill_color=lite_col,
                   fill_opacity=0.38, stroke_width=0)
    rim.move_to([-radius*0.24, radius*0.26, 0.002])
    g.add(rim)

    lbl = Text(label, font_size=int(radius*52),
               color=WHITE if _perceived_lum(color) < 0.5 else BLACK)
    lbl.move_to([0, 0, 0.004])
    g.add(lbl)

    return g


def _perceived_lum(hex_col: str) -> float:
    try:
        rgb = np.array(color_to_rgb(hex_col))
        return 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Counting animation helper
# ─────────────────────────────────────────────────────────────────────────────

class _CountUp(Animation):
    """
    Animates a Text label counting from old_val to new_val.
    Replaces the text in place.

    Parameters
    ----------
    mob      : Text mob to update
    old_val  : float
    new_val  : float
    fmt      : format string (default "{:.3f}")
    """

    def __init__(
        self,
        mob:     Text,
        old_val: float,
        new_val: float,
        fmt:     str = "{:.3f}",
        **kwargs,
    ):
        self.old_val = old_val
        self.new_val = new_val
        self.fmt     = fmt
        kwargs.setdefault("run_time", 0.80)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(mob, **kwargs)

    def interpolate_mobject(self, alpha: float):
        val = self.old_val + (self.new_val - self.old_val) * alpha
        self.mobject.become(
            Text(self.fmt.format(val),
                 font_size=self.mobject.font_size,
                 color=self.mobject.get_color(),
                 font="monospace")
            .move_to(self.mobject.get_center())
        )


# ─────────────────────────────────────────────────────────────────────────────
# The full demo scene
# ─────────────────────────────────────────────────────────────────────────────

class BayesDemo(Scene):
    """
    Full cinematic Bayes' theorem demonstration.

    Run with:
        manim -pql demo_bayes.py BayesDemo

    For high quality:
        manim -pqh demo_bayes.py BayesDemo
    """

    # ── Problem parameters ────────────────────────────────────────────
    N_RED_A   = 3
    N_BLUE_A  = 2
    N_RED_B   = 1
    N_BLUE_B  = 4
    P_PRIOR_A = 0.5
    P_RED_A   = 3 / 5    # 0.600
    P_RED_B   = 1 / 5    # 0.200

    def construct(self):

        # ── ACT 0: Cold open ──────────────────────────────────────────
        self._act0_cold_open()

        # ── ACT 1: Two urns setup ─────────────────────────────────────
        urn_a, urn_b = self._act1_setup()

        # ── ACT 2: Prior probability bar ──────────────────────────────
        prob_bar = self._act2_prior(urn_a, urn_b)

        # ── ACT 3: Draw a red ball ────────────────────────────────────
        ball_display = self._act3_draw(urn_a, urn_b, prob_bar)

        # ── ACT 4: Likelihood table ───────────────────────────────────
        like_table = self._act4_likelihood(urn_a, urn_b,
                                           prob_bar, ball_display)

        # ── ACT 5: Bayes formula — first update ───────────────────────
        prob_bar, formula = self._act5_bayes_formula(
            urn_a, urn_b, prob_bar, like_table, ball_display
        )

        # ── ACT 6: Posterior bar ──────────────────────────────────────
        prob_bar = self._act6_posterior_bar(
            urn_a, urn_b, prob_bar, formula
        )

        # ── ACT 7: Sequential updates ─────────────────────────────────
        prob_bar, trail = self._act7_sequential(
            urn_a, urn_b, prob_bar, like_table, ball_display
        )

        # ── ACT 8: Beta distribution ──────────────────────────────────
        self._act8_beta_curve(urn_a, urn_b, prob_bar, trail)

        # ── ACT 9: Decision and summary ───────────────────────────────
        self._act9_decision(urn_a, urn_b)

    # ─────────────────────────────────────────────────────────────────
    # Act 0 — Cold open
    # ─────────────────────────────────────────────────────────────────

    def _act0_cold_open(self):
        # Full-screen dark background
        bg = Rectangle(width=16, height=9,
                       fill_color=P["bg"],
                       fill_opacity=1.0, stroke_width=0)
        self.add(bg)

        # Mystery urn in darkness
        mystery_urn = _UrnSchematic(
            label="?", n_red=0, n_blue=0,
            fill_color="#303038",
            width=1.4, height=2.0,
        )
        mystery_urn.move_to([0, 0.2, 0])

        q_mark = Text("?", font_size=68,
                      color=P["subtitle_fg"])
        q_mark.move_to([0, 1.70, 0])

        self.play(FadeIn(mystery_urn, shift=UP*0.3), run_time=1.2)
        self.play(FadeIn(q_mark, shift=UP*0.15,
                         rate_func=rate_functions.ease_out_back),
                  run_time=0.9)
        self.wait(0.4)

        # Title question
        question = Text(
            "Which urn did this ball come from?",
            font_size=32, color=P["title_fg"],
        )
        question.move_to([0, -1.60, 0])
        self.play(Write(question, run_time=1.4))
        self.wait(0.8)

        # Drawn ball floats up
        ball = _drawn_ball_display(P["red_ball"], "RED", radius=0.32)
        ball.move_to([0, -0.50, 0])
        ball.set_opacity(0)
        self.play(FadeIn(ball, shift=UP*0.30, run_time=0.8))
        self.wait(0.6)

        self.play(
            FadeOut(mystery_urn),
            FadeOut(q_mark),
            FadeOut(question),
            FadeOut(ball),
            FadeOut(bg),
            run_time=0.8,
        )

    # ─────────────────────────────────────────────────────────────────
    # Act 1 — Two urns setup
    # ─────────────────────────────────────────────────────────────────

    def _act1_setup(self):
        # Section label
        act_lbl = _section_label("Act 1 — The Setup")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Urn A — terracotta
        urn_a = _UrnSchematic(
            label="Urn A",
            n_red=self.N_RED_A, n_blue=self.N_BLUE_A,
            fill_color=P["urn_a"],
            prior=self.P_PRIOR_A,
        )
        urn_a.move_to([-3.40, 0.40, 0])

        # Urn B — cobalt
        urn_b = _UrnSchematic(
            label="Urn B",
            n_red=self.N_RED_B, n_blue=self.N_BLUE_B,
            fill_color=P["urn_b"],
            prior=1 - self.P_PRIOR_A,
        )
        urn_b.move_to([3.40, 0.40, 0])

        self.play(
            FadeIn(urn_a, shift=LEFT*0.4),
            FadeIn(urn_b, shift=RIGHT*0.4),
            run_time=1.2,
        )

        # Description labels
        desc_a = Text(
            f"{self.N_RED_A} red,  {self.N_BLUE_A} blue",
            font_size=18, color=P["subtitle_fg"],
        )
        desc_b = Text(
            f"{self.N_RED_B} red,  {self.N_BLUE_B} blue",
            font_size=18, color=P["subtitle_fg"],
        )
        desc_a.next_to(urn_a, DOWN, buff=1.10)
        desc_b.next_to(urn_b, DOWN, buff=1.10)
        self.play(FadeIn(desc_a), FadeIn(desc_b), run_time=0.7)

        # "One urn chosen at random" label
        chosen_lbl = Text(
            "One urn was chosen at random. We don't know which.",
            font_size=22, color=P["title_fg"],
        )
        chosen_lbl.move_to([0, -2.60, 0])
        self.play(Write(chosen_lbl, run_time=1.5))
        self.wait(0.8)

        self.play(FadeOut(act_lbl), FadeOut(chosen_lbl),
                  FadeOut(desc_a), FadeOut(desc_b), run_time=0.5)

        self._urn_a = urn_a
        self._urn_b = urn_b
        return urn_a, urn_b

    # ─────────────────────────────────────────────────────────────────
    # Act 2 — Prior
    # ─────────────────────────────────────────────────────────────────

    def _act2_prior(self, urn_a, urn_b):
        act_lbl = _section_label("Act 2 — Prior Probability")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Shrink urns to top
        self.play(
            urn_a.animate.scale(0.72).move_to([-4.80, 1.60, 0]),
            urn_b.animate.scale(0.72).move_to([ 4.80, 1.60, 0]),
            run_time=0.90,
        )

        # Prior bar
        prob_bar = _ProbBar(
            p_a=self.P_PRIOR_A,
            width=6.0, height=0.55,
            title="Prior:  P(A) = P(B) = 0.500",
        )
        prob_bar.move_to([0, 0.20, 0])
        prob_bar.set_opacity(0)

        self.play(FadeIn(prob_bar, shift=UP*0.15,
                         run_time=1.0))

        # Formula badge
        try:
            prior_form = MathTex(
                r"P(A) = P(B) = \frac{1}{2} = 0.500"
                r"\quad\text{(equal prior)}",
                font_size=24, color=P["term_prior"],
            )
        except Exception:
            prior_form = Text(
                "P(A) = P(B) = 1/2 = 0.500  (equal prior)",
                font_size=20, color=P["term_prior"],
            )
        prior_form.move_to([0, -0.70, 0])
        self.play(Write(prior_form, run_time=1.2))
        self.wait(0.7)

        # Indicate both halves
        self.play(
            Indicate(prob_bar, color=P["term_prior"],
                     scale_factor=1.04, run_time=0.7)
        )
        self.wait(0.3)

        self.play(FadeOut(act_lbl), FadeOut(prior_form), run_time=0.5)
        return prob_bar

    # ─────────────────────────────────────────────────────────────────
    # Act 3 — Draw a red ball
    # ─────────────────────────────────────────────────────────────────

    def _act3_draw(self, urn_a, urn_b, prob_bar):
        act_lbl = _section_label("Act 3 — The Draw")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Rumble effect on both urns (we don't know which one)
        self.play(
            urn_a.animate.shift(LEFT*0.08),
            run_time=0.12,
        )
        self.play(
            urn_a.animate.shift(RIGHT*0.16),
            run_time=0.12,
        )
        self.play(
            urn_a.animate.shift(LEFT*0.08),
            run_time=0.10,
        )

        # Red ball emerges from bottom-centre
        ball_display = _drawn_ball_display(
            P["red_ball"], "RED", radius=0.42
        )
        ball_display.move_to([0, -0.50, 0])
        ball_display.set_opacity(0)

        self.play(
            FadeIn(ball_display, shift=UP*0.40,
                   rate_func=rate_functions.ease_out_back),
            run_time=1.0,
        )

        reveal_lbl = Text(
            "We drew a RED ball.",
            font_size=28, color=P["red_ball"],
        )
        reveal_lbl.move_to([0, -1.45, 0])
        self.play(Write(reveal_lbl, run_time=1.0))

        question = Text(
            "Which urn is more likely to have produced it?",
            font_size=20, color=P["title_fg"],
        )
        question.move_to([0, -2.10, 0])
        self.play(FadeIn(question, shift=UP*0.10,
                         run_time=0.8))
        self.wait(1.0)

        self.play(
            FadeOut(act_lbl),
            FadeOut(reveal_lbl),
            FadeOut(question),
            run_time=0.5,
        )
        return ball_display

    # ─────────────────────────────────────────────────────────────────
    # Act 4 — Likelihood table
    # ─────────────────────────────────────────────────────────────────

    def _act4_likelihood(self, urn_a, urn_b, prob_bar, ball_display):
        act_lbl = _section_label("Act 4 — Likelihood")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Move ball to side
        self.play(
            ball_display.animate.scale(0.65).move_to([4.60, -0.20, 0]),
            run_time=0.7,
        )

        # Build likelihood table
        like_table = _LikelihoodTable(
            p_red_a=self.P_RED_A,
            p_red_b=self.P_RED_B,
            p_a=self.P_PRIOR_A,
        )
        like_table.scale(0.88)
        like_table.move_to([0, -0.90, 0])
        like_table.set_opacity(0)

        self.play(FadeIn(like_table, shift=UP*0.15,
                         run_time=1.1))

        # Highlight the red row
        self.wait(0.4)
        red_row_hint = Text(
            "← P(red | Urn A) is 3× larger than P(red | Urn B)",
            font_size=17, color=P["tbl_val_red"],
        )
        red_row_hint.move_to([0, 0.95, 0])
        self.play(Write(red_row_hint, run_time=1.0))
        self.wait(0.8)

        # Highlight marginal
        marg_hint = Text(
            "Marginal P(red) = 0.600×0.5 + 0.200×0.5 = 0.400",
            font_size=16, color=P["tbl_val_marg"],
        )
        marg_hint.move_to([0, 1.38, 0])
        self.play(FadeIn(marg_hint, shift=UP*0.08,
                         run_time=0.8))
        self.wait(0.8)

        self.play(
            FadeOut(act_lbl),
            FadeOut(red_row_hint),
            FadeOut(marg_hint),
            run_time=0.5,
        )
        return like_table

    # ─────────────────────────────────────────────────────────────────
    # Act 5 — Bayes formula
    # ─────────────────────────────────────────────────────────────────

    def _act5_bayes_formula(
        self, urn_a, urn_b, prob_bar, like_table, ball_display
    ):
        act_lbl = _section_label("Act 5 — Bayes' Theorem")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Slide everything up to make room
        self.play(
            like_table.animate.scale(0.75).move_to([-3.60, -2.30, 0]),
            prob_bar.animate.move_to([0, 2.15, 0]),
            urn_a.animate.move_to([-5.80, 2.00, 0]),
            urn_b.animate.move_to([ 5.80, 2.00, 0]),
            run_time=0.80,
        )

        # Build Bayes formula
        formula = _BayesFormula(
            p_a=self.P_PRIOR_A,
            p_red_a=self.P_RED_A,
            p_red_b=self.P_RED_B,
        )
        formula.scale(0.88)
        formula.move_to([0.80, -0.10, 0])
        formula.set_opacity(0)

        self.play(FadeIn(formula, shift=UP*0.15,
                         run_time=1.2))
        self.wait(0.5)

        # Highlight terms one at a time
        self.play(
            Indicate(formula.sym_pieces[2],   # P(red|A)
                     color=P["term_like"],
                     scale_factor=1.15, run_time=0.7),
        )
        self.wait(0.2)
        self.play(
            Indicate(formula.sym_pieces[4],   # P(A)
                     color=P["term_prior"],
                     scale_factor=1.15, run_time=0.7),
        )
        self.wait(0.2)
        self.play(
            Indicate(formula.sym_pieces[6],   # P(red)
                     color=P["term_marg"],
                     scale_factor=1.15, run_time=0.7),
        )
        self.wait(0.3)

        # Highlight the result
        self.play(
            Flash(formula.result_pieces[2].get_center(),
                  color=P["term_post"],
                  line_length=0.16, num_lines=10,
                  run_time=0.7),
            Indicate(formula.result_pieces[2],
                     color=P["term_post"],
                     scale_factor=1.20, run_time=0.7),
        )
        self.wait(0.7)

        # Arrow from formula result to prob bar
        post_val = formula._posterior_a
        arr = Arrow(
            start=formula.result_pieces[2].get_center()
                  + np.array([0, 0.30, 0]),
            end  =prob_bar.get_center() + np.array([0, -0.40, 0]),
            stroke_color=P["term_post"],
            stroke_width=1.8, tip_length=0.16, buff=0.08,
        )
        self.play(Create(arr, run_time=0.6))
        self.wait(0.3)

        self.play(FadeOut(act_lbl), FadeOut(arr), run_time=0.5)
        return prob_bar, formula

    # ─────────────────────────────────────────────────────────────────
    # Act 6 — Posterior bar (first update)
    # ─────────────────────────────────────────────────────────────────

    def _act6_posterior_bar(self, urn_a, urn_b, prob_bar, formula):
        act_lbl = _section_label("Act 6 — Posterior Update")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        post_a = formula._posterior_a   # 0.750

        # Morph the prior bar into posterior
        new_bar = _ProbBar(
            p_a=post_a,
            width=6.0, height=0.55,
            title=f"Posterior:  P(A|red) = {post_a:.3f}   P(B|red) = {1-post_a:.3f}",
        )
        new_bar.move_to(prob_bar.get_center())

        self.play(
            ReplacementTransform(prob_bar, new_bar, run_time=1.4)
        )

        # Flash the dominant side
        self.play(
            Indicate(new_bar, color=P["bar_a"],
                     scale_factor=1.04, run_time=0.7)
        )
        self.wait(0.4)

        # Update urn probability badges
        new_badge_a = urn_a.update_prior_badge(post_a)
        new_badge_b = urn_b.update_prior_badge(1 - post_a)
        self.play(
            FadeIn(new_badge_a, shift=UP*0.06),
            FadeIn(new_badge_b, shift=UP*0.06),
            run_time=0.6,
        )
        self.wait(0.5)

        self.play(FadeOut(act_lbl), run_time=0.4)
        return new_bar

    # ─────────────────────────────────────────────────────────────────
    # Act 7 — Sequential updates
    # ─────────────────────────────────────────────────────────────────

    def _act7_sequential(
        self, urn_a, urn_b, prob_bar, like_table, ball_display
    ):
        act_lbl = _section_label("Act 7 — Sequential Updates")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Starting posterior after draw 1 (red) was P(A|red) = 0.75
        p_red_a = self.P_RED_A
        p_red_b = self.P_RED_B

        history = [("red  ←",  0.750)]   # after draw 1

        # We'll track P(A) as the running posterior
        p_a_current = 0.750

        draw_configs = [
            # (ball_color, ball_label, is_red)
            (P["red_ball"],  "RED",  True),
            (P["blue_ball"], "BLUE", False),
        ]

        for draw_num, (bcol, blbl, is_red) in enumerate(draw_configs,
                                                         start=2):
            # Show drawn ball
            ball = _drawn_ball_display(bcol, blbl, radius=0.30)
            ball.move_to([4.80, -0.20, 0])
            self.play(FadeIn(ball, shift=LEFT*0.20, run_time=0.6))

            # Compute updated posterior
            p_like_a = p_red_a if is_red else (1 - p_red_a)
            p_like_b = p_red_b if is_red else (1 - p_red_b)
            p_marg   = p_like_a * p_a_current + p_like_b * (1 - p_a_current)
            p_a_new  = (p_like_a * p_a_current) / max(p_marg, 1e-9)

            # Highlight relevant likelihood row in table
            self.play(
                Indicate(like_table,
                         color=bcol,
                         scale_factor=1.03,
                         run_time=0.6),
            )

            # Morph probability bar
            obs_str  = blbl.lower()
            bar_title = (
                f"After draw {draw_num} ({blbl}):  "
                f"P(A) = {p_a_new:.3f}   "
                f"P(B) = {1-p_a_new:.3f}"
            )
            new_bar = _ProbBar(
                p_a=p_a_new,
                width=6.0, height=0.55,
                title=bar_title,
            )
            new_bar.move_to(prob_bar.get_center())
            self.play(
                ReplacementTransform(prob_bar, new_bar, run_time=1.1)
            )
            prob_bar = new_bar

            history.append((f"{blbl.lower()}  ←", p_a_new))
            p_a_current = p_a_new

            self.play(FadeOut(ball), run_time=0.35)
            self.wait(0.35)

        # Build posterior trail
        trail = _PosteriorTrail(
            history=history,
            bar_w=3.0, bar_h=0.28, spacing=0.46,
        )
        trail.move_to([-1.50, -1.70, 0])
        trail.set_opacity(0)
        self.play(FadeIn(trail, shift=UP*0.10, run_time=0.9))
        self.wait(0.7)

        self.play(FadeOut(act_lbl), run_time=0.4)
        return prob_bar, trail

    # ─────────────────────────────────────────────────────────────────
    # Act 8 — Beta distribution posterior
    # ─────────────────────────────────────────────────────────────────

    def _act8_beta_curve(self, urn_a, urn_b, prob_bar, trail):
        act_lbl = _section_label("Act 8 — Continuous Posterior")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Transition: move prob bar + trail out
        self.play(
            prob_bar.animate.shift(DOWN*5).set_opacity(0),
            trail.animate.shift(DOWN*5).set_opacity(0),
            run_time=0.7,
        )
        self.remove(prob_bar, trail)

        # Beta-Binomial model:
        # Prior: Beta(1, 1) = Uniform
        # After 2 reds and 1 blue from chosen urn: Beta(3, 2)
        # (counts: successes = 2 red, failures = 1 blue → α=1+2=3, β=1+1=2)
        alpha_post = 3.0   # prior_α + successes
        beta_post  = 2.0   # prior_β + failures

        intro_lbl = Text(
            "Beta-Binomial model: 3 draws (2 red, 1 blue)\n"
            "Prior: Beta(1,1)  →  Posterior: Beta(3,2)",
            font_size=20, color=P["subtitle_fg"],
        )
        intro_lbl.move_to([0, 2.20, 0])
        self.play(Write(intro_lbl, run_time=1.2))

        beta_curve = _BetaCurve(
            alpha_param=alpha_post,
            beta_param=beta_post,
            width=6.0, height=3.0,
        )
        beta_curve.move_to([0, -0.20, 0])
        beta_curve.set_opacity(0)

        self.play(FadeIn(beta_curve, shift=UP*0.15, run_time=1.3))
        self.wait(0.5)

        # Mark the mode
        mode = (alpha_post - 1) / (alpha_post + beta_post - 2)
        mode_px = (mode - 0.5) * 6.0
        mode_line = DashedLine(
            start=[mode_px, -1.65, 0.010],
            end  =[mode_px,  1.20, 0.010],
            stroke_color=P["term_post"],
            stroke_width=2.0,
            dash_length=0.11,
        )
        mode_lbl = Text(
            f"Mode = {mode:.2f}",
            font_size=17, color=P["term_post"],
        )
        mode_lbl.move_to([mode_px + 0.90, 1.42, 0.010])
        self.play(
            Create(mode_line, run_time=0.6),
            FadeIn(mode_lbl, run_time=0.5),
        )
        self.wait(0.8)

        self.play(
            FadeOut(act_lbl),
            FadeOut(intro_lbl),
            FadeOut(mode_line),
            FadeOut(mode_lbl),
            run_time=0.5,
        )

        # Fade out beta curve for act 9
        self.play(FadeOut(beta_curve), run_time=0.5)

    # ─────────────────────────────────────────────────────────────────
    # Act 9 — Decision and summary
    # ─────────────────────────────────────────────────────────────────

    def _act9_decision(self, urn_a, urn_b):
        act_lbl = _section_label("Act 9 — Decision")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Bring urns back centre-stage
        self.play(
            urn_a.animate.scale(1.0 / 0.72).move_to([-3.40, 0.50, 0]),
            urn_b.animate.scale(1.0 / 0.72).move_to([ 3.40, 0.50, 0]),
            run_time=0.80,
        )

        # Final posterior bar
        # After 2 red + 1 blue draws:
        # Draw 1 red:  P(A)=0.5   → P(A|r1) = 0.600×0.5/0.4    = 0.750
        # Draw 2 red:  P(A)=0.75  → P(A|r2) = 0.600×0.75/0.500 ≈ 0.900
        # Draw 3 blue: P(A)=0.9   → P(A|b3) = 0.400×0.9/0.380  ≈ 0.947
        final_p_a = (self.P_RED_A**2 * (1-self.P_RED_A)
                     * self.P_PRIOR_A)
        final_p_b = (self.P_RED_B**2 * (1-self.P_RED_B)
                     * (1-self.P_PRIOR_A))
        norm      = final_p_a + final_p_b
        final_p_a /= norm
        final_p_b = 1 - final_p_a

        final_bar = _ProbBar(
            p_a=final_p_a, width=5.0, height=0.50,
            title=(f"Final posterior:  "
                   f"P(A|data) = {final_p_a:.3f}   "
                   f"P(B|data) = {final_p_b:.3f}"),
        )
        final_bar.move_to([0, -1.60, 0])
        self.play(FadeIn(final_bar, shift=UP*0.12, run_time=0.9))
        self.wait(0.4)

        # Odds statement
        odds = final_p_a / max(final_p_b, 1e-9)
        verdict = Text(
            f"Urn A is {odds:.1f}× more likely than Urn B.",
            font_size=27, color=P["decision_fg"],
        )
        verdict.move_to([0, -2.55, 0])
        self.play(Write(verdict, run_time=1.1))

        # Flash Urn A
        self.play(
            Indicate(urn_a, color=P["decision_fg"],
                     scale_factor=1.07, run_time=0.80),
            Flash(urn_a.get_center(),
                  color=P["decision_fg"],
                  line_length=0.22,
                  num_lines=12,
                  run_time=0.80),
        )

        # Decision badge on Urn A
        badge_bg = RoundedRectangle(
            width=2.60, height=0.55,
            corner_radius=0.10,
            fill_color=P["decision_bg"],
            fill_opacity=0.92,
            stroke_color=P["decision_fg"],
            stroke_width=1.8,
        )
        badge_bg.move_to([-3.40, 1.92, 0])
        badge_txt = Text("MOST LIKELY", font_size=22,
                         color=P["decision_fg"])
        badge_txt.move_to(badge_bg.get_center() + np.array([0,0,0.001]))
        badge = VGroup(badge_bg, badge_txt)
        badge.rotate(0.06)
        self.play(
            GrowFromCenter(badge,
                           rate_func=rate_functions.ease_out_back,
                           run_time=0.75),
        )
        self.wait(0.6)

        # Summary panel
        self.play(
            FadeOut(act_lbl),
            FadeOut(verdict),
            FadeOut(final_bar),
            FadeOut(badge),
            urn_a.animate.scale(0.60).move_to([-5.50, 1.80, 0]),
            urn_b.animate.scale(0.60).move_to([ 5.50, 1.80, 0]),
            run_time=0.70,
        )

        summary = _SummaryPanel(
            p_a_prior=self.P_PRIOR_A,
            p_red_a=self.P_RED_A,
            p_red_b=self.P_RED_B,
            p_a_posterior=final_p_a,
            panel_w=7.50,
        )
        summary.move_to([0, -0.20, 0])
        summary.set_opacity(0)
        self.play(FadeIn(summary, shift=UP*0.15, run_time=1.2))
        self.wait(2.5)

        # Closing fade
        self.play(
            FadeOut(summary),
            FadeOut(urn_a),
            FadeOut(urn_b),
            run_time=1.0,
        )

        # End card
        end_card = _title_card(
            "Bayes' Theorem",
            "Prior × Likelihood  →  Posterior",
            main_fs=42, sub_fs=24,
        )
        self.play(FadeIn(end_card, run_time=1.0))
        self.wait(2.0)
        self.play(FadeOut(end_card, run_time=0.8))