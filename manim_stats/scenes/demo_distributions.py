"""
manim_stats/scenes/demo_distributions.py
==========================================
DistributionsDemo — A complete, cinematic probability distributions
showcase scene for Manim statistics animations.

Story arc (7 acts)
------------------
Act 0  Distribution family tree
       Animated branching tree from "Probability Distribution" root:
         → Discrete  (Bernoulli, Binomial, Poisson, Geometric, Hypergeometric)
         → Continuous (Normal, t, Chi², F, Exponential, Gamma, Beta, Uniform,
                       Cauchy, Pareto, Log-Normal, Weibull)
       Each node pulses in with its canonical curve thumbnail.

Act 1  Discrete distributions gallery
       Five panels in sequence:
         Bernoulli, Binomial, Poisson, Geometric, Negative Binomial
       Each panel: PMF bars (3-D depth), CDF step function overlay,
       E[X] / Var[X] badges, parameter badge, real-world application tag.
       Parameter animation: p sweeps 0.1→0.9 for Bernoulli/Binomial,
       λ sweeps 1→8 for Poisson.

Act 2  Continuous bell-curve family
       Normal, Student-t (df=1,5,30), Chi-squared, F-distribution.
       Side-by-side PDF + CDF panels.
       Shaded area annotation: "P(a ≤ X ≤ b) = {val}"
       t-distribution tails visibly heavier than Normal.

Act 3  Skewed and bounded distributions
       Exponential, Gamma, Beta, Log-Normal, Weibull.
       Shape parameter animation: Gamma k sweeps 1→5.
       Beta α,β animation shows shift from U-shaped to bell-shaped.

Act 4  Heavy tails
       Zoom into the right tail region.
       Overlay: Normal, Exponential, Pareto, Cauchy.
       Log-scale y-axis reveals the dramatic tail differences.
       "Extreme events" annotation.

Act 5  Distribution relationships
       Six animated arrows with labels:
         Binomial(n,p) → Normal(np, np(1-p))   [CLT, n large]
         Binomial(n,p) → Poisson(λ=np)         [n→∞, p→0]
         Poisson(λ)    → Normal(λ,λ)            [λ large]
         Gamma(1,λ)    → Exponential(λ)         [special case]
         Gamma(k/2,2)  → Chi²(k)               [parameterisation]
         Beta(1,1)     → Uniform(0,1)           [special case]
       Each relationship animates: left distribution morphs into right.

Act 6  Parameter sweep gallery
       3×2 grid of live-updating distribution panels.
       Each sweeps its key parameter while the curve morphs.
       Colour gradient shows how shape changes with parameter.

Act 7  Comparison overlay + closing
       All continuous distributions overlaid on one axis (normalised).
       Colour-coded legend.
       Closing: "Each distribution tells a story about randomness."

Components (all self-contained)
--------------------------------
  _DistNode         — tree node with thumbnail curve + label
  _DistTree         — full branching family tree
  _PMFPanel         — discrete distribution panel (PMF bars + CDF)
  _PDFPanel         — continuous distribution panel (PDF + CDF)
  _StatsPanel       — E[X], Var[X], skewness, MGF sidebar
  _AppTag           — real-world application tag badge
  _RelArrow         — distribution relationship arrow
  _ParamSweep       — single distribution morphing under parameter change
  _TailComparison   — log-scale tail overlay

Dependencies
------------
  manim (CE or GL), numpy, scipy (optional)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable, Sequence
from functools import partial

from manim import (
    Scene,
    VGroup,
    Rectangle, RoundedRectangle,
    Circle, Annulus, Dot, Polygon, Line, DashedLine, Arrow,
    Text, MathTex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut, GrowFromCenter,
    Create, Write,
    Indicate, Flash,
    Transform, ReplacementTransform,
    ValueTracker,
    interpolate_color, color_to_rgb,
    always_redraw,
    BLACK, WHITE,
    GREY,   GREY_B,   GREY_C,
    RED,    RED_B,    RED_C,
    GREEN,  GREEN_B,  GREEN_C,
    BLUE,   BLUE_B,   BLUE_C,   BLUE_D,
    YELLOW, YELLOW_A,
    ORANGE, TEAL, TEAL_B,
    GOLD,   GOLD_B,   GOLD_C,
    PURPLE_A, PURPLE_B, MAROON, PINK,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    VMobject,
)

# ─────────────────────────────────────────────────────────────────────────────
# scipy — graceful fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy import stats as _sc
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _pdf(dist: str, x: np.ndarray, **kw) -> np.ndarray:
    if not _SCIPY:
        return np.exp(-0.5 * x**2) / np.sqrt(2 * PI)
    d = {
        "norm":    lambda: _sc.norm.pdf(x),
        "t":       lambda: _sc.t.pdf(x, df=kw["df"]),
        "chi2":    lambda: _sc.chi2.pdf(x, df=kw["df"]),
        "f":       lambda: _sc.f.pdf(x, dfn=kw["dfn"], dfd=kw["dfd"]),
        "expon":   lambda: _sc.expon.pdf(x, scale=1/kw.get("lam",1)),
        "gamma":   lambda: _sc.gamma.pdf(x, a=kw["k"], scale=1/kw.get("lam",1)),
        "beta":    lambda: _sc.beta.pdf(x, kw["a"], kw["b"]),
        "lognorm": lambda: _sc.lognorm.pdf(x, s=kw["sigma"], scale=np.exp(kw.get("mu",0))),
        "weibull": lambda: _sc.weibull_min.pdf(x, c=kw["k"]),
        "cauchy":  lambda: _sc.cauchy.pdf(x),
        "pareto":  lambda: _sc.pareto.pdf(x, b=kw.get("b",1)),
        "uniform": lambda: _sc.uniform.pdf(x, loc=kw.get("a",0), scale=kw.get("b",1)-kw.get("a",0)),
    }.get(dist)
    if d is None:
        return np.zeros_like(x)
    try:
        y = d()
        return np.where(np.isfinite(y), y, 0.0)
    except Exception:
        return np.zeros_like(x)


def _cdf(dist: str, x: np.ndarray, **kw) -> np.ndarray:
    if not _SCIPY:
        return 0.5 * (1 + np.vectorize(lambda z: float(
            np.sign(z) * (1 - np.exp(-0.7071068*abs(z)))
        ))(x))
    d = {
        "norm":    lambda: _sc.norm.cdf(x),
        "t":       lambda: _sc.t.cdf(x, df=kw["df"]),
        "chi2":    lambda: _sc.chi2.cdf(x, df=kw["df"]),
        "f":       lambda: _sc.f.cdf(x, dfn=kw["dfn"], dfd=kw["dfd"]),
        "expon":   lambda: _sc.expon.cdf(x, scale=1/kw.get("lam",1)),
        "gamma":   lambda: _sc.gamma.cdf(x, a=kw["k"], scale=1/kw.get("lam",1)),
        "beta":    lambda: _sc.beta.cdf(x, kw["a"], kw["b"]),
        "lognorm": lambda: _sc.lognorm.cdf(x, s=kw["sigma"], scale=np.exp(kw.get("mu",0))),
        "weibull": lambda: _sc.weibull_min.cdf(x, c=kw["k"]),
        "cauchy":  lambda: _sc.cauchy.cdf(x),
        "uniform": lambda: _sc.uniform.cdf(x, loc=kw.get("a",0), scale=kw.get("b",1)-kw.get("a",0)),
    }.get(dist)
    if d is None:
        return np.zeros_like(x)
    try:
        y = d()
        return np.where(np.isfinite(y), y, 0.0)
    except Exception:
        return np.zeros_like(x)


def _pmf(dist: str, k: np.ndarray, **kw) -> np.ndarray:
    if not _SCIPY:
        return np.ones_like(k, float) / len(k)
    d = {
        "bernoulli": lambda: _sc.bernoulli.pmf(k, kw["p"]),
        "binom":     lambda: _sc.binom.pmf(k, kw["n"], kw["p"]),
        "poisson":   lambda: _sc.poisson.pmf(k, kw["mu"]),
        "geom":      lambda: _sc.geom.pmf(k, kw["p"]),
        "nbinom":    lambda: _sc.nbinom.pmf(k, kw["r"], kw["p"]),
        "hypergeom": lambda: _sc.hypergeom.pmf(k, kw["M"], kw["n"], kw["N"]),
    }.get(dist)
    if d is None:
        return np.zeros_like(k, float)
    try:
        y = d()
        return np.where(np.isfinite(y), y, 0.0)
    except Exception:
        return np.zeros_like(k, float)


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

# Per-distribution canonical colours
DIST_COLORS = {
    # Discrete
    "bernoulli": "#3A8FD4",
    "binom":     "#2A6AB0",
    "poisson":   "#7040C8",
    "geom":      "#C85040",
    "nbinom":    "#C87840",
    "hypergeom": "#40A890",
    # Continuous
    "norm":      "#E07030",
    "t":         "#D04060",
    "chi2":      "#40A060",
    "f":         "#8040C0",
    "expon":     "#C8A020",
    "gamma":     "#3080C0",
    "beta":      "#C04080",
    "lognorm":   "#60B060",
    "weibull":   "#A06020",
    "cauchy":    "#E04040",
    "pareto":    "#8A2BE2",
    "uniform":   "#20A0A0",
}

P = {
    "bg":           "#080C12",
    "bg_panel":     "#0C1018",
    "panel_border": "#2A3848",
    "axis":         "#3A4A5A",
    "tick":         "#304050",
    "label":        "#8090A8",
    "title":        "#D8F0FF",
    "subtitle":     "#607888",
    "badge_bg":     "#0C1018",
    "cdf_line":     "#F0D060",
    "shaded_fill":  "#503010",
    "shaded_border":"#D0A040",
    "arrow_rel":    "#A0B0C0",
    "app_tag_bg":   "#101820",
    "app_tag_fg":   "#70A8C8",
    "stats_bg":     "#0E1420",
    "stats_border": "#2A3848",
    "stats_key":    "#6080A0",
    "stats_val":    "#C0D8F0",
    "tree_node_bg": "#101820",
    "tree_edge":    "#304050",
    "tree_discrete":"#3070C0",
    "tree_cont":    "#C05030",
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _curve_poly(x_px, y_px, baseline=0.0):
    return (
        [[float(x_px[0]), baseline, 0]]
        + [[float(x), float(y), 0] for x, y in zip(x_px, y_px)]
        + [[float(x_px[-1]), baseline, 0]]
    )


def _spine(x_px, y_px, color, sw=1.8, z=0.005):
    step  = max(1, len(x_px) // 80)
    pts   = [np.array([float(x_px[i]), float(y_px[i]), z])
             for i in range(0, len(x_px), step) if y_px[i] > 1e-5]
    if len(pts) < 2:
        return VGroup()
    mob = VMobject(stroke_color=color, stroke_width=sw,
                   stroke_opacity=0.85)
    mob.set_points_smoothly(pts)
    return mob


def _axis_line(x_lo, x_hi, y, color=None, sw=1.3):
    return Line([x_lo, y, 0.001], [x_hi, y, 0.001],
                stroke_color=color or P["axis"], stroke_width=sw)


def _ticks(vals, y, lo_raw, hi_raw, px_w, fmt="{:.1f}",
           fs=10, y_off=-0.22):
    grp = VGroup()
    for v in vals:
        xp = (v - lo_raw) / (hi_raw - lo_raw) * px_w - px_w/2
        grp.add(Line([xp, y-0.06, 0.001], [xp, y+0.06, 0.001],
                     stroke_color=P["tick"], stroke_width=0.9))
        grp.add(Text(fmt.format(v), font_size=fs, color=P["label"])
                .move_to([xp, y+y_off, 0.001]))
    return grp


# ─────────────────────────────────────────────────────────────────────────────
# _StatsPanel
# ─────────────────────────────────────────────────────────────────────────────

class _StatsPanel(VGroup):
    """
    Floating sidebar showing E[X], Var[X], skewness, support.

    Parameters
    ----------
    rows : list of (key, value) pairs
    title: str
    color: accent colour for left border
    """

    def __init__(
        self,
        rows:   list[tuple[str, str]],
        title:  str = "",
        color:  str = None,
        w:      float = 2.60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        color   = color or P["stats_border"]
        row_h   = 0.44
        total_h = len(rows) * row_h + (0.60 if title else 0.20)
        z0      = 0.001

        bg = RoundedRectangle(
            width=w, height=total_h,
            corner_radius=0.10,
            fill_color=P["stats_bg"],
            fill_opacity=0.93,
            stroke_color=P["stats_border"],
            stroke_width=0.8,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        strip = RoundedRectangle(
            width=0.07, height=total_h-0.04,
            corner_radius=0.035,
            fill_color=color, fill_opacity=0.85, stroke_width=0,
        )
        strip.move_to([-w/2+0.055, 0, z0+0.001])
        self.add(strip)

        y_top = total_h/2 - 0.28
        if title:
            self.add(Text(title, font_size=14, color=P["title"])
                     .move_to([0, y_top, z0+0.002]))
            self.add(Line([-w/2+0.12, y_top-0.22, z0+0.002],
                          [ w/2-0.08, y_top-0.22, z0+0.002],
                          stroke_color=P["stats_border"],
                          stroke_width=0.6))
            y_top -= 0.46

        for i, (k, v) in enumerate(rows):
            yr = y_top - i * row_h
            km = Text(k, font_size=12, color=P["stats_key"])
            vm = Text(v, font_size=12, color=P["stats_val"],
                      font="monospace")
            km.move_to([-w/2+0.18+km.width/2, yr, z0+0.003])
            vm.move_to([ w/2-0.08-vm.width/2, yr, z0+0.003])
            self.add(km, vm)


# ─────────────────────────────────────────────────────────────────────────────
# _AppTag
# ─────────────────────────────────────────────────────────────────────────────

class _AppTag(VGroup):
    """Small real-world application badge."""

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        lbl = Text(text, font_size=12, color=P["app_tag_fg"])
        bg  = RoundedRectangle(
            width=lbl.width+0.22, height=lbl.height+0.14,
            corner_radius=0.06,
            fill_color=P["app_tag_bg"], fill_opacity=0.90,
            stroke_color=P["app_tag_fg"], stroke_width=0.6,
        )
        bg.move_to(ORIGIN)
        lbl.move_to([0, 0, 0.001])
        self.add(bg, lbl)


# ─────────────────────────────────────────────────────────────────────────────
# _PMFPanel  —  discrete distribution
# ─────────────────────────────────────────────────────────────────────────────

class _PMFPanel(VGroup):
    """
    Discrete distribution panel.

    Layers:
      1. Background panel
      2. 3-D PMF bars (front face + right face + top cap + AO base)
      3. Probability labels above bars (for small support)
      4. CDF step function overlay (gold)
      5. Axis + ticks
      6. Title + parameter badge
      7. Stats sidebar
      8. Application tag

    Parameters
    ----------
    dist     : str   — distribution key
    k_vals   : array of integer support values
    dist_kw  : dict  — parameters forwarded to _pmf()
    title    : str
    app_tag  : str   — real-world application one-liner
    stats    : list of (key, val) pairs for sidebar
    width, height, baseline_y : geometry
    show_cdf : bool
    show_stats: bool
    """

    def __init__(
        self,
        dist:      str,
        k_vals:    np.ndarray,
        dist_kw:   dict,
        title:     str,
        app_tag:   str = "",
        stats:     list = None,
        width:     float = 4.0,
        height:    float = 2.5,
        baseline_y: float = 0.0,
        show_cdf:  bool  = True,
        show_stats: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        col = DIST_COLORS.get(dist, "#4080C0")

        probs   = _pmf(dist, k_vals, **dist_kw)
        max_p   = float(probs.max()) if probs.max() > 1e-9 else 1.0
        n_k     = len(k_vals)
        bar_w   = width / n_k
        dx      = bar_w * 0.07   # 3-D depth
        dark    = interpolate_color(col, BLACK, 0.45)
        lite    = interpolate_color(col, WHITE, 0.28)
        z0      = 0.001

        # ── Background ────────────────────────────────────────────────
        bg = RoundedRectangle(
            width=width+0.22, height=height+1.20,
            corner_radius=0.12,
            fill_color=P["bg_panel"], fill_opacity=0.92,
            stroke_color=P["panel_border"], stroke_width=0.7,
        )
        bg.move_to([0, baseline_y + (height+1.20)/2 - 0.60, z0])
        self.add(bg)

        # ── Axis ──────────────────────────────────────────────────────
        self.add(_axis_line(-width/2, width/2, baseline_y))

        # ── PMF bars ──────────────────────────────────────────────────
        cum_p = 0.0
        for i, (k, p) in enumerate(zip(k_vals, probs)):
            cum_p += p
            h   = (p / max_p) * height
            xl  = i * bar_w - width/2
            xr  = xl + bar_w
            xc  = xl + bar_w/2
            by  = baseline_y
            if h < 1e-4:
                continue

            # Front face
            self.add(Rectangle(
                width=bar_w-0.012, height=h,
                fill_color=col, fill_opacity=0.85,
                stroke_color=dark, stroke_width=0.5,
            ).move_to([xc, by+h/2, z0+0.002]))

            # Right side
            self.add(Polygon(
                [xr-0.006,    by,         z0+0.002],
                [xr-0.006+dx, by+dx*0.4,  z0+0.002],
                [xr-0.006+dx, by+h+dx*0.4,z0+0.002],
                [xr-0.006,    by+h,       z0+0.002],
                fill_color=dark, fill_opacity=0.78, stroke_width=0,
            ).shift([0,0,0.001]))

            # Top cap
            self.add(Polygon(
                [xl+0.006,    by+h,        z0+0.004],
                [xr-0.006,    by+h,        z0+0.004],
                [xr-0.006+dx, by+h+dx*0.4, z0+0.004],
                [xl+0.006+dx, by+h+dx*0.4, z0+0.004],
                fill_color=lite, fill_opacity=0.85, stroke_width=0,
            ))

            # AO base
            self.add(Rectangle(
                width=bar_w-0.012, height=0.038,
                fill_color=BLACK, fill_opacity=0.50, stroke_width=0,
            ).move_to([xc, by+0.019, z0+0.003]))

            # Probability label (only if enough space)
            if n_k <= 12 and p > max_p * 0.04:
                lbl = Text(f"{p:.3f}", font_size=9,
                           color=interpolate_color(col, WHITE, 0.50))
                lbl.move_to([xc, by+h+0.14, z0+0.006])
                self.add(lbl)

            # k label below axis
            self.add(Text(str(int(k)), font_size=10, color=P["label"])
                     .move_to([xc, by-0.22, z0+0.001]))

        # ── CDF step function ─────────────────────────────────────────
        if show_cdf:
            cum = 0.0
            for i, (k, p) in enumerate(zip(k_vals, probs)):
                xl  = i * bar_w - width/2
                xr  = xl + bar_w
                yp  = (cum / 1.0) * height + baseline_y
                yn  = ((cum + p) / 1.0) * height + baseline_y
                # Horizontal segment
                self.add(Line([xl, yp, z0+0.008], [xr, yp, z0+0.008],
                              stroke_color=P["cdf_line"],
                              stroke_width=1.4, stroke_opacity=0.80))
                # Vertical step
                self.add(Line([xr, yp, z0+0.008], [xr, yn, z0+0.008],
                              stroke_color=P["cdf_line"],
                              stroke_width=1.4, stroke_opacity=0.80))
                cum += p
            # Final horizontal to edge
            self.add(Line([width/2, height+baseline_y, z0+0.008],
                          [width/2+0.15, height+baseline_y, z0+0.008],
                          stroke_color=P["cdf_line"],
                          stroke_width=1.4, stroke_opacity=0.60))

        # ── Title + parameter badge ───────────────────────────────────
        self.add(Text(title, font_size=17, color=col)
                 .move_to([0, baseline_y+height+0.38, z0+0.003]))

        param_str = "  ".join(f"{k}={v}" for k, v in dist_kw.items())
        self.add(Text(param_str, font_size=13, color=P["label"])
                 .move_to([0, baseline_y+height+0.14, z0+0.003]))

        # ── Stats sidebar ─────────────────────────────────────────────
        if show_stats and stats:
            sp = _StatsPanel(stats, color=col, w=2.20)
            sp.move_to([width/2+1.30, baseline_y+height*0.40, 0])
            self.add(sp)

        # ── Application tag ───────────────────────────────────────────
        if app_tag:
            tag = _AppTag(app_tag)
            tag.move_to([0, baseline_y-0.55, z0+0.003])
            self.add(tag)


# ─────────────────────────────────────────────────────────────────────────────
# _PDFPanel  —  continuous distribution
# ─────────────────────────────────────────────────────────────────────────────

class _PDFPanel(VGroup):
    """
    Continuous distribution panel: PDF curve + optional CDF.

    Layers:
      1. Background panel
      2. AO under-shadow polygon
      3. Base fill polygon
      4. Inner lite band (subsurface sheen)
      5. Ridge spine VMobject
      6. Shaded region between x_shade_lo and x_shade_hi with badge
      7. CDF curve (dashed gold, right y-axis)
      8. Axis + ticks
      9. Title + parameter badge
      10. Stats sidebar
      11. Application tag

    Parameters
    ----------
    dist         : str   — distribution key
    x_lo, x_hi   : raw axis range
    dist_kw      : dict
    title        : str
    x_shade_lo, x_shade_hi : shaded region bounds (raw units)
    app_tag      : str
    stats        : sidebar rows
    width, height, baseline_y : geometry
    show_cdf     : bool
    show_shade   : bool
    """

    def __init__(
        self,
        dist:        str,
        x_lo:        float,
        x_hi:        float,
        dist_kw:     dict,
        title:       str,
        x_shade_lo:  float = None,
        x_shade_hi:  float = None,
        app_tag:     str   = "",
        stats:       list  = None,
        width:       float = 4.0,
        height:      float = 2.6,
        baseline_y:  float = 0.0,
        show_cdf:    bool  = True,
        show_shade:  bool  = True,
        n_pts:       int   = 400,
        **kwargs,
    ):
        super().__init__(**kwargs)
        col  = DIST_COLORS.get(dist, "#4080C0")
        dark = interpolate_color(col, BLACK, 0.50)
        lite = interpolate_color(col, WHITE, 0.30)
        z0   = 0.001

        raw_x   = np.linspace(x_lo, x_hi, n_pts)
        y_pdf   = _pdf(dist, raw_x, **dist_kw)
        y_pdf   = np.clip(y_pdf, 0, None)
        peak    = float(y_pdf.max()) if y_pdf.max() > 1e-9 else 1.0
        y_sc    = y_pdf / peak * height

        def to_px(v):
            return (v - x_lo) / (x_hi - x_lo) * width - width/2

        x_px = np.array([to_px(v) for v in raw_x])

        # ── Background ────────────────────────────────────────────────
        bg = RoundedRectangle(
            width=width+0.22, height=height+1.25,
            corner_radius=0.12,
            fill_color=P["bg_panel"], fill_opacity=0.92,
            stroke_color=P["panel_border"], stroke_width=0.7,
        )
        bg.move_to([0, baseline_y+(height+1.25)/2-0.62, z0])
        self.add(bg)

        # ── AO under-shadow ───────────────────────────────────────────
        verts_ao = _curve_poly(x_px, y_sc*0.94, baseline_y)
        if len(verts_ao) >= 3:
            self.add(Polygon(*verts_ao,
                             fill_color=dark,
                             fill_opacity=0.50, stroke_width=0)
                     .shift([0.04, -0.04, -0.001]))

        # ── Base fill ─────────────────────────────────────────────────
        verts = _curve_poly(x_px, y_sc, baseline_y)
        if len(verts) >= 3:
            self.add(Polygon(*verts,
                             fill_color=col,
                             fill_opacity=0.78, stroke_width=0)
                     .shift([0, 0, z0+0.001]))

        # ── Inner lite band ───────────────────────────────────────────
        y_inner = y_sc * 0.42
        verts_l = _curve_poly(x_px, y_inner, baseline_y)
        if len(verts_l) >= 3:
            self.add(Polygon(*verts_l,
                             fill_color=lite,
                             fill_opacity=0.25, stroke_width=0)
                     .shift([0, 0, z0+0.002]))

        # ── Ridge spine ───────────────────────────────────────────────
        self.add(_spine(x_px, y_sc, col, sw=1.8, z=z0+0.005))

        # ── Shaded area ───────────────────────────────────────────────
        if show_shade and x_shade_lo is not None and x_shade_hi is not None:
            mask = (raw_x >= x_shade_lo) & (raw_x <= x_shade_hi)
            xs2  = x_px[mask]
            ys2  = y_sc[mask]
            if len(xs2) >= 2:
                lo_px = to_px(x_shade_lo)
                hi_px = to_px(x_shade_hi)
                shade_v = (
                    [[lo_px, baseline_y, 0]]
                    + [[float(x), float(y), 0]
                       for x, y in zip(xs2, ys2)]
                    + [[hi_px, baseline_y, 0]]
                )
                self.add(Polygon(*shade_v,
                                 fill_color=P["shaded_fill"],
                                 fill_opacity=0.70, stroke_width=0)
                         .shift([0, 0, z0+0.003]))
                # Shaded region boundary lines
                for xb in [lo_px, hi_px]:
                    idx = np.argmin(np.abs(x_px - xb))
                    yb  = float(y_sc[idx])
                    self.add(DashedLine(
                        [xb, baseline_y, z0+0.006],
                        [xb, yb,         z0+0.006],
                        stroke_color=P["shaded_border"],
                        stroke_width=1.5,
                        dash_length=0.08,
                    ))
                # Probability badge
                if _SCIPY:
                    try:
                        from scipy.stats import norm as _n
                        p_shade = float(_cdf(dist, np.array([x_shade_hi]),
                                             **dist_kw)[0]
                                        - _cdf(dist, np.array([x_shade_lo]),
                                               **dist_kw)[0])
                        mid_px  = (to_px(x_shade_lo)+to_px(x_shade_hi))/2
                        mean_ys = float(ys2.mean()) if len(ys2) else 0
                        p_lbl   = Text(f"p = {p_shade:.3f}",
                                       font_size=13,
                                       color=P["shaded_border"])
                        p_lbl.move_to([mid_px,
                                       baseline_y+mean_ys*0.5+0.22,
                                       z0+0.008])
                        self.add(p_lbl)
                    except Exception:
                        pass

        # ── CDF curve ─────────────────────────────────────────────────
        if show_cdf:
            y_cdf   = _cdf(dist, raw_x, **dist_kw)
            y_cdf_sc = np.clip(y_cdf, 0, 1) * height
            pts3d   = [np.array([float(x_px[i]),
                                 float(y_cdf_sc[i])+baseline_y,
                                 z0+0.007])
                       for i in range(0, len(x_px), max(1, len(x_px)//80))
                       if np.isfinite(y_cdf[i])]
            if len(pts3d) >= 2:
                cdf_mob = VMobject(stroke_color=P["cdf_line"],
                                   stroke_width=1.6,
                                   stroke_opacity=0.72)
                cdf_mob.set_points_smoothly(pts3d)
                self.add(cdf_mob)
            # CDF label
            self.add(Text("CDF", font_size=10, color=P["cdf_line"])
                     .move_to([width/2-0.25,
                               baseline_y+height-0.12, z0+0.008]))

        # ── Axis + ticks ──────────────────────────────────────────────
        self.add(_axis_line(-width/2-0.1, width/2+0.1, baseline_y))
        n_ticks = 5
        tick_vs = np.linspace(x_lo, x_hi, n_ticks)
        for v in tick_vs:
            xp = to_px(float(v))
            self.add(Line([xp, baseline_y-0.06, z0+0.001],
                          [xp, baseline_y+0.06, z0+0.001],
                          stroke_color=P["tick"], stroke_width=0.9))
            self.add(Text(f"{v:.2g}", font_size=10, color=P["label"])
                     .move_to([xp, baseline_y-0.24, z0+0.001]))

        # ── Title + parameter badge ───────────────────────────────────
        self.add(Text(title, font_size=17, color=col)
                 .move_to([0, baseline_y+height+0.38, z0+0.003]))
        param_str = "  ".join(f"{k}={v}" for k, v in dist_kw.items())
        self.add(Text(param_str, font_size=12, color=P["label"])
                 .move_to([0, baseline_y+height+0.14, z0+0.003]))

        # ── Stats sidebar ─────────────────────────────────────────────
        if stats:
            sp = _StatsPanel(stats, color=col, w=2.20)
            sp.move_to([width/2+1.32, baseline_y+height*0.35, 0])
            self.add(sp)

        # ── Application tag ───────────────────────────────────────────
        if app_tag:
            tag = _AppTag(app_tag)
            tag.move_to([0, baseline_y-0.52, z0+0.003])
            self.add(tag)

        # Store for morph use
        self._to_px   = to_px
        self._col     = col
        self._raw_x   = raw_x
        self._y_sc    = y_sc
        self._x_px    = x_px
        self._baseline = baseline_y
        self._height  = height
        self._width   = width


# ─────────────────────────────────────────────────────────────────────────────
# _DistTree  —  family tree diagram
# ─────────────────────────────────────────────────────────────────────────────

class _DistTree(VGroup):
    """
    Animated family tree of probability distributions.

    Root: "Probability Distributions"
    Two branches: Discrete | Continuous
    Leaves: individual distributions with thumbnail curves.

    Parameters
    ----------
    scale : float — overall scale factor
    """

    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        def _node(text: str, col: str, w=1.80, h=0.40) -> VGroup:
            g   = VGroup()
            bg  = RoundedRectangle(width=w, height=h,
                                   corner_radius=0.08,
                                   fill_color=P["tree_node_bg"],
                                   fill_opacity=0.94,
                                   stroke_color=col, stroke_width=1.0)
            bg.move_to(ORIGIN)
            lbl = Text(text, font_size=int(13*scale), color=col)
            lbl.move_to([0, 0, 0.001])
            g.add(bg, lbl)
            return g

        def _edge(start, end):
            return Line(start, end,
                        stroke_color=P["tree_edge"],
                        stroke_width=0.9)

        # Root
        root = _node("Probability Distributions",
                     P["title"], w=2.80*scale, h=0.46*scale)
        root.move_to([0, 3.20*scale, 0])
        self.add(root)

        # Branch nodes
        disc_node = _node("Discrete", P["tree_discrete"],
                          w=1.60*scale, h=0.40*scale)
        disc_node.move_to([-4.0*scale, 2.00*scale, 0])
        cont_node = _node("Continuous", P["tree_cont"],
                          w=1.60*scale, h=0.40*scale)
        cont_node.move_to([ 4.0*scale, 2.00*scale, 0])
        self.add(disc_node, cont_node)

        self.add(_edge(root.get_bottom(), disc_node.get_top()))
        self.add(_edge(root.get_bottom(), cont_node.get_top()))

        # Discrete leaves
        disc_leaves = [
            ("Bernoulli",   DIST_COLORS["bernoulli"]),
            ("Binomial",    DIST_COLORS["binom"]),
            ("Poisson",     DIST_COLORS["poisson"]),
            ("Geometric",   DIST_COLORS["geom"]),
            ("Hypergeom.",  DIST_COLORS["hypergeom"]),
        ]
        n_d  = len(disc_leaves)
        x_d  = np.linspace(-6.8*scale, -1.2*scale, n_d)
        for j, (name, col) in enumerate(disc_leaves):
            leaf = _node(name, col, w=1.30*scale, h=0.36*scale)
            leaf.move_to([x_d[j], 0.70*scale, 0])
            self.add(leaf)
            self.add(_edge(disc_node.get_bottom(), leaf.get_top()))

        # Continuous leaves (two rows)
        cont_row1 = [
            ("Normal",      DIST_COLORS["norm"]),
            ("t",           DIST_COLORS["t"]),
            ("Chi-squared", DIST_COLORS["chi2"]),
            ("F",           DIST_COLORS["f"]),
            ("Uniform",     DIST_COLORS["uniform"]),
            ("Exponential", DIST_COLORS["expon"]),
        ]
        cont_row2 = [
            ("Gamma",       DIST_COLORS["gamma"]),
            ("Beta",        DIST_COLORS["beta"]),
            ("Log-Normal",  DIST_COLORS["lognorm"]),
            ("Weibull",     DIST_COLORS["weibull"]),
            ("Cauchy",      DIST_COLORS["cauchy"]),
            ("Pareto",      DIST_COLORS["pareto"]),
        ]
        for row_idx, row in enumerate([cont_row1, cont_row2]):
            n_c  = len(row)
            x_c  = np.linspace(1.0*scale, 7.4*scale, n_c)
            y_c  = (0.70 - row_idx * 1.10) * scale
            for j, (name, col) in enumerate(row):
                leaf = _node(name, col, w=1.22*scale, h=0.34*scale)
                leaf.move_to([x_c[j], y_c, 0])
                self.add(leaf)
                parent = cont_node if row_idx == 0 else self.submobjects[
                    3 + len(disc_leaves) + j   # first cont-row node
                ]
                self.add(_edge(
                    cont_node.get_bottom() if row_idx == 0
                    else self.submobjects[3+len(disc_leaves)+j].get_bottom(),
                    leaf.get_top()
                ))

        self.scale(scale * 0.72)


# ─────────────────────────────────────────────────────────────────────────────
# _TailComparison  —  log-scale right-tail overlay
# ─────────────────────────────────────────────────────────────────────────────

class _TailComparison(VGroup):
    """
    Log-scale right-tail overlay for four distributions.

    X-axis: raw value (2 to 8)
    Y-axis: log₁₀(pdf)
    Each distribution: ridge spine in its canonical colour.
    Annotations: "Light tail" / "Heavy tail" arrows.
    """

    def __init__(
        self,
        width:  float = 6.0,
        height: float = 3.2,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0     = 0.001
        x_lo   = 1.5
        x_hi   = 7.0
        y_log_lo = -7.0
        y_log_hi =  0.0
        raw_x  = np.linspace(x_lo, x_hi, 300)

        def to_px_x(v):
            return (v - x_lo) / (x_hi - x_lo) * width - width/2

        def to_px_y(log_v):
            return (log_v - y_log_lo) / (y_log_hi - y_log_lo) * height + baseline_y

        # Background
        bg = RoundedRectangle(
            width=width+0.24, height=height+0.80,
            corner_radius=0.12,
            fill_color=P["bg_panel"], fill_opacity=0.93,
            stroke_color=P["panel_border"], stroke_width=0.7,
        )
        bg.move_to([0, baseline_y+height/2-0.10, z0])
        self.add(bg)

        # Axes
        self.add(_axis_line(-width/2, width/2, baseline_y))
        self.add(Line([-width/2, baseline_y, z0+0.001],
                      [-width/2, baseline_y+height, z0+0.001],
                      stroke_color=P["axis"], stroke_width=1.2))

        # Y ticks (log scale)
        for log_v in [-6, -5, -4, -3, -2, -1, 0]:
            yp = to_px_y(float(log_v))
            if baseline_y <= yp <= baseline_y + height:
                self.add(Line([-width/2-0.07, yp, z0+0.001],
                              [-width/2+0.07, yp, z0+0.001],
                              stroke_color=P["tick"],
                              stroke_width=0.8))
                self.add(Text(f"10^{{{log_v}}}", font_size=9,
                              color=P["label"])
                         .move_to([-width/2-0.40, yp, z0+0.001]))

        # X ticks
        for xv in [2, 3, 4, 5, 6, 7]:
            xp = to_px_x(float(xv))
            self.add(Line([xp, baseline_y-0.06, z0+0.001],
                          [xp, baseline_y+0.06, z0+0.001],
                          stroke_color=P["tick"], stroke_width=0.8))
            self.add(Text(str(xv), font_size=10, color=P["label"])
                     .move_to([xp, baseline_y-0.25, z0+0.001]))

        # Plot each distribution tail
        configs = [
            ("norm",   {},                         "Normal"),
            ("expon",  {"lam": 1.0},               "Exponential"),
            ("cauchy", {},                          "Cauchy"),
            ("pareto", {"b": 1.0},                 "Pareto(b=1)"),
        ]

        for dist, kw, label in configs:
            col  = DIST_COLORS.get(dist, "#8080FF")
            y_pdf = _pdf(dist, raw_x, **kw)
            y_pdf = np.clip(y_pdf, 1e-10, None)
            log_y = np.log10(y_pdf)
            log_y = np.clip(log_y, y_log_lo, y_log_hi)

            pts3d = [
                np.array([to_px_x(float(raw_x[i])),
                          to_px_y(float(log_y[i])), z0+0.005])
                for i in range(len(raw_x))
                if (x_lo <= raw_x[i] <= x_hi
                    and log_y[i] > y_log_lo + 0.2)
            ]
            if len(pts3d) >= 2:
                sp = VMobject(stroke_color=col,
                              stroke_width=2.0, stroke_opacity=0.90)
                sp.set_points_smoothly(pts3d)
                self.add(sp)

            # End label
            if len(pts3d) > 0:
                lbl = Text(label, font_size=11, color=col)
                end = pts3d[-1]
                lbl.move_to([end[0]+0.05+lbl.width/2,
                             end[1], z0+0.006])
                self.add(lbl)

        # Annotations
        self.add(Text("Tail Behaviour (log scale)",
                      font_size=16, color=P["title"])
                 .move_to([0, baseline_y+height+0.30, z0+0.003]))
        self.add(Text("← Light tail", font_size=12,
                      color=DIST_COLORS["norm"])
                 .move_to([-1.20, baseline_y+0.35, z0+0.003]))
        self.add(Text("← Heavy tail", font_size=12,
                      color=DIST_COLORS["cauchy"])
                 .move_to([1.80, baseline_y+height*0.55, z0+0.003]))
        self.add(Text("x", font_size=13, color=P["label"])
                 .move_to([0, baseline_y-0.38, z0+0.002]))
        self.add(Text("log₁₀ f(x)", font_size=12, color=P["label"])
                 .rotate(PI/2)
                 .move_to([-width/2-0.65, baseline_y+height/2, z0+0.002]))


# ─────────────────────────────────────────────────────────────────────────────
# _RelArrow  —  distribution relationship
# ─────────────────────────────────────────────────────────────────────────────

class _RelArrow(VGroup):
    """
    Arrow connecting two distributions with a relationship label.

    Parameters
    ----------
    start_pos, end_pos : np.ndarray
    label    : main relationship text
    condition: condition text (smaller, below label)
    color    : arrow colour
    """

    def __init__(
        self,
        start_pos: np.ndarray,
        end_pos:   np.ndarray,
        label:     str,
        condition: str = "",
        color:     str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        col = color or P["arrow_rel"]
        z0  = 0.005

        arr = Arrow(
            start=start_pos, end=end_pos,
            stroke_color=col, stroke_width=1.8,
            tip_length=0.16, buff=0.10,
        )
        self.add(arr)

        mid = (start_pos + end_pos) / 2
        lbl = Text(label, font_size=13, color=col)
        lbl.move_to(mid + UP*0.28)
        self.add(lbl)

        if condition:
            cond = Text(condition, font_size=11,
                        color=interpolate_color(col, WHITE, 0.30))
            cond.move_to(mid - UP*0.12)
            self.add(cond)


# ─────────────────────────────────────────────────────────────────────────────
# _ComparisonOverlay  —  all continuous PDFs on one axis
# ─────────────────────────────────────────────────────────────────────────────

class _ComparisonOverlay(VGroup):
    """
    All continuous distributions overlaid on a single normalised axis.

    Each curve: ridge spine in its canonical colour.
    Legend: coloured dots + names.

    Parameters
    ----------
    width, height, baseline_y : plot geometry
    """

    def __init__(
        self,
        width:  float = 8.0,
        height: float = 3.0,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        configs = [
            ("norm",   -4.0,  4.0, {},                   "Normal(0,1)"),
            ("t",      -4.5,  4.5, {"df": 5},            "t(df=5)"),
            ("expon",   0.0,  5.0, {"lam": 1.0},         "Exp(λ=1)"),
            ("gamma",   0.0,  8.0, {"k": 3, "lam": 1.0}, "Gamma(3,1)"),
            ("beta",    0.0,  1.0, {"a": 2, "b": 5},     "Beta(2,5)"),
            ("uniform", 0.0,  1.0, {"a": 0.0, "b": 1.0}, "Uniform(0,1)"),
            ("lognorm", 0.0,  5.0, {"sigma": 0.8, "mu": 0.0}, "LogNorm"),
            ("cauchy", -4.0,  4.0, {},                   "Cauchy"),
        ]

        # Global x range: normalise each to its own x_range but all on [−4,4]
        global_lo, global_hi = -4.0, 4.0
        raw_x_global = np.linspace(global_lo, global_hi, 500)

        def to_px(v):
            return ((v - global_lo) / (global_hi - global_lo)
                    * width - width/2)

        # Background
        bg = RoundedRectangle(
            width=width+0.24, height=height+0.80,
            corner_radius=0.12,
            fill_color=P["bg_panel"], fill_opacity=0.92,
            stroke_color=P["panel_border"], stroke_width=0.7,
        )
        bg.move_to([0, baseline_y+height/2-0.10, z0])
        self.add(bg)

        self.add(_axis_line(-width/2, width/2, baseline_y))

        # X ticks
        for xv in [-3,-2,-1,0,1,2,3]:
            xp = to_px(float(xv))
            self.add(Line([xp, baseline_y-0.06, z0+0.001],
                          [xp, baseline_y+0.06, z0+0.001],
                          stroke_color=P["tick"], stroke_width=0.8))
            self.add(Text(str(xv), font_size=10, color=P["label"])
                     .move_to([xp, baseline_y-0.24, z0+0.001]))

        for dist, x_lo, x_hi, kw, name in configs:
            col   = DIST_COLORS.get(dist, "#8080FF")
            raw_x = np.linspace(x_lo, x_hi, 400)
            y_pdf = _pdf(dist, raw_x, **kw)
            y_pdf = np.where(np.isfinite(y_pdf), y_pdf, 0.0)
            peak  = float(y_pdf.max()) if y_pdf.max() > 1e-9 else 1.0

            # Map raw x to global scale (linear, so just shift/scale)
            raw_frac = (raw_x - x_lo) / (x_hi - x_lo)
            raw_global = raw_frac * (global_hi - global_lo) + global_lo
            y_sc = y_pdf / peak * height

            step  = max(1, len(raw_global)//80)
            pts3d = [
                np.array([to_px(float(raw_global[i])),
                          float(y_sc[i]) + baseline_y,
                          z0+0.004])
                for i in range(0, len(raw_global), step)
                if (y_sc[i] > 0.01
                    and global_lo <= raw_global[i] <= global_hi)
            ]
            if len(pts3d) >= 2:
                sp = VMobject(stroke_color=col,
                              stroke_width=1.8, stroke_opacity=0.88)
                sp.set_points_smoothly(pts3d)
                self.add(sp)

        # Title
        self.add(Text("Distribution Shape Comparison",
                      font_size=18, color=P["title"])
                 .move_to([0, baseline_y+height+0.30, z0+0.003]))

        # Legend (two columns)
        n_per_col = 4
        for i, (dist, *_, name) in enumerate(configs):
            col = DIST_COLORS.get(dist, "#8080FF")
            col_idx = i // n_per_col
            row_idx = i % n_per_col
            lx = -1.80 + col_idx * 3.60
            ly = baseline_y - 0.58 - row_idx * 0.30
            self.add(Dot(radius=0.06,
                         point=[lx-0.40, ly, z0+0.003],
                         color=col, fill_opacity=1.0))
            self.add(Text(name, font_size=11, color=col)
                     .move_to([lx+0.30, ly, z0+0.003]))


# ─────────────────────────────────────────────────────────────────────────────
# The full demo scene
# ─────────────────────────────────────────────────────────────────────────────

class DistributionsDemo(Scene):
    """
    Complete probability distributions cinematic showcase.

    Run with:
        manim -pql demo_distributions.py DistributionsDemo
    For high quality:
        manim -pqh demo_distributions.py DistributionsDemo
    """

    def construct(self):
        self._setup_bg()
        self._act0_family_tree()
        self._act1_discrete_gallery()
        self._act2_bell_curves()
        self._act3_skewed_bounded()
        self._act4_heavy_tails()
        self._act5_relationships()
        self._act6_param_sweep()
        self._act7_comparison_closing()

    # ─────────────────────────────────────────────────────────────────
    # Shared
    # ─────────────────────────────────────────────────────────────────

    def _setup_bg(self):
        bg = Rectangle(width=16, height=9,
                       fill_color=P["bg"],
                       fill_opacity=1.0, stroke_width=0)
        self.add(bg)
        self._bg = bg

    def _section(self, text: str):
        lbl = Text(text, font_size=16, color=P["subtitle"])
        lbl.to_corner(LEFT + UP, buff=0.22)
        return lbl

    def _fade_scene(self, *mobs, rt=0.55):
        self.play(*[FadeOut(m) for m in mobs if m is not None],
                  run_time=rt)

    # ─────────────────────────────────────────────────────────────────
    # Act 0 — Family tree
    # ─────────────────────────────────────────────────────────────────

    def _act0_family_tree(self):
        lbl = self._section("Act 0 — Distribution Family Tree")
        self.play(FadeIn(lbl, run_time=0.4))

        tree = _DistTree(scale=1.0)
        tree.scale(0.82).center().shift(UP * 0.30)
        tree.set_opacity(0)

        self.play(FadeIn(tree, shift=UP*0.15, run_time=1.4))
        self.wait(0.6)

        # Pulse the two branch nodes
        for node_idx in [1, 2]:   # disc_node, cont_node
            self.play(
                Indicate(tree.submobjects[node_idx],
                         scale_factor=1.08, run_time=0.45)
            )
            self.wait(0.08)

        self.wait(0.7)
        hook = Text(
            "Each distribution is a model for a different kind of randomness.",
            font_size=22, color=P["title"],
        )
        hook.move_to([0, -3.20, 0])
        self.play(Write(hook, run_time=1.2))
        self.wait(0.9)

        self._fade_scene(tree, hook, lbl)

    # ─────────────────────────────────────────────────────────────────
    # Act 1 — Discrete gallery
    # ─────────────────────────────────────────────────────────────────

    def _act1_discrete_gallery(self):
        lbl = self._section("Act 1 — Discrete Distributions")
        self.play(FadeIn(lbl, run_time=0.4))

        panels_data = [
            dict(
                dist="bernoulli",
                k_vals=np.array([0, 1]),
                dist_kw={"p": 0.35},
                title="Bernoulli(p=0.35)",
                app_tag="▸ Coin flip, A/B test outcome",
                stats=[
                    ("E[X]",    "p = 0.350"),
                    ("Var[X]",  "p(1−p) = 0.228"),
                    ("Support", "{0, 1}"),
                    ("Skew",    "(1−2p)/√(pq)"),
                ],
            ),
            dict(
                dist="binom",
                k_vals=np.arange(0, 16),
                dist_kw={"n": 15, "p": 0.40},
                title="Binomial(n=15, p=0.4)",
                app_tag="▸ # successes in n independent trials",
                stats=[
                    ("E[X]",    "np = 6.00"),
                    ("Var[X]",  "np(1−p) = 3.60"),
                    ("Support", "{0,…,n}"),
                    ("Skew",    "(1−2p)/√(npq)"),
                ],
            ),
            dict(
                dist="poisson",
                k_vals=np.arange(0, 16),
                dist_kw={"mu": 4.0},
                title="Poisson(λ=4)",
                app_tag="▸ Events per interval (calls, arrivals)",
                stats=[
                    ("E[X]",    "λ = 4.00"),
                    ("Var[X]",  "λ = 4.00"),
                    ("Support", "{0, 1, 2, …}"),
                    ("Skew",    "1/√λ"),
                ],
            ),
            dict(
                dist="geom",
                k_vals=np.arange(1, 14),
                dist_kw={"p": 0.30},
                title="Geometric(p=0.3)",
                app_tag="▸ Trials until first success",
                stats=[
                    ("E[X]",    "1/p = 3.33"),
                    ("Var[X]",  "(1−p)/p² = 7.78"),
                    ("Support", "{1, 2, 3, …}"),
                    ("Skew",    "(2−p)/√(1−p)"),
                ],
            ),
            dict(
                dist="nbinom",
                k_vals=np.arange(0, 18),
                dist_kw={"r": 3, "p": 0.40},
                title="NegBinom(r=3, p=0.4)",
                app_tag="▸ Failures before r-th success",
                stats=[
                    ("E[X]",    "r(1−p)/p = 4.50"),
                    ("Var[X]",  "r(1−p)/p² = 11.25"),
                    ("Support", "{0, 1, 2, …}"),
                    ("Skew",    "(2−p)/√(r(1−p))"),
                ],
            ),
        ]

        for i, kw in enumerate(panels_data):
            panel = _PMFPanel(
                **kw,
                width=5.0, height=2.60,
                baseline_y=0.0,
                show_cdf=True,
                show_stats=True,
            )
            panel.scale(0.82).center().shift(DOWN * 0.15)
            panel.set_opacity(0)
            self.play(FadeIn(panel, shift=UP*0.12, run_time=0.80))
            self.wait(0.70)
            if i < len(panels_data) - 1:
                self.play(FadeOut(panel, run_time=0.40))

        self.play(FadeOut(panel, lbl, run_time=0.50))

    # ─────────────────────────────────────────────────────────────────
    # Act 2 — Bell-curve family
    # ─────────────────────────────────────────────────────────────────

    def _act2_bell_curves(self):
        lbl = self._section("Act 2 — Bell-Curve Family")
        self.play(FadeIn(lbl, run_time=0.4))

        bell_configs = [
            dict(
                dist="norm",
                x_lo=-4.0, x_hi=4.0, dist_kw={},
                title="Normal(μ=0, σ=1)",
                x_shade_lo=-1.0, x_shade_hi=1.0,
                app_tag="▸ Measurement error, heights, test scores",
                stats=[
                    ("E[X]",    "μ = 0"),
                    ("Var[X]",  "σ² = 1"),
                    ("Support", "(−∞, +∞)"),
                    ("Skew",    "0  (symmetric)"),
                    ("MGF",     "exp(μt + σ²t²/2)"),
                ],
            ),
            dict(
                dist="t",
                x_lo=-4.5, x_hi=4.5, dist_kw={"df": 5},
                title="Student-t(df=5)",
                x_shade_lo=-2.0, x_shade_hi=2.0,
                app_tag="▸ Small-sample inference, unknown σ",
                stats=[
                    ("E[X]",    "0  (df > 1)"),
                    ("Var[X]",  "df/(df−2)"),
                    ("Support", "(−∞, +∞)"),
                    ("Skew",    "0  (symmetric)"),
                    ("Notes",   "Heavier tails than N"),
                ],
            ),
            dict(
                dist="chi2",
                x_lo=0.0, x_hi=18.0, dist_kw={"df": 5},
                title="Chi-Squared(df=5)",
                x_shade_lo=0.0, x_shade_hi=11.07,
                app_tag="▸ Goodness-of-fit, variance estimation",
                stats=[
                    ("E[X]",    "df = 5"),
                    ("Var[X]",  "2·df = 10"),
                    ("Support", "[0, +∞)"),
                    ("Skew",    "√(8/df)"),
                ],
            ),
            dict(
                dist="f",
                x_lo=0.01, x_hi=6.0, dist_kw={"dfn": 5, "dfd": 10},
                title="F(d₁=5, d₂=10)",
                x_shade_lo=0.0, x_shade_hi=3.33,
                app_tag="▸ ANOVA, comparing variances",
                stats=[
                    ("E[X]",    "d₂/(d₂−2) ≈ 1.25"),
                    ("Var[X]",  "complex"),
                    ("Support", "[0, +∞)"),
                    ("Skew",    "> 0  (right skewed)"),
                ],
            ),
        ]

        panels = []
        for cfg in bell_configs:
            p = _PDFPanel(**cfg, width=5.0, height=2.60,
                          baseline_y=0.0,
                          show_cdf=True, show_shade=True)
            p.scale(0.82).center().shift(DOWN*0.10)
            p.set_opacity(0)
            panels.append(p)

        for i, panel in enumerate(panels):
            self.play(FadeIn(panel, shift=UP*0.12, run_time=0.80))
            self.wait(0.75)
            if i < len(panels) - 1:
                self.play(FadeOut(panel, run_time=0.40))

        # Side-by-side: Normal vs t, showing heavier tails
        n_panel = _PDFPanel(
            dist="norm", x_lo=-4.5, x_hi=4.5, dist_kw={},
            title="Normal(0,1)",
            width=3.6, height=2.3, baseline_y=0.0,
            show_cdf=False, show_shade=False,
        )
        t1_panel = _PDFPanel(
            dist="t", x_lo=-4.5, x_hi=4.5, dist_kw={"df": 1},
            title="t(df=1)  ← Cauchy",
            width=3.6, height=2.3, baseline_y=0.0,
            show_cdf=False, show_shade=False,
        )
        n_panel.scale(0.80).move_to([-3.20, 0.30, 0])
        t1_panel.scale(0.80).move_to([ 3.20, 0.30, 0])

        self.play(
            FadeOut(panels[-1], run_time=0.35),
            FadeIn(n_panel,  shift=LEFT*0.12, run_time=0.70),
            FadeIn(t1_panel, shift=RIGHT*0.12, run_time=0.70),
        )

        cmp_msg = Text(
            "df → ∞:  t-distribution converges to Normal",
            font_size=19, color=P["subtitle"],
        )
        cmp_msg.move_to([0, -2.55, 0])
        self.play(Write(cmp_msg, run_time=0.90))
        self.wait(0.80)

        self._fade_scene(n_panel, t1_panel, cmp_msg, lbl)

    # ─────────────────────────────────────────────────────────────────
    # Act 3 — Skewed and bounded distributions
    # ─────────────────────────────────────────────────────────────────

    def _act3_skewed_bounded(self):
        lbl = self._section("Act 3 — Skewed & Bounded Distributions")
        self.play(FadeIn(lbl, run_time=0.4))

        skewed_configs = [
            dict(
                dist="expon", x_lo=0.0, x_hi=5.0,
                dist_kw={"lam": 1.0},
                title="Exponential(λ=1)",
                x_shade_lo=0.0, x_shade_hi=1.0,
                app_tag="▸ Time between events, survival analysis",
                stats=[
                    ("E[X]",    "1/λ = 1.00"),
                    ("Var[X]",  "1/λ² = 1.00"),
                    ("Support", "[0, +∞)"),
                    ("Skew",    "2  (constant!)"),
                    ("Memoryless","P(X>s+t|X>s)=P(X>t)"),
                ],
            ),
            dict(
                dist="gamma", x_lo=0.0, x_hi=14.0,
                dist_kw={"k": 3, "lam": 1.0},
                title="Gamma(k=3, λ=1)",
                x_shade_lo=1.0, x_shade_hi=6.0,
                app_tag="▸ Sum of exponentials, queuing theory",
                stats=[
                    ("E[X]",    "k/λ = 3.00"),
                    ("Var[X]",  "k/λ² = 3.00"),
                    ("Support", "[0, +∞)"),
                    ("Skew",    "2/√k ≈ 1.15"),
                ],
            ),
            dict(
                dist="beta", x_lo=0.0, x_hi=1.0,
                dist_kw={"a": 2, "b": 5},
                title="Beta(α=2, β=5)",
                x_shade_lo=0.10, x_shade_hi=0.50,
                app_tag="▸ Proportions, Bayesian prior for p",
                stats=[
                    ("E[X]",    "α/(α+β) = 0.286"),
                    ("Var[X]",  "αβ/((α+β)²(α+β+1))"),
                    ("Support", "[0, 1]"),
                    ("Skew",    "f(α,β)"),
                ],
            ),
            dict(
                dist="lognorm", x_lo=0.01, x_hi=6.0,
                dist_kw={"sigma": 0.8, "mu": 0.0},
                title="Log-Normal(μ=0, σ=0.8)",
                x_shade_lo=0.5, x_shade_hi=2.5,
                app_tag="▸ Stock prices, income distribution",
                stats=[
                    ("E[X]",    "exp(μ+σ²/2) ≈ 1.38"),
                    ("Var[X]",  "(eˢ²−1)·e^(2μ+σ²)"),
                    ("Support", "(0, +∞)"),
                    ("Skew",    "> 0  (right skewed)"),
                ],
            ),
            dict(
                dist="weibull", x_lo=0.0, x_hi=3.5,
                dist_kw={"k": 2},
                title="Weibull(k=2)",
                x_shade_lo=0.5, x_shade_hi=2.0,
                app_tag="▸ Reliability, failure-time modelling",
                stats=[
                    ("E[X]",    "Γ(1+1/k)"),
                    ("Var[X]",  "Γ(1+2/k)−[E[X]]²"),
                    ("Support", "[0, +∞)"),
                    ("k<1",     "DFR (decreasing hazard)"),
                    ("k>1",     "IFR (increasing hazard)"),
                ],
            ),
        ]

        for i, cfg in enumerate(skewed_configs):
            panel = _PDFPanel(**cfg, width=5.0, height=2.6,
                              baseline_y=0.0,
                              show_cdf=True, show_shade=True)
            panel.scale(0.80).center().shift(DOWN*0.15)
            panel.set_opacity(0)
            self.play(FadeIn(panel, shift=UP*0.12, run_time=0.75))
            self.wait(0.70)
            if i < len(skewed_configs) - 1:
                self.play(FadeOut(panel, run_time=0.38))

        # Beta parameter sweep: α=β=0.5 (U-shape) → α=β=1 (Uniform) → α=β=5 (bell)
        beta_params = [
            ({"a": 0.5, "b": 0.5}, "Beta(0.5,0.5) — U-shape"),
            ({"a": 1.0, "b": 1.0}, "Beta(1,1)  = Uniform(0,1)"),
            ({"a": 2.0, "b": 5.0}, "Beta(2,5)  — right skew"),
            ({"a": 5.0, "b": 2.0}, "Beta(5,2)  — left skew"),
            ({"a": 5.0, "b": 5.0}, "Beta(5,5)  — bell shape"),
        ]

        beta_sweep_lbl = Text(
            "Beta distribution  — parameter sweep",
            font_size=20, color=DIST_COLORS["beta"],
        )
        beta_sweep_lbl.move_to([0, 2.90, 0])
        self.play(
            FadeOut(panel, run_time=0.35),
            FadeIn(beta_sweep_lbl, run_time=0.40),
        )

        prev = None
        for kw, subtitle in beta_params:
            new_panel = _PDFPanel(
                dist="beta", x_lo=0.0, x_hi=1.0,
                dist_kw=kw, title=subtitle,
                width=5.0, height=2.5, baseline_y=0.0,
                show_cdf=False, show_shade=False,
                show_stats=False,
            )
            new_panel.scale(0.85).center().shift(DOWN*0.10)
            if prev is None:
                new_panel.set_opacity(0)
                self.play(FadeIn(new_panel, run_time=0.55))
            else:
                self.play(ReplacementTransform(prev, new_panel,
                                               run_time=0.75))
            prev = new_panel
            self.wait(0.50)

        self._fade_scene(prev, beta_sweep_lbl, lbl)

    # ─────────────────────────────────────────────────────────────────
    # Act 4 — Heavy tails
    # ─────────────────────────────────────────────────────────────────

    def _act4_heavy_tails(self):
        lbl = self._section("Act 4 — Heavy Tails")
        self.play(FadeIn(lbl, run_time=0.4))

        tail_plot = _TailComparison(
            width=7.0, height=3.5, baseline_y=-0.20,
        )
        tail_plot.scale(0.90).center().shift(UP*0.15)
        tail_plot.set_opacity(0)

        self.play(FadeIn(tail_plot, shift=UP*0.12, run_time=1.10))

        msg1 = Text(
            "On a log scale, tail weight becomes clearly visible.",
            font_size=19, color=P["subtitle"],
        )
        msg1.move_to([0, -2.90, 0])
        self.play(Write(msg1, run_time=1.0))
        self.wait(0.5)

        msg2 = Text(
            "Cauchy and Pareto: extreme values are far more probable.",
            font_size=18, color=DIST_COLORS["cauchy"],
        )
        msg2.move_to([0, -3.50, 0])
        self.play(FadeIn(msg2, shift=UP*0.08, run_time=0.7))
        self.wait(1.0)

        self._fade_scene(tail_plot, msg1, msg2, lbl)

    # ─────────────────────────────────────────────────────────────────
    # Act 5 — Distribution relationships
    # ─────────────────────────────────────────────────────────────────

    def _act5_relationships(self):
        lbl = self._section("Act 5 — Distribution Relationships")
        self.play(FadeIn(lbl, run_time=0.4))

        relations = [
            # (name_a, pos_a, name_b, pos_b, label, condition, color)
            ("Binomial(n,p)",  np.array([-5.0,  2.0, 0]),
             "Normal(np, np(1-p))", np.array([-1.0,  2.0, 0]),
             "CLT", "n large", DIST_COLORS["binom"]),

            ("Binomial(n,p)",  np.array([-5.0,  2.0, 0]),
             "Poisson(λ=np)",  np.array([-5.0, -0.4, 0]),
             "Rare event", "n→∞, p→0", DIST_COLORS["poisson"]),

            ("Poisson(λ)",     np.array([-5.0, -0.4, 0]),
             "Normal(λ,λ)",    np.array([-1.0, -0.4, 0]),
             "CLT", "λ large", DIST_COLORS["norm"]),

            ("Gamma(1,λ)",     np.array([ 1.8,  2.0, 0]),
             "Exponential(λ)", np.array([ 5.0,  2.0, 0]),
             "k=1", "special case", DIST_COLORS["expon"]),

            ("Gamma(k/2,½)",   np.array([ 1.8,  0.0, 0]),
             "Chi²(k)",        np.array([ 5.0,  0.0, 0]),
             "reparameterisation", "", DIST_COLORS["chi2"]),

            ("Beta(1,1)",      np.array([ 1.8, -2.0, 0]),
             "Uniform(0,1)",   np.array([ 5.0, -2.0, 0]),
             "α=β=1", "special case", DIST_COLORS["uniform"]),
        ]

        # Draw all nodes first
        node_mobs = {}
        for item in relations:
            for name, pos in [(item[0], item[1]), (item[2], item[3])]:
                if name not in node_mobs:
                    col = P["subtitle"]
                    bg  = RoundedRectangle(
                        width=2.20, height=0.42,
                        corner_radius=0.08,
                        fill_color=P["bg_panel"],
                        fill_opacity=0.90,
                        stroke_color=col, stroke_width=0.8,
                    )
                    bg.move_to(pos)
                    nm  = Text(name, font_size=11, color=P["title"])
                    nm.move_to(pos + np.array([0, 0, 0.001]))
                    node_grp = VGroup(bg, nm)
                    node_mobs[name] = node_grp

        all_nodes = VGroup(*node_mobs.values())
        self.play(FadeIn(all_nodes, run_time=0.80))

        # Animate arrows one by one
        arrow_mobs = []
        for name_a, pos_a, name_b, pos_b, rel_lbl, cond, col in relations:
            arr = _RelArrow(
                start_pos=pos_a + np.array([1.12, 0, 0]),
                end_pos=pos_b   - np.array([1.12, 0, 0]),
                label=rel_lbl, condition=cond, color=col,
            )
            arr.set_opacity(0)
            self.play(FadeIn(arr, run_time=0.55))
            arrow_mobs.append(arr)
            self.wait(0.25)

        self.wait(0.8)
        self._fade_scene(all_nodes, *arrow_mobs, lbl)

    # ─────────────────────────────────────────────────────────────────
    # Act 6 — Parameter sweep gallery
    # ─────────────────────────────────────────────────────────────────

    def _act6_param_sweep(self):
        lbl = self._section("Act 6 — Parameter Sweep Gallery")
        self.play(FadeIn(lbl, run_time=0.4))

        # 3 columns × 2 rows of mini panels, each sweeping a parameter
        sweep_configs = [
            # (dist, x_lo, x_hi, param_name, param_values, base_kw, title)
            ("norm",   -4.5,  4.5, "σ",
             [0.5, 1.0, 1.5, 2.0, 2.5],
             {},   "Normal — vary σ",
             lambda s: {"sigma_scale": s}),

            ("t",      -5.0,  5.0, "df",
             [1, 3, 5, 15, 30],
             {},   "Student-t — vary df",
             lambda d: {"df": d}),

            ("expon",   0.0,  6.0, "λ",
             [0.5, 1.0, 2.0, 3.0],
             {},   "Exponential — vary λ",
             lambda l: {"lam": l}),

            ("gamma",   0.0, 12.0, "k",
             [1, 2, 3, 5, 8],
             {},   "Gamma — vary k",
             lambda k: {"k": k, "lam": 1.0}),

            ("beta",    0.0,  1.0, "α=β",
             [0.5, 1.0, 2.0, 4.0, 8.0],
             {},   "Beta — vary α=β",
             lambda v: {"a": v, "b": v}),

            ("weibull", 0.0,  3.5, "k",
             [0.5, 1.0, 1.5, 2.0, 3.5],
             {},   "Weibull — vary k",
             lambda k: {"k": k}),
        ]

        # Build 3×2 grid
        n_cols, n_rows = 3, 2
        col_w, row_h   = 4.60, 3.10
        panels_grid    = []

        for idx, (dist, xlo, xhi, pname, pvals, bkw,
                  title, kw_fn) in enumerate(sweep_configs):
            ci = idx % n_cols
            ri = idx // n_cols
            cx = (ci - 1) * col_w
            cy = (0.5 - ri) * row_h + 0.20

            col_base = DIST_COLORS.get(dist, "#4080C0")
            sweep_grp = VGroup()

            # Draw one curve per parameter value, with colour gradient
            n_v = len(pvals)
            bg = RoundedRectangle(
                width=col_w-0.20, height=row_h-0.30,
                corner_radius=0.10,
                fill_color=P["bg_panel"], fill_opacity=0.90,
                stroke_color=P["panel_border"], stroke_width=0.6,
            )
            bg.move_to([cx, cy, 0])
            sweep_grp.add(bg)

            raw_x = np.linspace(xlo, xhi, 300)
            for j, v in enumerate(pvals):
                frac   = j / max(n_v-1, 1)
                col    = interpolate_color(
                    interpolate_color(col_base, BLACK, 0.40),
                    interpolate_color(col_base, WHITE, 0.30),
                    frac,
                )
                kw     = kw_fn(v)
                if dist == "norm":
                    sig = kw.get("sigma_scale", 1.0)
                    y_pdf = np.exp(-0.5*(raw_x/sig)**2)/(sig*np.sqrt(2*PI))
                else:
                    y_pdf = _pdf(dist, raw_x, **kw)
                y_pdf  = np.where(np.isfinite(y_pdf), y_pdf, 0.0)
                peak   = float(y_pdf.max()) if y_pdf.max()>1e-9 else 1.0
                ph     = (row_h - 1.00) * 0.88
                y_sc   = y_pdf / peak * ph
                pw     = col_w - 0.40
                x_px   = (raw_x-xlo)/(xhi-xlo)*pw - pw/2

                step   = max(1, len(x_px)//60)
                pts3d  = [np.array([float(x_px[k]),
                                    float(y_sc[k])+cy-ph*0.42, 0.003])
                          for k in range(0, len(x_px), step)
                          if y_sc[k] > 0.005]
                if len(pts3d) >= 2:
                    sp = VMobject(stroke_color=col,
                                  stroke_width=1.5, stroke_opacity=0.88)
                    sp.set_points_smoothly(pts3d)
                    sweep_grp.add(sp)

            # Axis
            pw = col_w - 0.40
            sweep_grp.add(Line(
                [cx - pw/2, cy - (row_h-1.0)*0.42, 0.002],
                [cx + pw/2, cy - (row_h-1.0)*0.42, 0.002],
                stroke_color=P["axis"], stroke_width=1.1,
            ))

            # Title
            sweep_grp.add(
                Text(title, font_size=14, color=col_base)
                .move_to([cx, cy + (row_h-1.0)*0.50 + 0.14, 0.003])
            )

            # Parameter gradient bar
            bar_w  = pw * 0.80
            bar_h2 = 0.10
            for j in range(20):
                frac2 = j / 19
                col2  = interpolate_color(
                    interpolate_color(col_base, BLACK, 0.40),
                    interpolate_color(col_base, WHITE, 0.30),
                    frac2,
                )
                bar_seg = Rectangle(
                    width=bar_w/20, height=bar_h2,
                    fill_color=col2, fill_opacity=0.85, stroke_width=0,
                )
                bar_seg.move_to([cx - bar_w/2 + bar_w/40 + j*bar_w/20,
                                  cy + (row_h-1.0)*0.50 + 0.38, 0.003])
                sweep_grp.add(bar_seg)
            sweep_grp.add(
                Text(f"{pname}={pvals[0]}", font_size=9,
                     color=P["label"])
                .move_to([cx - bar_w/2,
                           cy + (row_h-1.0)*0.50 + 0.60, 0.003])
            )
            sweep_grp.add(
                Text(f"{pname}={pvals[-1]}", font_size=9,
                     color=P["label"])
                .move_to([cx + bar_w/2,
                           cy + (row_h-1.0)*0.50 + 0.60, 0.003])
            )

            panels_grid.append(sweep_grp)

        all_grid = VGroup(*panels_grid)
        all_grid.scale(0.90).center()
        all_grid.set_opacity(0)

        self.play(FadeIn(all_grid, shift=UP*0.12, run_time=1.20))
        self.wait(1.20)

        # Pulse each panel
        for panel in panels_grid:
            self.play(Indicate(panel, scale_factor=1.04,
                               run_time=0.35))
            self.wait(0.06)

        self.wait(0.6)
        self._fade_scene(all_grid, lbl)

    # ─────────────────────────────────────────────────────────────────
    # Act 7 — Comparison overlay + closing
    # ─────────────────────────────────────────────────────────────────

    def _act7_comparison_closing(self):
        lbl = self._section("Act 7 — Shape Comparison & Closing")
        self.play(FadeIn(lbl, run_time=0.4))

        overlay = _ComparisonOverlay(
            width=8.0, height=3.2, baseline_y=0.0,
        )
        overlay.scale(0.88).center().shift(UP*0.20)
        overlay.set_opacity(0)

        self.play(FadeIn(overlay, shift=UP*0.12, run_time=1.20))
        self.wait(0.70)

        # Flash each curve colour
        closing_msg = Text(
            "Each distribution encodes assumptions about the data-generating process.",
            font_size=20, color=P["title"],
        )
        closing_msg.move_to([0, -3.20, 0])
        self.play(Write(closing_msg, run_time=1.2))
        self.wait(0.8)

        sub_msg = Text(
            "Choosing the right distribution is the foundation of statistical modelling.",
            font_size=18, color=P["subtitle"],
        )
        sub_msg.move_to([0, -3.80, 0])
        self.play(FadeIn(sub_msg, shift=UP*0.08, run_time=0.8))
        self.wait(1.0)

        self._fade_scene(overlay, closing_msg, sub_msg, lbl)

        # End card
        bg_end = Rectangle(width=16, height=9,
                            fill_color=P["bg"],
                            fill_opacity=1.0, stroke_width=0)
        title_end = Text("Probability Distributions",
                         font_size=44, color=P["title"])
        sub_end   = Text(
            "From Bernoulli to Pareto — each tells a story",
            font_size=24, color=P["subtitle"],
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