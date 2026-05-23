"""
manim_stats/scenes/demo_clt.py
================================
CLTDemo — A complete, cinematic Central Limit Theorem demonstration scene.

Story arc (8 acts)
------------------
Act 0  Title card + hook
       "No matter the shape of the population…"
       Three wildly different population curves pulse onto screen.
       "…the sample mean always converges to normal."

Act 1  Population zoo
       Three panels: Uniform, Exponential, Bimodal.
       Each curve animates in with its μ, σ badges.
       Population dots scatter on a number line below each curve.

Act 2  What is a sample mean?
       Focus on the Exponential panel.
       n=5 dots are drawn one by one from the distribution.
       A bracket collapses to the mean.  Mean dot labelled x̄.
       Formula: x̄ = (x₁ + x₂ + … + xₙ) / n

Act 3  Build one histogram — slow
       10 full DrawSample→ExtractMean→AddToHistogram cycles.
       Each bar flash highlighted.
       Live k-counter badge updates.

Act 4  Convergence speed-up
       "Let's draw 100 samples..."
       Fast burst: bars blur upward.
       At k=50:  theoretical curve appears dashed.
       At k=100: curve becomes solid, histogram clearly matches.
       Overlay reveal animation.

Act 5  The n effect
       Side-by-side: n=1, n=5, n=30, n=100.
       Each panel pre-loaded with 80 samples.
       All four histograms shown simultaneously.
       SE arrow shrinks from left to right.
       "SE = σ/√n" formula with live substitution.

Act 6  Population independence
       Three simultaneous panels (Uniform, Exponential, Bimodal).
       All run 80 fast samples in parallel.
       All three histograms converge to normal.
       "The shape of the population doesn't matter."

Act 7  Normal Q-Q plot
       After the Exponential histogram converges,
       a Q-Q plot appears beside it.
       Theoretical normal line vs empirical quantiles.
       Points hug the line — visual proof of normality.

Act 8  CLT formal statement + closing
       Full formal statement animates in term by term.
       "√n (X̄ − μ) / σ → N(0,1) as n → ∞"
       Each symbol colour-coded.
       Closing title card.

Scene uses
----------
  manim_stats.inference.sampling_dist  — SamplingDistribution3D,
    BuildPopulation, RunCLT, NarrowSE, ConvergenceRace,
    make_n_effect_row, make_clt_comparison

Dependencies
------------
  manim (CE or GL), numpy, scipy (optional)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List

from manim import (
    Scene,
    VGroup, Group,
    Rectangle, RoundedRectangle, Square,
    Circle, Annulus, Dot, Polygon, Line, DashedLine, Arc,
    Arrow, DoubleArrow,
    Text, MathTex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut, GrowFromCenter,
    Create, Write, Uncreate,
    Indicate, Flash,
    Rotate,
    Transform, ReplacementTransform,
    ValueTracker,
    interpolate_color, color_to_rgb,
    always_redraw,
    BLACK, WHITE,
    GREY,  GREY_A,  GREY_B,  GREY_C,  GREY_D,
    RED,   RED_A,   RED_B,   RED_C,   RED_D,
    GREEN, GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E,
    BLUE,  BLUE_A,  BLUE_B,  BLUE_C,  BLUE_D,  BLUE_E,
    YELLOW, YELLOW_A, YELLOW_E,
    ORANGE, TEAL, TEAL_A, TEAL_B, TEAL_C,
    GOLD,  GOLD_A,  GOLD_B,  GOLD_C,  GOLD_D,
    PURPLE_A, PURPLE_B, MAROON, PINK,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    ManimColor,
    VMobject,
)

# ─────────────────────────────────────────────────────────────────────────────
# Import sampling_dist components (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from inference.sampling_dist import (
        SamplingDistribution3D,
        BuildPopulation,
        RunCLT,
        NarrowSE,
        ConvergenceRace,
        FlashBin,
        make_n_effect_row,
        make_clt_comparison,
        _PopulationPanel,
        _HistogramPanel,
        _SampleStrip,
        _pop_pdf,
        _pop_stats,
        _pop_x_range,
        _map_x,
        _curve_verts,
        _sample_pop,
        PAL as SPAL,
    )
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    SPAL = {}


# ─────────────────────────────────────────────────────────────────────────────
# Scene palette
# ─────────────────────────────────────────────────────────────────────────────

P = {
    # Population colours (one per distribution)
    "uniform_fill":   "#1A3A5C",
    "uniform_ridge":  "#70B4EE",
    "exp_fill":       "#2A1A5C",
    "exp_ridge":      "#B070EE",
    "bimodal_fill":   "#1A3A20",
    "bimodal_ridge":  "#70EE90",

    # Sample dots
    "dot":            "#F0C040",
    "dot_mean":       "#FF6040",
    "mean_beam":      "#FF9060",

    # Histogram
    "bar":            "#3A7AC8",
    "bar_lite":       "#60A0F0",
    "bar_side":       "#1A3A68",
    "bar_flash":      "#FFD060",

    # Theory curve
    "theory":         "#F0C040",
    "theory_dash":    "#806010",

    # SE arrow
    "se":             "#D4AF37",
    "se_label":       "#F8E090",

    # Formula terms
    "f_xbar":         "#70C8FF",
    "f_mu":           "#FFD060",
    "f_sigma":        "#FF9060",
    "f_n":            "#90FF90",
    "f_norm":         "#E080FF",
    "f_eq":           "#C8D8E8",

    # QQ plot
    "qq_dot":         "#60C0FF",
    "qq_line":        "#FFD060",
    "qq_axis":        "#506070",

    # Title / UI
    "title":          "#D8F0FF",
    "subtitle":       "#8090A8",
    "badge_bg":       "#0C1018",
    "badge_border":   "#2A3848",
    "section":        "#607080",
    "bg":             "#080C12",
    "bg_panel":       "#0C1018",
    "panel_border":   "#2A3848",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared scene helpers
# ─────────────────────────────────────────────────────────────────────────────

def _title_card(main: str, sub: str = "",
                main_fs: int = 44, sub_fs: int = 22) -> VGroup:
    g  = VGroup()
    bg = Rectangle(width=16, height=9,
                   fill_color=P["bg"],
                   fill_opacity=1.0, stroke_width=0)
    g.add(bg)
    ml = Text(main, font_size=main_fs, color=P["title"])
    ml.move_to([0, 0.35 if sub else 0, 0])
    g.add(ml)
    if sub:
        sl = Text(sub, font_size=sub_fs, color=P["subtitle"])
        sl.move_to([0, -0.50, 0])
        g.add(sl)
    return g


def _section_label(text: str) -> Text:
    return Text(text, font_size=16, color=P["section"])


def _badge(text: str, color: str = None) -> VGroup:
    """Small floating info badge."""
    g   = VGroup()
    col = color or P["subtitle"]
    lbl = Text(text, font_size=15, color=col)
    bg  = RoundedRectangle(
        width=lbl.width + 0.22,
        height=lbl.height + 0.14,
        corner_radius=0.06,
        fill_color=P["badge_bg"],
        fill_opacity=0.90,
        stroke_color=col,
        stroke_width=0.7,
    )
    bg.move_to(ORIGIN)
    lbl.move_to([0, 0, 0.001])
    g.add(bg, lbl)
    return g


def _norm_pdf_scene(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * PI))


# ─────────────────────────────────────────────────────────────────────────────
# Mini population curve (self-contained, no sampling_dist dependency)
# ─────────────────────────────────────────────────────────────────────────────

class _MiniPopCurve(VGroup):
    """
    Compact population curve panel with:
      - Base fill polygon (3 layers)
      - Ridge spine
      - μ and σ badges
      - Title label
      - Optional dot strip below

    Completely self-contained — does not require sampling_dist.
    """

    def __init__(
        self,
        pop_type:   str,
        params:     dict,
        title:      str,
        fill_color: str,
        ridge_color: str,
        width:      float = 3.5,
        height:     float = 2.2,
        baseline_y: float = 0.0,
        show_dots:  bool  = True,
        n_dots:     int   = 60,
        seed:       int   = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pop_type  = pop_type
        self._params    = params
        self._width     = width
        self._height    = height
        self._baseline  = baseline_y

        x_lo, x_hi = self._x_range(pop_type, params)
        raw_x       = np.linspace(x_lo, x_hi, 400)
        y_raw       = self._pdf(pop_type, params, raw_x)
        y_raw       = np.where(np.isfinite(y_raw), y_raw, 0.0)
        peak        = float(y_raw.max()) if y_raw.max() > 1e-9 else 1.0
        y_scaled    = y_raw / peak * height
        x_px        = (raw_x - x_lo) / (x_hi - x_lo) * width - width / 2

        ao_col = interpolate_color(fill_color, BLACK, 0.55)

        # ── 1. AO under-layer ─────────────────────────────────────────
        verts_ao = self._verts(x_px, y_scaled * 0.95, baseline_y)
        if len(verts_ao) >= 3:
            self.add(Polygon(*verts_ao,
                             fill_color=ao_col,
                             fill_opacity=0.55, stroke_width=0).shift([0.04,-0.04,0]))

        # ── 2. Base fill ──────────────────────────────────────────────
        verts = self._verts(x_px, y_scaled, baseline_y)
        if len(verts) >= 3:
            self.add(Polygon(*verts,
                             fill_color=fill_color,
                             fill_opacity=0.82, stroke_width=0))

        # ── 3. Inner lite band ────────────────────────────────────────
        lite_col = interpolate_color(fill_color, WHITE, 0.22)
        verts_l  = self._verts(x_px, y_scaled * 0.45, baseline_y)
        if len(verts_l) >= 3:
            self.add(Polygon(*verts_l,
                             fill_color=lite_col,
                             fill_opacity=0.28, stroke_width=0,
                             ).shift([0, 0, 0.001]))

        # ── 4. Ridge spine ────────────────────────────────────────────
        step  = max(1, len(x_px) // 80)
        pts3d = [np.array([float(x_px[i]), float(y_scaled[i]), 0.004])
                 for i in range(0, len(x_px), step) if y_scaled[i] > 0.01]
        if len(pts3d) >= 2:
            spine = VMobject(stroke_color=ridge_color,
                             stroke_width=1.8, stroke_opacity=0.85)
            spine.set_points_smoothly(pts3d)
            self.add(spine)

        # ── 5. Axis ───────────────────────────────────────────────────
        self.add(Line(
            [-width/2, baseline_y, 0.001],
            [ width/2, baseline_y, 0.001],
            stroke_color=P["qq_axis"], stroke_width=1.3,
        ))

        # ── 6. Population dots below axis ─────────────────────────────
        if show_dots:
            rng    = np.random.default_rng(seed)
            sample = _sample_pop(pop_type, params, n_dots, rng) if _SD_AVAILABLE else (
                rng.uniform(x_lo, x_hi, n_dots)
            )
            for val in sample:
                val  = float(np.clip(val, x_lo, x_hi))
                xdot = (val - x_lo) / (x_hi - x_lo) * width - width / 2
                ydot = baseline_y - np.random.uniform(0.06, 0.20)
                d    = Dot(radius=0.035,
                           point=[xdot, ydot, 0.002],
                           color=P["dot"],
                           fill_opacity=0.60)
                self.add(d)

        # ── 7. Title + stat badges ────────────────────────────────────
        self.add(Text(title, font_size=17, color=ridge_color)
                 .move_to([0, baseline_y + height + 0.35, 0.003]))

        mu, sigma = self._stats(pop_type, params)
        stat_str  = f"μ={mu:.2f}  σ={sigma:.2f}"
        self.add(Text(stat_str, font_size=13, color=P["subtitle"])
                 .move_to([0, baseline_y + height + 0.08, 0.003]))

        self._mu    = mu
        self._sigma = sigma
        self._x_lo  = x_lo
        self._x_hi  = x_hi

    # ──── static helpers ─────────────────────────────────────────────

    @staticmethod
    def _verts(x_px, y_scaled, baseline):
        return (
            [[float(x_px[0]), baseline, 0]]
            + [[float(x), float(y), 0]
               for x, y in zip(x_px, y_scaled)]
            + [[float(x_px[-1]), baseline, 0]]
        )

    @staticmethod
    def _pdf(pop_type, params, x):
        if _SD_AVAILABLE:
            return _pop_pdf(pop_type, params, x)
        # fallback: normal
        mu = params.get("mu", 0.0)
        sig = params.get("sigma", 1.0)
        return _norm_pdf_scene(x, mu, sig)

    @staticmethod
    def _stats(pop_type, params):
        if _SD_AVAILABLE:
            return _pop_stats(pop_type, params)
        return params.get("mu", 0.0), params.get("sigma", 1.0)

    @staticmethod
    def _x_range(pop_type, params):
        if _SD_AVAILABLE:
            return _pop_x_range(pop_type, params)
        mu  = params.get("mu", 0.0)
        sig = params.get("sigma", 1.0)
        return mu - 4*sig, mu + 4*sig


# ─────────────────────────────────────────────────────────────────────────────
# Mini histogram (self-contained 3-D style bars)
# ─────────────────────────────────────────────────────────────────────────────

class _MiniHistogram(VGroup):
    """
    Compact 3-D histogram from a list of sample means.

    Bars have: front face, right side face, top cap, base AO shadow.
    Includes the theoretical N(μ, σ²/n) overlay curve.
    """

    def __init__(
        self,
        means:      np.ndarray,
        mu:         float,
        sigma_n:    float,
        n_bins:     int   = 18,
        width:      float = 3.5,
        height:     float = 2.2,
        baseline_y: float = 0.0,
        show_theory: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        means = np.asarray(means)
        if len(means) == 0:
            return

        # Bin setup: ±3.5 SE around mu
        x_lo = mu - 3.8 * sigma_n
        x_hi = mu + 3.8 * sigma_n
        edges   = np.linspace(x_lo, x_hi, n_bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        counts, _ = np.histogram(means, bins=edges)
        max_count = max(counts.max(), 1)
        bin_w_px  = width / n_bins
        dx        = bin_w_px * 0.07   # 3-D right-face depth
        cap_h     = 0.040

        # ── Axis ──────────────────────────────────────────────────────
        self.add(Line(
            [-width/2, baseline_y, 0.001],
            [ width/2, baseline_y, 0.001],
            stroke_color=P["qq_axis"], stroke_width=1.2,
        ))

        # Tick marks
        for i in range(n_bins + 1):
            xp = i * bin_w_px - width/2
            self.add(Line(
                [xp, baseline_y - 0.05, 0.001],
                [xp, baseline_y + 0.05, 0.001],
                stroke_color=P["qq_axis"], stroke_width=0.7,
            ))

        # ── Bars ──────────────────────────────────────────────────────
        for i, count in enumerate(counts):
            if count == 0:
                continue
            h   = (count / max_count) * height
            xl  = i * bin_w_px - width / 2
            xr  = xl + bin_w_px
            by  = baseline_y

            # Front face
            self.add(Rectangle(
                width=bin_w_px - 0.012,
                height=h,
                fill_color=P["bar"],
                fill_opacity=0.88,
                stroke_color=P["bar_side"],
                stroke_width=0.4,
            ).move_to([xl + bin_w_px/2, by + h/2, 0.002]))

            # Right side
            side = Polygon(
                [xr - 0.006, by,   0.002],
                [xr - 0.006 + dx, by + dx*0.5,   0.002],
                [xr - 0.006 + dx, by + h + dx*0.5, 0.002],
                [xr - 0.006, by + h, 0.002],
                fill_color=P["bar_side"],
                fill_opacity=0.80, stroke_width=0,
            )
            side.shift([0, 0, 0.001])
            self.add(side)

            # Top cap
            top = Polygon(
                [xl + 0.006,     by + h,           0.004],
                [xr - 0.006,     by + h,           0.004],
                [xr - 0.006 + dx, by + h + dx*0.5, 0.004],
                [xl + 0.006 + dx, by + h + dx*0.5, 0.004],
                fill_color=P["bar_lite"],
                fill_opacity=0.88, stroke_width=0,
            )
            self.add(top)

            # AO base shadow
            self.add(Rectangle(
                width=bin_w_px - 0.012,
                height=0.040,
                fill_color="#040608",
                fill_opacity=0.55, stroke_width=0,
            ).move_to([xl + bin_w_px/2, by + 0.020, 0.003]))

        # ── Theoretical overlay ───────────────────────────────────────
        if show_theory and len(means) >= 5:
            raw_x  = np.linspace(x_lo, x_hi, 300)
            y_pdf  = _norm_pdf_scene(raw_x, mu, sigma_n)
            y_pdf  = np.where(np.isfinite(y_pdf), y_pdf, 0.0)
            bin_w_raw = (x_hi - x_lo) / n_bins
            scale  = (len(means) * bin_w_raw
                      * height / max_count)
            x_px   = (raw_x - x_lo) / (x_hi - x_lo) * width - width/2
            y_px   = y_pdf * scale

            # Fill
            verts = (
                [[float(x_px[0]), baseline_y, 0]]
                + [[float(x), float(y), 0]
                   for x, y in zip(x_px, y_px)]
                + [[float(x_px[-1]), baseline_y, 0]]
            )
            if len(verts) >= 3:
                self.add(Polygon(*verts,
                                 fill_color="#2A1A00",
                                 fill_opacity=0.35,
                                 stroke_width=0).shift([0,0,0.006]))

            # Curve
            step  = max(1, len(x_px)//80)
            pts3d = [np.array([float(x_px[i]), float(y_px[i]), 0.008])
                     for i in range(0, len(x_px), step)
                     if y_px[i] > 0.005]
            if len(pts3d) >= 2:
                sp = VMobject(stroke_color=P["theory"],
                              stroke_width=2.0, stroke_opacity=0.90)
                sp.set_points_smoothly(pts3d)
                self.add(sp)

        self._counts   = counts
        self._n_bins   = n_bins
        self._x_lo     = x_lo
        self._x_hi     = x_hi
        self._width    = width
        self._height   = height
        self._baseline = baseline_y


# ─────────────────────────────────────────────────────────────────────────────
# QQ plot (normal quantile–quantile)
# ─────────────────────────────────────────────────────────────────────────────

class _QQPlot(VGroup):
    """
    Normal Q-Q plot comparing empirical quantiles of sample means
    to theoretical N(0,1) quantiles.

    Components:
      1. Axes (x = theoretical, y = empirical)
      2. Reference line (45° through mean/std)
      3. Data dots
      4. Confidence band (dashed lines either side)
      5. Title and axis labels
    """

    def __init__(
        self,
        means:      np.ndarray,
        mu:         float,
        sigma_n:    float,
        width:      float = 3.5,
        height:     float = 3.0,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if len(means) < 4:
            return

        z0    = 0.001
        n     = len(means)
        sorted_means = np.sort(means)

        # Theoretical quantiles (Filliben's approximation)
        probs = (np.arange(1, n+1) - 0.375) / (n + 0.25)
        try:
            from scipy.stats import norm as _scnorm
            z_theory = _scnorm.ppf(probs)
        except ImportError:
            # Rational approximation for normal ppf
            def _ppf_approx(p):
                if p <= 0: return -4.0
                if p >= 1: return  4.0
                t = np.sqrt(-2 * np.log(min(p, 1-p)))
                c = [2.515517, 0.802853, 0.010328]
                d = [1.432788, 0.189269, 0.001308]
                z = t - (c[0]+c[1]*t+c[2]*t**2)/(1+d[0]*t+d[1]*t**2+d[2]*t**3)
                return z if p > 0.5 else -z
            z_theory = np.array([_ppf_approx(p) for p in probs])

        # Standardise empirical quantiles
        emp_std   = float(np.std(sorted_means)) or 1.0
        emp_mean  = float(np.mean(sorted_means))
        z_emp     = (sorted_means - emp_mean) / emp_std

        # Axis range
        z_min = min(float(z_theory.min()), float(z_emp.min()), -3.0)
        z_max = max(float(z_theory.max()), float(z_emp.max()),  3.0)
        rng   = z_max - z_min

        def to_px_x(z): return (z - z_min) / rng * width  - width/2
        def to_px_y(z): return (z - z_min) / rng * height + baseline_y

        # ── Axes ──────────────────────────────────────────────────────
        x_ax = Line(
            [-width/2, baseline_y, z0+0.001],
            [ width/2, baseline_y, z0+0.001],
            stroke_color=P["qq_axis"], stroke_width=1.4,
        )
        y_ax = Line(
            [-width/2, baseline_y,          z0+0.001],
            [-width/2, baseline_y + height, z0+0.001],
            stroke_color=P["qq_axis"], stroke_width=1.4,
        )
        self.add(x_ax, y_ax)

        # Axis ticks
        for tv in [-2, -1, 0, 1, 2]:
            xp = to_px_x(float(tv))
            yp = to_px_y(float(tv))
            if z_min <= tv <= z_max:
                self.add(Line([xp, baseline_y-0.06, z0+0.001],
                              [xp, baseline_y+0.06, z0+0.001],
                              stroke_color=P["qq_axis"], stroke_width=0.9))
                self.add(Text(str(tv), font_size=10,
                              color=P["subtitle"])
                         .move_to([xp, baseline_y - 0.22, z0+0.001]))
                self.add(Line([-width/2-0.06, yp, z0+0.001],
                              [-width/2+0.06, yp, z0+0.001],
                              stroke_color=P["qq_axis"], stroke_width=0.9))

        # ── 45° reference line ────────────────────────────────────────
        ref_lo = [to_px_x(z_min), to_px_y(z_min), z0+0.002]
        ref_hi = [to_px_x(z_max), to_px_y(z_max), z0+0.002]
        ref_line = Line(ref_lo, ref_hi,
                        stroke_color=P["qq_line"],
                        stroke_width=1.8, stroke_opacity=0.85)
        self.add(ref_line)

        # ── Confidence band (±1.36/√n envelope) ──────────────────────
        env_half = 1.36 / np.sqrt(n)
        for sign in [-1, 1]:
            band_pts = [
                np.array([to_px_x(zt),
                          to_px_y(zt + sign*env_half), z0+0.002])
                for zt in np.linspace(z_min, z_max, 80)
            ]
            band = VMobject(stroke_color=P["theory_dash"] if sign == 1
                            else P["theory_dash"],
                            stroke_width=1.0,
                            stroke_opacity=0.50)
            band.set_points_smoothly(band_pts)
            self.add(band)

        # ── Data dots ─────────────────────────────────────────────────
        for zt, ze in zip(z_theory, z_emp):
            xp = to_px_x(float(zt))
            yp = to_px_y(float(ze))
            if (-width/2 <= xp <= width/2
                    and baseline_y <= yp <= baseline_y + height):
                d = Dot(radius=0.045,
                        point=[xp, yp, z0+0.004],
                        color=P["qq_dot"],
                        fill_opacity=0.80)
                self.add(d)

        # ── Labels ────────────────────────────────────────────────────
        self.add(Text("Normal Q-Q Plot", font_size=15,
                      color=P["title"])
                 .move_to([0, baseline_y + height + 0.30, z0+0.003]))
        self.add(Text("Theoretical quantiles", font_size=11,
                      color=P["subtitle"])
                 .move_to([0, baseline_y - 0.42, z0+0.002]))
        self.add(Text("Sample quantiles", font_size=11,
                      color=P["subtitle"])
                 .rotate(PI/2)
                 .move_to([-width/2 - 0.42,
                            baseline_y + height/2, z0+0.002]))


# ─────────────────────────────────────────────────────────────────────────────
# Live sample-mean counter badge
# ─────────────────────────────────────────────────────────────────────────────

class _LiveCounter(VGroup):
    """
    Live k-counter + SE badge.

    Shows:
      k = {n}  samples drawn
      SE = σ/√n = {val}
    """

    def __init__(
        self,
        k:      int,
        se_val: float,
        n:      int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        bg = RoundedRectangle(
            width=3.20, height=0.88,
            corner_radius=0.09,
            fill_color=P["badge_bg"],
            fill_opacity=0.92,
            stroke_color=P["badge_border"],
            stroke_width=0.8,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        rows = [
            (f"k = {k}  samples drawn", P["subtitle"]),
            (f"SE = σ/√{n} = {se_val:.4f}", P["se_label"]),
        ]
        for i, (txt, col) in enumerate(rows):
            lbl = Text(txt, font_size=14, color=col,
                       font="monospace")
            lbl.move_to([0, 0.20 - i * 0.38, z0+0.002])
            self.add(lbl)


# ─────────────────────────────────────────────────────────────────────────────
# CLT formal statement
# ─────────────────────────────────────────────────────────────────────────────

class _CLTStatement(VGroup):
    """
    Colour-coded formal CLT statement.

    √n (X̄ − μ) / σ  →  N(0,1)  as  n → ∞

    Arranged in three lines:
      Line 1: symbolic statement
      Line 2: "Equivalently:  X̄ ~ N(μ, σ²/n)"
      Line 3: conditions
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        z0 = 0.001

        bg = RoundedRectangle(
            width=9.50, height=3.80,
            corner_radius=0.14,
            fill_color=P["bg_panel"],
            fill_opacity=0.96,
            stroke_color=P["badge_border"],
            stroke_width=1.0,
        )
        bg.move_to(ORIGIN)
        self.add(bg)

        # Strip
        strip = RoundedRectangle(
            width=0.09, height=3.72,
            corner_radius=0.045,
            fill_color=P["f_norm"],
            fill_opacity=0.85, stroke_width=0,
        )
        strip.move_to([-4.70, 0, z0+0.001])
        self.add(strip)

        title = Text("Central Limit Theorem",
                     font_size=24, color=P["title"])
        title.move_to([0, 1.55, z0+0.002])
        self.add(title)

        self.add(Line(
            [-4.60, 1.20, z0+0.002],
            [ 4.60, 1.20, z0+0.002],
            stroke_color=P["badge_border"], stroke_width=0.7,
        ))

        # ── Line 1: main formula ──────────────────────────────────────
        pieces1 = [
            ("\\frac{\\sqrt{n}\\,(\\bar{X}", P["f_xbar"]),
            ("\\;-\\;",                       P["f_eq"]),
            ("\\mu",                          P["f_mu"]),
            (")}{",                           P["f_eq"]),
            ("\\sigma",                       P["f_sigma"]),
            ("}",                             P["f_eq"]),
            ("\\;\\xrightarrow{d}\\;",        P["f_eq"]),
            ("\\mathcal{N}(0,\\,1)",          P["f_norm"]),
            ("\\quad\\text{as }",             P["f_eq"]),
            ("n\\to\\infty",                  P["f_n"]),
        ]
        self.sym_pieces = VGroup()
        x_cur = -4.40
        for tex_str, col in pieces1:
            try:
                m = MathTex(tex_str, font_size=28, color=col)
            except Exception:
                m = Text(tex_str.replace("\\",""), font_size=22, color=col)
            m.move_to([x_cur + m.width/2, 0.52, z0+0.003])
            self.sym_pieces.add(m)
            x_cur += m.width + 0.06
        self.add(self.sym_pieces)

        # ── Line 2: equivalently ──────────────────────────────────────
        try:
            eq2 = MathTex(
                r"\text{Equivalently:}\quad"
                r"\bar{X} \;\sim\; \mathcal{N}\!\left(\mu,\;"
                r"\frac{\sigma^2}{n}\right)",
                font_size=24, color=P["f_eq"],
            )
        except Exception:
            eq2 = Text("Equivalently:  X̄ ~ N(μ, σ²/n)",
                       font_size=20, color=P["f_eq"])
        eq2.move_to([0, -0.18, z0+0.003])
        self.add(eq2)

        # ── Line 3: conditions ────────────────────────────────────────
        cond = Text(
            "Conditions:  finite μ and σ²,  i.i.d. samples,"
            "  n sufficiently large",
            font_size=15, color=P["subtitle"],
        )
        cond.move_to([0, -0.90, z0+0.003])
        self.add(cond)

        # ── Line 4: interpretation ────────────────────────────────────
        interp = Text(
            "The sampling distribution of the mean is normal"
            " regardless of the population shape.",
            font_size=15, color=P["subtitle"],
        )
        interp.move_to([0, -1.40, z0+0.003])
        self.add(interp)


# ─────────────────────────────────────────────────────────────────────────────
# SE effect panel (n=1 → n=100, SE arrow shrinks)
# ─────────────────────────────────────────────────────────────────────────────

class _SEPanel(VGroup):
    """
    One column of the n-effect comparison showing a histogram
    with SE arrow and n label.

    Parameters
    ----------
    means   : np.ndarray
    mu, sigma_n : distribution parameters for overlay
    n       : sample size (label)
    width, height, baseline_y : geometry
    """

    def __init__(
        self,
        means:      np.ndarray,
        mu:         float,
        sigma_n:    float,
        n:          int,
        width:      float = 2.80,
        height:     float = 2.20,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        # Histogram
        hist = _MiniHistogram(
            means=means,
            mu=mu, sigma_n=sigma_n,
            n_bins=16,
            width=width, height=height,
            baseline_y=baseline_y,
        )
        self.add(hist)

        # SE arrow
        px_per_unit = width / (7.6 * sigma_n)  # ±3.8 SE range → full width
        se_px_half  = sigma_n * px_per_unit
        arr = DoubleArrow(
            start=[-se_px_half, baseline_y - 0.50, z0+0.002],
            end  =[ se_px_half, baseline_y - 0.50, z0+0.002],
            stroke_color=P["se"], stroke_width=1.8,
            tip_length=0.13, buff=0,
        )
        self.add(arr)

        se_lbl = Text(f"SE={sigma_n:.3f}", font_size=12,
                      color=P["se_label"])
        se_lbl.move_to([0, baseline_y - 0.82, z0+0.002])
        self.add(se_lbl)

        # n label
        n_lbl = Text(f"n = {n}", font_size=18, color=P["title"])
        n_lbl.move_to([0, baseline_y + height + 0.30, z0+0.003])
        self.add(n_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# The full demo scene
# ─────────────────────────────────────────────────────────────────────────────

class CLTDemo(Scene):
    """
    Central Limit Theorem cinematic demonstration.

    Run with:
        manim -pql demo_clt.py CLTDemo
    For high quality:
        manim -pqh demo_clt.py CLTDemo
    """

    # ── Population parameters ─────────────────────────────────────────
    POP_TYPE  = "exponential"
    POP_PARAMS = {"lam": 1.0}
    N_MAIN    = 5
    SIGMA_POP = 1.0          # σ for exponential(lam=1)
    MU_POP    = 1.0          # μ for exponential(lam=1)

    def construct(self):
        self._act0_title()
        self._act1_population_zoo()
        self._act2_what_is_sample_mean()
        self._act3_build_histogram_slow()
        self._act4_convergence_speedup()
        self._act5_n_effect()
        self._act6_population_independence()
        self._act7_qq_plot()
        self._act8_clt_statement()

    # ─────────────────────────────────────────────────────────────────
    # Act 0 — Title / hook
    # ─────────────────────────────────────────────────────────────────

    def _act0_title(self):
        bg = Rectangle(width=16, height=9,
                       fill_color=P["bg"],
                       fill_opacity=1.0, stroke_width=0)
        self.add(bg)

        # Three population curves pulse in
        pop_specs = [
            ("uniform",     {"a": 0.0, "b": 1.0},
             "Uniform",     P["uniform_fill"], P["uniform_ridge"]),
            ("exponential", {"lam": 1.0},
             "Exponential", P["exp_fill"],     P["exp_ridge"]),
            ("bimodal",     {"mu1": -1.8, "mu2": 1.8,
                              "sigma": 0.6, "mix": 0.5},
             "Bimodal",     P["bimodal_fill"], P["bimodal_ridge"]),
        ]
        curves = VGroup()
        for i, (pt, pm, title, fc, rc) in enumerate(pop_specs):
            c = _MiniPopCurve(pt, pm, title, fc, rc,
                              width=2.6, height=1.6,
                              baseline_y=0.0,
                              show_dots=False)
            c.scale(0.90)
            c.move_to([(i-1) * 3.6, 0.60, 0])
            curves.add(c)

        self.play(
            *[FadeIn(c, shift=UP*0.20,
                     rate_func=rate_functions.ease_out_back,
                     run_time=1.0)
              for c in curves],
        )
        self.wait(0.3)

        # Hook text
        hook = Text("No matter the shape of the population…",
                    font_size=27, color=P["subtitle"])
        hook.move_to([0, -1.30, 0])
        self.play(Write(hook, run_time=1.2))
        self.wait(0.4)

        punchline = Text(
            "…the sampling distribution of the mean converges to normal.",
            font_size=25, color=P["title"],
        )
        punchline.move_to([0, -1.95, 0])
        self.play(FadeIn(punchline, shift=UP*0.12, run_time=0.9))
        self.wait(0.7)

        # Main title
        main_title = Text("Central Limit Theorem",
                          font_size=46, color=P["title"])
        main_title.move_to([0, -3.00, 0])
        self.play(FadeIn(main_title,
                         rate_func=rate_functions.ease_out_back,
                         run_time=0.8))
        self.wait(0.9)

        self.play(
            FadeOut(curves),
            FadeOut(hook),
            FadeOut(punchline),
            FadeOut(main_title),
            FadeOut(bg),
            run_time=0.70,
        )

    # ─────────────────────────────────────────────────────────────────
    # Act 1 — Population zoo
    # ─────────────────────────────────────────────────────────────────

    def _act1_population_zoo(self):
        act_lbl = _section_label("Act 1 — Population Zoo")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        zoo_specs = [
            ("uniform",     {"a": 0.0, "b": 1.0},
             "Uniform[0,1]",   P["uniform_fill"], P["uniform_ridge"]),
            ("exponential", {"lam": 1.0},
             "Exp(λ=1)",        P["exp_fill"],     P["exp_ridge"]),
            ("bimodal",     {"mu1": -1.8, "mu2": 1.8,
                              "sigma": 0.6, "mix": 0.5},
             "Bimodal",         P["bimodal_fill"], P["bimodal_ridge"]),
        ]

        self._zoo_curves = VGroup()
        for i, (pt, pm, title, fc, rc) in enumerate(zoo_specs):
            c = _MiniPopCurve(
                pt, pm, title, fc, rc,
                width=3.4, height=2.2,
                baseline_y=0.10,
                show_dots=True, n_dots=50, seed=i*7,
            )
            c.move_to([(i-1) * 4.60, 0.50, 0])
            self._zoo_curves.add(c)

        for c in self._zoo_curves:
            self.play(FadeIn(c, shift=UP*0.18,
                             run_time=0.75))

        # Stat badges
        badges = VGroup()
        for i, (pt, pm, *_) in enumerate(zoo_specs):
            if _SD_AVAILABLE:
                mu, sigma = _pop_stats(pt, pm)
            else:
                mu, sigma = 0.5, 0.29
            b = _badge(f"μ={mu:.2f}  σ={sigma:.2f}",
                       zoo_specs[i][4])
            b.move_to([(i-1)*4.60, -1.65, 0])
            badges.add(b)
        self.play(FadeIn(badges, run_time=0.7))
        self.wait(0.8)

        # "All different shapes — all will produce normal sample means"
        msg = Text(
            "All very different shapes. "
            "The CLT says their sample means will all be normal.",
            font_size=20, color=P["subtitle"],
        )
        msg.move_to([0, -2.60, 0])
        self.play(Write(msg, run_time=1.3))
        self.wait(0.8)

        self.play(
            FadeOut(badges), FadeOut(msg),
            FadeOut(act_lbl),
            run_time=0.5,
        )
        # Keep zoo_curves visible but shrink for next act

    # ─────────────────────────────────────────────────────────────────
    # Act 2 — What is a sample mean?
    # ─────────────────────────────────────────────────────────────────

    def _act2_what_is_sample_mean(self):
        act_lbl = _section_label("Act 2 — What is a Sample Mean?")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Focus on exponential curve (index 1)
        exp_curve = self._zoo_curves[1]
        others    = VGroup(self._zoo_curves[0], self._zoo_curves[2])

        self.play(
            others.animate.set_opacity(0.22),
            exp_curve.animate.scale(1.15).move_to([-3.20, 0.55, 0]),
            run_time=0.80,
        )

        # Number line below the curve
        n_strip = 5
        rng     = np.random.default_rng(99)
        lam     = 1.0
        sample  = rng.exponential(1.0/lam, n_strip)
        sample  = np.clip(sample, 0.05, 5.0)

        x_lo, x_hi = 0.0, 5.0
        strip_w     = 3.2
        strip_y     = -1.75
        strip_cx    = -3.20

        # Number line
        nline = Line(
            [strip_cx - strip_w/2, strip_y, 0],
            [strip_cx + strip_w/2, strip_y, 0],
            stroke_color=P["qq_axis"], stroke_width=1.4,
        )
        self.add(nline)
        for rv in [0, 1, 2, 3, 4, 5]:
            xp = (rv / 5.0) * strip_w - strip_w/2 + strip_cx
            self.add(Line([xp, strip_y-0.07, 0.001],
                          [xp, strip_y+0.07, 0.001],
                          stroke_color=P["qq_axis"],
                          stroke_width=1.0))
            self.add(Text(str(rv), font_size=11,
                          color=P["subtitle"])
                     .move_to([xp, strip_y-0.26, 0.001]))

        # Drop dots one by one
        dot_mobs = []
        for j, val in enumerate(sample):
            xp  = (val / (x_hi - x_lo)) * strip_w - strip_w/2 + strip_cx
            ystart = strip_y + 1.10
            d   = Dot(radius=0.09,
                      point=[xp, ystart, 0.005],
                      color=P["dot"],
                      fill_opacity=0.90)
            # Subscript label
            try:
                lbl = MathTex(f"x_{{{j+1}}}",
                              font_size=16, color=P["dot"])
            except Exception:
                lbl = Text(f"x{j+1}", font_size=13, color=P["dot"])
            lbl.move_to([xp, ystart + 0.25, 0.006])

            self.play(
                FadeIn(d, shift=DOWN*1.00,
                       rate_func=rate_functions.ease_in_cubic,
                       run_time=0.30),
                FadeIn(lbl, run_time=0.25),
            )
            dot_mobs.append((d, lbl, xp))
            self.wait(0.08)

        # Bracket collapses to mean
        mean_val = float(np.mean(sample))
        mean_px  = (mean_val / (x_hi - x_lo)) * strip_w - strip_w/2 + strip_cx

        self.wait(0.3)
        bracket_l = Line(
            [dot_mobs[0][2],  strip_y + 0.25, 0.006],
            [dot_mobs[-1][2], strip_y + 0.25, 0.006],
            stroke_color=P["f_xbar"], stroke_width=1.8,
        )
        self.play(Create(bracket_l, run_time=0.55))

        # Bracket ticks
        for xp2 in [dot_mobs[0][2], dot_mobs[-1][2]]:
            self.add(Line(
                [xp2, strip_y + 0.17, 0.006],
                [xp2, strip_y + 0.33, 0.006],
                stroke_color=P["f_xbar"], stroke_width=1.6,
            ))

        # Mean dot appears
        mean_dot = Dot(radius=0.13,
                       point=[mean_px, strip_y + 0.25, 0.008],
                       color=P["dot_mean"],
                       fill_opacity=1.0)
        self.play(
            GrowFromCenter(mean_dot,
                           rate_func=rate_functions.ease_out_back,
                           run_time=0.55)
        )

        # x̄ label
        try:
            xbar_lbl = MathTex(r"\bar{x} = " + f"{mean_val:.2f}",
                               font_size=22, color=P["dot_mean"])
        except Exception:
            xbar_lbl = Text(f"x̄ = {mean_val:.2f}",
                            font_size=18, color=P["dot_mean"])
        xbar_lbl.move_to([mean_px, strip_y + 0.62, 0.008])
        self.play(FadeIn(xbar_lbl, shift=UP*0.08, run_time=0.5))

        # Formula
        try:
            formula = MathTex(
                r"\bar{x} = \frac{x_1 + x_2 + \cdots + x_n}{n}",
                font_size=26, color=P["f_eq"],
            )
        except Exception:
            formula = Text("x̄ = (x₁ + … + xₙ) / n",
                           font_size=22, color=P["f_eq"])
        formula.move_to([1.80, -0.10, 0])
        self.play(Write(formula, run_time=1.1))
        self.wait(0.9)

        # Mean dot beams upward (preview of histogram)
        beam = Line(
            [mean_px, strip_y + 0.35, 0.008],
            [mean_px, strip_y + 2.20, 0.008],
            stroke_color=P["mean_beam"],
            stroke_width=1.4, stroke_opacity=0.60,
        )
        self.play(Create(beam, run_time=0.45))
        self.wait(0.30)

        # Clean up act 2
        act2_group = VGroup(
            nline, bracket_l, mean_dot, xbar_lbl,
            formula, beam,
            *[d for d,l,_ in dot_mobs],
            *[l for _,l,_ in dot_mobs],
        )
        self.play(
            FadeOut(act2_group),
            others.animate.set_opacity(1.0),
            exp_curve.animate.scale(1/1.15).move_to([0, 0.50, 0]),
            FadeOut(act_lbl),
            run_time=0.65,
        )

    # ─────────────────────────────────────────────────────────────────
    # Act 3 — Build histogram slow (using SamplingDistribution3D)
    # ─────────────────────────────────────────────────────────────────

    def _act3_build_histogram_slow(self):
        act_lbl = _section_label("Act 3 — Building the Histogram")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Fade out zoo, build full SD panel
        self.play(
            FadeOut(self._zoo_curves),
            run_time=0.50,
        )

        if not _SD_AVAILABLE:
            # Fallback: show message
            msg = Text("(Install manim_stats to see live CLT simulation)",
                       font_size=20, color=P["subtitle"])
            msg.move_to(ORIGIN)
            self.play(FadeIn(msg, run_time=0.5))
            self.wait(1.5)
            self.play(FadeOut(msg, act_lbl, run_time=0.4))
            return

        # Build full SamplingDistribution3D
        self._sd = SamplingDistribution3D.exponential(
            lam=1.0, n=self.N_MAIN,
            n_bins=18,
            pop_panel_w=3.8,
            hist_panel_w=4.8,
            panel_h=2.6,
            baseline_y=0.3,
            max_samples=120,
            show_strip=True,
            show_se=True,
            show_panel=True,
            seed=42,
        )
        self._sd.center()
        self._sd.shift(DOWN * 0.20)
        self.add(self._sd)

        # Build population
        self.play(BuildPopulation(self._sd, run_time=1.8))
        self.play(FadeIn(self._sd.clt_arrow, run_time=0.5))
        self.wait(0.3)

        # Counter badge
        sigma_n = self.SIGMA_POP / np.sqrt(self.N_MAIN)
        counter = _LiveCounter(k=0, se_val=sigma_n, n=self.N_MAIN)
        counter.move_to([0, -2.80, 0])
        self.play(FadeIn(counter, run_time=0.4))

        # 10 slow iterations
        for k in range(1, 11):
            self.play(RunCLT(self._sd, run_time=1.10))

            # Update counter every 2 steps
            if k % 2 == 0:
                new_counter = _LiveCounter(
                    k=k, se_val=sigma_n, n=self.N_MAIN
                )
                new_counter.move_to(counter.get_center())
                self.play(
                    ReplacementTransform(counter, new_counter,
                                         run_time=0.25)
                )
                counter = new_counter

        self.wait(0.5)
        self.play(FadeOut(act_lbl), FadeOut(counter), run_time=0.4)

    # ─────────────────────────────────────────────────────────────────
    # Act 4 — Convergence speed-up
    # ─────────────────────────────────────────────────────────────────

    def _act4_convergence_speedup(self):
        act_lbl = _section_label("Act 4 — Convergence")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        if not _SD_AVAILABLE or not hasattr(self, "_sd"):
            self.play(FadeOut(act_lbl), run_time=0.3)
            return

        # "Let's speed up…" banner
        speed_banner = Text(
            "Let's draw 90 more samples…",
            font_size=25, color=P["subtitle"],
        )
        speed_banner.move_to([0, -2.70, 0])
        self.play(FadeIn(speed_banner, run_time=0.5))
        self.wait(0.4)

        # Fast burst: 50 samples
        for _ in range(50):
            self.play(RunCLT(self._sd, run_time=0.06, fast=True))

        # Theory curve appears dashed after 50
        self._sd.hist_panel.set_theory_curve(
            self._sd._mu_pop, self._sd._se, self._sd.k_drawn
        )
        if self._sd.hist_panel._theory_curve is not None:
            self.play(
                FadeIn(self._sd.hist_panel._theory_curve,
                       run_time=0.6)
            )

        at50_badge = _badge(f"k = 50  (theory: dashed)",
                            P["theory_dash"])
        at50_badge.move_to([0, -2.70, 0])
        self.play(
            ReplacementTransform(speed_banner, at50_badge,
                                 run_time=0.5)
        )
        self.wait(0.4)

        # 40 more → curve goes solid
        for _ in range(40):
            self.play(RunCLT(self._sd, run_time=0.04, fast=True))

        self._sd.hist_panel.set_theory_curve(
            self._sd._mu_pop, self._sd._se, self._sd.k_drawn
        )

        at100_badge = _badge(
            f"k = {self._sd.k_drawn}  (theory: solid)",
            P["theory"]
        )
        at100_badge.move_to([0, -2.70, 0])
        self.play(
            ReplacementTransform(at50_badge, at100_badge,
                                 run_time=0.5)
        )
        self.wait(0.5)

        # Converged message
        conv_msg = Text(
            "The histogram is converging to the theoretical normal curve.",
            font_size=20, color=P["title"],
        )
        conv_msg.move_to([0, -3.40, 0])
        self.play(Write(conv_msg, run_time=1.0))
        self.wait(0.8)

        self.play(
            FadeOut(act_lbl),
            FadeOut(at100_badge),
            FadeOut(conv_msg),
            FadeOut(self._sd),
            run_time=0.65,
        )

    # ─────────────────────────────────────────────────────────────────
    # Act 5 — The n effect
    # ─────────────────────────────────────────────────────────────────

    def _act5_n_effect(self):
        act_lbl = _section_label("Act 5 — The Effect of n")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        n_values = [1, 5, 30, 100]
        rng      = np.random.default_rng(7)
        k_pre    = 120

        panels = VGroup()
        spacing = 3.40
        for i, n in enumerate(n_values):
            sigma_n = self.SIGMA_POP / np.sqrt(n)
            means   = np.array([
                float(np.mean(rng.exponential(
                    1.0/self.POP_PARAMS["lam"], n
                )))
                for _ in range(k_pre)
            ])
            pan = _SEPanel(
                means=means,
                mu=self.MU_POP, sigma_n=sigma_n,
                n=n,
                width=2.80, height=2.20,
                baseline_y=0.0,
            )
            pan.move_to([(i - 1.5) * spacing, 0.40, 0])
            panels.add(pan)

        self.play(
            *[FadeIn(p, shift=UP*0.15,
                     run_time=0.60 + i*0.10)
              for i, p in enumerate(panels)],
        )
        self.wait(0.5)

        # SE = σ/√n formula reveal
        try:
            se_form = MathTex(
                r"\mathrm{SE} = \frac{\sigma}{\sqrt{n}}"
                r"\quad\Rightarrow\quad"
                r"\text{larger }n\Rightarrow\text{narrower distribution}",
                font_size=24, color=P["se_label"],
            )
        except Exception:
            se_form = Text(
                "SE = σ/√n  →  larger n → narrower distribution",
                font_size=20, color=P["se_label"],
            )
        se_form.move_to([0, -2.20, 0])
        self.play(Write(se_form, run_time=1.2))
        self.wait(0.8)

        # Live SE values
        for i, n in enumerate(n_values):
            sigma_n = self.SIGMA_POP / np.sqrt(n)
            self.play(
                Indicate(panels[i], color=P["se"],
                         scale_factor=1.05, run_time=0.45),
            )
            self.wait(0.08)
        self.wait(0.5)

        self.play(
            FadeOut(panels),
            FadeOut(se_form),
            FadeOut(act_lbl),
            run_time=0.55,
        )

    # ─────────────────────────────────────────────────────────────────
    # Act 6 — Population independence
    # ─────────────────────────────────────────────────────────────────

    def _act6_population_independence(self):
        act_lbl = _section_label("Act 6 — Population Independence")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        pop_specs = [
            ("uniform",     {"a": 0.0, "b": 1.0},
             P["uniform_fill"], P["uniform_ridge"]),
            ("exponential", {"lam": 1.0},
             P["exp_fill"],     P["exp_ridge"]),
            ("bimodal",     {"mu1": -1.8, "mu2": 1.8,
                              "sigma": 0.6, "mix": 0.5},
             P["bimodal_fill"], P["bimodal_ridge"]),
        ]

        rng     = np.random.default_rng(31)
        n_samp  = 10
        k_fast  = 100
        spacing = 4.50
        panels  = VGroup()

        for i, (pt, pm, fc, rc) in enumerate(pop_specs):
            if _SD_AVAILABLE:
                mu, sigma = _pop_stats(pt, pm)
            else:
                mu, sigma = 0.5, 0.3
            sigma_n = sigma / np.sqrt(n_samp)

            # Generate means
            means = []
            for _ in range(k_fast):
                if _SD_AVAILABLE:
                    s = _sample_pop(pt, pm, n_samp, rng)
                else:
                    s = rng.normal(mu, sigma, n_samp)
                means.append(float(np.mean(s)))
            means = np.array(means)

            # Mini population curve
            pop_c = _MiniPopCurve(
                pt, pm, "", fc, rc,
                width=3.0, height=1.30,
                baseline_y=0.0,
                show_dots=False,
            )

            # Mini histogram
            hist = _MiniHistogram(
                means=means,
                mu=mu, sigma_n=sigma_n,
                n_bins=16,
                width=3.0, height=1.80,
                baseline_y=0.0,
                show_theory=True,
            )

            # Arrow between them
            arr = Arrow(
                start=[0, -0.20, 0],
                end  =[0, -0.80, 0],
                stroke_color=P["subtitle"],
                stroke_width=1.4, tip_length=0.16, buff=0,
            )
            try:
                arr_lbl = MathTex(rf"n={n_samp}", font_size=14,
                                  color=P["subtitle"])
            except Exception:
                arr_lbl = Text(f"n={n_samp}", font_size=12,
                               color=P["subtitle"])
            arr_lbl.next_to(arr, RIGHT, buff=0.08)

            col_grp = VGroup()
            pop_c.move_to([0, 1.60, 0])
            arr.move_to([0, 0.38, 0])
            arr_lbl.move_to([0.60, 0.38, 0])
            hist.move_to([0, -1.00, 0])
            col_grp.add(pop_c, arr, arr_lbl, hist)
            col_grp.move_to([(i-1) * spacing, 0.20, 0])
            panels.add(col_grp)

        self.play(
            *[FadeIn(p, shift=UP*0.15, run_time=0.70)
              for p in panels],
        )
        self.wait(0.6)

        msg = Text(
            "All three converge to a normal shape. "
            "The population doesn't matter.",
            font_size=22, color=P["title"],
        )
        msg.move_to([0, -3.0, 0])
        self.play(Write(msg, run_time=1.2))
        self.wait(0.8)

        # Flash all histograms simultaneously
        self.play(
            *[Indicate(p.submobjects[-1],
                       color=P["theory"],
                       scale_factor=1.04,
                       run_time=0.6)
              for p in panels],
        )
        self.wait(0.5)

        self.play(
            FadeOut(panels),
            FadeOut(msg),
            FadeOut(act_lbl),
            run_time=0.60,
        )

    # ─────────────────────────────────────────────────────────────────
    # Act 7 — Normal Q-Q plot
    # ─────────────────────────────────────────────────────────────────

    def _act7_qq_plot(self):
        act_lbl = _section_label("Act 7 — Normal Q-Q Plot")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        # Generate a large sample of means from exponential
        rng     = np.random.default_rng(88)
        n_samp  = 30
        k_total = 200
        lam     = 1.0
        means   = np.array([
            float(np.mean(rng.exponential(1.0/lam, n_samp)))
            for _ in range(k_total)
        ])
        sigma_n = (1.0/lam) / np.sqrt(n_samp)
        mu      = 1.0 / lam

        # Histogram on left
        hist = _MiniHistogram(
            means=means,
            mu=mu, sigma_n=sigma_n,
            n_bins=20,
            width=4.0, height=3.0,
            baseline_y=-0.30,
            show_theory=True,
        )
        hist.move_to([-3.30, 0.35, 0])

        # Label
        hist_title = Text(
            f"n={n_samp}, k={k_total} samples\nExp(λ=1) population",
            font_size=16, color=P["subtitle"],
        )
        hist_title.move_to([-3.30, 2.30, 0])

        self.play(FadeIn(hist, shift=LEFT*0.15, run_time=0.9))
        self.play(FadeIn(hist_title, run_time=0.5))
        self.wait(0.4)

        # Arrow
        arr = Arrow(
            start=[-1.0, 0.35, 0],
            end  =[ 0.6, 0.35, 0],
            stroke_color=P["subtitle"],
            stroke_width=1.6, tip_length=0.18, buff=0,
        )
        self.play(Create(arr, run_time=0.5))

        # QQ plot on right
        qq = _QQPlot(
            means=means,
            mu=mu, sigma_n=sigma_n,
            width=3.8, height=3.0,
            baseline_y=-0.30,
        )
        qq.move_to([3.10, 0.35, 0])
        qq.set_opacity(0)
        self.play(FadeIn(qq, shift=RIGHT*0.15, run_time=1.0))
        self.wait(0.5)

        # Highlight that points hug the line
        qq_msg = Text(
            "Points hug the 45° line  →  distribution is approximately normal.",
            font_size=19, color=P["title"],
        )
        qq_msg.move_to([0, -2.70, 0])
        self.play(Write(qq_msg, run_time=1.1))
        self.wait(0.9)

        self.play(
            FadeOut(hist), FadeOut(hist_title),
            FadeOut(arr), FadeOut(qq),
            FadeOut(qq_msg), FadeOut(act_lbl),
            run_time=0.60,
        )

    # ─────────────────────────────────────────────────────────────────
    # Act 8 — Formal CLT statement + closing
    # ─────────────────────────────────────────────────────────────────

    def _act8_clt_statement(self):
        act_lbl = _section_label("Act 8 — The Formal Statement")
        act_lbl.to_corner(LEFT + UP, buff=0.22)
        self.play(FadeIn(act_lbl, run_time=0.4))

        stmt = _CLTStatement()
        stmt.scale(0.94)
        stmt.move_to([0, 0.10, 0])
        stmt.set_opacity(0)

        self.play(FadeIn(stmt, shift=UP*0.15, run_time=1.2))
        self.wait(0.5)

        # Highlight formula terms one by one
        highlight_sequence = [
            (0, P["f_xbar"],  "sample mean"),
            (2, P["f_mu"],    "population mean"),
            (4, P["f_sigma"], "population std dev"),
            (7, P["f_norm"],  "standard normal"),
            (9, P["f_n"],     "n → ∞"),
        ]
        for idx, col, _ in highlight_sequence:
            if idx < len(stmt.sym_pieces):
                self.play(
                    Indicate(stmt.sym_pieces[idx],
                             color=col,
                             scale_factor=1.18,
                             run_time=0.55),
                )
                self.wait(0.10)
        self.wait(0.7)

        # Legend for colour codes
        legend_items = [
            (P["f_xbar"],   "X̄  = sample mean"),
            (P["f_mu"],     "μ  = population mean"),
            (P["f_sigma"],  "σ  = population std dev"),
            (P["f_n"],      "n  = sample size"),
            (P["f_norm"],   "N(0,1)  = standard normal"),
        ]
        legend = VGroup()
        for j, (col, txt) in enumerate(legend_items):
            dot = Dot(radius=0.06,
                      point=[-4.30, -2.20 + j*0.32, 0.001],
                      color=col, fill_opacity=1.0)
            lbl = Text(txt, font_size=13, color=col)
            lbl.move_to([-3.60 + lbl.width/2, -2.20 + j*0.32, 0.001])
            legend.add(dot, lbl)
        self.play(FadeIn(legend, run_time=0.7))
        self.wait(1.0)

        # Closing statement
        closing = Text(
            "This is why the normal distribution appears everywhere in statistics.",
            font_size=22, color=P["title"],
        )
        closing.move_to([0, -3.30, 0])
        self.play(Write(closing, run_time=1.2))
        self.wait(1.0)

        self.play(
            FadeOut(stmt),
            FadeOut(legend),
            FadeOut(closing),
            FadeOut(act_lbl),
            run_time=0.70,
        )

        # End card
        end_bg = Rectangle(width=16, height=9,
                           fill_color=P["bg"],
                           fill_opacity=1.0, stroke_width=0)
        end_title = Text("Central Limit Theorem",
                         font_size=46, color=P["title"])
        try:
            end_sub = MathTex(
                r"\bar{X} \;\overset{d}{\longrightarrow}\; "
                r"\mathcal{N}\!\left(\mu,\,\frac{\sigma^2}{n}\right)",
                font_size=34, color=P["f_norm"],
            )
        except Exception:
            end_sub = Text("X̄ → N(μ, σ²/n)", font_size=28,
                           color=P["f_norm"])
        end_title.move_to([0,  0.55, 0])
        end_sub.move_to([0, -0.45, 0])

        self.play(
            FadeIn(end_bg, run_time=0.5),
            FadeIn(end_title, run_time=0.7),
        )
        self.play(FadeIn(end_sub,
                         rate_func=rate_functions.ease_out_back,
                         run_time=0.8))
        self.wait(2.0)
        self.play(FadeOut(VGroup(end_bg, end_title, end_sub)),
                  run_time=0.9)