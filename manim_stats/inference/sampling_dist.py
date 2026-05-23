"""
manim_stats/inference/sampling_dist.py
========================================
SamplingDistribution3D — A highly detailed, animated Central Limit
Theorem and sampling distribution visualisation for Manim.

Primary use cases
-----------------
  Central Limit Theorem demonstration  — any population → normal sampling dist
  Standard error intuition             — SE = σ/√n shrinks with n
  Sampling variability                 — watch individual means scatter
  Point estimation                     — sample mean as an estimator of μ
  Bootstrap intuition                  — repeated resampling

Design goals
------------
Population panel (left)
  * Full rendered source distribution curve — supports Normal, Uniform,
    Exponential, Chi-squared, Bimodal, and Bernoulli shapes.
  * Three-layer curve: base fill polygon + AO tail darkening + ridge spine.
  * Population parameter badges: μ, σ, shape description.
  * "Sample zone" bracket: a highlighted sub-region with tick marks
    showing one concrete sample being drawn.

Sample strip (centre)
  * Dot strip: individual sample observations rendered as small coloured
    circles on a number line, dropping in one by one.
  * After all n dots land, a bracket collapses from both ends to the mean,
    which is highlighted in a bright colour.
  * Mean dot shoots upward from the strip to the histogram.

Sampling distribution histogram (right)
  * Histogram bars with 3-D depth: bright top face cap, darker right
    side face, and a subtle AO shadow polygon at the base.
  * Bars grow upward from 0 (animated height).
  * Relative frequency or count labelling above each bar.
  * Bin highlight: when a new mean lands the bar flashes.

Theoretical overlay
  * N(μ, σ²/n) normal curve drawn over the histogram.
  * Dashed until enough samples accumulate (< 10), solid after.
  * Curve animates toward the histogram as bars fill in.

Standard error annotation
  * DoubleArrow bracket spanning ±1 SE around the theoretical mean.
  * "SE = σ/√n = {value:.3f}" badge above the arrow.
  * Arrow shrinks when n increases (NarrowSE animation).

Live stats panel
  * Floating badge showing:
      Samples drawn:  k
      Current x̄:    value
      True μ:         value
      SE = σ/√n:     value
      n per sample:   n

CLT label
  * Central arrow from population panel to sampling dist panel.
  * Label: "n = {n}, repeat {k} times" — updates live.

Animations
----------
  BuildPopulation   — population curve grows from baseline
  DrawSample        — n dots drop onto the sample strip one by one
  ExtractMean       — bracket collapses → mean highlighted → mean dot rises
  AddToHistogram    — mean dot lands in correct bin, bar grows by 1 unit
  RunCLT            — orchestrates DrawSample→ExtractMean→AddToHistogram
                      for k iterations, updating stats panel
  NarrowSE          — increase n: sample strip widens/narrows, SE arrow
                      shrinks, theoretical curve narrows
  ConvergenceRace   — animate bars rapidly filling toward normal (speedup)
  FlashBin          — pulse the bar that just received a new count

Population constructors
-----------------------
  SamplingDistribution3D.normal(mu, sigma, n, **kw)
  SamplingDistribution3D.uniform(a, b, n, **kw)
  SamplingDistribution3D.exponential(lam, n, **kw)
  SamplingDistribution3D.chi_squared(df, n, **kw)
  SamplingDistribution3D.bimodal(mu1, mu2, sigma, mix, n, **kw)
  SamplingDistribution3D.bernoulli(p, n, **kw)

Dependencies
------------
  manim (CE or GL), numpy, scipy (optional)

Usage
-----
    from manim_stats.inference.sampling_dist import (
        SamplingDistribution3D, RunCLT, NarrowSE
    )

    class CLTScene(Scene):
        def construct(self):
            sd = SamplingDistribution3D.exponential(lam=1.0, n=5)
            self.play(BuildPopulation(sd))
            for _ in range(30):
                self.play(RunCLT(sd, run_time=0.6))
            self.play(NarrowSE(sd, new_n=30, run_time=2.5))
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable, Literal, Sequence

from manim import (
    VGroup,
    Rectangle, RoundedRectangle, Square,
    Circle, Annulus, Dot, Polygon, Line, DashedLine,
    Arrow, DoubleArrow,
    Text, MathTex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut,
    Create, Write,
    Indicate, Flash,
    ValueTracker,
    interpolate_color, color_to_rgb,
    always_redraw,
    WHITE, BLACK,
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
)

# ─────────────────────────────────────────────────────────────────────────────
# scipy — graceful fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy import stats as _stats
    def _norm_pdf(x, mu, sigma):
        return _stats.norm.pdf(x, loc=mu, scale=sigma)
    def _norm_ppf(p, mu, sigma):
        return float(_stats.norm.ppf(p, loc=mu, scale=sigma))
    def _sample_pop(pop_type, params, n, rng):
        if pop_type == "normal":
            return rng.normal(params["mu"], params["sigma"], n)
        elif pop_type == "uniform":
            return rng.uniform(params["a"], params["b"], n)
        elif pop_type == "exponential":
            return rng.exponential(1.0 / params["lam"], n)
        elif pop_type == "chi2":
            return rng.chisquare(params["df"], n)
        elif pop_type == "bimodal":
            mask = rng.random(n) < params["mix"]
            s = np.where(
                mask,
                rng.normal(params["mu1"], params["sigma"], n),
                rng.normal(params["mu2"], params["sigma"], n),
            )
            return s
        elif pop_type == "bernoulli":
            return rng.binomial(1, params["p"], n).astype(float)
        return rng.normal(0, 1, n)
except ImportError:
    def _norm_pdf(x, mu, sigma):
        return (np.exp(-0.5*((x-mu)/sigma)**2)
                / (sigma * np.sqrt(2*PI)))
    def _norm_ppf(p, mu, sigma):
        z = 0.0
        for _ in range(60):
            z -= (_norm_pdf(np.array([z]),0,1)[0]
                  - _norm_pdf(np.array([z]),0,1)[0]) / (
                _norm_pdf(np.array([z]),0,1)[0]+1e-14)
        return mu + z*sigma
    def _sample_pop(pop_type, params, n, rng):
        return rng.normal(params.get("mu",0), params.get("sigma",1), n)


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

PAL = {
    # Population curve
    "pop_fill":       "#1A3A5C",
    "pop_fill_lite":  "#2E6AAA",
    "pop_ridge":      "#70B4EE",
    "pop_ao":         "#081828",
    "pop_label":      "#90C8F8",

    # Sample strip dots
    "dot_default":    "#E8B840",
    "dot_mean":       "#FF6040",
    "dot_highlight":  "#FFE080",
    "strip_line":     "#405060",
    "bracket":        "#C8D8E8",
    "mean_beam":      "#FF8060",

    # Histogram
    "bar_top":        "#3A7AC8",
    "bar_top_lite":   "#60A0F0",
    "bar_side":       "#1A3A68",
    "bar_ao":         "#080C18",
    "bar_flash":      "#FFD060",
    "bar_border":     "#2A5090",

    # Theoretical curve
    "theory_solid":   "#F0C040",
    "theory_dashed":  "#907020",
    "theory_fill":    "#3A2800",

    # SE arrow
    "se_arrow":       "#D4AF37",
    "se_label":       "#F8E090",
    "se_badge_bg":    "#1A1208",

    # Stats panel
    "panel_bg":       "#0C1018",
    "panel_border":   "#2A3848",
    "panel_key":      "#7090A8",
    "panel_val":      "#C8E0F8",
    "panel_title":    "#D8F0FF",

    # CLT arrow
    "clt_arrow":      "#A0B8D0",
    "clt_label":      "#C8E0F8",

    # Axis
    "axis":           "#506070",
    "axis_label":     "#90A8B8",
    "tick":           "#405060",
}


# ─────────────────────────────────────────────────────────────────────────────
# Population PDF functions
# ─────────────────────────────────────────────────────────────────────────────

def _pop_pdf(pop_type: str, params: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate the population PDF on array x."""
    if pop_type == "normal":
        return _norm_pdf(x, params["mu"], params["sigma"])
    elif pop_type == "uniform":
        a, b = params["a"], params["b"]
        return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
    elif pop_type == "exponential":
        lam = params["lam"]
        return np.where(x >= 0, lam * np.exp(-lam * x), 0.0)
    elif pop_type == "chi2":
        df = params["df"]
        try:
            from scipy.stats import chi2
            return chi2.pdf(x, df)
        except ImportError:
            # Rough chi2 pdf via gamma approximation
            k = df / 2
            with np.errstate(all="ignore"):
                import math
                y = np.where(
                    x > 0,
                    (x**(k-1) * np.exp(-x/2)) / (2**k * math.gamma(k)),
                    0.0,
                )
            return y
    elif pop_type == "bimodal":
        m1 = _norm_pdf(x, params["mu1"], params["sigma"])
        m2 = _norm_pdf(x, params["mu2"], params["sigma"])
        mix = params.get("mix", 0.5)
        return mix * m1 + (1 - mix) * m2
    elif pop_type == "bernoulli":
        p = params["p"]
        # Discrete: approximate with two spikes
        y = np.zeros_like(x, dtype=float)
        for val, prob in [(0.0, 1-p), (1.0, p)]:
            mask = np.abs(x - val) < 0.08
            y[mask] = prob / 0.16   # normalise spike height
        return y
    return _norm_pdf(x, 0, 1)


def _pop_stats(pop_type: str, params: dict) -> tuple[float, float]:
    """Return (mu, sigma) for the population."""
    if pop_type == "normal":
        return params["mu"], params["sigma"]
    elif pop_type == "uniform":
        a, b = params["a"], params["b"]
        return (a+b)/2, (b-a)/np.sqrt(12)
    elif pop_type == "exponential":
        lam = params["lam"]
        return 1.0/lam, 1.0/lam
    elif pop_type == "chi2":
        df = params["df"]
        return float(df), float(np.sqrt(2*df))
    elif pop_type == "bimodal":
        mix = params.get("mix", 0.5)
        mu  = mix*params["mu1"] + (1-mix)*params["mu2"]
        s2  = (mix*(params["sigma"]**2 + params["mu1"]**2)
               + (1-mix)*(params["sigma"]**2 + params["mu2"]**2)
               - mu**2)
        return mu, float(np.sqrt(max(s2, 1e-6)))
    elif pop_type == "bernoulli":
        p = params["p"]
        return p, float(np.sqrt(p*(1-p)))
    return 0.0, 1.0


def _pop_x_range(pop_type: str, params: dict) -> tuple[float, float]:
    """Suggest a good x-range for plotting this population."""
    mu, sigma = _pop_stats(pop_type, params)
    if pop_type == "uniform":
        a, b = params["a"], params["b"]
        pad = (b-a)*0.15
        return a-pad, b+pad
    elif pop_type == "exponential":
        return 0.0, 5.0/params["lam"]
    elif pop_type == "chi2":
        df = params["df"]
        return 0.0, df + 5*np.sqrt(2*df)
    elif pop_type == "bernoulli":
        return -0.5, 1.5
    elif pop_type == "bimodal":
        lo = min(params["mu1"], params["mu2"]) - 3.5*params["sigma"]
        hi = max(params["mu1"], params["mu2"]) + 3.5*params["sigma"]
        return lo, hi
    return mu - 4*sigma, mu + 4*sigma


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _curve_verts(x_px, y_px, baseline=0.0):
    return (
        [[float(x_px[0]), baseline, 0]]
        + [[float(x), float(y), 0] for x, y in zip(x_px, y_px)]
        + [[float(x_px[-1]), baseline, 0]]
    )


def _map_x(raw_x, x_lo, x_hi, plot_w):
    """Map raw x values to Manim world x coordinates."""
    return (np.asarray(raw_x) - x_lo) / (x_hi - x_lo) * plot_w - plot_w/2


# ─────────────────────────────────────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────────────────────────────────────

class _PopulationPanel(VGroup):
    """
    Left panel: population distribution curve with parameter badges.

    Layers:
      1. Panel background rounded rect
      2. Base fill polygon (curve area)
      3. AO tail darkening (left and right tails)
      4. Ridge spine (VMobject along curve top)
      5. Peak cap circle
      6. Parameter badges: μ, σ, distribution name
      7. "Population" title label

    Parameters
    ----------
    pop_type   : str        — distribution family
    params     : dict       — distribution parameters
    panel_w    : float      — panel width (Manim units)
    panel_h    : float      — panel height (curve max height)
    baseline_y : float
    """

    def __init__(
        self,
        pop_type:  str,
        params:    dict,
        panel_w:   float = 4.5,
        panel_h:   float = 3.0,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pop_type  = pop_type
        self._params    = params
        self._panel_w   = panel_w
        self._panel_h   = panel_h
        self._baseline  = baseline_y

        x_lo, x_hi = _pop_x_range(pop_type, params)
        raw_x = np.linspace(x_lo, x_hi, 500)
        x_px  = _map_x(raw_x, x_lo, x_hi, panel_w)

        y_raw  = _pop_pdf(pop_type, params, raw_x)
        y_raw  = np.where(np.isfinite(y_raw), y_raw, 0.0)
        peak   = float(y_raw.max()) if y_raw.max() > 1e-12 else 1.0
        scale_y = panel_h / peak
        y_px    = y_raw * scale_y

        # Store for sample-zone overlay
        self._raw_x   = raw_x
        self._x_px    = x_px
        self._y_px    = y_px
        self._x_lo    = x_lo
        self._x_hi    = x_hi
        self._scale_y = scale_y

        # ── Panel background ──────────────────────────────────────────
        bg = RoundedRectangle(
            width=panel_w + 0.25,
            height=panel_h + 1.10,
            corner_radius=0.12,
            fill_color=PAL["pop_ao"],
            fill_opacity=0.55,
            stroke_color=PAL["strip_line"],
            stroke_width=0.8,
        )
        bg.move_to([0, baseline_y + panel_h/2 - 0.10, -0.002])
        self.add(bg)

        # ── Base fill polygon ─────────────────────────────────────────
        verts = _curve_verts(x_px, y_px, baseline_y)
        if len(verts) >= 3:
            base = Polygon(
                *verts,
                fill_color=PAL["pop_fill"],
                fill_opacity=0.85,
                stroke_width=0,
            )
            base.shift([0, 0, 0.001])
            self.add(base)

        # ── AO darkening at tails ─────────────────────────────────────
        ao_w = (x_px[-1] - x_px[0]) * 0.10
        for x_edge, x_inner in [
            (x_px[0],  x_px[0]  + ao_w),
            (x_px[-1], x_px[-1] - ao_w),
        ]:
            xl, xr = min(x_edge, x_inner), max(x_edge, x_inner)
            mask = (x_px >= xl) & (x_px <= xr)
            xs, ys = x_px[mask], y_px[mask]
            if len(xs) >= 2:
                av = ([[xl, baseline_y, 0]]
                      + [[float(x),float(y),0] for x,y in zip(xs,ys)]
                      + [[xr, baseline_y, 0]])
                ao = Polygon(*av,
                             fill_color=PAL["pop_ao"],
                             fill_opacity=0.48, stroke_width=0)
                ao.shift([0, 0, 0.002])
                self.add(ao)

        # ── Ridge spine ───────────────────────────────────────────────
        step = max(1, len(x_px)//80)
        pts3d = [np.array([float(x_px[i]), float(y_px[i]), 0.004])
                 for i in range(0, len(x_px), step) if y_px[i] > 1e-4]
        if len(pts3d) >= 2:
            spine = VMobject(stroke_color=PAL["pop_ridge"],
                             stroke_width=1.8, stroke_opacity=0.80)
            spine.set_points_smoothly(pts3d)
            self.add(spine)

        # ── Peak cap ──────────────────────────────────────────────────
        peak_i   = int(np.argmax(y_px))
        peak_pos = [float(x_px[peak_i]),
                    float(y_px[peak_i]) + 0.03, 0.005]
        cap = Circle(radius=0.05,
                     fill_color=PAL["pop_ridge"],
                     fill_opacity=0.90, stroke_width=0)
        cap.move_to(peak_pos)
        self.add(cap)

        # ── Axis line ─────────────────────────────────────────────────
        ax = Line(
            start=[-panel_w/2 - 0.15, baseline_y, 0.001],
            end  =[ panel_w/2 + 0.15, baseline_y, 0.001],
            stroke_color=PAL["axis"], stroke_width=1.5,
        )
        self.add(ax)

        # ── Axis ticks ────────────────────────────────────────────────
        mu_pop, _ = _pop_stats(pop_type, params)
        for rv in [x_lo, mu_pop, x_hi]:
            xp  = float(_map_x(rv, x_lo, x_hi, panel_w))
            tk  = Line([xp, baseline_y-0.07, 0.001],
                       [xp, baseline_y+0.07, 0.001],
                       stroke_color=PAL["tick"], stroke_width=1.2)
            lbl = Text(f"{rv:.1f}", font_size=12,
                       color=PAL["axis_label"])
            lbl.move_to([xp, baseline_y - 0.27, 0.001])
            self.add(tk, lbl)

        # ── Parameter badges ──────────────────────────────────────────
        mu_v, sigma_v = _pop_stats(pop_type, params)
        badge_y = baseline_y + panel_h + 0.25
        for i, (sym, val) in enumerate(
            [("\\mu", mu_v), ("\\sigma", sigma_v)]
        ):
            try:
                bm = MathTex(f"{sym} = {val:.3f}",
                             font_size=20, color=PAL["pop_label"])
            except Exception:
                bm = Text(f"{sym}={val:.3f}", font_size=17,
                          color=PAL["pop_label"])
            bm.move_to([-panel_w/4 + i*panel_w/2,
                        badge_y, 0.003])
            self.add(bm)

        # ── Title ─────────────────────────────────────────────────────
        title_map = {
            "normal": "Normal Population",
            "uniform": "Uniform Population",
            "exponential": "Exponential Population",
            "chi2": "Chi-Squared Population",
            "bimodal": "Bimodal Population",
            "bernoulli": "Bernoulli Population",
        }
        title = Text(title_map.get(pop_type, "Population"),
                     font_size=18, color=PAL["panel_title"])
        title.move_to([0, baseline_y + panel_h + 0.62, 0.003])
        self.add(title)

        # Save geometry for use by animations
        self._mu_pop    = mu_v
        self._sigma_pop = sigma_v


class _SampleStrip(VGroup):
    """
    The sample observation strip: a number line with dots.

    Components:
      1. Horizontal number line with tick marks
      2. n Dot objects (added one by one during DrawSample)
      3. Bracket that collapses to the mean (added during ExtractMean)
      4. Mean dot (highlighted, larger)

    Parameters
    ----------
    x_lo, x_hi : raw axis range
    strip_w    : Manim units width
    y          : vertical position
    n_samples  : sample size (number of dots)
    """

    def __init__(
        self,
        x_lo:      float,
        x_hi:      float,
        strip_w:   float,
        y:         float,
        n_samples: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._x_lo     = x_lo
        self._x_hi     = x_hi
        self._strip_w  = strip_w
        self._y        = y
        self._n        = n_samples
        self._dots: list = []
        self._mean_dot = None

        # ── Number line ───────────────────────────────────────────────
        nline = Line(
            start=[-strip_w/2, y, 0],
            end  =[ strip_w/2, y, 0],
            stroke_color=PAL["strip_line"],
            stroke_width=1.5,
        )
        self.add(nline)

        # Tick marks
        n_ticks = 6
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            rv   = x_lo + frac * (x_hi - x_lo)
            xp   = frac * strip_w - strip_w/2
            tk   = Line([xp, y-0.06, 0.001],
                        [xp, y+0.06, 0.001],
                        stroke_color=PAL["tick"], stroke_width=1.0)
            lbl  = Text(f"{rv:.1f}", font_size=11,
                        color=PAL["axis_label"])
            lbl.move_to([xp, y - 0.22, 0.001])
            self.add(tk, lbl)

    def raw_to_px(self, v: float) -> float:
        frac = (v - self._x_lo) / (self._x_hi - self._x_lo)
        return frac * self._strip_w - self._strip_w/2

    def add_dot(
        self,
        raw_val: float,
        color: str = None,
        dot_r: float = 0.07,
    ) -> Dot:
        """Create and register a dot at raw_val."""
        color  = color or PAL["dot_default"]
        xp     = self.raw_to_px(raw_val)
        # Jitter y slightly to avoid perfect overlap
        y_jitt = self._y + np.random.uniform(0.04, 0.22)
        dot    = Dot(radius=dot_r,
                     point=[xp, y_jitt, 0.005],
                     color=color,
                     fill_opacity=0.88)
        self._dots.append(dot)
        self.add(dot)
        return dot

    def make_mean_dot(self, mean_raw: float) -> Dot:
        """Create the highlighted mean dot."""
        xp  = self.raw_to_px(mean_raw)
        dot = Dot(radius=0.11,
                  point=[xp, self._y + 0.12, 0.010],
                  color=PAL["dot_mean"],
                  fill_opacity=1.0)
        self._mean_dot = dot
        self.add(dot)
        return dot

    def clear_dots(self):
        """Remove all sample dots (but keep the number line)."""
        for d in self._dots:
            if d in self.submobjects:
                self.remove(d)
        self._dots.clear()
        if self._mean_dot and self._mean_dot in self.submobjects:
            self.remove(self._mean_dot)
        self._mean_dot = None


class _HistogramPanel(VGroup):
    """
    Right panel: sampling distribution histogram with 3-D bar effect.

    Each bar has three faces:
      top_cap    — bright thin rectangle (the "top face")
      side_face  — slightly darker rectangle on the right side
      front_face — the main bar (body colour)
    Plus a base AO shadow below each bar.

    Parameters
    ----------
    x_lo, x_hi : raw x range for the histogram
    n_bins     : number of histogram bins
    panel_w    : Manim units width
    max_bar_h  : maximum bar height (Manim units) at full count
    baseline_y : y of the histogram baseline
    max_count  : expected maximum count (for bar scaling)
    """

    def __init__(
        self,
        x_lo:      float,
        x_hi:      float,
        n_bins:    int,
        panel_w:   float,
        max_bar_h: float,
        baseline_y: float,
        max_count:  int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._x_lo      = x_lo
        self._x_hi      = x_hi
        self._n_bins    = n_bins
        self._panel_w   = panel_w
        self._max_bar_h = max_bar_h
        self._baseline  = baseline_y
        self._max_count = max_count

        # Bin edges and centres
        self._edges   = np.linspace(x_lo, x_hi, n_bins + 1)
        self._centres = (self._edges[:-1] + self._edges[1:]) / 2
        self._bin_w   = panel_w / n_bins          # bar width in px
        self._counts  = np.zeros(n_bins, dtype=int)

        # 3-D depth parameters
        self._depth_x = self._bin_w * 0.08        # right-face overhang
        self._cap_h   = 0.045                     # top face height

        # Pre-create bar groups (initially zero height)
        self._bars: list[VGroup] = []
        for i in range(n_bins):
            bar_grp = self._make_bar(i, 0)
            self._bars.append(bar_grp)
            self.add(bar_grp)

        # Axis
        ax = Line(
            start=[-panel_w/2 - 0.15, baseline_y, 0.001],
            end  =[ panel_w/2 + 0.15, baseline_y, 0.001],
            stroke_color=PAL["axis"], stroke_width=1.5,
        )
        self.add(ax)

        # Axis ticks + labels
        for i in range(n_bins + 1):
            xp  = float(i * self._bin_w - panel_w/2)
            rv  = self._edges[i]
            tk  = Line([xp, baseline_y-0.06, 0.001],
                       [xp, baseline_y+0.06, 0.001],
                       stroke_color=PAL["tick"], stroke_width=0.9)
            lbl = Text(f"{rv:.2f}", font_size=10,
                       color=PAL["axis_label"])
            lbl.rotate(PI/4)
            lbl.move_to([xp, baseline_y - 0.28, 0.001])
            self.add(tk, lbl)

        # Count labels above bars (initially empty)
        self._count_labels: list = [None] * n_bins

        # Theoretical curve placeholder (added later)
        self._theory_curve = None

    def _bar_left_px(self, i: int) -> float:
        return i * self._bin_w - self._panel_w/2

    def _bar_height(self, count: int) -> float:
        """Convert count to bar height in Manim units."""
        return (count / max(self._max_count, 1)) * self._max_bar_h

    def _make_bar(self, i: int, count: int) -> VGroup:
        """Build a 3-D bar at bin i with given count."""
        grp   = VGroup()
        xl    = self._bar_left_px(i)
        h     = self._bar_height(count)
        bw    = self._bin_w
        by    = self._baseline
        dx    = self._depth_x
        cap_h = self._cap_h

        if h < 1e-4:
            return grp

        # ── Front face (main bar body) ────────────────────────────────
        front = Rectangle(
            width=bw - 0.015,
            height=h,
            fill_color=PAL["bar_top"],
            fill_opacity=0.88,
            stroke_color=PAL["bar_border"],
            stroke_width=0.5,
        )
        front.move_to([xl + bw/2, by + h/2, 0.002])
        grp.add(front)

        # ── Right side face ───────────────────────────────────────────
        side_pts = [
            [xl + bw - 0.008,    by,       0.002],
            [xl + bw - 0.008 + dx, by + dx*0.5, 0.002],
            [xl + bw - 0.008 + dx, by + h + dx*0.5, 0.002],
            [xl + bw - 0.008,    by + h,   0.002],
        ]
        side = Polygon(*side_pts,
                       fill_color=PAL["bar_side"],
                       fill_opacity=0.82, stroke_width=0)
        side.shift([0, 0, 0.001])
        grp.add(side)

        # ── Top cap ───────────────────────────────────────────────────
        top_pts = [
            [xl + 0.008,         by + h,           0.004],
            [xl + bw - 0.008,    by + h,           0.004],
            [xl + bw - 0.008 + dx, by + h + dx*0.5, 0.004],
            [xl + 0.008 + dx,    by + h + dx*0.5,  0.004],
        ]
        top = Polygon(*top_pts,
                      fill_color=PAL["bar_top_lite"],
                      fill_opacity=0.90, stroke_width=0)
        grp.add(top)

        # ── AO shadow at base ─────────────────────────────────────────
        ao_h = 0.055
        ao   = Rectangle(
            width=bw - 0.015,
            height=ao_h,
            fill_color=PAL["bar_ao"],
            fill_opacity=0.55,
            stroke_width=0,
        )
        ao.move_to([xl + bw/2, by + ao_h/2, 0.003])
        grp.add(ao)

        return grp

    def update_bar(self, i: int):
        """Rebuild bar i to reflect current count."""
        # Remove old bar
        old = self._bars[i]
        if old in self.submobjects:
            self.remove(old)
        # Build new
        new = self._make_bar(i, int(self._counts[i]))
        self._bars[i] = new
        self.add(new)

        # Count label
        if self._count_labels[i] is not None:
            if self._count_labels[i] in self.submobjects:
                self.remove(self._count_labels[i])
        xl  = self._bar_left_px(i)
        h   = self._bar_height(int(self._counts[i]))
        lbl = Text(str(int(self._counts[i])),
                   font_size=11, color=PAL["panel_val"])
        lbl.move_to([xl + self._bin_w/2,
                     self._baseline + h + 0.18, 0.008])
        self._count_labels[i] = lbl
        self.add(lbl)

    def add_count(self, raw_mean: float) -> int:
        """
        Increment the bin containing raw_mean.
        Returns the bin index.
        """
        i = int(np.searchsorted(self._edges[1:], raw_mean))
        i = np.clip(i, 0, self._n_bins - 1)
        self._counts[i] += 1
        self.update_bar(i)
        return int(i)

    def bin_top_position(self, i: int) -> np.ndarray:
        """World position of the top-centre of bar i."""
        xl = self._bar_left_px(i)
        h  = self._bar_height(int(self._counts[i]))
        return np.array([xl + self._bin_w/2,
                         self._baseline + h + 0.10, 0.0])

    def set_theory_curve(
        self,
        mu: float,
        sigma_n: float,
        n_curves: int = 0,
    ) -> Optional[VGroup]:
        """
        Draw or update the theoretical N(mu, sigma_n²) curve overlay.
        ``n_curves`` determines dashed (< 10) vs solid style.
        """
        if self._theory_curve is not None:
            if self._theory_curve in self.submobjects:
                self.remove(self._theory_curve)

        raw_x = np.linspace(self._x_lo, self._x_hi, 400)
        y_pdf = _norm_pdf(raw_x, mu, sigma_n)
        y_pdf = np.where(np.isfinite(y_pdf), y_pdf, 0.0)

        # Scale: area of pdf = 1; histogram total area = sum(counts) * bin_width
        bin_w_raw = (self._x_hi - self._x_lo) / self._n_bins
        total     = int(self._counts.sum())
        if total < 2:
            return None

        scale_y = (total * bin_w_raw
                   * self._max_bar_h / self._max_count
                   / (self._x_hi - self._x_lo) * self._n_bins)

        x_px    = _map_x(raw_x, self._x_lo, self._x_hi, self._panel_w)
        y_px    = y_pdf * scale_y

        curve_grp = VGroup()
        is_solid  = n_curves >= 10

        # Fill area (subtle)
        verts = _curve_verts(x_px, y_px, self._baseline)
        if len(verts) >= 3:
            fill = Polygon(*verts,
                           fill_color=PAL["theory_fill"],
                           fill_opacity=0.30,
                           stroke_width=0)
            fill.shift([0, 0, 0.006])
            curve_grp.add(fill)

        # Curve line
        step  = max(1, len(x_px)//100)
        pts3d = [np.array([float(x_px[j]), float(y_px[j]), 0.008])
                 for j in range(0, len(x_px), step) if y_px[j] > 0.005]
        if len(pts3d) >= 2:
            col = PAL["theory_solid"] if is_solid else PAL["theory_dashed"]
            sp  = VMobject(stroke_color=col,
                           stroke_width=2.2 if is_solid else 1.5,
                           stroke_opacity=0.90 if is_solid else 0.60)
            sp.set_points_smoothly(pts3d)
            curve_grp.add(sp)

        self._theory_curve = curve_grp
        self.add(curve_grp)
        return curve_grp


class _SEArrow(VGroup):
    """
    Double-headed bracket arrow spanning ±1 SE around the sampling mean.

    Components:
      1. DoubleArrow
      2. SE badge: "SE = σ/√n = {value:.3f}"
      3. Bracket ticks at each end

    Parameters
    ----------
    mu_px  : Manim x of the sampling distribution mean
    se_px  : Manim units width of 1 SE
    y      : vertical position
    se_val : raw SE value (for label)
    n      : sample size (for label)
    """

    def __init__(
        self,
        mu_px:  float,
        se_px:  float,
        y:      float,
        se_val: float,
        n:      int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.012

        arr = DoubleArrow(
            start=[mu_px - se_px, y, z0],
            end  =[mu_px + se_px, y, z0],
            stroke_color=PAL["se_arrow"],
            stroke_width=2.0,
            tip_length=0.14,
            buff=0,
        )
        self.add(arr)

        # Bracket ticks at ends
        for xp in [mu_px - se_px, mu_px + se_px]:
            tk = Line([xp, y - 0.10, z0+0.001],
                      [xp, y + 0.10, z0+0.001],
                      stroke_color=PAL["se_arrow"], stroke_width=1.5)
            self.add(tk)

        # Badge
        try:
            lbl = MathTex(
                rf"\mathrm{{SE}} = \sigma/\sqrt{{n}} = {se_val:.3f}",
                font_size=18, color=PAL["se_label"]
            )
        except Exception:
            lbl = Text(f"SE = σ/√n = {se_val:.3f}",
                       font_size=15, color=PAL["se_label"])
        bg = RoundedRectangle(
            width=lbl.width + 0.22,
            height=lbl.height + 0.14,
            corner_radius=0.06,
            fill_color=PAL["se_badge_bg"],
            fill_opacity=0.90,
            stroke_color=PAL["se_arrow"],
            stroke_width=0.7,
        )
        bg.move_to([mu_px, y + 0.35, z0 + 0.001])
        lbl.move_to([mu_px, y + 0.35, z0 + 0.002])
        self.add(bg, lbl)


class _StatsPanel(VGroup):
    """
    Floating live-stats panel showing current simulation state.

    Rows:
      Samples drawn:  k
      Sample size n:  n
      Current x̄:    value
      True μ:         value
      SE = σ/√n:     value
      Empirical SE:   std(means) so far

    Parameters
    ----------
    panel_w, panel_h : dimensions
    """

    def __init__(
        self,
        k:        int   = 0,
        n:        int   = 1,
        x_bar:    float = 0.0,
        true_mu:  float = 0.0,
        se_true:  float = 1.0,
        se_emp:   float = 0.0,
        panel_w:  float = 2.40,
        panel_h:  float = 2.20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        bg = RoundedRectangle(
            width=panel_w, height=panel_h,
            corner_radius=0.10,
            fill_color=PAL["panel_bg"],
            fill_opacity=0.92,
            stroke_color=PAL["panel_border"],
            stroke_width=0.8,
        )
        bg.move_to([0, 0, z0])
        self.add(bg)

        # Left border strip
        strip = RoundedRectangle(
            width=0.08, height=panel_h-0.04,
            corner_radius=0.04,
            fill_color=PAL["bar_top"],
            fill_opacity=0.85, stroke_width=0,
        )
        strip.move_to([-panel_w/2+0.06, 0, z0+0.001])
        self.add(strip)

        title = Text("CLT Simulation", font_size=15,
                     color=PAL["panel_title"])
        title.move_to([0, panel_h/2 - 0.22, z0+0.002])
        self.add(title)

        self.add(Line(
            [-panel_w/2+0.14, panel_h/2-0.42, z0+0.002],
            [ panel_w/2-0.08, panel_h/2-0.42, z0+0.002],
            stroke_color=PAL["panel_border"], stroke_width=0.6,
        ))

        rows = [
            ("Samples drawn",  str(k)),
            ("n per sample",   str(n)),
            ("Current x̄",    f"{x_bar:.3f}"),
            ("True μ",         f"{true_mu:.3f}"),
            ("SE = σ/√n",      f"{se_true:.4f}"),
            ("Empirical SE",   f"{se_emp:.4f}" if k>1 else "—"),
        ]
        row_h = (panel_h - 0.80) / len(rows)
        y0    = panel_h/2 - 0.60
        for i, (key, val) in enumerate(rows):
            yr = y0 - i * row_h
            km = Text(key, font_size=12, color=PAL["panel_key"])
            vm = Text(val, font_size=12, color=PAL["panel_val"],
                      font="monospace")
            km.move_to([-panel_w/2+0.22+km.width/2, yr, z0+0.003])
            vm.move_to([ panel_w/2-0.10-vm.width/2, yr, z0+0.003])
            self.add(km, vm)


# ─────────────────────────────────────────────────────────────────────────────
# SamplingDistribution3D  ──  the main export
# ─────────────────────────────────────────────────────────────────────────────

class SamplingDistribution3D(VGroup):
    """
    Full Central Limit Theorem / sampling distribution visualisation.

    Layout (left → right):
      [Population panel] → [CLT arrow] → [Sample strip] → [Histogram panel]

    Parameters
    ----------
    pop_type   : str    — "normal" | "uniform" | "exponential" |
                          "chi2" | "bimodal" | "bernoulli"
    params     : dict   — distribution parameters
    n          : int    — sample size per draw
    n_bins     : int    — histogram bins (default 20)
    pop_panel_w: float  — population panel width
    hist_panel_w: float — histogram panel width
    panel_h    : float  — curve / bar maximum height
    baseline_y : float
    max_samples: int    — expected total samples (for bar scaling)
    seed       : int    — RNG seed for reproducibility
    show_strip : bool   — show the sample dot strip
    show_se    : bool   — show the SE arrow on histogram
    show_panel : bool   — show the stats panel

    Attributes
    ----------
    pop_panel     : _PopulationPanel
    sample_strip  : _SampleStrip | None
    hist_panel    : _HistogramPanel
    se_arrow      : _SEArrow | None
    stats_panel   : _StatsPanel | None
    clt_arrow     : Arrow
    n_tracker     : ValueTracker
    k_drawn       : int   — samples drawn so far
    means         : list  — running list of drawn sample means
    current_sample: np.ndarray | None

    Class methods
    -------------
    SamplingDistribution3D.normal(mu, sigma, n, **kw)
    SamplingDistribution3D.uniform(a, b, n, **kw)
    SamplingDistribution3D.exponential(lam, n, **kw)
    SamplingDistribution3D.chi_squared(df, n, **kw)
    SamplingDistribution3D.bimodal(mu1, mu2, sigma, mix, n, **kw)
    SamplingDistribution3D.bernoulli(p, n, **kw)
    """

    def __init__(
        self,
        pop_type:    str  = "normal",
        params:      dict = None,
        n:           int  = 5,
        n_bins:      int  = 20,
        pop_panel_w: float = 4.0,
        hist_panel_w: float = 5.0,
        panel_h:     float = 3.0,
        baseline_y:  float = 0.0,
        max_samples: int   = 100,
        seed:        int   = 42,
        show_strip:  bool  = True,
        show_se:     bool  = True,
        show_panel:  bool  = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        params = params or {"mu": 0.0, "sigma": 1.0}
        self._pop_type    = pop_type
        self._params      = params
        self._n           = n
        self._n_bins      = n_bins
        self._baseline    = baseline_y
        self._panel_h     = panel_h
        self._max_samples = max_samples
        self._rng         = np.random.default_rng(seed)
        self.k_drawn      = 0
        self.means: list[float] = []
        self.current_sample: Optional[np.ndarray] = None
        self.n_tracker    = ValueTracker(float(n))

        # Population statistics
        self._mu_pop, self._sigma_pop = _pop_stats(pop_type, params)
        self._se           = self._sigma_pop / np.sqrt(n)
        x_lo_pop, x_hi_pop = _pop_x_range(pop_type, params)

        # Sampling distribution x-range: ±4 SE around mu
        x_lo_hist = self._mu_pop - 4.5 * self._se
        x_hi_hist = self._mu_pop + 4.5 * self._se

        # ── Layout positions ──────────────────────────────────────────
        gap         = 0.45
        strip_w     = hist_panel_w * 0.70
        strip_y     = baseline_y - 1.30
        pop_cx      = -(pop_panel_w/2 + gap + strip_w/2 + gap + hist_panel_w/2) / 2
        hist_cx     = (hist_panel_w/2 + gap + strip_w/2 + gap + pop_panel_w/2) / 2

        # Total width centres
        total_w = pop_panel_w + strip_w + hist_panel_w + 2*gap
        pop_cx  = -total_w/2 + pop_panel_w/2
        hist_cx =  total_w/2 - hist_panel_w/2
        strip_cx = 0.0

        # ── Population panel ──────────────────────────────────────────
        self.pop_panel = _PopulationPanel(
            pop_type=pop_type,
            params=params,
            panel_w=pop_panel_w,
            panel_h=panel_h,
            baseline_y=baseline_y,
        )
        self.pop_panel.shift([pop_cx, 0, 0])
        self.add(self.pop_panel)

        # ── CLT arrow ─────────────────────────────────────────────────
        clt_x_start = pop_cx + pop_panel_w/2 + 0.10
        clt_x_end   = hist_cx - hist_panel_w/2 - 0.10
        clt_y       = baseline_y + panel_h * 0.55
        self.clt_arrow = Arrow(
            start=[clt_x_start, clt_y, 0],
            end  =[clt_x_end,   clt_y, 0],
            stroke_color=PAL["clt_arrow"],
            stroke_width=2.0,
            tip_length=0.20,
            buff=0,
        )
        try:
            clt_lbl = MathTex(
                rf"n = {n},\;\text{{repeat}}",
                font_size=18, color=PAL["clt_label"]
            )
        except Exception:
            clt_lbl = Text(f"n={n}, repeat",
                           font_size=15, color=PAL["clt_label"])
        clt_lbl.next_to(self.clt_arrow, UP, buff=0.10)
        self._clt_label = clt_lbl
        self.add(self.clt_arrow, clt_lbl)

        # ── Sample strip ──────────────────────────────────────────────
        self.sample_strip = None
        if show_strip:
            self.sample_strip = _SampleStrip(
                x_lo=x_lo_pop,
                x_hi=x_hi_pop,
                strip_w=strip_w,
                y=strip_y,
                n_samples=n,
            )
            self.sample_strip.shift([strip_cx, 0, 0])
            self.add(self.sample_strip)

        # ── Histogram panel ───────────────────────────────────────────
        self.hist_panel = _HistogramPanel(
            x_lo=x_lo_hist,
            x_hi=x_hi_hist,
            n_bins=n_bins,
            panel_w=hist_panel_w,
            max_bar_h=panel_h * 0.88,
            baseline_y=baseline_y,
            max_count=max(max_samples // n_bins * 3, 5),
        )
        self.hist_panel.shift([hist_cx, 0, 0])
        self._hist_cx  = hist_cx
        self._hist_x_lo = x_lo_hist
        self._hist_x_hi = x_hi_hist
        self.add(self.hist_panel)

        # ── Histogram title ───────────────────────────────────────────
        try:
            htitle = MathTex(
                r"\text{Sampling Distribution of }\bar{X}",
                font_size=20, color=PAL["panel_title"],
            )
        except Exception:
            htitle = Text("Sampling Distribution of X̄",
                          font_size=17, color=PAL["panel_title"])
        htitle.move_to([hist_cx,
                        baseline_y + panel_h + 0.62, 0.003])
        self.add(htitle)

        # ── SE arrow ─────────────────────────────────────────────────
        self.se_arrow = None
        if show_se:
            mu_hist_px = self._hist_px(self._mu_pop)
            se_px_w    = self._se * (hist_panel_w / (x_hi_hist - x_lo_hist))
            self.se_arrow = _SEArrow(
                mu_px=hist_cx + mu_hist_px,
                se_px=se_px_w,
                y=baseline_y - 0.58,
                se_val=self._se,
                n=n,
            )
            self.add(self.se_arrow)

        # ── Stats panel ───────────────────────────────────────────────
        self.stats_panel = None
        if show_panel:
            self.stats_panel = _StatsPanel(
                k=0, n=n,
                x_bar=self._mu_pop,
                true_mu=self._mu_pop,
                se_true=self._se,
            )
            self.stats_panel.move_to([strip_cx, baseline_y - 2.30, 0])
            self.add(self.stats_panel)

        # Internals
        self._strip_cx   = strip_cx
        self._pop_cx     = pop_cx
        self._x_lo_pop   = x_lo_pop
        self._x_hi_pop   = x_hi_pop
        self._hist_panel_w = hist_panel_w

    # ─────────────────────────────────────────────────────────────────
    # Coordinate helpers
    # ─────────────────────────────────────────────────────────────────

    def _hist_px(self, raw_val: float) -> float:
        """Manim x relative to hist panel centre."""
        return _map_x(raw_val,
                      self._hist_x_lo, self._hist_x_hi,
                      self._hist_panel_w)

    # ─────────────────────────────────────────────────────────────────
    # Sample generation
    # ─────────────────────────────────────────────────────────────────

    def draw_new_sample(self) -> np.ndarray:
        """Draw n observations from the population. Stores in self.current_sample."""
        sample = _sample_pop(self._pop_type, self._params,
                             self._n, self._rng)
        self.current_sample = sample
        return sample

    def commit_mean(self) -> float:
        """Add current sample mean to means list. Returns the mean."""
        if self.current_sample is None:
            self.draw_new_sample()
        mean = float(np.mean(self.current_sample))
        self.means.append(mean)
        self.k_drawn += 1
        return mean

    def update_stats_panel(self):
        """Rebuild the stats panel with current counts."""
        if self.stats_panel is None:
            return
        if self.stats_panel in self.submobjects:
            self.remove(self.stats_panel)
        pos = self.stats_panel.get_center()
        emp_se = float(np.std(self.means)) if len(self.means) > 1 else 0.0
        x_bar  = float(self.means[-1]) if self.means else self._mu_pop
        self.stats_panel = _StatsPanel(
            k=self.k_drawn, n=self._n,
            x_bar=x_bar,
            true_mu=self._mu_pop,
            se_true=self._se,
            se_emp=emp_se,
        )
        self.stats_panel.move_to(pos)
        self.add(self.stats_panel)

    # ─────────────────────────────────────────────────────────────────
    # Class-method constructors
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def normal(cls, mu: float = 0.0, sigma: float = 1.0,
               n: int = 5, **kw) -> "SamplingDistribution3D":
        return cls("normal", {"mu": mu, "sigma": sigma}, n=n, **kw)

    @classmethod
    def uniform(cls, a: float = 0.0, b: float = 1.0,
                n: int = 5, **kw) -> "SamplingDistribution3D":
        return cls("uniform", {"a": a, "b": b}, n=n, **kw)

    @classmethod
    def exponential(cls, lam: float = 1.0,
                    n: int = 5, **kw) -> "SamplingDistribution3D":
        return cls("exponential", {"lam": lam}, n=n, **kw)

    @classmethod
    def chi_squared(cls, df: int = 4,
                    n: int = 5, **kw) -> "SamplingDistribution3D":
        return cls("chi2", {"df": df}, n=n, **kw)

    @classmethod
    def bimodal(
        cls,
        mu1: float = -2.0, mu2: float = 2.0,
        sigma: float = 0.8, mix: float = 0.5,
        n: int = 5, **kw,
    ) -> "SamplingDistribution3D":
        return cls("bimodal",
                   {"mu1": mu1, "mu2": mu2,
                    "sigma": sigma, "mix": mix},
                   n=n, **kw)

    @classmethod
    def bernoulli(cls, p: float = 0.3,
                  n: int = 20, **kw) -> "SamplingDistribution3D":
        return cls("bernoulli", {"p": p}, n=n, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Animations
# ─────────────────────────────────────────────────────────────────────────────

class BuildPopulation(Animation):
    """
    Population curve grows upward from the baseline.

    α ∈ [0.00, 0.65] — curve scales up from 0
    α ∈ [0.55, 1.00] — parameter badges and title fade in
    """

    def __init__(self, sd: SamplingDistribution3D, **kwargs):
        self.sd = sd
        kwargs.setdefault("run_time", 2.0)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(sd.pop_panel, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        scale  = max(min(alpha / 0.65, 1.0), 1e-5)
        badge_a = np.clip((alpha - 0.55) / 0.45, 0, 1)
        self.mobject.scale(
            [1, scale, 1],
            about_point=[
                self.mobject.get_center()[0],
                self.sd._baseline, 0,
            ],
        )
        # Fade in badges (last few submobjects)
        for sub in list(self.mobject.submobjects)[-5:]:
            sub.set_opacity(badge_a)


class DrawSample(Succession):
    """
    Drop n observation dots onto the sample strip one by one.
    Each dot falls from the population curve level to the strip.

    Parameters
    ----------
    sd         : SamplingDistribution3D
    sample     : np.ndarray | None  — if None, calls sd.draw_new_sample()
    dot_rt     : float              — time per dot drop
    stagger    : float              — delay between dots
    """

    def __init__(
        self,
        sd:     SamplingDistribution3D,
        sample: Optional[np.ndarray] = None,
        dot_rt: float = 0.18,
        stagger: float = 0.04,
        **kwargs,
    ):
        if sample is None:
            sample = sd.draw_new_sample()
        else:
            sd.current_sample = sample

        if sd.sample_strip is None:
            super().__init__(FadeIn(VGroup(), run_time=0.01), **kwargs)
            return

        # Clear previous dots
        sd.sample_strip.clear_dots()

        anims = []
        for val in sample:
            # Clamp to strip range
            val = float(np.clip(val, sd._x_lo_pop, sd._x_hi_pop))
            dot = sd.sample_strip.add_dot(val)
            dot.set_opacity(0)
            anims.append(FadeIn(dot, shift=DOWN * 0.18, run_time=dot_rt))

        kwargs.setdefault("run_time",
                          len(anims) * (dot_rt + stagger))
        super().__init__(*anims, **kwargs)


class ExtractMean(Animation):
    """
    Bracket collapses from both ends to the mean.
    Mean dot brightens. Mean dot then rises toward the histogram.

    Parameters
    ----------
    sd       : SamplingDistribution3D
    run_time : float
    """

    def __init__(self, sd: SamplingDistribution3D, **kwargs):
        self.sd = sd
        kwargs.setdefault("run_time", 0.90)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)

        # Ensure sample and mean dot exist
        if sd.current_sample is None:
            sd.draw_new_sample()
        mean_raw = float(np.mean(sd.current_sample))
        self._mean_raw = mean_raw

        if sd.sample_strip is not None:
            self._mean_dot = sd.sample_strip.make_mean_dot(mean_raw)
            self._mean_dot.set_opacity(0)
        else:
            self._mean_dot = None

        mob = sd.sample_strip if sd.sample_strip is not None else VGroup()
        super().__init__(mob, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        # Fade in mean dot
        if self._mean_dot is not None:
            md_alpha = min(1.0, alpha * 2.5)
            # Re-find mean dot in submobjects (it was added during __init__)
            for sub in self.mobject.submobjects:
                if isinstance(sub, Dot) and sub.get_color() == PAL["dot_mean"]:
                    sub.set_opacity(md_alpha)
                    # Scale up slightly
                    sub.scale(1.0 + 0.25 * np.sin(alpha * PI))
                    break

        # Shrink sample dots toward mean (collapse animation)
        collapse = min(1.0, alpha / 0.80)
        for sub in self.mobject.submobjects:
            if isinstance(sub, Dot) and sub.get_color() != PAL["dot_mean"]:
                if collapse > 0:
                    sub.set_opacity(1.0 - collapse * 0.65)


class AddToHistogram(Animation):
    """
    Mean dot flies from the strip to its histogram bin.
    The bar grows by 1 unit. The bin flashes.

    Parameters
    ----------
    sd       : SamplingDistribution3D
    mean_val : float | None   — if None uses sd.means[-1] or commits new
    run_time : float
    """

    def __init__(
        self,
        sd:       SamplingDistribution3D,
        mean_val: Optional[float] = None,
        **kwargs,
    ):
        self.sd = sd

        if mean_val is None:
            mean_val = sd.commit_mean()
        self._mean_val = mean_val

        # Determine target bin and bar top position
        bin_i          = sd.hist_panel.add_count(mean_val)
        self._bin_i    = bin_i
        self._target   = sd.hist_panel.bin_top_position(bin_i)
        self._target   = self._target + np.array([sd._hist_cx, 0, 0])

        # Start position: above sample strip
        if sd.sample_strip is not None:
            sxp = sd.sample_strip.raw_to_px(
                float(np.clip(mean_val, sd._x_lo_pop, sd._x_hi_pop))
            )
            self._start = np.array([
                sxp + sd._strip_cx,
                sd._baseline - 1.0,
                0.015,
            ])
        else:
            self._start = sd.hist_panel.bin_top_position(bin_i)

        # Travelling dot
        self._trav_dot = Dot(
            radius=0.10,
            point=self._start,
            color=PAL["dot_mean"],
            fill_opacity=1.0,
        )
        sd.add(self._trav_dot)

        kwargs.setdefault("run_time", 0.65)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(self._trav_dot, **kwargs)

    def interpolate_mobject(self, alpha: float):
        # Parabolic arc from start to target
        h_arc   = abs(self._target[1] - self._start[1]) * 0.40
        pos     = (self._start * (1 - alpha)
                   + self._target * alpha
                   + UP * h_arc * 4 * alpha * (1 - alpha))
        self.mobject.move_to(pos)
        # Fade out near destination
        self.mobject.set_opacity(1.0 - alpha * 0.6)

    def clean_up_from_scene(self, scene) -> None:
        super().clean_up_from_scene(scene)
        if self._trav_dot in self.sd.submobjects:
            self.sd.remove(self._trav_dot)
        # Update theory curve and stats panel
        if self.sd.k_drawn >= 2:
            self.sd.hist_panel.set_theory_curve(
                self.sd._mu_pop,
                self.sd._se,
                self.sd.k_drawn,
            )
        self.sd.update_stats_panel()


class RunCLT(Succession):
    """
    One full CLT iteration:
      DrawSample → ExtractMean → AddToHistogram

    Parameters
    ----------
    sd        : SamplingDistribution3D
    run_time  : float  — total time for the full iteration
    fast      : bool   — if True, skip dot animation (instant add)
    """

    def __init__(
        self,
        sd:       SamplingDistribution3D,
        run_time: float = 1.5,
        fast:     bool  = False,
        **kwargs,
    ):
        sample   = sd.draw_new_sample()
        mean_val = float(np.mean(sample))

        if fast:
            # Commit directly without animation
            sd.current_sample = sample
            sd.commit_mean()
            sd.hist_panel.set_theory_curve(
                sd._mu_pop, sd._se, sd.k_drawn)
            sd.update_stats_panel()
            anims = [FadeIn(VGroup(), run_time=0.01)]
        else:
            ds_rt  = run_time * 0.38
            em_rt  = run_time * 0.28
            ah_rt  = run_time * 0.34
            anims = [
                DrawSample(sd, sample=sample,
                           dot_rt=ds_rt / max(sd._n, 1),
                           run_time=ds_rt),
                ExtractMean(sd, run_time=em_rt),
                AddToHistogram(sd, mean_val=mean_val,
                               run_time=ah_rt),
            ]

        kwargs.setdefault("run_time", run_time)
        super().__init__(*anims, **kwargs)


class FlashBin(AnimationGroup):
    """
    Flash the histogram bar at bin index i.

    Parameters
    ----------
    sd       : SamplingDistribution3D
    bin_i    : int
    run_time : float
    """

    def __init__(
        self,
        sd:    SamplingDistribution3D,
        bin_i: int,
        **kwargs,
    ):
        bar  = sd.hist_panel._bars[bin_i]
        col  = PAL["bar_flash"]
        kwargs.setdefault("run_time", 0.40)
        super().__init__(
            Indicate(bar, color=col, scale_factor=1.06,
                     run_time=kwargs["run_time"]),
            **kwargs,
        )


class NarrowSE(Animation):
    """
    Increase n → SE shrinks → histogram bars narrow → SE arrow shrinks.

    Morphs the entire visualisation into a rebuilt one.

    Parameters
    ----------
    sd      : SamplingDistribution3D
    new_n   : int
    run_time: float
    """

    def __init__(
        self,
        sd:     SamplingDistribution3D,
        new_n:  int,
        **kwargs,
    ):
        self.sd    = sd
        self.new_n = new_n
        # Build a fresh SD with the new n (no samples yet)
        self.target = type(sd)(
            pop_type=sd._pop_type,
            params=sd._params,
            n=new_n,
            n_bins=sd._n_bins,
            pop_panel_w=sd.pop_panel._panel_w,
            hist_panel_w=sd._hist_panel_w,
            panel_h=sd._panel_h,
            baseline_y=sd._baseline,
            max_samples=sd._max_samples,
            show_strip=sd.sample_strip is not None,
            show_se=sd.se_arrow is not None,
            show_panel=sd.stats_panel is not None,
        )
        self.target.move_to(sd.get_center())
        kwargs.setdefault("run_time", 2.5)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(sd, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(1 - alpha)
        self.target.set_opacity(alpha)
        if alpha >= 1.0:
            self.mobject.become(self.target)
            self.sd._n  = self.new_n
            self.sd._se = self.sd._sigma_pop / np.sqrt(self.new_n)
            self.sd.n_tracker.set_value(float(self.new_n))


class ConvergenceRace(Succession):
    """
    Rapidly animate k_total CLT iterations in fast mode,
    then fade in the final theoretical curve prominently.

    Parameters
    ----------
    sd        : SamplingDistribution3D
    k_total   : int    — total samples to draw
    burst_rt  : float  — time per fast iteration
    """

    def __init__(
        self,
        sd:       SamplingDistribution3D,
        k_total:  int   = 100,
        burst_rt: float = 0.04,
        **kwargs,
    ):
        # Burst phase: draw all samples in fast mode
        burst_anims = []
        for _ in range(k_total):
            burst_anims.append(
                RunCLT(sd, run_time=burst_rt, fast=True)
            )

        # Final theory curve reveal
        def _reveal_theory(sd=sd):
            sd.hist_panel.set_theory_curve(
                sd._mu_pop, sd._se, sd.k_drawn)
            return FadeIn(
                sd.hist_panel._theory_curve or VGroup(),
                run_time=1.0,
            )

        anims = burst_anims + [_reveal_theory()]
        kwargs.setdefault("run_time",
                          k_total * burst_rt + 1.0)
        super().__init__(*anims, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factories
# ─────────────────────────────────────────────────────────────────────────────

def make_clt_comparison(
    pop_specs: list[tuple[str, dict]],
    n: int = 10,
    n_bins: int = 16,
    k_samples: int = 60,
    panel_scale: float = 0.52,
    seed: int = 42,
) -> VGroup:
    """
    Create a row of SamplingDistribution3D objects, one per population,
    with k_samples already drawn — for a side-by-side CLT comparison.

    Parameters
    ----------
    pop_specs   : list of (pop_type, params) tuples
    n           : sample size
    n_bins      : histogram bins per panel
    k_samples   : samples to pre-draw (fast mode)
    panel_scale : scale factor per panel
    seed        : RNG seed

    Returns
    -------
    VGroup of SamplingDistribution3D objects.
    """
    grp   = VGroup()
    pw    = 3.0   # pop panel width per item
    hw    = 3.5   # hist panel width per item
    gap   = 0.3
    step  = (pw + hw + gap * 3) * panel_scale

    for i, (pt, pm) in enumerate(pop_specs):
        sd = SamplingDistribution3D(
            pop_type=pt, params=pm, n=n,
            n_bins=n_bins,
            pop_panel_w=pw, hist_panel_w=hw,
            max_samples=k_samples + 10,
            show_strip=False,
            show_se=True,
            show_panel=False,
            seed=seed + i,
        )
        # Fast-draw k_samples
        for _ in range(k_samples):
            sd.draw_new_sample()
            sd.commit_mean()
            sd.hist_panel.add_count(sd.means[-1])
        sd.hist_panel.set_theory_curve(
            sd._mu_pop, sd._se, sd.k_drawn)

        sd.scale(panel_scale)
        sd.move_to([i * step, 0, 0])
        grp.add(sd)

    grp.center()
    return grp


def make_n_effect_row(
    pop_type: str = "exponential",
    params:   dict = None,
    n_values: list[int] = None,
    k_samples: int = 80,
    panel_scale: float = 0.55,
    seed: int = 0,
) -> VGroup:
    """
    Create a row of SamplingDistribution3D objects varying n,
    showing how increasing n narrows the sampling distribution.

    Parameters
    ----------
    pop_type, params : source population
    n_values         : list of sample sizes (default [1, 5, 25, 100])
    k_samples        : samples to pre-draw per panel
    panel_scale, seed

    Returns
    -------
    VGroup of SamplingDistribution3D objects with n labels.
    """
    if params is None:
        params = {"lam": 1.0}
    if n_values is None:
        n_values = [1, 5, 25, 100]

    grp   = VGroup()
    pw, hw = 2.8, 3.2
    step  = (pw + hw + 0.5) * panel_scale

    for i, n in enumerate(n_values):
        sd = SamplingDistribution3D(
            pop_type=pop_type, params=params, n=n,
            n_bins=16,
            pop_panel_w=pw, hist_panel_w=hw,
            max_samples=k_samples + 10,
            show_strip=False, show_se=True, show_panel=False,
            seed=seed + i * 13,
        )
        for _ in range(k_samples):
            sd.draw_new_sample()
            sd.commit_mean()
            sd.hist_panel.add_count(sd.means[-1])
        sd.hist_panel.set_theory_curve(sd._mu_pop, sd._se, k_samples)

        sd.scale(panel_scale)
        sd.move_to([i * step, 0, 0])

        # n label below
        nlbl = Text(f"n = {n}", font_size=18,
                    color=PAL["clt_label"])
        nlbl.next_to(sd, DOWN, buff=0.12)
        grp.add(sd, nlbl)

    grp.center()
    return grp


# ─────────────────────────────────────────────────────────────────────────────
# Demo Scene  (run with: manim -pql sampling_dist.py SamplingDistDemo)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from manim import Scene

    class SamplingDistDemo(Scene):
        """Full CLT demonstration showcase."""

        def construct(self):

            # ── Exponential population, n=5 ──────────────────────────
            sd = SamplingDistribution3D.exponential(
                lam=1.0, n=5,
                n_bins=18,
                pop_panel_w=3.8,
                hist_panel_w=4.8,
                panel_h=2.8,
                baseline_y=0.5,
                max_samples=80,
                show_strip=True,
                show_se=True,
                show_panel=True,
            )
            sd.center()
            self.add(sd)

            # Build the population curve
            self.play(BuildPopulation(sd, run_time=1.8))
            self.play(FadeIn(sd.clt_arrow, run_time=0.5))
            self.wait(0.3)

            # Run 8 full iterations with animation
            for _ in range(8):
                self.play(RunCLT(sd, run_time=1.2))
            self.wait(0.5)

            # Flash the most-filled bin
            top_bin = int(np.argmax(sd.hist_panel._counts))
            self.play(FlashBin(sd, top_bin, run_time=0.5))
            self.wait(0.4)

            # Convergence race: 40 more fast samples
            self.play(ConvergenceRace(sd, k_total=40,
                                      burst_rt=0.03))
            self.wait(0.6)

            # Increase n → SE narrows
            self.play(NarrowSE(sd, new_n=25, run_time=2.5))
            self.wait(0.8)

            self.play(FadeOut(sd))

            # ── n-effect comparison row ───────────────────────────────
            row = make_n_effect_row(
                pop_type="exponential",
                params={"lam": 1.0},
                n_values=[1, 5, 30, 100],
                k_samples=100,
                panel_scale=0.48,
            )
            row.center()
            self.play(FadeIn(row, shift=UP*0.3), run_time=1.0)
            self.wait(2.5)

            self.play(FadeOut(row))

            # ── CLT comparison across population shapes ───────────────
            comparison = make_clt_comparison(
                pop_specs=[
                    ("uniform",     {"a": 0.0, "b": 1.0}),
                    ("exponential", {"lam": 1.0}),
                    ("bimodal",     {"mu1": -2.0, "mu2": 2.0,
                                     "sigma": 0.7, "mix": 0.5}),
                ],
                n=20, k_samples=80,
                panel_scale=0.48,
            )
            comparison.center()
            self.play(FadeIn(comparison, shift=UP*0.2), run_time=1.0)
            self.wait(3.0)

except ImportError:
    pass