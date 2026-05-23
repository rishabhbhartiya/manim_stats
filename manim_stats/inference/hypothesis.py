"""
manim_stats/inference/hypothesis.py
====================================
HypothesisTest3D — A highly detailed, statistically rigorous hypothesis
test visualisation for Manim statistics animations.

Primary use cases
-----------------
  One-sample Z-test              — known σ
  One-sample t-test              — unknown σ, small n
  Two-sample t-test              — independent samples
  Paired t-test                  — dependent samples
  Chi-square goodness-of-fit     — categorical data
  F-test / one-way ANOVA         — comparing variances / group means
  Two-tailed and one-tailed      — full symmetry support

Design goals
------------
Test statistic distribution
  * Correct curve shape per test:
      Z / t  → bell (t has heavier tails, parameterised by df)
      χ²     → right-skewed gamma-family curve
      F      → right-skewed with two df parameters
  * Three rendering layers per curve:
      – base fill polygon     (body colour)
      – AO tail darkening     (ambient occlusion near tails)
      – bright ridge spine    (lit top edge)
  * Distribution label badge: "Z ~ N(0,1)", "t ~ t(df=14)", etc.

Rejection region
  * Filled + hatched tail area beyond the critical value.
  * Hatch direction: ↗ for right tail, ↘ for left tail.
  * "Glow" effect: 3 concentric slightly-wider transparent versions
    of the critical line, decreasing in opacity — simulates neon glow.
  * Rejection zone label: "Rejection\nRegion  (α = 0.05)".

Critical value line
  * DashedLine with glow halo (3 concentric lines).
  * Tick + label at the x-axis: "z* = 1.645".
  * Bracket arc annotating the rejection tail width.

Observed statistic marker
  * Solid vertical line, coloured green (fail to reject) or red (reject).
  * Glow halo matching the decision colour.
  * Value label: "z_obs = 2.31" in a badge at the x-axis.
  * "Needle" drop animation: line falls from above.

p-value region
  * Separately coloured area between z_obs and the tail boundary.
  * Distinct from rejection region — uses a contrasting purple/teal fill.
  * p-value badge: "p = 0.0104" with a bracket connecting to the region.
  * For two-tailed: both p-value regions shown, p = 2 × one-tail area.

Test info panel (left sidebar)
  * Dark rounded-rect panel listing:
      – Test name            (e.g. "One-Sample Z-Test")
      – Hypotheses           H₀: μ = μ₀, H₁: μ > μ₀
      – Test statistic       Z = (x̄ − μ₀) / (σ/√n)
      – Observed value       z_obs = 2.31
      – Critical value       z* = 1.645
      – p-value              p = 0.0104
      – Significance level   α = 0.05
      – Decision             ✓ / ✗ badge
  * Panel has a coloured left border matching the decision colour.

Decision badge
  * Large centred badge that slams in during RevealDecision:
      "REJECT H₀"          — vivid red, bold
      "FAIL TO REJECT H₀"  — steel blue, bold
  * Semi-transparent rounded rect background.
  * Stamp-style rotation wobble on entry.

Animations
----------
  BuildTest        — axis grows, curve fills upward, critical lines appear
  DropStatistic    — stat marker falls from above the curve (ease-in)
  RevealPValue     — p-value region fills in from the stat line outward
  RevealDecision   — verdict badge stamps in with a rotation wobble
  SweepStatistic   — stat marker slides to new observed value
  ChangeAlpha      — critical line moves, rejection region resizes
  BuildInfoPanel   — panel types in line by line (Succession of Writes)
  CompareTests     — two HypothesisTest3D side by side (factory)

Dependencies
------------
  manim (CE or GL), numpy, scipy

Usage
-----
    from manim_stats.inference.hypothesis import (
        HypothesisTest3D, BuildTest, DropStatistic,
        RevealPValue, RevealDecision
    )

    class ZTestScene(Scene):
        def construct(self):
            ht = HypothesisTest3D.z_test(
                x_bar=52.3, mu0=50.0, sigma=8.0, n=25,
                alpha=0.05, tail="right",
            )
            self.play(BuildTest(ht))
            self.play(DropStatistic(ht))
            self.play(RevealPValue(ht))
            self.play(RevealDecision(ht))
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Literal, Sequence

from manim import (
    VGroup,
    Rectangle, RoundedRectangle,
    Circle, Annulus, Polygon, Line, DashedLine,
    Arrow, DoubleArrow,
    Text, MathTex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut,
    Create, Write, Uncreate,
    Indicate, Flash,
    Rotate,
    interpolate_color, color_to_rgb,
    WHITE, BLACK,
    GREY,  GREY_A,  GREY_B,  GREY_C,  GREY_D,
    RED,   RED_A,   RED_B,   RED_C,   RED_D,   RED_E,
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
# scipy — graceful fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.stats import norm as _scipy_norm
    from scipy.stats import t    as _scipy_t
    from scipy.stats import chi2 as _scipy_chi2
    from scipy.stats import f    as _scipy_f

    def _pdf(dist: str, x: np.ndarray, **kw) -> np.ndarray:
        if dist == "norm": return _scipy_norm.pdf(x)
        if dist == "t":    return _scipy_t.pdf(x, df=kw["df"])
        if dist == "chi2": return _scipy_chi2.pdf(x, df=kw["df"])
        if dist == "f":    return _scipy_f.pdf(x, dfn=kw["dfn"], dfd=kw["dfd"])
        return _scipy_norm.pdf(x)

    def _cdf(dist: str, x: float, **kw) -> float:
        if dist == "norm": return float(_scipy_norm.cdf(x))
        if dist == "t":    return float(_scipy_t.cdf(x, df=kw["df"]))
        if dist == "chi2": return float(_scipy_chi2.cdf(x, df=kw["df"]))
        if dist == "f":    return float(_scipy_f.cdf(x, dfn=kw["dfn"], dfd=kw["dfd"]))
        return float(_scipy_norm.cdf(x))

    def _ppf(dist: str, p: float, **kw) -> float:
        if dist == "norm": return float(_scipy_norm.ppf(p))
        if dist == "t":    return float(_scipy_t.ppf(p, df=kw["df"]))
        if dist == "chi2": return float(_scipy_chi2.ppf(p, df=kw["df"]))
        if dist == "f":    return float(_scipy_f.ppf(p, dfn=kw["dfn"], dfd=kw["dfd"]))
        return float(_scipy_norm.ppf(p))

except ImportError:
    # Minimal fallbacks (normal only)
    def _pdf(dist, x, **kw):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * PI)
    def _cdf(dist, x, **kw):
        from math import erf
        return 0.5 * (1 + erf(x / np.sqrt(2)))
    def _ppf(dist, p, **kw):
        # Newton–Raphson for normal
        z = 0.0
        for _ in range(60):
            z -= (_cdf(dist, z, **kw) - p) / (
                _pdf(dist, np.array([z]), **kw)[0] + 1e-14)
        return float(z)


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

PAL = {
    # Null distribution
    "dist_fill":      "#1B3A6A",
    "dist_fill_lite": "#2E6AB0",
    "dist_ridge":     "#7AB8F0",
    "dist_ao":        "#0A1828",

    # Rejection region
    "rej_fill":       "#6A1818",
    "rej_hatch":      "#A02020",
    "rej_border":     "#FF4040",

    # Critical line glow layers
    "crit_glow_1":    "#FF8080",   # brightest, innermost
    "crit_glow_2":    "#C04040",
    "crit_glow_3":    "#802020",
    "crit_label":     "#F0C0C0",
    "crit_badge_bg":  "#280808",

    # Observed statistic
    "obs_reject":     "#FF3030",   # reject H₀
    "obs_keep":       "#30A050",   # fail to reject
    "obs_glow_r":     "#FF8080",
    "obs_glow_k":     "#80D090",
    "obs_badge_bg":   "#0A1A0A",

    # p-value region
    "pval_fill":      "#5020A0",
    "pval_hatch":     "#8040C0",
    "pval_label":     "#C090FF",
    "pval_badge_bg":  "#180830",

    # Axis
    "axis":           "#8090A8",
    "axis_label":     "#B0C0D0",
    "tick":           "#607080",

    # Info panel
    "panel_bg":       "#0C1018",
    "panel_border":   "#2A3848",
    "panel_title":    "#D0E8FF",
    "panel_key":      "#8090A8",
    "panel_val":      "#C8D8E8",
    "panel_formula":  "#A0B8D0",

    # Decision badge
    "reject_bg":      "#3A0808",
    "reject_fg":      "#FF5050",
    "keep_bg":        "#081830",
    "keep_fg":        "#5090E0",

    # General
    "bg":             "#080C12",
    "white":          "#E8F0F8",
}


# ─────────────────────────────────────────────────────────────────────────────
# Distribution x-range configuration per test type
# ─────────────────────────────────────────────────────────────────────────────

_DIST_X_RANGE = {
    "norm":  (-4.2,  4.2),
    "t":     (-4.5,  4.5),
    "chi2":  (0.0,   None),   # right-only; max depends on df
    "f":     (0.0,   None),
}

_DIST_LABEL = {
    "norm": r"Z \sim \mathcal{N}(0,\,1)",
    "t":    r"t \sim t_{{df}}",          # {df} filled at runtime
    "chi2": r"\chi^2 \sim \chi^2_{{df}}",
    "f":    r"F \sim F_{{d_1,\,d_2}}",
}


# ─────────────────────────────────────────────────────────────────────────────
# Clipping helpers (shared with error_types)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_line(x0,y0,x1,y1,xmin,xmax,ymin,ymax):
    dx,dy = x1-x0, y1-y0
    p = [-dx, dx, -dy, dy]
    q = [x0-xmin, xmax-x0, y0-ymin, ymax-y0]
    t0,t1 = 0.0,1.0
    for pi,qi in zip(p,q):
        if abs(pi)<1e-10:
            if qi<0: return None
        else:
            t = qi/pi
            if pi<0: t0=max(t0,t)
            else:    t1=min(t1,t)
    if t0>t1: return None
    return (x0+t0*dx, y0+t0*dy), (x0+t1*dx, y0+t1*dy)


def _hatch(x_lo,x_hi,y_lo,y_hi,angle=PI/4,spacing=0.10,
           color=WHITE, sw=0.9, opacity=0.50, z=0.004):
    grp   = VGroup()
    ca,sa = np.cos(angle), np.sin(angle)
    perp  = np.array([-sa, ca])
    corners = np.array([[x_lo,y_lo],[x_hi,y_lo],[x_hi,y_hi],[x_lo,y_hi]])
    projs   = corners @ perp
    t = projs.min()
    while t <= projs.max() + spacing:
        ox,oy = perp[0]*t, perp[1]*t
        d = np.hypot(x_hi-x_lo, y_hi-y_lo)*1.5
        pts = _clip_line(ox-ca*d, oy-sa*d, ox+ca*d, oy+sa*d,
                         x_lo, x_hi, y_lo, y_hi)
        if pts:
            (ax,ay),(bx,by) = pts
            if abs(bx-ax)+abs(by-ay) > 1e-4:
                grp.add(Line([ax,ay,z],[bx,by,z],
                             stroke_color=color, stroke_width=sw,
                             stroke_opacity=opacity))
        t += spacing
    return grp


def _area_verts(x_vals, y_vals, x_lo, x_hi, baseline=0.0):
    mask = (x_vals >= x_lo) & (x_vals <= x_hi)
    xs, ys = x_vals[mask], y_vals[mask]
    if len(xs) < 2:
        return []
    def _iy(xv):
        i = np.argmin(np.abs(x_vals - xv))
        return float(y_vals[i])
    xs = np.concatenate([[x_lo], xs, [x_hi]])
    ys = np.concatenate([[_iy(x_lo)], ys, [_iy(x_hi)]])
    return [[float(x_lo), baseline, 0]] + \
           [[float(x), float(y), 0] for x,y in zip(xs,ys)] + \
           [[float(x_hi), baseline, 0]]


# ─────────────────────────────────────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────────────────────────────────────

class _DistributionCurve(VGroup):
    """
    The null-hypothesis test statistic distribution curve.

    Supports Z (norm), t, chi², and F distributions.
    Three visual layers: base fill, AO tail darkening, ridge spine.

    Parameters
    ----------
    dist     : "norm" | "t" | "chi2" | "f"
    x_vals   : Manim world x-coordinates (pre-computed)
    raw_x    : raw distribution x values
    y_vals   : pdf heights (already scaled to plot_height)
    baseline : y of x-axis
    z_base   : z-layer offset
    dist_kw  : df / dfn / dfd passed to _pdf
    """

    def __init__(
        self,
        dist:     str,
        x_vals:   np.ndarray,
        raw_x:    np.ndarray,
        y_vals:   np.ndarray,
        baseline: float = 0.0,
        z_base:   float = 0.001,
        dist_kw:  dict  = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        dist_kw = dist_kw or {}

        # ── 1. Base fill ──────────────────────────────────────────────
        verts = [[float(x_vals[0]), baseline, 0]] + \
                [[float(x), float(y), 0] for x,y in zip(x_vals, y_vals)] + \
                [[float(x_vals[-1]), baseline, 0]]
        if len(verts) >= 3:
            base = Polygon(
                *verts,
                fill_color=PAL["dist_fill"],
                fill_opacity=0.82,
                stroke_width=0,
            )
            base.shift([0, 0, z_base])
            self.add(base)

        # ── 2. AO tail darkening ──────────────────────────────────────
        ao_frac = 0.12
        x_span  = float(x_vals[-1] - x_vals[0])
        for x_edge, x_inner in [
            (x_vals[0],  x_vals[0]  + x_span * ao_frac),
            (x_vals[-1], x_vals[-1] - x_span * ao_frac),
        ]:
            x_lo = min(x_edge, x_inner)
            x_hi = max(x_edge, x_inner)
            av   = _area_verts(x_vals, y_vals, x_lo, x_hi, baseline)
            if len(av) >= 3:
                ao = Polygon(*av,
                             fill_color=PAL["dist_ao"],
                             fill_opacity=0.45, stroke_width=0)
                ao.shift([0, 0, z_base + 0.001])
                self.add(ao)

        # ── 3. Ridge spine ────────────────────────────────────────────
        step  = max(1, len(x_vals) // 100)
        pts3d = [np.array([float(x_vals[i]), float(y_vals[i]),
                           z_base + 0.003])
                 for i in range(0, len(x_vals), step)
                 if y_vals[i] > 1e-4]
        if len(pts3d) >= 2:
            spine = VMobject(stroke_color=PAL["dist_ridge"],
                             stroke_width=1.8, stroke_opacity=0.80)
            spine.set_points_smoothly(pts3d)
            self.add(spine)

        # ── 4. Distribution label badge ───────────────────────────────
        # Find the peak x position
        peak_i  = int(np.argmax(y_vals))
        peak_px = float(x_vals[peak_i])
        peak_py = float(y_vals[peak_i])

        dist_str = _DIST_LABEL.get(dist, r"H_0")
        if dist == "t" and "df" in dist_kw:
            dist_str = dist_str.replace("{df}", str(dist_kw["df"]))
        elif dist == "chi2" and "df" in dist_kw:
            dist_str = dist_str.replace("{df}", str(dist_kw["df"]))
        elif dist == "f" and "dfn" in dist_kw:
            d1, d2 = dist_kw.get("dfn",1), dist_kw.get("dfd",1)
            dist_str = dist_str.replace("{d_1,\\,d_2}", f"{d1},\\,{d2}")

        try:
            dlbl = MathTex(dist_str, font_size=22,
                           color=PAL["dist_ridge"])
        except Exception:
            dlbl = Text(dist, font_size=18, color=PAL["dist_ridge"])
        dlbl.move_to([peak_px, peak_py + 0.30, z_base + 0.005])
        self.add(dlbl)


class _RejectionRegion(VGroup):
    """
    Hatched + filled rejection tail region.

    Layers:
      1. Fill polygon
      2. Diagonal hatch lines
      3. Glowing critical-value boundary line (3 halos)
      4. "Rejection Region (α = …)" label

    Parameters
    ----------
    x_vals, raw_x, y_vals : distribution arrays
    x_crit_px  : Manim x of critical value
    x_crit_raw : raw critical value
    tail       : "right" | "left" | "both"
    alpha      : significance level
    x_lo_px, x_hi_px : full plot x bounds in Manim units
    baseline   : y of x-axis
    """

    def __init__(
        self,
        x_vals:    np.ndarray,
        raw_x:     np.ndarray,
        y_vals:    np.ndarray,
        x_crit_px: float,
        x_crit_raw: float,
        tail:      Literal["right","left","both"],
        alpha:     float,
        x_lo_px:   float,
        x_hi_px:   float,
        baseline:  float = 0.0,
        x_crit_lo_px:  float = None,   # left crit for two-tailed
        x_crit_lo_raw: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.004

        def _add_tail_region(xl_px, xr_px, hangle):
            av = _area_verts(x_vals, y_vals, xl_px, xr_px, baseline)
            if len(av) >= 3:
                fill = Polygon(*av,
                               fill_color=PAL["rej_fill"],
                               fill_opacity=0.65,
                               stroke_width=0)
                fill.shift([0, 0, z0])
                self.add(fill)

            # Hatch
            mask = (x_vals >= xl_px) & (x_vals <= xr_px)
            ymax = float(y_vals[mask].max()) if mask.sum() > 0 else 0.05
            self.add(_hatch(xl_px, xr_px, baseline, ymax * 1.02,
                            angle=hangle, spacing=0.09,
                            color=PAL["rej_hatch"], sw=1.0,
                            opacity=0.55, z=z0 + 0.002))

        def _add_glow_line(xp, side):
            """Three-halo glow around the critical line."""
            ymax = float(y_vals.max()) * 1.12
            for width, opacity in [(4.5, 0.18), (2.8, 0.35), (1.4, 0.80)]:
                gl = Line(
                    start=[xp, baseline - 0.05, z0 + 0.005],
                    end  =[xp, ymax,             z0 + 0.005],
                    stroke_color=PAL["rej_border"],
                    stroke_width=width,
                    stroke_opacity=opacity,
                )
                self.add(gl)

        def _add_crit_badge(xp, xv_raw):
            """Critical value tick + label badge at axis."""
            tick = Line([xp, baseline - 0.08, z0+0.006],
                        [xp, baseline + 0.08, z0+0.006],
                        stroke_color=PAL["crit_label"], stroke_width=2.0)
            self.add(tick)
            try:
                lbl = MathTex(f"z_c = {xv_raw:.3f}",
                              font_size=17, color=PAL["crit_label"])
            except Exception:
                lbl = Text(f"zc={xv_raw:.3f}", font_size=14,
                           color=PAL["crit_label"])
            bg = RoundedRectangle(
                width=lbl.width+0.18, height=lbl.height+0.12,
                corner_radius=0.05,
                fill_color=PAL["crit_badge_bg"], fill_opacity=0.90,
                stroke_color=PAL["rej_border"], stroke_width=0.7)
            by = baseline - 0.40
            bg.move_to([xp, by, z0+0.006])
            lbl.move_to([xp, by, z0+0.007])
            self.add(bg, lbl)

        # ── Right tail ────────────────────────────────────────────────
        if tail in ("right", "both"):
            _add_tail_region(x_crit_px, x_hi_px, PI/4)
            _add_glow_line(x_crit_px, "right")
            _add_crit_badge(x_crit_px, x_crit_raw)

            # Rejection label
            mid_px = (x_crit_px + x_hi_px) / 2
            ymax_r = float(y_vals[x_vals >= x_crit_px].max()
                           if (x_vals >= x_crit_px).any() else 0.1)
            rej_str = f"α{'/2' if tail=='both' else ''} = {alpha/2 if tail=='both' else alpha:.3f}"
            rlbl = Text(f"Rejection Region\n{rej_str}",
                        font_size=15, color=PAL["crit_label"])
            rlbl.move_to([mid_px, ymax_r * 0.55 + 0.28, z0+0.008])
            self.add(rlbl)

        # ── Left tail (two-tailed) ────────────────────────────────────
        if tail == "both" and x_crit_lo_px is not None:
            _add_tail_region(x_lo_px, x_crit_lo_px, -PI/4)
            _add_glow_line(x_crit_lo_px, "left")
            _add_crit_badge(x_crit_lo_px,
                            x_crit_lo_raw if x_crit_lo_raw else -x_crit_raw)
            # Left rejection label
            mid_lo = (x_lo_px + x_crit_lo_px) / 2
            ymax_l = float(y_vals[x_vals <= x_crit_lo_px].max()
                           if (x_vals <= x_crit_lo_px).any() else 0.1)
            llbl = Text(f"Rejection Region\nα/2 = {alpha/2:.3f}",
                        font_size=15, color=PAL["crit_label"])
            llbl.move_to([mid_lo, ymax_l * 0.55 + 0.28, z0+0.008])
            self.add(llbl)


class _ObservedStatLine(VGroup):
    """
    Vertical marker for the observed test statistic.

    Layers:
      1. Glow halo  (3 concentric lines, colour = decision)
      2. Main line  (solid, full height)
      3. Value badge at axis
      4. Arrow pointing into curve from above

    Parameters
    ----------
    x_px     : Manim x of the observed statistic
    x_raw    : raw value (for label)
    y_top    : top of the line
    baseline : y of axis
    rejects  : bool  — True = red, False = green
    stat_sym : str   — symbol for the statistic (e.g. "z", "t", "\\chi^2")
    """

    def __init__(
        self,
        x_px:    float,
        x_raw:   float,
        y_top:   float,
        baseline: float,
        rejects:  bool,
        stat_sym: str = "z",
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.012

        col      = PAL["obs_reject"] if rejects else PAL["obs_keep"]
        glow_col = PAL["obs_glow_r"] if rejects else PAL["obs_glow_k"]

        # ── Glow halo ─────────────────────────────────────────────────
        for sw, op in [(6.0, 0.12), (3.8, 0.28), (2.0, 0.65)]:
            gl = Line(
                start=[x_px, baseline,     z0],
                end  =[x_px, y_top + 0.10, z0],
                stroke_color=glow_col,
                stroke_width=sw,
                stroke_opacity=op,
            )
            self.add(gl)

        # ── Main line ─────────────────────────────────────────────────
        main_line = Line(
            start=[x_px, baseline,     z0 + 0.001],
            end  =[x_px, y_top + 0.10, z0 + 0.001],
            stroke_color=col,
            stroke_width=1.8,
        )
        self.add(main_line)

        # ── Value badge ───────────────────────────────────────────────
        try:
            vlbl = MathTex(f"{stat_sym}_{{\\mathrm{{obs}}}} = {x_raw:.3f}",
                           font_size=17, color=col)
        except Exception:
            vlbl = Text(f"{stat_sym}_obs = {x_raw:.3f}",
                        font_size=14, color=col)
        bg = RoundedRectangle(
            width=vlbl.width+0.20, height=vlbl.height+0.12,
            corner_radius=0.05,
            fill_color=PAL["obs_badge_bg"], fill_opacity=0.90,
            stroke_color=col, stroke_width=0.8)
        by = baseline - 0.72
        bg.move_to([x_px, by, z0+0.002])
        vlbl.move_to([x_px, by, z0+0.003])
        self.add(bg, vlbl)

        # ── Drop arrow (from above curve) ─────────────────────────────
        drop_arr = Arrow(
            start=[x_px, y_top + 0.55, z0 + 0.004],
            end  =[x_px, y_top + 0.14, z0 + 0.004],
            stroke_color=col, stroke_width=1.5,
            tip_length=0.13, buff=0,
        )
        self.add(drop_arr)

        self._x_px   = x_px
        self._y_top  = y_top
        self._col    = col


class _PValueRegion(VGroup):
    """
    The p-value shaded region: from the observed statistic to the tail.

    Uses a contrasting purple/teal fill distinct from the rejection region.
    Includes a bracket + "p = …" badge.

    Parameters
    ----------
    x_vals, raw_x, y_vals : arrays
    x_obs_px   : Manim x of observed stat
    x_tail_px  : Manim x of the outermost plot edge (tail direction)
    x_obs_raw  : raw observed value
    p_value    : computed p-value
    tail       : "right" | "left"
    baseline   : y of axis
    """

    def __init__(
        self,
        x_vals:  np.ndarray,
        raw_x:   np.ndarray,
        y_vals:  np.ndarray,
        x_obs_px:  float,
        x_tail_px: float,
        x_obs_raw: float,
        p_value:   float,
        tail:      Literal["right","left"],
        baseline:  float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.008

        xl = min(x_obs_px, x_tail_px)
        xr = max(x_obs_px, x_tail_px)

        # ── Fill ──────────────────────────────────────────────────────
        av = _area_verts(x_vals, y_vals, xl, xr, baseline)
        if len(av) >= 3:
            fill = Polygon(*av,
                           fill_color=PAL["pval_fill"],
                           fill_opacity=0.60, stroke_width=0)
            fill.shift([0, 0, z0])
            self.add(fill)

        # ── Hatch ─────────────────────────────────────────────────────
        mask = (x_vals >= xl) & (x_vals <= xr)
        ymax = float(y_vals[mask].max()) if mask.sum() > 0 else 0.05
        ang  = PI/4 if tail == "right" else -PI/4
        self.add(_hatch(xl, xr, baseline, ymax*1.02,
                        angle=ang, spacing=0.085,
                        color=PAL["pval_hatch"], sw=0.85,
                        opacity=0.50, z=z0+0.002))

        # ── p-value badge + bracket ───────────────────────────────────
        mid_px = (xl + xr) / 2
        mean_y = float(y_vals[mask].mean()) if mask.sum() > 0 else 0.05

        p_str  = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"
        try:
            plbl = MathTex(p_str, font_size=20, color=PAL["pval_label"])
        except Exception:
            plbl = Text(p_str, font_size=17, color=PAL["pval_label"])
        pbg = RoundedRectangle(
            width=plbl.width+0.22, height=plbl.height+0.14,
            corner_radius=0.06,
            fill_color=PAL["pval_badge_bg"], fill_opacity=0.92,
            stroke_color=PAL["pval_hatch"], stroke_width=0.9)
        label_y = mean_y * 0.55 + 0.50
        plbl.move_to([mid_px, label_y, z0+0.010])
        pbg.move_to([mid_px, label_y, z0+0.009])
        self.add(pbg, plbl)

        # Bracket leader from badge to region
        leader = Line(
            start=[mid_px, label_y - plbl.height/2 - 0.05, z0+0.009],
            end  =[mid_px, mean_y*0.30 + baseline + 0.04,  z0+0.009],
            stroke_color=PAL["pval_hatch"], stroke_width=1.2,
            stroke_opacity=0.70,
        )
        self.add(leader)


class _HypothesisLabel(VGroup):
    """
    Floating H₀ / H₁ hypothesis badge in the top-left corner.

    Displays:
      H₀: μ = μ₀ = 50.0
      H₁: μ > μ₀    (or ≠, or <)
    """

    def __init__(
        self,
        h0_str: str,
        h1_str: str,
        x:      float,
        y:      float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.025

        for i, (sym, s) in enumerate(
            [("H_0", h0_str), ("H_1", h1_str)]
        ):
            row = VGroup()
            try:
                sym_m = MathTex(sym + ":", font_size=22,
                                color=PAL["dist_ridge"])
                txt_m = MathTex(s,       font_size=20,
                                color=PAL["dist_ridge"])
            except Exception:
                sym_m = Text(sym.replace("_","") + ":",
                             font_size=18, color=PAL["dist_ridge"])
                txt_m = Text(s, font_size=16, color=PAL["dist_ridge"])
            txt_m.next_to(sym_m, RIGHT, buff=0.10)
            row.add(sym_m, txt_m)
            row.move_to([x + row.width/2, y - i * 0.42, z0])
            self.add(row)

        bg = RoundedRectangle(
            width=self.width + 0.30,
            height=self.height + 0.22,
            corner_radius=0.08,
            fill_color=PAL["panel_bg"],
            fill_opacity=0.85,
            stroke_color=PAL["panel_border"],
            stroke_width=0.8,
        )
        bg.move_to([x + bg.width/2 - 0.15,
                    y - 0.21, z0 - 0.001])
        self.add(bg)


class _InfoPanel(VGroup):
    """
    Left-sidebar info panel listing test details.

    Structure (top to bottom):
      ┌──────────────────────────────┐  ← coloured left border strip
      │ Test Name                    │
      │ ─────────────────────────    │
      │ H₀:  μ = 50.0               │
      │ H₁:  μ > 50.0               │
      │ ─────────────────────────    │
      │ Stat: Z = (x̄−μ₀)/(σ/√n)   │
      │ ─────────────────────────    │
      │ z_obs  =  2.310             │
      │ z*     =  1.645             │
      │ p      =  0.0104            │
      │ α      =  0.0500            │
      │ ─────────────────────────    │
      │ ✓ / ✗  REJECT / FAIL        │
      └──────────────────────────────┘

    Parameters
    ----------
    rows       : list of (key, value) tuples
    title      : panel title
    rejects    : bool  — colours the decision row and border
    panel_w, panel_h : dimensions
    """

    def __init__(
        self,
        rows:    list[tuple[str, str]],
        title:   str = "Hypothesis Test",
        rejects: bool = False,
        panel_w: float = 2.80,
        panel_h: float = 4.20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0     = 0.001
        border_col = PAL["obs_reject"] if rejects else PAL["obs_keep"]

        # ── Background ────────────────────────────────────────────────
        bg = RoundedRectangle(
            width=panel_w, height=panel_h,
            corner_radius=0.12,
            fill_color=PAL["panel_bg"],
            fill_opacity=0.94,
            stroke_color=PAL["panel_border"],
            stroke_width=0.8,
        )
        bg.move_to([0, 0, z0])
        self.add(bg)

        # ── Coloured left border strip ────────────────────────────────
        strip = RoundedRectangle(
            width=0.10, height=panel_h - 0.04,
            corner_radius=0.05,
            fill_color=border_col,
            fill_opacity=0.90,
            stroke_width=0,
        )
        strip.move_to([-panel_w/2 + 0.07, 0, z0 + 0.001])
        self.add(strip)

        # ── Title ─────────────────────────────────────────────────────
        title_mob = Text(title, font_size=16,
                         color=PAL["panel_title"],
                         font="sans-serif")
        title_mob.move_to([0, panel_h/2 - 0.28, z0 + 0.002])
        self.add(title_mob)

        # Divider
        self.add(Line([-panel_w/2+0.18, panel_h/2-0.50, z0+0.002],
                      [ panel_w/2-0.10, panel_h/2-0.50, z0+0.002],
                      stroke_color=PAL["panel_border"], stroke_width=0.7))

        # ── Rows ──────────────────────────────────────────────────────
        row_h    = (panel_h - 1.10) / max(len(rows), 1)
        y_start  = panel_h/2 - 0.75

        for i, (key, val) in enumerate(rows):
            y_row = y_start - i * row_h

            # Separator before decision row
            if key.startswith("Decision"):
                self.add(Line(
                    [-panel_w/2+0.18, y_row+row_h*0.55, z0+0.002],
                    [ panel_w/2-0.10, y_row+row_h*0.55, z0+0.002],
                    stroke_color=PAL["panel_border"], stroke_width=0.6))

            key_mob = Text(key, font_size=13, color=PAL["panel_key"])
            key_mob.move_to([-panel_w/2 + 0.30 + key_mob.width/2,
                              y_row, z0+0.003])
            val_col = (border_col if key.startswith("Decision")
                       else PAL["panel_val"])
            val_mob = Text(val, font_size=13, color=val_col,
                           font="monospace")
            val_mob.move_to([panel_w/2 - 0.12 - val_mob.width/2,
                             y_row, z0+0.003])
            self.add(key_mob, val_mob)

        self._rejects = rejects
        self._border_col = border_col


class _DecisionBadge(VGroup):
    """
    Large verdict badge: "REJECT H₀" or "FAIL TO REJECT H₀".

    Stamp-style: bold text on a semi-transparent rounded rect,
    rotated ±5° for visual punch.

    Parameters
    ----------
    rejects  : bool
    rotation : float  — stamp tilt in radians (default 0.09)
    """

    def __init__(
        self,
        rejects:  bool,
        rotation: float = 0.09,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0    = 0.030
        text  = "REJECT H₀" if rejects else "FAIL TO REJECT H₀"
        fg    = PAL["reject_fg"]   if rejects else PAL["keep_fg"]
        bg_c  = PAL["reject_bg"]   if rejects else PAL["keep_bg"]
        border= PAL["obs_reject"]  if rejects else PAL["obs_keep"]

        lbl = Text(text, font_size=38, color=fg,
                   font="sans-serif")
        lbl.move_to([0, 0, z0 + 0.001])

        bg = RoundedRectangle(
            width=lbl.width + 0.55,
            height=lbl.height + 0.35,
            corner_radius=0.14,
            fill_color=bg_c,
            fill_opacity=0.90,
            stroke_color=border,
            stroke_width=2.2,
        )
        bg.move_to([0, 0, z0])
        self.add(bg, lbl)
        self.rotate(rotation if rejects else -rotation)

        self._rejects = rejects


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisTest3D  ──  the main export
# ─────────────────────────────────────────────────────────────────────────────

class HypothesisTest3D(VGroup):
    """
    Full hypothesis test visualisation: null distribution, rejection
    region, observed statistic, p-value region, info panel, and
    decision badge.

    Parameters
    ----------
    dist         : "norm" | "t" | "chi2" | "f"
    obs_stat     : float   — observed test statistic value
    tail         : "right" | "left" | "both"
    alpha        : float   — significance level (default 0.05)
    dist_kw      : dict    — df / dfn / dfd for non-normal distributions
    plot_width   : float   — Manim units for the full plot width
    plot_height  : float   — maximum curve height
    baseline_y   : float   — y of x-axis
    test_name    : str     — shown in info panel title
    h0_str, h1_str : str   — LaTeX hypothesis strings
    stat_sym     : str     — LaTeX symbol for the statistic
    formula_str  : str     — formula shown in info panel
    show_panel   : bool    — show the info panel sidebar
    show_badge   : bool    — show the decision badge from the start

    Attributes
    ----------
    curve         : _DistributionCurve
    rej_region    : _RejectionRegion
    obs_line      : _ObservedStatLine   (initially hidden)
    pval_region   : _PValueRegion       (initially hidden)
    hyp_label     : _HypothesisLabel
    info_panel    : _InfoPanel | None
    decision_badge: _DecisionBadge      (initially hidden)
    rejects       : bool
    p_value       : float
    obs_stat      : float
    crit_val      : float

    Class methods
    -------------
    z_test(x_bar, mu0, sigma, n, alpha, tail, **kw)
    one_sample_t(x_bar, mu0, s, n, alpha, tail, **kw)
    two_sample_t(x1, x2, s1, s2, n1, n2, alpha, tail, **kw)
    chi_square(observed, expected, alpha, **kw)
    f_test(s1_sq, s2_sq, n1, n2, alpha, **kw)
    """

    def __init__(
        self,
        dist:        str   = "norm",
        obs_stat:    float = 2.0,
        tail:        Literal["right","left","both"] = "right",
        alpha:       float = 0.05,
        dist_kw:     dict  = None,
        plot_width:  float = 10.0,
        plot_height: float = 3.5,
        baseline_y:  float = 0.0,
        test_name:   str   = "Hypothesis Test",
        h0_str:      str   = r"\mu = \mu_0",
        h1_str:      str   = r"\mu > \mu_0",
        stat_sym:    str   = "z",
        formula_str: str   = r"Z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}",
        show_panel:  bool  = True,
        show_badge:  bool  = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        dist_kw    = dist_kw or {}
        self._dist     = dist
        self._dist_kw  = dist_kw
        self._obs_stat = obs_stat
        self._tail     = tail
        self._alpha    = alpha
        self._baseline = baseline_y
        self._pw       = plot_width
        self._ph       = plot_height

        # ── x-axis range for this distribution ───────────────────────
        x_lo_raw, x_hi_raw = self._get_x_range(dist, dist_kw, obs_stat)
        raw_x = np.linspace(x_lo_raw, x_hi_raw, 700)

        # Unit scale
        unit  = plot_width / (x_hi_raw - x_lo_raw)
        self._unit   = unit
        self._x0_raw = x_lo_raw

        def px(v): return (v - x_lo_raw) * unit - plot_width / 2
        self._px = px

        x_vals = (raw_x - x_lo_raw) * unit - plot_width / 2

        # PDF values, scaled to plot_height
        y_raw   = _pdf(dist, raw_x, **dist_kw)
        y_raw   = np.where(np.isfinite(y_raw), y_raw, 0.0)
        peak    = float(y_raw.max()) if y_raw.max() > 1e-12 else 1.0
        scale_y = plot_height / peak
        y_vals  = y_raw * scale_y

        # ── Critical value(s) ─────────────────────────────────────────
        if tail == "right":
            xc = _ppf(dist, 1 - alpha, **dist_kw)
            xc_lo = None
        elif tail == "left":
            xc = _ppf(dist, alpha, **dist_kw)
            xc_lo = None
        else:  # both
            xc    = _ppf(dist, 1 - alpha/2, **dist_kw)
            xc_lo = _ppf(dist, alpha/2, **dist_kw)

        self.crit_val = xc

        # ── p-value ───────────────────────────────────────────────────
        if tail == "right":
            p = 1 - _cdf(dist, obs_stat, **dist_kw)
        elif tail == "left":
            p = _cdf(dist, obs_stat, **dist_kw)
        else:
            p = 2 * (1 - _cdf(dist, abs(obs_stat), **dist_kw))
        p = float(np.clip(p, 0.0, 1.0))
        self.p_value = p

        # ── Decision ──────────────────────────────────────────────────
        self.rejects = p < alpha
        self.obs_stat = obs_stat

        # ── Axis line ─────────────────────────────────────────────────
        ax = Line(
            start=[-plot_width/2 - 0.3, baseline_y, 0],
            end  =[ plot_width/2 + 0.3, baseline_y, 0],
            stroke_color=PAL["axis"], stroke_width=1.8,
        )
        self.add(ax)
        # Arrow tip
        ax_arr = Arrow(
            start=[plot_width/2 + 0.1, baseline_y, 0],
            end  =[plot_width/2 + 0.40, baseline_y, 0],
            stroke_color=PAL["axis"], stroke_width=1.6,
            tip_length=0.18, buff=0,
        )
        self.add(ax_arr)

        # Axis ticks
        self._add_axis_ticks(x_lo_raw, x_hi_raw, px, baseline_y, dist)

        # ── Distribution curve ────────────────────────────────────────
        self.curve = _DistributionCurve(
            dist=dist, x_vals=x_vals, raw_x=raw_x,
            y_vals=y_vals, baseline=baseline_y,
            z_base=0.001, dist_kw=dist_kw,
        )
        self.add(self.curve)

        # ── Rejection region ──────────────────────────────────────────
        self.rej_region = _RejectionRegion(
            x_vals=x_vals, raw_x=raw_x, y_vals=y_vals,
            x_crit_px=px(xc), x_crit_raw=xc,
            tail=tail, alpha=alpha,
            x_lo_px=float(x_vals[0]),
            x_hi_px=float(x_vals[-1]),
            baseline=baseline_y,
            x_crit_lo_px=px(xc_lo) if xc_lo is not None else None,
            x_crit_lo_raw=xc_lo,
        )
        self.add(self.rej_region)

        # ── Hypothesis label ──────────────────────────────────────────
        self.hyp_label = _HypothesisLabel(
            h0_str=h0_str, h1_str=h1_str,
            x=-plot_width/2 + 0.10,
            y=plot_height * 0.90 + baseline_y,
        )
        self.add(self.hyp_label)

        # ── Observed statistic line (hidden until DropStatistic) ──────
        obs_px    = px(obs_stat)
        peak_here = float(y_vals[np.argmin(np.abs(x_vals - obs_px))])
        y_top     = max(peak_here, plot_height * 0.80)

        self.obs_line = _ObservedStatLine(
            x_px=obs_px, x_raw=obs_stat,
            y_top=y_top, baseline=baseline_y,
            rejects=self.rejects, stat_sym=stat_sym,
        )
        self.obs_line.set_opacity(0)
        self.add(self.obs_line)

        # ── p-value region (hidden until RevealPValue) ────────────────
        if tail == "right":
            tail_px = float(x_vals[-1])
            p_tail  = "right"
        elif tail == "left":
            tail_px = float(x_vals[0])
            p_tail  = "left"
        else:
            # Show right tail p-value region
            tail_px = float(x_vals[-1])
            p_tail  = "right"

        self.pval_region = _PValueRegion(
            x_vals=x_vals, raw_x=raw_x, y_vals=y_vals,
            x_obs_px=obs_px, x_tail_px=tail_px,
            x_obs_raw=obs_stat, p_value=p,
            tail=p_tail, baseline=baseline_y,
        )
        self.pval_region.set_opacity(0)
        self.add(self.pval_region)

        # ── Decision badge (hidden until RevealDecision) ──────────────
        self.decision_badge = _DecisionBadge(rejects=self.rejects)
        self.decision_badge.move_to(
            [0, baseline_y + plot_height * 0.50, 0.032]
        )
        self.decision_badge.set_opacity(0 if not show_badge else 1)
        self.add(self.decision_badge)

        # ── Info panel ────────────────────────────────────────────────
        self.info_panel = None
        if show_panel:
            p_str  = (f"{p:.4f}" if p >= 0.0001 else "< 0.0001")
            dec_str = ("✓ REJECT H₀" if self.rejects
                       else "✗ FAIL TO REJECT")
            rows = [
                ("H₀ :",     h0_str.replace("\\", "").replace("{","").replace("}","")),
                ("H₁ :",     h1_str.replace("\\", "").replace("{","").replace("}","")),
                ("Stat :",   formula_str.replace("\\","").replace("{","").replace("}","")),
                ("z_obs :",  f"{obs_stat:+.4f}"),
                ("z* :",     f"{xc:+.4f}"),
                ("p :",      p_str),
                ("α :",      f"{alpha:.4f}"),
                ("Decision", dec_str),
            ]
            self.info_panel = _InfoPanel(
                rows=rows,
                title=test_name,
                rejects=self.rejects,
                panel_w=2.90,
                panel_h=min(4.50, 0.52 * len(rows) + 1.0),
            )
            self.info_panel.move_to(
                [plot_width/2 + 1.75,
                 baseline_y + plot_height * 0.22,
                 0]
            )
            self.add(self.info_panel)

        # Store for rebuild
        self._x_vals  = x_vals
        self._y_vals  = y_vals
        self._raw_x   = raw_x
        self._xc      = xc
        self._xc_lo   = xc_lo
        self._x_lo_raw = x_lo_raw
        self._x_hi_raw = x_hi_raw

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_x_range(dist: str, kw: dict, obs: float) -> tuple:
        """Return (x_lo, x_hi) for the raw axis."""
        if dist == "norm":
            return (-4.5, 4.5)
        elif dist == "t":
            return (-5.0, 5.0)
        elif dist == "chi2":
            df = kw.get("df", 5)
            hi = max(df * 3.0, obs * 1.35, 25.0)
            return (0.0, hi)
        elif dist == "f":
            dfn = kw.get("dfn", 2)
            dfd = kw.get("dfd", 10)
            hi  = max(dfn * 4.0, obs * 1.35, 10.0)
            return (0.001, hi)
        return (-4.5, 4.5)

    def _add_axis_ticks(
        self,
        x_lo: float, x_hi: float,
        px, baseline_y: float, dist: str,
    ):
        """Add tick marks and labels to the x-axis."""
        # Determine nice tick spacing
        span = x_hi - x_lo
        step = 1.0
        if span > 20:  step = 5.0
        elif span > 10: step = 2.0
        elif span < 4:  step = 0.5

        val = np.ceil(x_lo / step) * step
        while val <= x_hi + 1e-9:
            xp = px(val)
            self.add(Line(
                start=[xp, baseline_y - 0.09, 0.001],
                end  =[xp, baseline_y + 0.09, 0.001],
                stroke_color=PAL["tick"], stroke_width=1.3,
            ))
            lbl_str = (f"{val:.0f}" if step >= 1 else f"{val:.1f}")
            lbl = Text(lbl_str, font_size=14, color=PAL["axis_label"])
            lbl.move_to([xp, baseline_y - 0.32, 0.001])
            self.add(lbl)
            val = round(val + step, 10)

    # ─────────────────────────────────────────────────────────────────
    # Class-method constructors
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def z_test(
        cls,
        x_bar: float, mu0: float,
        sigma: float, n: int,
        alpha: float = 0.05,
        tail: Literal["right","left","both"] = "right",
        **kwargs,
    ) -> "HypothesisTest3D":
        """
        One-sample Z-test (σ known).
        z = (x̄ − μ₀) / (σ / √n)
        """
        se      = sigma / np.sqrt(n)
        z_obs   = (x_bar - mu0) / se
        tail_sym = {"right":">", "left":"<", "both":"\\neq"}[tail]
        return cls(
            dist="norm", obs_stat=z_obs,
            tail=tail, alpha=alpha,
            test_name="One-Sample Z-Test",
            h0_str=rf"\mu = {mu0:.2f}",
            h1_str=rf"\mu {tail_sym} {mu0:.2f}",
            stat_sym="z",
            formula_str=r"Z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}",
            **kwargs,
        )

    @classmethod
    def one_sample_t(
        cls,
        x_bar: float, mu0: float,
        s:     float, n:   int,
        alpha: float = 0.05,
        tail: Literal["right","left","both"] = "right",
        **kwargs,
    ) -> "HypothesisTest3D":
        """One-sample t-test (σ unknown)."""
        se      = s / np.sqrt(n)
        t_obs   = (x_bar - mu0) / se
        df      = n - 1
        tail_sym = {"right":">", "left":"<", "both":"\\neq"}[tail]
        return cls(
            dist="t", obs_stat=t_obs,
            dist_kw={"df": df},
            tail=tail, alpha=alpha,
            test_name=f"One-Sample t-Test (df={df})",
            h0_str=rf"\mu = {mu0:.2f}",
            h1_str=rf"\mu {tail_sym} {mu0:.2f}",
            stat_sym="t",
            formula_str=r"t = \frac{\bar{x}-\mu_0}{s/\sqrt{n}}",
            **kwargs,
        )

    @classmethod
    def two_sample_t(
        cls,
        x1: float, x2: float,
        s1: float, s2: float,
        n1: int,   n2: int,
        alpha: float = 0.05,
        tail: Literal["right","left","both"] = "both",
        equal_var: bool = False,
        **kwargs,
    ) -> "HypothesisTest3D":
        """Two-sample Welch t-test (unequal variances by default)."""
        if equal_var:
            sp  = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            se  = sp * np.sqrt(1/n1 + 1/n2)
            df  = n1 + n2 - 2
        else:
            se_sq = s1**2/n1 + s2**2/n2
            se    = np.sqrt(se_sq)
            df    = int((se_sq**2) / (
                (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
            ))
        t_obs = (x1 - x2) / se
        tail_sym = {"right":">", "left":"<", "both":"\\neq"}[tail]
        return cls(
            dist="t", obs_stat=t_obs,
            dist_kw={"df": df},
            tail=tail, alpha=alpha,
            test_name=f"Two-Sample t-Test (df={df})",
            h0_str=r"\mu_1 = \mu_2",
            h1_str=rf"\mu_1 {tail_sym} \mu_2",
            stat_sym="t",
            formula_str=r"t = \frac{\bar{x}_1-\bar{x}_2}{s_p\sqrt{1/n_1+1/n_2}}",
            **kwargs,
        )

    @classmethod
    def chi_square(
        cls,
        observed: Sequence[float],
        expected: Sequence[float],
        alpha: float = 0.05,
        **kwargs,
    ) -> "HypothesisTest3D":
        """Chi-square goodness-of-fit test."""
        O    = np.array(observed, dtype=float)
        E    = np.array(expected, dtype=float)
        chi2 = float(np.sum((O - E)**2 / E))
        df   = len(O) - 1
        return cls(
            dist="chi2", obs_stat=chi2,
            dist_kw={"df": df},
            tail="right", alpha=alpha,
            test_name=f"Chi-Square GoF (df={df})",
            h0_str=r"O_i = E_i \;\forall i",
            h1_str=r"O_i \neq E_i \;\text{for some } i",
            stat_sym=r"\chi^2",
            formula_str=r"\chi^2 = \sum\frac{(O_i-E_i)^2}{E_i}",
            **kwargs,
        )

    @classmethod
    def f_test(
        cls,
        s1_sq: float, s2_sq: float,
        n1: int,      n2: int,
        alpha: float = 0.05,
        **kwargs,
    ) -> "HypothesisTest3D":
        """F-test for equality of variances."""
        F    = s1_sq / s2_sq
        dfn  = n1 - 1
        dfd  = n2 - 1
        return cls(
            dist="f", obs_stat=F,
            dist_kw={"dfn": dfn, "dfd": dfd},
            tail="right", alpha=alpha,
            test_name=f"F-Test (df₁={dfn}, df₂={dfd})",
            h0_str=r"\sigma_1^2 = \sigma_2^2",
            h1_str=r"\sigma_1^2 \neq \sigma_2^2",
            stat_sym="F",
            formula_str=r"F = \frac{s_1^2}{s_2^2}",
            **kwargs,
        )

    # ─────────────────────────────────────────────────────────────────
    # Rebuild helper
    # ─────────────────────────────────────────────────────────────────

    def rebuild(
        self,
        obs_stat: Optional[float] = None,
        alpha:    Optional[float] = None,
    ) -> "HypothesisTest3D":
        """Return a new HypothesisTest3D with updated parameters."""
        return HypothesisTest3D(
            dist=self._dist,
            obs_stat=obs_stat if obs_stat is not None else self._obs_stat,
            tail=self._tail,
            alpha=alpha if alpha is not None else self._alpha,
            dist_kw=self._dist_kw,
            plot_width=self._pw,
            plot_height=self._ph,
            baseline_y=self._baseline,
            show_panel=self.info_panel is not None,
            show_badge=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Animations
# ─────────────────────────────────────────────────────────────────────────────

class BuildTest(Animation):
    """
    Axis appears → curve fills upward from baseline →
    rejection region fades in → hypothesis label fades in.

    α ∈ [0.00, 0.50] — curve grows from baseline
    α ∈ [0.45, 0.80] — rejection region fades in
    α ∈ [0.75, 1.00] — hypothesis label fades in
    """

    def __init__(self, ht: HypothesisTest3D, **kwargs):
        self.ht = ht
        kwargs.setdefault("run_time", 2.4)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(ht, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())

        # Phase 1: curve grows upward
        curve_a = min(1.0, alpha / 0.50)
        scale   = max(curve_a, 1e-5)
        self.ht.curve.scale(
            [1, scale, 1],
            about_point=[
                self.ht.curve.get_center()[0],
                self.ht._baseline, 0
            ],
        )

        # Phase 2: rejection region
        rej_a = np.clip((alpha - 0.45) / 0.35, 0, 1)
        self.ht.rej_region.set_opacity(rej_a)

        # Phase 3: hypothesis label
        hyp_a = np.clip((alpha - 0.75) / 0.25, 0, 1)
        self.ht.hyp_label.set_opacity(hyp_a)

        # Info panel fades throughout
        if self.ht.info_panel is not None:
            self.ht.info_panel.set_opacity(
                np.clip((alpha - 0.50) / 0.50, 0, 1)
            )


class DropStatistic(Animation):
    """
    The observed-statistic line falls from above the curve
    to its final position (ease-in).

    The line starts at y = baseline + plot_height * 1.8 and
    falls to its correct position.

    α ∈ [0.00, 0.85] — line falls into position
    α ∈ [0.80, 1.00] — value badge fades in
    """

    def __init__(self, ht: HypothesisTest3D, **kwargs):
        self.ht      = ht
        self._start_y_off = ht._ph * 1.8   # start above the curve
        kwargs.setdefault("run_time", 1.4)
        kwargs.setdefault("rate_func", rate_functions.ease_in_cubic)
        super().__init__(ht.obs_line, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(1.0)

        # Vertical drop: from start_y_off above to 0 offset
        drop_a  = min(1.0, alpha / 0.85)
        y_off   = self._start_y_off * (1 - drop_a)
        self.mobject.shift([0, y_off, 0])

        # Badge opacity
        badge_a = np.clip((alpha - 0.80) / 0.20, 0, 1)
        # Badge is the last two submobjects (bg + text)
        for sub in list(self.mobject.submobjects)[-2:]:
            sub.set_opacity(badge_a)


class RevealPValue(Animation):
    """
    p-value region fills in from the observed statistic outward
    toward the tail (left→right or right→left depending on tail).

    α ∈ [0.00, 0.70] — region expands from stat line outward
    α ∈ [0.65, 1.00] — p-value badge fades in
    """

    def __init__(self, ht: HypothesisTest3D, **kwargs):
        self.ht = ht
        kwargs.setdefault("run_time", 1.2)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(ht.pval_region, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())

        # Expand opacity
        fill_a  = min(1.0, alpha / 0.70)
        badge_a = np.clip((alpha - 0.65) / 0.35, 0, 1)

        self.mobject.set_opacity(fill_a)
        # Badge is last submobjects
        for sub in list(self.mobject.submobjects)[-2:]:
            sub.set_opacity(badge_a)


class RevealDecision(Animation):
    """
    Decision badge stamps in with a rotation wobble.

    Starts scaled to 0, rotated ±15°, fades/grows to full size
    with a slight bounce overshoot.

    Parameters
    ----------
    ht       : HypothesisTest3D
    run_time : float
    """

    def __init__(self, ht: HypothesisTest3D, **kwargs):
        self.ht = ht
        kwargs.setdefault("run_time", 0.90)
        kwargs.setdefault("rate_func", rate_functions.ease_out_back)
        super().__init__(ht.decision_badge, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        scale   = alpha
        tilt    = (1 - alpha) * 0.26 * (1 if self.ht.rejects else -1)
        self.mobject.scale(
            max(scale, 1e-5),
            about_point=self.ht.decision_badge.get_center(),
        )
        self.mobject.rotate(tilt, about_point=self.mobject.get_center())
        self.mobject.set_opacity(alpha)


class SweepStatistic(Animation):
    """
    Slide the observed-statistic line to a new value,
    morphing the p-value region and decision badge.

    Parameters
    ----------
    ht          : HypothesisTest3D
    new_obs     : float   — new observed statistic value
    run_time    : float
    """

    def __init__(
        self,
        ht: HypothesisTest3D,
        new_obs: float,
        **kwargs,
    ):
        self.ht      = ht
        self.new_obs = new_obs
        self.target  = ht.rebuild(obs_stat=new_obs)
        self.target.move_to(ht.get_center())
        kwargs.setdefault("run_time", 1.6)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(ht, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(1 - alpha * 0.4)
        self.target.set_opacity(alpha * 0.4 + (0.6 if alpha > 0.5 else 0))

        if alpha >= 1.0:
            self.mobject.become(self.target)
            self.ht._obs_stat = self.new_obs


class ChangeAlpha(Animation):
    """
    Move the critical value by changing α — rejection region resizes,
    critical line slides, p-value comparison updates.

    Parameters
    ----------
    ht        : HypothesisTest3D
    new_alpha : float
    run_time  : float
    """

    def __init__(
        self,
        ht: HypothesisTest3D,
        new_alpha: float,
        **kwargs,
    ):
        self.ht        = ht
        self.new_alpha = new_alpha
        self.target    = ht.rebuild(alpha=new_alpha)
        self.target.move_to(ht.get_center())
        kwargs.setdefault("run_time", 1.8)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_sine)
        super().__init__(ht, **kwargs)

    def interpolate_mobject(self, alpha_t: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(1 - alpha_t)
        self.target.set_opacity(alpha_t)
        if alpha_t >= 1.0:
            self.mobject.become(self.target)
            self.ht._alpha = self.new_alpha


class BuildInfoPanel(Succession):
    """
    Animate the info panel lines appearing one by one (typewriter effect).

    Each row fades in sequentially.

    Parameters
    ----------
    ht        : HypothesisTest3D
    per_row_rt: float — time per row
    """

    def __init__(
        self,
        ht: HypothesisTest3D,
        per_row_rt: float = 0.18,
        **kwargs,
    ):
        if ht.info_panel is None:
            super().__init__(FadeIn(VGroup(), run_time=0.1), **kwargs)
            return

        panel = ht.info_panel
        # Hide all rows initially
        rows = panel.submobjects[3:]   # skip bg, strip, title
        for r in rows:
            r.set_opacity(0)

        anims = [
            FadeIn(panel.submobjects[0], run_time=0.3),   # bg
            FadeIn(panel.submobjects[1], run_time=0.2),   # strip
            FadeIn(panel.submobjects[2], run_time=0.2),   # title
        ]
        for r in rows:
            anims.append(FadeIn(r, shift=RIGHT*0.08,
                                run_time=per_row_rt))

        kwargs.setdefault("run_time",
                          sum(a.run_time for a in anims))
        super().__init__(*anims, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factories
# ─────────────────────────────────────────────────────────────────────────────

def compare_tests(
    ht1: HypothesisTest3D,
    ht2: HypothesisTest3D,
    spacing: float = 0.5,
    scale: float = 0.72,
) -> VGroup:
    """
    Place two HypothesisTest3D objects side by side for comparison.

    Useful for showing e.g. one-tailed vs two-tailed, or
    different sample sizes.

    Parameters
    ----------
    ht1, ht2 : the two tests
    spacing  : gap between them
    scale    : scale factor applied to each

    Returns
    -------
    VGroup centred at origin.
    """
    ht1.scale(scale)
    ht2.scale(scale)
    total_w = ht1.width + ht2.width + spacing
    ht1.move_to([-total_w/2 + ht1.width/2, 0, 0])
    ht2.move_to([ total_w/2 - ht2.width/2, 0, 0])
    grp = VGroup(ht1, ht2)
    grp.center()
    return grp


def make_full_sequence(
    ht: HypothesisTest3D,
) -> Succession:
    """
    Full animation sequence for a single hypothesis test:
    BuildTest → DropStatistic → RevealPValue → BuildInfoPanel →
    RevealDecision.

    Returns a Succession ready to pass to scene.play().
    """
    return Succession(
        BuildTest(ht,          run_time=2.2),
        DropStatistic(ht,      run_time=1.3),
        RevealPValue(ht,       run_time=1.1),
        BuildInfoPanel(ht,     per_row_rt=0.16),
        RevealDecision(ht,     run_time=0.85),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo Scene  (run with: manim -pql hypothesis.py HypothesisDemo)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from manim import Scene

    class HypothesisDemo(Scene):
        """Showcase of HypothesisTest3D for all test types."""

        def construct(self):

            # ── 1. One-sample Z-test (right-tailed, rejects) ──────────
            zt = HypothesisTest3D.z_test(
                x_bar=52.3, mu0=50.0, sigma=8.0, n=25,
                alpha=0.05, tail="right",
                plot_width=9.0, plot_height=3.2,
                baseline_y=-1.8,
                show_panel=True,
            )
            zt.center()
            self.play(make_full_sequence(zt))
            self.wait(0.8)

            # ── Change α: watch critical line slide ───────────────────
            self.play(ChangeAlpha(zt, new_alpha=0.01, run_time=1.8))
            self.wait(0.5)
            self.play(ChangeAlpha(zt, new_alpha=0.10, run_time=1.8))
            self.wait(0.5)

            # ── Sweep the statistic ───────────────────────────────────
            self.play(SweepStatistic(zt, new_obs=1.2, run_time=1.6))
            self.wait(0.4)
            self.play(SweepStatistic(zt, new_obs=2.8, run_time=1.6))
            self.wait(0.8)
            self.play(FadeOut(zt))

            # ── 2. One-sample t-test (two-tailed) ─────────────────────
            tt = HypothesisTest3D.one_sample_t(
                x_bar=54.0, mu0=50.0, s=9.5, n=12,
                alpha=0.05, tail="both",
                plot_width=9.0, plot_height=3.2,
                baseline_y=-1.8,
                show_panel=True,
            )
            tt.center()
            self.play(make_full_sequence(tt))
            self.wait(0.8)
            self.play(FadeOut(tt))

            # ── 3. Chi-square goodness-of-fit ─────────────────────────
            cst = HypothesisTest3D.chi_square(
                observed=[18, 22, 30, 14, 16],
                expected=[20, 20, 20, 20, 20],
                alpha=0.05,
                plot_width=9.0, plot_height=3.2,
                baseline_y=-1.8,
                show_panel=True,
            )
            cst.center()
            self.play(make_full_sequence(cst))
            self.wait(0.8)
            self.play(FadeOut(cst))

            # ── 4. Two-sample t-test comparison (one-tailed vs two) ───
            tt1 = HypothesisTest3D.two_sample_t(
                x1=55.0, x2=50.0,
                s1=8.0,  s2=9.0,
                n1=20,   n2=18,
                alpha=0.05, tail="right",
                plot_width=8.0, plot_height=2.8,
                baseline_y=-1.5, show_panel=False,
            )
            tt2 = HypothesisTest3D.two_sample_t(
                x1=55.0, x2=50.0,
                s1=8.0,  s2=9.0,
                n1=20,   n2=18,
                alpha=0.05, tail="both",
                plot_width=8.0, plot_height=2.8,
                baseline_y=-1.5, show_panel=False,
            )
            side_by_side = compare_tests(tt1, tt2, spacing=0.4, scale=0.68)
            side_by_side.center()
            self.play(
                AnimationGroup(
                    BuildTest(tt1, run_time=1.8),
                    BuildTest(tt2, run_time=1.8),
                )
            )
            self.play(
                AnimationGroup(
                    DropStatistic(tt1, run_time=1.1),
                    DropStatistic(tt2, run_time=1.1),
                )
            )
            self.play(
                AnimationGroup(
                    RevealPValue(tt1, run_time=0.9),
                    RevealPValue(tt2, run_time=0.9),
                )
            )
            self.play(
                AnimationGroup(
                    RevealDecision(tt1, run_time=0.8),
                    RevealDecision(tt2, run_time=0.8),
                )
            )
            self.wait(2.0)

except ImportError:
    pass