"""
manim_stats/inference/error_types.py
=====================================
TypeITypeII — Highly detailed, statistically rigorous visualisation of
Type I error (α), Type II error (β), and Power (1−β) for Manim
statistics animations.

Primary use cases
-----------------
  Hypothesis testing foundations  — visual definition of α and β
  Power analysis                  — how n, δ, σ, α trade off
  Effect size demonstrations      — shifting H₁ changes β
  Two-tailed tests                — symmetric critical regions
  Neyman–Pearson framework        — decision rules and error types

Design goals
------------
Distribution curves
  * Two filled normal curves: H₀ (null, blue family) and H₁ (alternative,
    red/orange family), each rendered with three layers:
      – base fill polygon      — the full curve area
      – bright ridge spine     — thin bright line along the curve peak
      – tail AO darkening      — darker gradient strips at far tails
  * Both curves share the same x-axis and are drawn at the same scale.
  * Curve width (σ_eff = σ/√n) is controlled by a ValueTracker for n,
    enabling live NarrowCurves animation.

Critical region
  * Vertical dashed line at the critical value xₒ.
  * Hatched rejection region: parallel diagonal lines clipped to the
    right tail of H₀ (or both tails for two-tailed), giving a "forbidden
    zone" feel distinct from the error shading.
  * Critical value badge: rounded-rect label at the base of the line.

Error / power regions (all with distinct hatch + fill)
  * α region  — area under H₀ in the rejection zone.
                Bright red fill + dark red diagonal hatch (↗).
  * β region  — area under H₁ in the acceptance zone.
                Gold/orange fill + orange hatch (↘).
  * 1−β (Power) region — area under H₁ in the rejection zone.
                Green fill + green hatch (↗), same direction as α.
  * 1−α region — area under H₀ in the acceptance zone.
                Blue fill + blue hatch (↘).

Annotation system
  * Each region has:
      – region_label  (e.g. "α = 0.050")  floating above the region
      – bracket line  with a fine leader arrow pointing into the region
      – formula badge (e.g. "P(reject H₀ | H₀ true)")

Effect size arrow
  * DoubleArrow between the two distribution means (μ₀ → μ₁).
  * Label "δ" above the arrow, and computed value below.

Decision table panel
  * 2×2 grid floating to the right:
      Rows = Reality (H₀ true / H₁ true)
      Cols = Decision (Fail to Reject / Reject H₀)
      Cells = 1−α, α, β, 1−β with live numeric values.

Two-tailed support
  * Symmetric critical regions, two α/2 annotations, correct β.

Animations
----------
  BuildDistributions — curves grow from flat line upward (both together)
  RevealAlpha        — α region fills and label appears
  RevealBeta         — β region fills and label appears
  RevealPower        — 1−β region fills and label appears
  RevealAll          — orchestrated sequence of all four reveals
  ShiftH1            — moves the H₁ distribution (changes δ), β shrinks/grows
  NarrowCurves       — increases n, both curves narrow, β shrinks
  SweepAlpha         — moves the critical value, α grows/shrinks, β inversely
  FlashDecision      — highlights one cell of the decision table
  BuildDecisionTable — animates the 2×2 table appearing

Dependencies
------------
  manim (CE or GL), numpy, scipy

Usage
-----
    from manim_stats.inference.error_types import (
        TypeITypeII, BuildDistributions, RevealAll, NarrowCurves
    )

    class PowerScene(Scene):
        def construct(self):
            viz = TypeITypeII(
                mu0=0.0, mu1=2.0, sigma=1.0, n=1,
                alpha=0.05, two_tailed=False,
            )
            self.play(BuildDistributions(viz))
            self.play(RevealAll(viz))
            self.play(NarrowCurves(viz, new_n=16, run_time=3.0))
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Literal, List, Tuple

from manim import (
    VGroup,
    Rectangle, RoundedRectangle, Square,
    Circle, Annulus, Polygon, Line, DashedLine, Arrow, DoubleArrow,
    Text, MathTex, Tex,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut, GrowFromCenter,
    Create, Write, Uncreate,
    Indicate, Flash,
    ValueTracker,
    interpolate_color, color_to_rgb,
    always_redraw,
    WHITE, BLACK,
    GREY,   GREY_A,   GREY_B,   GREY_C,   GREY_D,
    RED,    RED_A,    RED_B,    RED_C,    RED_D,    RED_E,
    GREEN,  GREEN_A,  GREEN_B,  GREEN_C,  GREEN_D,  GREEN_E,
    BLUE,   BLUE_A,   BLUE_B,   BLUE_C,   BLUE_D,   BLUE_E,
    YELLOW, YELLOW_A, YELLOW_B, YELLOW_E,
    ORANGE, TEAL, TEAL_A, TEAL_B,
    GOLD,   GOLD_A,   GOLD_B,   GOLD_C,   GOLD_D,
    MAROON, PURPLE_A, PURPLE_B, PINK,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    ManimColor,
)

# ─────────────────────────────────────────────────────────────────────────────
# scipy — graceful fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.stats import norm as _norm
    def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return _norm.pdf(x, loc=mu, scale=sigma)
    def _norm_cdf(x: float, mu: float, sigma: float) -> float:
        return float(_norm.cdf(x, loc=mu, scale=sigma))
    def _norm_ppf(p: float, mu: float, sigma: float) -> float:
        return float(_norm.ppf(p, loc=mu, scale=sigma))
except ImportError:
    def _norm_pdf(x, mu, sigma):
        return (np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                / (sigma * np.sqrt(2 * PI)))
    def _norm_cdf(x, mu, sigma):
        return float(0.5 * (1 + np.vectorize(
            lambda t: 2 / np.sqrt(PI) * sum(
                (-1)**k * t**(2*k+1) / (np.math.factorial(k) * (2*k+1))
                for k in range(50)
            )
        )((x - mu) / (sigma * np.sqrt(2)))))
    def _norm_ppf(p, mu, sigma):
        # rough inverse via Newton–Raphson
        x = mu
        for _ in range(50):
            x -= (_norm_cdf(x, mu, sigma) - p) / (_norm_pdf(np.array([x]), mu, sigma)[0] + 1e-12)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    # H₀ distribution
    "h0_fill":      "#1B4F8A",
    "h0_fill_lite": "#4A90D9",
    "h0_ridge":     "#90C8F8",
    "h0_tail_ao":   "#0D2040",

    # H₁ distribution
    "h1_fill":      "#8A2B1B",
    "h1_fill_lite": "#D9604A",
    "h1_ridge":     "#F8B090",
    "h1_tail_ao":   "#401008",

    # Region fills
    "alpha_fill":   "#CC2222",      # α: Type I — vivid red
    "alpha_hatch":  "#8A0808",
    "beta_fill":    "#C87820",      # β: Type II — gold/amber
    "beta_hatch":   "#7A4A08",
    "power_fill":   "#1A9A50",      # 1−β: Power — green
    "power_hatch":  "#0A5828",
    "correct_fill": "#1B4F8A",      # 1−α: correct rejection — blue
    "correct_hatch":"#0D2040",

    # Critical line
    "crit_line":    "#F0E8C0",
    "crit_badge_bg":"#2A2010",
    "crit_badge_fg":"#F0E8C0",

    # Effect arrow
    "delta_arrow":  "#D4AF37",
    "delta_label":  "#F0E8C0",

    # Annotations
    "label_alpha":  "#FF6060",
    "label_beta":   "#F0A030",
    "label_power":  "#40CC80",
    "label_correct":"#70B0F0",

    # Axis / general
    "axis":         "#A0A8B0",
    "axis_label":   "#C8D0D8",
    "bg_panel":     "#0E1420",
    "table_border": "#3A4050",
    "table_header": "#1E2840",
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _curve_polygon_verts(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    baseline_y: float = 0.0,
) -> list[list[float]]:
    """
    Vertices for a filled curve polygon:
    left baseline → curve top → right baseline → close.
    """
    top   = [[float(x), float(y), 0] for x, y in zip(x_vals, y_vals)]
    left  = [[float(x_vals[0]),  baseline_y, 0]]
    right = [[float(x_vals[-1]), baseline_y, 0]]
    return left + top + right


def _clipped_area_verts(
    x_vals:  np.ndarray,
    y_vals:  np.ndarray,
    x_lo:    float,
    x_hi:    float,
    baseline_y: float = 0.0,
) -> list[list[float]]:
    """
    Vertices for the area under a curve between x_lo and x_hi.
    Returns empty list if region is negligibly small.
    """
    mask = (x_vals >= x_lo) & (x_vals <= x_hi)
    xs   = x_vals[mask]
    ys   = y_vals[mask]
    if len(xs) < 2:
        return []

    # Interpolate boundary points precisely
    def _interp_y(x_target: float) -> float:
        idx = np.searchsorted(x_vals, x_target)
        idx = np.clip(idx, 1, len(x_vals) - 1)
        x0, x1 = x_vals[idx-1], x_vals[idx]
        y0, y1 = y_vals[idx-1], y_vals[idx]
        t = (x_target - x0) / (x1 - x0 + 1e-12)
        return float(y0 + t * (y1 - y0))

    xs = np.concatenate([[x_lo], xs, [x_hi]])
    ys = np.concatenate(
        [[_interp_y(x_lo)], ys, [_interp_y(x_hi)]]
    )

    top   = [[float(x), float(y), 0] for x, y in zip(xs, ys)]
    left  = [[float(x_lo), baseline_y, 0]]
    right = [[float(x_hi), baseline_y, 0]]
    return left + top + right


def _hatch_lines(
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    angle: float = PI / 4,
    spacing: float = 0.12,
    stroke_color: str = WHITE,
    stroke_width: float = 0.8,
    stroke_opacity: float = 0.55,
    z: float = 0.004,
) -> VGroup:
    """
    Diagonal hatch lines inside an axis-aligned bounding box,
    clipped to [x_lo, x_hi] × [y_lo, y_hi].

    Each line is drawn at `angle` to horizontal, spaced `spacing` apart
    (measured perpendicular to the lines).
    """
    grp   = VGroup()
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # Perpendicular direction
    perp  = np.array([-sin_a, cos_a])
    # Project bounding box corners onto perp axis
    corners = np.array([
        [x_lo, y_lo], [x_hi, y_lo],
        [x_hi, y_hi], [x_lo, y_hi],
    ])
    projs   = corners @ perp
    p_min, p_max = projs.min(), projs.max()

    t = p_min
    while t <= p_max + spacing:
        # Line origin (point on the perpendicular axis)
        ox = perp[0] * t
        oy = perp[1] * t
        # Extend along the line direction far enough to cross the box
        diag = np.sqrt((x_hi - x_lo)**2 + (y_hi - y_lo)**2) * 1.5
        sx = ox - cos_a * diag
        sy = oy - sin_a * diag
        ex = ox + cos_a * diag
        ey = oy + sin_a * diag
        # Clip to bounding box using Cohen–Sutherland (simplified)
        pts  = _clip_line_to_box(sx, sy, ex, ey, x_lo, x_hi, y_lo, y_hi)
        if pts is not None:
            (cx0, cy0), (cx1, cy1) = pts
            if abs(cx1 - cx0) > 1e-4 or abs(cy1 - cy0) > 1e-4:
                grp.add(Line(
                    start=[cx0, cy0, z],
                    end  =[cx1, cy1, z],
                    stroke_color=stroke_color,
                    stroke_width=stroke_width,
                    stroke_opacity=stroke_opacity,
                ))
        t += spacing
    return grp


def _clip_line_to_box(
    x0: float, y0: float,
    x1: float, y1: float,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
) -> Optional[tuple]:
    """Liang–Barsky line clipping. Returns ((x0,y0),(x1,y1)) or None."""
    dx, dy = x1 - x0, y1 - y0
    p = [-dx,  dx, -dy,  dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-10:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return (
        (x0 + t0 * dx, y0 + t0 * dy),
        (x0 + t1 * dx, y0 + t1 * dy),
    )


def _world_to_px(val: float, x_center: float, unit: float) -> float:
    """Convert an x-axis value to Manim world x-coordinate."""
    return (val - x_center) * unit


# ─────────────────────────────────────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────────────────────────────────────

class _NormalCurve(VGroup):
    """
    A single filled normal distribution curve.

    Layers (back → front):
      1. base_poly    — full curve fill polygon
      2. ao_left      — dark AO strip at the left tail
      3. ao_right     — dark AO strip at the right tail
      4. ridge_spine  — bright 1px line along the curve top
      5. peak_cap     — small bright circle at the distribution peak

    Parameters
    ----------
    mu, sigma    : distribution parameters
    x_vals       : pre-computed x array (Manim world units)
    raw_x        : raw axis values corresponding to x_vals
    scale_y      : vertical scaling factor (curves are normalised to 1)
    fill_color, fill_lite, ridge_color, ao_color : colours
    baseline_y   : y of the axis
    z_base       : base z-layer for this curve
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        x_vals: np.ndarray,
        raw_x:  np.ndarray,
        scale_y: float,
        fill_color: str,
        fill_lite:  str,
        ridge_color: str,
        ao_color:    str,
        baseline_y:  float = 0.0,
        z_base:      float = 0.0,
        opacity:     float = 0.82,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mu      = mu
        self._sigma   = sigma
        self._scale_y = scale_y

        y_raw = _norm_pdf(raw_x, mu, sigma)
        y_vals = y_raw * scale_y

        # ── 1. Base fill polygon ──────────────────────────────────────
        verts = _curve_polygon_verts(x_vals, y_vals, baseline_y)
        if len(verts) >= 3:
            base = Polygon(
                *verts,
                fill_color=fill_color,
                fill_opacity=opacity,
                stroke_width=0,
            )
            base.move_to(base.get_center() + np.array([0, 0, z_base]))
            self.add(base)

        # ── 2 & 3. AO darkening at tails ─────────────────────────────
        ao_w   = (raw_x[-1] - raw_x[0]) * 0.12
        for side in [-1, 1]:
            x_lim = raw_x[0]  if side < 0 else raw_x[-1]
            x_mid = x_lim + side * ao_w * (-1 if side < 0 else 1) * (-1)
            # ao region: from tail edge to ao_w inward
            if side < 0:
                x_lo, x_hi = raw_x[0], raw_x[0] + ao_w
            else:
                x_lo, x_hi = raw_x[-1] - ao_w, raw_x[-1]

            # Map to world coords
            mask = (raw_x >= x_lo) & (raw_x <= x_hi)
            if mask.sum() < 2:
                continue
            ao_verts = _clipped_area_verts(
                x_vals, y_vals, x_vals[mask][0], x_vals[mask][-1], baseline_y
            )
            if len(ao_verts) >= 3:
                ao = Polygon(
                    *ao_verts,
                    fill_color=ao_color,
                    fill_opacity=0.38,
                    stroke_width=0,
                )
                ao.move_to(ao.get_center() + np.array([0, 0, z_base + 0.001]))
                self.add(ao)

        # ── 4. Ridge spine (bright outline along the curve top) ───────
        spine_pts = [
            [float(x_vals[i]), float(y_vals[i]), z_base + 0.003]
            for i in range(0, len(x_vals), max(1, len(x_vals)//80))
            if y_vals[i] > 0.003
        ]
        if len(spine_pts) >= 2:
            from manim import VMobject
            spine = VMobject(stroke_color=ridge_color,
                             stroke_width=1.5, stroke_opacity=0.70)
            spine.set_points_smoothly([np.array(p) for p in spine_pts])
            self.add(spine)

        # ── 5. Peak cap ───────────────────────────────────────────────
        peak_x_raw = mu
        peak_x_px  = x_vals[np.argmin(np.abs(raw_x - peak_x_raw))]
        peak_y_px  = float(y_vals[np.argmin(np.abs(raw_x - peak_x_raw))])
        peak_cap = Circle(
            radius=0.045,
            fill_color=ridge_color,
            fill_opacity=0.85,
            stroke_width=0,
        )
        peak_cap.move_to([peak_x_px, peak_y_px + 0.01, z_base + 0.004])
        self.add(peak_cap)


class _ErrorRegion(VGroup):
    """
    A shaded + hatched region under a normal curve between x_lo and x_hi.

    Layers:
      1. fill_polygon  — filled area
      2. hatch_lines   — diagonal hatch (direction and colour per region type)
      3. region_label  — e.g. "α = 0.050"
      4. formula_badge — e.g. "P(reject H₀ | H₀ true)"
      5. leader_arrow  — arrow from label into region

    Parameters
    ----------
    x_vals, raw_x, y_vals : arrays for the parent distribution
    x_lo_raw, x_hi_raw    : raw axis bounds for this region
    fill_color, hatch_color : colours
    hatch_angle    : diagonal direction (PI/4 = ↗, -PI/4 = ↘)
    label_text     : e.g. "α = 0.050"
    formula_text   : e.g. "P(reject H₀ | H₀ true)"
    label_side     : "above" | "below" | "left" | "right"
    baseline_y     : y of x-axis
    z_base         : z-layer
    """

    def __init__(
        self,
        x_vals:    np.ndarray,
        raw_x:     np.ndarray,
        y_vals:    np.ndarray,
        x_lo_raw:  float,
        x_hi_raw:  float,
        fill_color:  str,
        hatch_color: str,
        hatch_angle: float = PI / 4,
        label_text:  str   = "",
        formula_text: str  = "",
        label_color: str   = WHITE,
        label_side:  Literal["above","below","left","right"] = "above",
        baseline_y:  float = 0.0,
        z_base:      float = 0.001,
        fill_opacity: float = 0.50,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Map raw bounds to world pixel bounds
        # (raw_x and x_vals are in 1-to-1 correspondence)
        def raw_to_px(rv: float) -> float:
            idx = np.argmin(np.abs(raw_x - rv))
            return float(x_vals[idx])

        x_lo_px = raw_to_px(x_lo_raw)
        x_hi_px = raw_to_px(x_hi_raw)

        # ── 1. Fill polygon ───────────────────────────────────────────
        verts = _clipped_area_verts(
            x_vals, y_vals, x_lo_px, x_hi_px, baseline_y
        )
        if len(verts) >= 3:
            fill = Polygon(
                *verts,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                stroke_width=0,
            )
            fill.move_to(fill.get_center() + np.array([0, 0, z_base]))
            self.add(fill)

        # ── 2. Hatch lines ─────────────────────────────────────────────
        # Find y extent of the region
        mask  = (raw_x >= x_lo_raw) & (raw_x <= x_hi_raw)
        y_max = float(y_vals[mask].max()) if mask.sum() > 0 else 0.05
        hatch = _hatch_lines(
            x_lo=x_lo_px, x_hi=x_hi_px,
            y_lo=baseline_y, y_hi=y_max * 1.02,
            angle=hatch_angle,
            spacing=0.10,
            stroke_color=hatch_color,
            stroke_width=0.9,
            stroke_opacity=0.50,
            z=z_base + 0.002,
        )
        self.add(hatch)

        # ── 3. Region label ───────────────────────────────────────────
        # Centroid x of the region, positioned above or below
        mid_x_px   = (x_lo_px + x_hi_px) / 2
        mask2      = (raw_x >= x_lo_raw) & (raw_x <= x_hi_raw)
        region_h   = float(y_vals[mask2].mean()) if mask2.sum() > 0 else 0.05

        if label_text:
            lbl = Text(
                label_text,
                font_size=20,
                color=label_color,
                font="sans-serif",
            )
            if label_side == "above":
                lbl_y = region_h * 0.55 + 0.30
            elif label_side == "below":
                lbl_y = baseline_y - 0.38
            elif label_side == "left":
                lbl_y = region_h * 0.40
            else:
                lbl_y = region_h * 0.40

            lbl.move_to([mid_x_px, lbl_y, z_base + 0.010])
            self.add(lbl)
            self._label_pos = np.array([mid_x_px, lbl_y, z_base + 0.010])

            # Leader arrow from label down into region
            arrow_tip_y = region_h * 0.28 + baseline_y
            if abs(lbl_y - arrow_tip_y) > 0.08:
                leader = Arrow(
                    start=[mid_x_px, lbl_y - lbl.height / 2 - 0.04,
                           z_base + 0.009],
                    end  =[mid_x_px, arrow_tip_y + 0.04,
                           z_base + 0.009],
                    stroke_color=interpolate_color(label_color, BLACK, 0.30),
                    stroke_width=1.4,
                    tip_length=0.12,
                    buff=0,
                )
                self.add(leader)

        # ── 4. Formula badge ─────────────────────────────────────────
        if formula_text:
            try:
                formula = MathTex(
                    formula_text, font_size=16,
                    color=interpolate_color(label_color, WHITE, 0.25)
                )
            except Exception:
                formula = Text(
                    formula_text, font_size=14,
                    color=interpolate_color(label_color, WHITE, 0.25)
                )
            f_bg = RoundedRectangle(
                width=formula.width + 0.20,
                height=formula.height + 0.12,
                corner_radius=0.05,
                fill_color=PALETTE["bg_panel"],
                fill_opacity=0.80,
                stroke_color=label_color,
                stroke_width=0.6,
                stroke_opacity=0.55,
            )
            if label_text:
                f_y = self._label_pos[1] + 0.32
            else:
                f_y = region_h * 0.55 + 0.30
            formula.move_to([mid_x_px, f_y, z_base + 0.011])
            f_bg.move_to([mid_x_px, f_y, z_base + 0.010])
            self.add(f_bg, formula)


class _CriticalLine(VGroup):
    """
    Vertical dashed critical-value line with a label badge.

    Layers:
      1. dashed_line  — full-height dashed vertical
      2. upper_arrow  — small upward-pointing arrow at the top
      3. badge        — rounded-rect label at the axis
      4. xc_label     — "xₒ = {value:.3f}" text in badge

    Parameters
    ----------
    x_px       : Manim world x-coordinate
    y_bottom   : y of axis baseline
    y_top      : y of curve apex area
    x_raw      : raw axis value (for badge label)
    side       : "right" | "left" (which side is rejection)
    """

    def __init__(
        self,
        x_px:     float,
        y_bottom: float,
        y_top:    float,
        x_raw:    float,
        side:     Literal["right","left"] = "right",
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.015

        # ── Dashed line ───────────────────────────────────────────────
        dline = DashedLine(
            start=[x_px, y_bottom - 0.12, z0],
            end  =[x_px, y_top   + 0.18, z0],
            stroke_color=PALETTE["crit_line"],
            stroke_width=2.2,
            dash_length=0.14,
            dashed_ratio=0.55,
        )
        self.add(dline)

        # ── Arrow tip at top ──────────────────────────────────────────
        tip = Arrow(
            start=[x_px, y_top + 0.05, z0],
            end  =[x_px, y_top + 0.30, z0],
            stroke_color=PALETTE["crit_line"],
            stroke_width=1.6,
            tip_length=0.12,
            buff=0,
        )
        self.add(tip)

        # ── Badge at axis ─────────────────────────────────────────────
        sign_str = "≥" if side == "right" else "≤"
        badge_str = f"$x_c = {x_raw:.3f}$"
        try:
            badge_lbl = MathTex(
                badge_str.replace("$",""),
                font_size=18,
                color=PALETTE["crit_badge_fg"],
            )
        except Exception:
            badge_lbl = Text(
                f"xc = {x_raw:.3f}",
                font_size=16,
                color=PALETTE["crit_badge_fg"],
            )
        badge_bg = RoundedRectangle(
            width=badge_lbl.width + 0.22,
            height=badge_lbl.height + 0.14,
            corner_radius=0.06,
            fill_color=PALETTE["crit_badge_bg"],
            fill_opacity=0.92,
            stroke_color=PALETTE["crit_line"],
            stroke_width=0.8,
        )
        badge_y = y_bottom - 0.42
        badge_bg.move_to([x_px, badge_y, z0 + 0.001])
        badge_lbl.move_to([x_px, badge_y, z0 + 0.002])
        self.add(badge_bg, badge_lbl)

        # ── Rejection-side annotation arrow ───────────────────────────
        arr_dir = RIGHT if side == "right" else LEFT
        rej_lbl = Text(
            "Rejection\nregion" if side == "right" else "Rejection\nregion",
            font_size=14,
            color=PALETTE["crit_line"],
        )
        rej_lbl.move_to(
            [x_px + arr_dir[0] * 0.9, y_top + 0.05, z0 + 0.002]
        )
        self.add(rej_lbl)


class _EffectArrow(VGroup):
    """
    Double-headed arrow between μ₀ and μ₁, labelled δ.

    Layers:
      1. double_arrow  — DoubleArrow from μ₀_px to μ₁_px
      2. delta_label   — "δ" above the arrow
      3. value_label   — "= μ₁ − μ₀ = {value:.2f}" below

    Parameters
    ----------
    mu0_px, mu1_px : Manim world x-coords of the two means
    y              : vertical position of the arrow
    delta          : μ₁ − μ₀ (raw units)
    """

    def __init__(
        self,
        mu0_px: float,
        mu1_px: float,
        y:      float,
        delta:  float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.018

        arr = DoubleArrow(
            start=[mu0_px, y, z0],
            end  =[mu1_px, y, z0],
            stroke_color=PALETTE["delta_arrow"],
            stroke_width=2.0,
            tip_length=0.15,
            buff=0,
        )
        self.add(arr)

        mid_px = (mu0_px + mu1_px) / 2

        try:
            d_lbl = MathTex(r"\delta", font_size=28,
                            color=PALETTE["delta_label"])
        except Exception:
            d_lbl = Text("δ", font_size=24,
                         color=PALETTE["delta_label"])
        d_lbl.move_to([mid_px, y + 0.28, z0 + 0.001])
        self.add(d_lbl)

        val_lbl = Text(
            f"= μ₁ − μ₀ = {delta:.2f}",
            font_size=15,
            color=PALETTE["delta_label"],
        )
        val_lbl.move_to([mid_px, y - 0.22, z0 + 0.001])
        self.add(val_lbl)


class _DecisionTable(VGroup):
    """
    2×2 decision table: Reality (rows) × Decision (columns).

    Structure:
      ┌───────────────┬──────────────┬──────────────┐
      │               │ Fail Reject  │   Reject H₀  │
      ├───────────────┼──────────────┼──────────────┤
      │ H₀ True       │  1−α (TN)   │   α  (FP)    │
      ├───────────────┼──────────────┼──────────────┤
      │ H₁ True       │  β  (FN)    │  1−β (TP)    │
      └───────────────┴──────────────┴──────────────┘

    Each cell shows: symbol, name, and computed numeric value.

    Parameters
    ----------
    alpha, beta    : computed Type I / II error rates
    cell_w, cell_h : cell dimensions (Manim units)
    """

    def __init__(
        self,
        alpha:  float,
        beta:   float,
        cell_w: float = 1.60,
        cell_h: float = 0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0 = 0.001

        power   = 1 - beta
        correct = 1 - alpha

        # Cell data: (symbol, name, value, fill_color, text_color)
        cells = [
            # Row 0: H₀ True
            ("1−α", "Correct\nReject",  correct, PALETTE["correct_fill"],
             PALETTE["label_correct"]),
            ("α",   "Type I\nError",    alpha,   PALETTE["alpha_fill"],
             PALETTE["label_alpha"]),
            # Row 1: H₁ True
            ("β",   "Type II\nError",   beta,    PALETTE["beta_fill"],
             PALETTE["label_beta"]),
            ("1−β", "Power",            power,   PALETTE["power_fill"],
             PALETTE["label_power"]),
        ]

        row_labels  = ["H₀ True", "H₁ True"]
        col_labels  = ["Fail to\nReject H₀", "Reject H₀"]
        row_header_w = cell_w * 0.80

        total_w = row_header_w + 2 * cell_w
        total_h = cell_h * 3   # header + 2 data rows

        # ── Background panel ──────────────────────────────────────────
        bg = RoundedRectangle(
            width=total_w + 0.22,
            height=total_h + 0.22,
            corner_radius=0.10,
            fill_color=PALETTE["bg_panel"],
            fill_opacity=0.92,
            stroke_color=PALETTE["table_border"],
            stroke_width=1.2,
        )
        bg.move_to([0, 0, z0])
        self.add(bg)

        # ── Column headers ────────────────────────────────────────────
        x_cols  = [
            row_header_w / 2,                         # row-label column
            row_header_w + cell_w / 2,                # col 0
            row_header_w + cell_w * 3 / 2,            # col 1
        ]
        y_header = total_h / 2 - cell_h / 2

        # Empty top-left cell
        _draw_cell(self, x_cols[0] - total_w/2, y_header,
                   row_header_w, cell_h, PALETTE["table_header"],
                   "", z0)
        # Top-centre label
        _draw_label(self, x_cols[0] - total_w/2, y_header,
                    "Decision", 16, PALETTE["axis_label"], z0 + 0.002)

        for ci, col_lbl in enumerate(col_labels):
            xc = x_cols[ci + 1] - total_w / 2
            _draw_cell(self, xc, y_header,
                       cell_w, cell_h, PALETTE["table_header"], "", z0)
            _draw_label(self, xc, y_header,
                        col_lbl, 15, PALETTE["axis_label"], z0 + 0.002)

        # ── Row headers + data cells ──────────────────────────────────
        for ri in range(2):
            yc = total_h / 2 - cell_h * (ri + 1.5)
            # Row header
            _draw_cell(self, x_cols[0] - total_w/2, yc,
                       row_header_w, cell_h, PALETTE["table_header"], "", z0)
            _draw_label(self, x_cols[0] - total_w/2, yc,
                        row_labels[ri], 15, PALETTE["axis_label"], z0+0.002)

            for ci in range(2):
                sym, name, val, fc, tc = cells[ri * 2 + ci]
                xc = x_cols[ci + 1] - total_w / 2
                _draw_cell(self, xc, yc, cell_w, cell_h,
                           interpolate_color(fc, BLACK, 0.55), "", z0)
                # Symbol
                try:
                    sym_mob = MathTex(sym, font_size=22, color=tc)
                except Exception:
                    sym_mob = Text(sym, font_size=18, color=tc)
                sym_mob.move_to([xc, yc + 0.14, z0 + 0.003])
                self.add(sym_mob)
                # Name
                name_mob = Text(name, font_size=11,
                                color=interpolate_color(tc, WHITE, 0.25))
                name_mob.move_to([xc, yc - 0.05, z0 + 0.003])
                self.add(name_mob)
                # Value
                val_mob = Text(
                    f"{val:.3f}",
                    font_size=13,
                    color=tc,
                    font="monospace",
                )
                val_mob.move_to([xc, yc - 0.28, z0 + 0.003])
                self.add(val_mob)

        # ── Grid lines ────────────────────────────────────────────────
        for xi in [row_header_w, row_header_w + cell_w]:
            xpx = xi - total_w / 2
            self.add(Line(
                start=[xpx, -total_h / 2, z0 + 0.001],
                end  =[xpx,  total_h / 2, z0 + 0.001],
                stroke_color=PALETTE["table_border"],
                stroke_width=0.8,
            ))
        for yi in [total_h / 2 - cell_h, total_h / 2 - 2 * cell_h]:
            self.add(Line(
                start=[-total_w / 2, yi, z0 + 0.001],
                end  =[ total_w / 2, yi, z0 + 0.001],
                stroke_color=PALETTE["table_border"],
                stroke_width=0.8,
            ))


def _draw_cell(
    parent: VGroup, x: float, y: float,
    w: float, h: float, color: str, text: str, z: float,
):
    rect = Rectangle(
        width=w, height=h,
        fill_color=color, fill_opacity=1.0,
        stroke_color=PALETTE["table_border"],
        stroke_width=0.6,
    )
    rect.move_to([x, y, z])
    parent.add(rect)


def _draw_label(
    parent: VGroup, x: float, y: float,
    text: str, fs: int, color: str, z: float,
):
    lbl = Text(text, font_size=fs, color=color)
    lbl.move_to([x, y, z])
    parent.add(lbl)


# ─────────────────────────────────────────────────────────────────────────────
# TypeITypeII  ──  the main export
# ─────────────────────────────────────────────────────────────────────────────

class TypeITypeII(VGroup):
    """
    Full Type I / Type II error visualisation.

    Parameters
    ----------
    mu0 : float
        Null hypothesis mean.
    mu1 : float
        Alternative hypothesis mean.
    sigma : float
        Common population standard deviation.
    n : int
        Sample size.  Effective SE = sigma / sqrt(n).
    alpha : float
        Significance level (Type I error rate).  Default 0.05.
    two_tailed : bool
        If True, use symmetric two-tailed critical region.
    plot_width : float
        Total width of the plot in Manim units.  Default 10.0.
    plot_height : float
        Maximum curve height.  Default 3.0.
    x_sigma_range : float
        How many σ_eff to show on each side of the midpoint.  Default 3.8.
    show_table : bool
        Show the decision table to the right.  Default True.
    show_effect_arrow : bool
        Show the δ arrow between means.  Default True.
    show_all_regions : bool
        Show α, β, 1−β, 1−α simultaneously from construction.
    baseline_y : float
        y-position of the x-axis baseline.  Default 0.0.

    Attributes
    ----------
    h0_curve, h1_curve      : _NormalCurve
    crit_line                : _CriticalLine (or VGroup of two for two-tailed)
    alpha_region             : _ErrorRegion
    beta_region              : _ErrorRegion
    power_region             : _ErrorRegion
    correct_region           : _ErrorRegion
    effect_arrow             : _EffectArrow | None
    decision_table           : _DecisionTable | None
    n_tracker                : ValueTracker  — track current n
    alpha_tracker            : ValueTracker  — track current alpha
    xc_raw                   : float         — critical value in raw units
    computed_alpha            : float
    computed_beta             : float
    computed_power            : float
    sigma_eff                : float

    Class methods
    -------------
    TypeITypeII.z_test(mu0, mu1, sigma, n, alpha, **kw)
    TypeITypeII.t_test(mu0, mu1, sigma, n, alpha, **kw)
    """

    def __init__(
        self,
        mu0: float = 0.0,
        mu1: float = 2.0,
        sigma: float = 1.0,
        n: int = 1,
        alpha: float = 0.05,
        two_tailed: bool = False,
        plot_width: float = 10.0,
        plot_height: float = 3.0,
        x_sigma_range: float = 3.8,
        show_table: bool = True,
        show_effect_arrow: bool = True,
        show_all_regions: bool = False,
        baseline_y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._mu0          = mu0
        self._mu1          = mu1
        self._sigma        = sigma
        self._n            = n
        self._alpha        = alpha
        self._two_tailed   = two_tailed
        self._plot_width   = plot_width
        self._plot_height  = plot_height
        self._baseline_y   = baseline_y

        # Value trackers for live updates
        self.n_tracker     = ValueTracker(float(n))
        self.alpha_tracker = ValueTracker(alpha)

        self._build(show_table, show_effect_arrow, show_all_regions)

    # ─────────────────────────────────────────────────────────────────
    # Core build
    # ─────────────────────────────────────────────────────────────────

    def _build(
        self,
        show_table: bool,
        show_effect_arrow: bool,
        show_all_regions: bool,
    ):
        """Construct all visual components from current parameters."""
        mu0          = self._mu0
        mu1          = self._mu1
        sigma        = self._sigma
        n            = self._n
        alpha        = self._alpha
        two_tailed   = self._two_tailed
        baseline_y   = self._baseline_y
        plot_width   = self._plot_width
        plot_height  = self._plot_height

        # ── Derived quantities ────────────────────────────────────────
        sigma_eff    = sigma / np.sqrt(n)
        self.sigma_eff = sigma_eff

        # Critical value(s)
        if two_tailed:
            xc_hi = _norm_ppf(1 - alpha / 2, mu0, sigma_eff)
            xc_lo = _norm_ppf(    alpha / 2, mu0, sigma_eff)
        else:
            xc_hi = _norm_ppf(1 - alpha, mu0, sigma_eff)
            xc_lo = -np.inf
        self.xc_raw = xc_hi

        # Compute true α, β, power
        if two_tailed:
            self.computed_alpha = (
                _norm_cdf(xc_lo, mu0, sigma_eff)
                + (1 - _norm_cdf(xc_hi, mu0, sigma_eff))
            )
            self.computed_beta  = (
                _norm_cdf(xc_hi, mu1, sigma_eff)
                - _norm_cdf(xc_lo, mu1, sigma_eff)
            )
        else:
            self.computed_alpha = 1 - _norm_cdf(xc_hi, mu0, sigma_eff)
            self.computed_beta  = _norm_cdf(xc_hi, mu1, sigma_eff)
        self.computed_power = 1 - self.computed_beta

        # ── x-axis setup ──────────────────────────────────────────────
        x_mid    = (mu0 + mu1) / 2
        x_range  = self._plot_width / 2 + sigma_eff * 0.5
        # We span from the leftmost to rightmost relevant point
        x_lo_raw = min(mu0, mu1) - 3.8 * sigma_eff
        x_hi_raw = max(mu0, mu1) + 3.8 * sigma_eff
        raw_x    = np.linspace(x_lo_raw, x_hi_raw, 600)

        # Unit: Manim units per raw axis unit
        unit     = plot_width / (x_hi_raw - x_lo_raw)
        self._unit   = unit
        self._raw_x  = raw_x

        def to_px(rv: float) -> float:
            return (rv - x_lo_raw) * unit - plot_width / 2

        self._to_px = to_px
        x_vals   = (raw_x - x_lo_raw) * unit - plot_width / 2

        # y scaling: normalise curves so peak = plot_height
        peak_pdf = float(_norm_pdf(np.array([mu0]), mu0, sigma_eff)[0])
        scale_y  = plot_height / peak_pdf

        y0_vals = _norm_pdf(raw_x, mu0, sigma_eff) * scale_y
        y1_vals = _norm_pdf(raw_x, mu1, sigma_eff) * scale_y

        # ── Axis line ─────────────────────────────────────────────────
        axis_line = Line(
            start=[-plot_width / 2 - 0.3, baseline_y, 0],
            end  =[ plot_width / 2 + 0.3, baseline_y, 0],
            stroke_color=PALETTE["axis"],
            stroke_width=1.8,
        )
        self.add(axis_line)

        # ── Mean tick marks ───────────────────────────────────────────
        for mu, lbl_str in [(mu0, r"\mu_0"), (mu1, r"\mu_1")]:
            px = to_px(mu)
            tk = Line(
                start=[px, baseline_y - 0.10, 0.001],
                end  =[px, baseline_y + 0.10, 0.001],
                stroke_color=PALETTE["axis"],
                stroke_width=1.5,
            )
            self.add(tk)
            try:
                ml = MathTex(lbl_str, font_size=20,
                             color=PALETTE["axis_label"])
            except Exception:
                ml = Text(lbl_str.replace("\\","").replace("{","").replace("}",""),
                          font_size=17, color=PALETTE["axis_label"])
            ml.move_to([px, baseline_y - 0.38, 0.001])
            self.add(ml)

        # ── H₀ curve ──────────────────────────────────────────────────
        self.h0_curve = _NormalCurve(
            mu=to_px(mu0), sigma=sigma_eff * unit,
            x_vals=x_vals, raw_x=raw_x,
            scale_y=scale_y,
            fill_color=PALETTE["h0_fill"],
            fill_lite=PALETTE["h0_fill_lite"],
            ridge_color=PALETTE["h0_ridge"],
            ao_color=PALETTE["h0_tail_ao"],
            baseline_y=baseline_y,
            z_base=0.002,
            opacity=0.75,
        )
        self.add(self.h0_curve)

        # ── H₁ curve ──────────────────────────────────────────────────
        self.h1_curve = _NormalCurve(
            mu=to_px(mu1), sigma=sigma_eff * unit,
            x_vals=x_vals, raw_x=raw_x,
            scale_y=scale_y,
            fill_color=PALETTE["h1_fill"],
            fill_lite=PALETTE["h1_fill_lite"],
            ridge_color=PALETTE["h1_ridge"],
            ao_color=PALETTE["h1_tail_ao"],
            baseline_y=baseline_y,
            z_base=0.003,
            opacity=0.75,
        )
        self.add(self.h1_curve)

        # ── Curve labels ──────────────────────────────────────────────
        for mu, label_str, ridge_col in [
            (mu0, r"H_0", PALETTE["h0_ridge"]),
            (mu1, r"H_1", PALETTE["h1_ridge"]),
        ]:
            px    = to_px(mu)
            py    = float(y0_vals[np.argmin(np.abs(raw_x - mu))]) if mu == mu0 \
                    else float(y1_vals[np.argmin(np.abs(raw_x - mu))])
            try:
                cl = MathTex(label_str, font_size=28, color=ridge_col)
            except Exception:
                cl = Text(label_str, font_size=24, color=ridge_col)
            cl.move_to([px, py + 0.30, 0.020])
            self.add(cl)

        # ── Critical line(s) ──────────────────────────────────────────
        y_top = plot_height * 0.88
        self.crit_lines = VGroup()
        self.crit_line  = _CriticalLine(
            x_px=to_px(xc_hi),
            y_bottom=baseline_y,
            y_top=y_top,
            x_raw=xc_hi,
            side="right",
        )
        self.crit_lines.add(self.crit_line)
        if two_tailed:
            crit_lo = _CriticalLine(
                x_px=to_px(xc_lo),
                y_bottom=baseline_y,
                y_top=y_top,
                x_raw=xc_lo,
                side="left",
            )
            self.crit_lines.add(crit_lo)
        self.add(self.crit_lines)

        # ── Error regions ─────────────────────────────────────────────
        # α: under H₀ in rejection zone (right of xc, or both tails)
        self.alpha_region = _ErrorRegion(
            x_vals=x_vals, raw_x=raw_x, y_vals=y0_vals,
            x_lo_raw=xc_hi, x_hi_raw=x_hi_raw,
            fill_color=PALETTE["alpha_fill"],
            hatch_color=PALETTE["alpha_hatch"],
            hatch_angle=PI / 4,
            label_text=f"α = {self.computed_alpha:.3f}",
            formula_text=r"P(\text{reject }H_0 | H_0)",
            label_color=PALETTE["label_alpha"],
            label_side="above",
            baseline_y=baseline_y,
            z_base=0.006,
            fill_opacity=0.55,
        )

        # β: under H₁ in acceptance zone (left of xc)
        self.beta_region = _ErrorRegion(
            x_vals=x_vals, raw_x=raw_x, y_vals=y1_vals,
            x_lo_raw=x_lo_raw, x_hi_raw=xc_hi,
            fill_color=PALETTE["beta_fill"],
            hatch_color=PALETTE["beta_hatch"],
            hatch_angle=-PI / 4,
            label_text=f"β = {self.computed_beta:.3f}",
            formula_text=r"P(\text{fail reject}|H_1)",
            label_color=PALETTE["label_beta"],
            label_side="above",
            baseline_y=baseline_y,
            z_base=0.006,
            fill_opacity=0.55,
        )

        # 1−β (Power): under H₁ in rejection zone (right of xc)
        self.power_region = _ErrorRegion(
            x_vals=x_vals, raw_x=raw_x, y_vals=y1_vals,
            x_lo_raw=xc_hi, x_hi_raw=x_hi_raw,
            fill_color=PALETTE["power_fill"],
            hatch_color=PALETTE["power_hatch"],
            hatch_angle=PI / 4,
            label_text=f"1−β = {self.computed_power:.3f}",
            formula_text=r"P(\text{reject }H_0 | H_1)",
            label_color=PALETTE["label_power"],
            label_side="above",
            baseline_y=baseline_y,
            z_base=0.006,
            fill_opacity=0.55,
        )

        # 1−α (Correct non-rejection): under H₀ in acceptance zone
        self.correct_region = _ErrorRegion(
            x_vals=x_vals, raw_x=raw_x, y_vals=y0_vals,
            x_lo_raw=x_lo_raw, x_hi_raw=xc_hi,
            fill_color=PALETTE["correct_fill"],
            hatch_color=PALETTE["correct_hatch"],
            hatch_angle=-PI / 4,
            label_text=f"1−α = {1-self.computed_alpha:.3f}",
            formula_text=r"P(\text{fail reject}|H_0)",
            label_color=PALETTE["label_correct"],
            label_side="below",
            baseline_y=baseline_y,
            z_base=0.005,
            fill_opacity=0.38,
        )

        if show_all_regions:
            self.add(self.correct_region, self.beta_region,
                     self.alpha_region, self.power_region)
        else:
            # Regions start hidden; animations reveal them
            for r in [self.alpha_region, self.beta_region,
                      self.power_region, self.correct_region]:
                r.set_opacity(0)
                self.add(r)

        # ── Effect size arrow ─────────────────────────────────────────
        self.effect_arrow = None
        if show_effect_arrow and mu0 != mu1:
            arrow_y = plot_height * 0.30
            self.effect_arrow = _EffectArrow(
                mu0_px=to_px(mu0),
                mu1_px=to_px(mu1),
                y=baseline_y + arrow_y,
                delta=mu1 - mu0,
            )
            self.add(self.effect_arrow)

        # ── Decision table ────────────────────────────────────────────
        self.decision_table = None
        if show_table:
            self.decision_table = _DecisionTable(
                alpha=self.computed_alpha,
                beta=self.computed_beta,
                cell_w=1.55,
                cell_h=0.72,
            )
            self.decision_table.scale(0.88)
            self.decision_table.move_to(
                [plot_width / 2 + 2.80, baseline_y + plot_height * 0.30, 0]
            )
            self.add(self.decision_table)

        # Store for re-build
        self._x_vals  = x_vals
        self._y0_vals = y0_vals
        self._y1_vals = y1_vals
        self._x_lo_raw = x_lo_raw
        self._x_hi_raw = x_hi_raw

    # ─────────────────────────────────────────────────────────────────
    # Rebuild with new parameters
    # ─────────────────────────────────────────────────────────────────

    def rebuild(
        self,
        n:      Optional[int]   = None,
        alpha:  Optional[float] = None,
        mu1:    Optional[float] = None,
        sigma:  Optional[float] = None,
    ) -> "TypeITypeII":
        """
        Return a fresh TypeITypeII with updated parameters.
        Used internally by NarrowCurves and SweepAlpha.
        """
        return TypeITypeII(
            mu0=self._mu0,
            mu1=mu1    if mu1    is not None else self._mu1,
            sigma=sigma if sigma is not None else self._sigma,
            n=n        if n      is not None else self._n,
            alpha=alpha if alpha is not None else self._alpha,
            two_tailed=self._two_tailed,
            plot_width=self._plot_width,
            plot_height=self._plot_height,
            baseline_y=self._baseline_y,
            show_table=self.decision_table is not None,
            show_effect_arrow=self.effect_arrow is not None,
            show_all_regions=True,
        )

    # ─────────────────────────────────────────────────────────────────
    # Class-method constructors
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def z_test(
        cls,
        mu0: float, mu1: float,
        sigma: float, n: int,
        alpha: float = 0.05,
        **kwargs,
    ) -> "TypeITypeII":
        """One-sided Z-test visualisation (σ known)."""
        return cls(mu0=mu0, mu1=mu1, sigma=sigma, n=n,
                   alpha=alpha, two_tailed=False, **kwargs)

    @classmethod
    def t_test(
        cls,
        mu0: float, mu1: float,
        sigma: float, n: int,
        alpha: float = 0.05,
        **kwargs,
    ) -> "TypeITypeII":
        """One-sided t-test visualisation (uses t critical value)."""
        # For visual purposes use normal (large-n approx) or scipy t
        try:
            from scipy.stats import t as tdist
            xc = float(tdist.ppf(1 - alpha, df=n-1)) * sigma / np.sqrt(n) + mu0
        except ImportError:
            xc = _norm_ppf(1 - alpha, mu0, sigma / np.sqrt(n))
        return cls(mu0=mu0, mu1=mu1, sigma=sigma, n=n,
                   alpha=alpha, two_tailed=False, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Animations
# ─────────────────────────────────────────────────────────────────────────────

class BuildDistributions(Animation):
    """
    Both distribution curves grow upward from the baseline simultaneously.
    The critical line materialises at the end.

    Parameters
    ----------
    viz      : TypeITypeII
    run_time : float
    """

    def __init__(self, viz: TypeITypeII, **kwargs):
        self.viz = viz
        kwargs.setdefault("run_time", 2.2)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        grp = VGroup(viz.h0_curve, viz.h1_curve)
        super().__init__(grp, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        # Scale y from 0 → 1 (curves grow upward from baseline)
        baseline = self.viz._baseline_y
        self.mobject.scale(
            [1, alpha if alpha > 1e-4 else 1e-4, 1],
            about_point=[
                self.mobject.get_center()[0],
                baseline,
                0,
            ],
        )
        # Critical lines fade in during last 30%
        crit_alpha = max(0.0, (alpha - 0.70) / 0.30)
        self.viz.crit_lines.set_opacity(crit_alpha)


class RevealAlpha(Animation):
    """Fade in the α (Type I error) region."""

    def __init__(self, viz: TypeITypeII, **kwargs):
        self.viz = viz
        kwargs.setdefault("run_time", 1.0)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(viz.alpha_region, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(alpha)


class RevealBeta(Animation):
    """Fade in the β (Type II error) region."""

    def __init__(self, viz: TypeITypeII, **kwargs):
        self.viz = viz
        kwargs.setdefault("run_time", 1.0)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(viz.beta_region, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(alpha)


class RevealPower(Animation):
    """Fade in the 1−β (Power) region."""

    def __init__(self, viz: TypeITypeII, **kwargs):
        self.viz = viz
        kwargs.setdefault("run_time", 1.0)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(viz.power_region, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(alpha)


class RevealAll(Succession):
    """
    Orchestrated sequence: curves → α → 1−α → β → 1−β → table.

    Parameters
    ----------
    viz      : TypeITypeII
    run_time : float — total (distributes across sub-animations)
    """

    def __init__(self, viz: TypeITypeII, **kwargs):
        anims = [
            BuildDistributions(viz, run_time=2.0),
            FadeIn(viz.correct_region, run_time=0.8),
            RevealAlpha(viz, run_time=0.9),
            FadeIn(viz.beta_region,    run_time=0.9),
            RevealPower(viz, run_time=0.9),
        ]
        if viz.effect_arrow is not None:
            anims.append(FadeIn(viz.effect_arrow, run_time=0.7))
        if viz.decision_table is not None:
            anims.append(FadeIn(viz.decision_table, shift=RIGHT * 0.3,
                                run_time=0.8))
        kwargs.setdefault(
            "run_time", sum(a.run_time for a in anims)
        )
        super().__init__(*anims, **kwargs)


class ShiftH1(Animation):
    """
    Smoothly move the H₁ distribution to a new mean ``new_mu1``,
    morphing the β and power regions accordingly.

    Implementation: interpolates between the current viz and a freshly
    rebuilt one using ``become()``.

    Parameters
    ----------
    viz     : TypeITypeII
    new_mu1 : float
    run_time: float
    """

    def __init__(
        self,
        viz: TypeITypeII,
        new_mu1: float,
        **kwargs,
    ):
        self.viz     = viz
        self.new_mu1 = new_mu1
        self.target  = viz.rebuild(mu1=new_mu1)
        self.target.move_to(viz.get_center())
        kwargs.setdefault("run_time", 2.0)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(viz, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(
            self.starting_mobject.copy()
            if alpha < 0.5
            else self.target.copy()
        )
        # Smooth cross-fade
        self.starting_mobject.set_opacity(1 - alpha)
        self.target.set_opacity(alpha)
        if alpha >= 1.0:
            self.viz._mu1 = self.new_mu1


class NarrowCurves(Animation):
    """
    Increase sample size n → both curves narrow, β shrinks, power grows.
    Morphs the entire visualisation into a rebuilt one.

    Parameters
    ----------
    viz     : TypeITypeII
    new_n   : int   — new sample size
    run_time: float
    """

    def __init__(
        self,
        viz: TypeITypeII,
        new_n: int,
        **kwargs,
    ):
        self.viz    = viz
        self.new_n  = new_n
        self.target = viz.rebuild(n=new_n)
        self.target.move_to(viz.get_center())
        kwargs.setdefault("run_time", 2.5)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        super().__init__(viz, **kwargs)

    def interpolate_mobject(self, alpha: float):
        # Cross-fade between old and new
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(1 - alpha)
        self.target.set_opacity(alpha)

        if alpha >= 1.0:
            self.mobject.become(self.target)
            self.viz._n = self.new_n
            self.viz.n_tracker.set_value(float(self.new_n))


class SweepAlpha(Animation):
    """
    Move the critical value to correspond to a new significance level,
    morphing α, β, and power regions.

    Parameters
    ----------
    viz       : TypeITypeII
    new_alpha : float   — new significance level
    run_time  : float
    """

    def __init__(
        self,
        viz: TypeITypeII,
        new_alpha: float,
        **kwargs,
    ):
        self.viz       = viz
        self.new_alpha = new_alpha
        self.target    = viz.rebuild(alpha=new_alpha)
        self.target.move_to(viz.get_center())
        kwargs.setdefault("run_time", 2.0)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_sine)
        super().__init__(viz, **kwargs)

    def interpolate_mobject(self, alpha_t: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.set_opacity(1 - alpha_t)
        self.target.set_opacity(alpha_t)

        if alpha_t >= 1.0:
            self.mobject.become(self.target)
            self.viz._alpha = self.new_alpha
            self.viz.alpha_tracker.set_value(self.new_alpha)


class FlashDecision(AnimationGroup):
    """
    Highlight one cell of the decision table with a bright flash.

    Parameters
    ----------
    viz        : TypeITypeII
    cell       : "alpha" | "beta" | "power" | "correct"
    run_time   : float
    """

    _CELL_MAP = {
        "alpha":   PALETTE["label_alpha"],
        "beta":    PALETTE["label_beta"],
        "power":   PALETTE["label_power"],
        "correct": PALETTE["label_correct"],
    }

    def __init__(
        self,
        viz: TypeITypeII,
        cell: Literal["alpha","beta","power","correct"] = "alpha",
        **kwargs,
    ):
        col = self._CELL_MAP.get(cell, WHITE)
        region_map = {
            "alpha":   viz.alpha_region,
            "beta":    viz.beta_region,
            "power":   viz.power_region,
            "correct": viz.correct_region,
        }
        region = region_map.get(cell, viz.alpha_region)

        anims = [
            Indicate(region, color=col, scale_factor=1.03, run_time=0.8),
        ]
        if viz.decision_table is not None:
            anims.append(
                Flash(
                    viz.decision_table.get_center()
                    + np.array([0.5, 0, 0]),
                    color=col,
                    line_length=0.15,
                    num_lines=8,
                    run_time=0.8,
                )
            )
        kwargs.setdefault("run_time", 0.8)
        super().__init__(*anims, **kwargs)


class BuildDecisionTable(Animation):
    """
    Animate the 2×2 decision table growing from the centre outward.

    Parameters
    ----------
    viz      : TypeITypeII
    run_time : float
    """

    def __init__(self, viz: TypeITypeII, **kwargs):
        self.viz = viz
        kwargs.setdefault("run_time", 1.2)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        tbl = viz.decision_table if viz.decision_table is not None else VGroup()
        super().__init__(tbl, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.scale(
            alpha if alpha > 1e-4 else 1e-4,
            about_point=self.mobject.get_center(),
        )
        self.mobject.set_opacity(alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factories
# ─────────────────────────────────────────────────────────────────────────────

def make_power_analysis_grid(
    mu0: float = 0.0,
    mu1_values: list[float] = None,
    n_values:   list[int]   = None,
    sigma: float = 1.0,
    alpha: float = 0.05,
    cell_scale: float = 0.38,
) -> VGroup:
    """
    Grid of TypeITypeII visualisations varying δ (columns) and n (rows),
    for a compact power analysis overview.

    Parameters
    ----------
    mu0       : null mean
    mu1_values: list of alternative means (grid columns)
    n_values  : list of sample sizes (grid rows)
    sigma, alpha : fixed
    cell_scale : scale factor per cell

    Returns
    -------
    VGroup — rows × columns grid of TypeITypeII objects.
    """
    if mu1_values is None:
        mu1_values = [1.0, 2.0, 3.0]
    if n_values is None:
        n_values = [1, 4, 16]

    grid = VGroup()
    col_w = 5.0
    row_h = 3.5

    for ri, n in enumerate(n_values):
        for ci, mu1 in enumerate(mu1_values):
            viz = TypeITypeII(
                mu0=mu0, mu1=mu1,
                sigma=sigma, n=n,
                alpha=alpha,
                two_tailed=False,
                plot_width=4.0,
                plot_height=2.0,
                show_table=False,
                show_effect_arrow=False,
                show_all_regions=True,
            )
            viz.scale(cell_scale)
            viz.move_to([ci * col_w * cell_scale,
                         -ri * row_h * cell_scale, 0])
            grid.add(viz)

    grid.center()
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Demo Scene  (run with: manim -pql error_types.py ErrorTypesDemo)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from manim import Scene, DEGREES

    class ErrorTypesDemo(Scene):
        """Full showcase of TypeITypeII visualisation and animations."""

        def construct(self):

            # ── Base visualisation ────────────────────────────────────
            viz = TypeITypeII(
                mu0=0.0, mu1=2.5,
                sigma=1.0, n=1,
                alpha=0.05,
                two_tailed=False,
                plot_width=9.0,
                plot_height=3.2,
                show_table=True,
                show_effect_arrow=True,
                show_all_regions=False,
                baseline_y=-1.5,
            )
            viz.center()
            self.add(viz)

            # ── Reveal all regions in sequence ────────────────────────
            self.play(RevealAll(viz))
            self.wait(0.8)

            # ── Flash the α cell ──────────────────────────────────────
            self.play(FlashDecision(viz, cell="alpha"))
            self.wait(0.4)
            self.play(FlashDecision(viz, cell="beta"))
            self.wait(0.4)
            self.play(FlashDecision(viz, cell="power"))
            self.wait(0.6)

            # ── Increase n: watch β shrink ────────────────────────────
            self.play(NarrowCurves(viz, new_n=9, run_time=2.8))
            self.wait(0.6)
            self.play(NarrowCurves(viz, new_n=25, run_time=2.5))
            self.wait(0.8)

            # ── Sweep α: trade-off ────────────────────────────────────
            self.play(SweepAlpha(viz, new_alpha=0.01, run_time=2.0))
            self.wait(0.5)
            self.play(SweepAlpha(viz, new_alpha=0.10, run_time=2.0))
            self.wait(0.8)

            # ── Shift H₁ closer (harder test) ────────────────────────
            self.play(ShiftH1(viz, new_mu1=1.5, run_time=2.0))
            self.wait(0.5)

            # ── Power analysis grid ───────────────────────────────────
            self.play(FadeOut(viz))
            grid = make_power_analysis_grid(
                mu0=0.0,
                mu1_values=[1.0, 2.0, 3.0],
                n_values=[1, 4, 16],
                sigma=1.0, alpha=0.05,
                cell_scale=0.36,
            )
            grid.center()
            self.play(FadeIn(grid, shift=UP * 0.3), run_time=1.0)
            self.wait(2.5)

except ImportError:
    pass