"""
manim_stats/distributions/cdf_viz.py
======================================
Dedicated CDF (Cumulative Distribution Function) visualisation system
for the Manim Statistics Extension.

This module is separate from ``base_dist.py`` because the CDF is not just
a different colouring of the PDF — it requires wholly different geometry,
annotation logic, animation sequencing, and interactive probing tools.

Architecture
------------
CDFViz3D                        ← master class
    ├── CDFCurveLayer           ← smooth monotone curve (continuous)
    ├── StepFunction3D          ← staircase geometry (discrete)
    ├── CDFFillLayer            ← shaded area below curve
    ├── CDFAnnotationSystem     ← quartile / p-level horizontal guides
    ├── ProbabilityReadout3D    ← L-shaped x↔p live probe
    ├── CDFComparisonLayer      ← multi-CDF overlay + KS statistic
    ├── QuantileFunction3D      ← inverse CDF curve
    ├── CDFDecompositionPanel   ← running PMF sum panel (discrete only)
    ├── ECDFLayer               ← empirical CDF from raw data
    └── SurvivalFunctionLayer   ← S(x) = 1 - F(x) mirrored overlay

Key design principles
---------------------
• Every sub-layer is independently add / remove / animate-able.
• ``x_tracker`` and ``p_tracker`` are public ValueTrackers driving
  the live readout probe so scenes can animate them directly.
• Works for BOTH continuous and discrete DistributionFunction objects.
• The KS statistic layer works between any two CDFs (theoretical or ECDF).
• All geometry is in 3-D scene space via axes.c2p().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Literal,
    Optional, Sequence, Tuple, Union,
)

import numpy as np
from numpy.typing import ArrayLike

from manim import (
    VGroup, VMobject, Mobject,
    Line3D, Arrow3D, Dot3D,
    Line, DashedLine, Polygon,
    Text, MathTex, DecimalNumber,
    RoundedRectangle, Rectangle,
    ParametricFunction,
    Animation, AnimationGroup, Succession,
    Create, FadeIn, FadeOut, Write, Transform,
    DrawBorderThenFill, GrowArrow,
    UpdateFromAlphaFunc, always_redraw,
    ORIGIN, UP, DOWN, LEFT, RIGHT, IN, OUT,
    DEGREES, PI, TAU,
    WHITE, BLACK, GRAY,
    interpolate_color, smooth, there_and_back,
    ValueTracker, rate_functions,
)

from ..core.base import (
    StatsObject3D,
    StatsTheme, StatsColorPalette,
    MaterialConfig, MaterialApplicator,
    AnimationConfig, BuildStyle,
    HighlightStyle, HighlightSystem,
    ThemeMode,
)
from ..core.math_utils import (
    DistributionFunction, DistributionResult,
    format_stat_value, FloatArray,
)
from ..axes.axes3d import StatsAxes3D, AxisID, GridStyle


# ─────────────────────────────────────────────────────────────────────────────
# 0.  ENUMERATIONS & CONFIG DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

class CDFDisplayMode(Enum):
    CDF      = auto()   # F(x) = P(X ≤ x)
    SURVIVAL = auto()   # S(x) = 1 - F(x)
    LOG_CDF  = auto()   # log F(x)  — linearises Weibull / Exponential
    LOG_SF   = auto()   # log S(x)  — log-survival (hazard analysis)
    BOTH     = auto()   # F(x) and S(x) simultaneously


class StepJumpStyle(Enum):
    """How the vertical jumps in a discrete step CDF are drawn."""
    DASHED   = auto()   # dashed vertical line between steps
    SOLID    = auto()   # solid thin line
    ARROW    = auto()   # small upward arrow
    NONE     = auto()   # no jump line — just open/filled circles


@dataclass
class CDFCurveConfig:
    """Visual style for the main CDF curve."""
    color:             Optional[str]  = None   # None → theme.primary
    stroke_width:      float          = 3.0
    stroke_opacity:    float          = 1.0
    n_points:          int            = 500    # curve resolution
    # Fill below the curve
    show_fill:         bool           = True
    fill_color:        Optional[str]  = None
    fill_opacity:      float          = 0.12
    # Y=1 reference line
    show_y1_line:      bool           = True
    y1_color:          Optional[str]  = None
    y1_width:          float          = 1.2
    y1_opacity:        float          = 0.45
    # Y=0 baseline
    show_baseline:     bool           = True
    baseline_color:    Optional[str]  = None
    baseline_width:    float          = 1.2
    baseline_opacity:  float          = 0.45


@dataclass
class StepConfig:
    """Style for discrete staircase CDF."""
    step_color:          Optional[str]  = None   # horizontal segments
    step_width:          float          = 2.8
    step_opacity:        float          = 1.0
    jump_style:          StepJumpStyle  = StepJumpStyle.DASHED
    jump_color:          Optional[str]  = None
    jump_width:          float          = 1.4
    jump_opacity:        float          = 0.60
    open_dot_radius:     float          = 0.055  # bottom of jump (exclusive)
    closed_dot_radius:   float          = 0.065  # top of jump (inclusive)
    open_dot_color:      Optional[str]  = None
    closed_dot_color:    Optional[str]  = None
    show_step_labels:    bool           = True   # cumulative prob labels
    step_label_font_size: float         = 18
    step_label_decimals: int            = 3
    step_label_offset:   float          = 0.28   # right of each step
    show_increment_labels: bool         = False  # P(X=k) increment on jump
    increment_font_size:   float        = 15


@dataclass
class ReadoutConfig:
    """Style for the L-shaped x↔p readout probe."""
    x_line_color:      Optional[str]   = None   # None → theme.accent
    p_line_color:      Optional[str]   = None   # None → theme.secondary
    line_width:        float           = 1.8
    line_opacity:      float           = 0.85
    dot_radius:        float           = 0.07
    dot_color:         Optional[str]   = None
    show_x_label:      bool            = True
    show_p_label:      bool            = True
    label_font_size:   float           = 22
    label_decimals:    int             = 4
    show_coordinates:  bool            = True   # "(x, F(x))" floating label
    coordinate_font:   float           = 20


@dataclass
class ComparisonConfig:
    """Style for multi-CDF overlay and KS statistic."""
    colors:            Optional[List[str]] = None  # per-CDF colours
    show_ks_stat:      bool               = True
    ks_arrow_color:    Optional[str]      = None
    ks_label_font:     float              = 22
    show_ks_region:    bool               = True   # shaded area between CDFs
    ks_region_opacity: float              = 0.20
    line_width:        float              = 2.5
    line_opacity:      float              = 0.80


@dataclass
class QuantileFnConfig:
    """Style for the inverse CDF (quantile function) curve."""
    color:             Optional[str]  = None   # None → theme.secondary
    stroke_width:      float          = 2.5
    stroke_opacity:    float          = 0.80
    show_diagonal:     bool           = True   # y = x reference line
    diagonal_color:    Optional[str]  = None
    diagonal_opacity:  float          = 0.30
    label:             str            = r"F^{-1}(p)"


@dataclass
class ECDFConfig:
    """Style for the empirical CDF overlay."""
    color:             Optional[str]  = None   # None → theme.positive
    stroke_width:      float          = 2.0
    stroke_opacity:    float          = 0.75
    show_dots:         bool           = True
    dot_radius:        float          = 0.045
    label:             str            = r"\hat{F}_n(x)"
    label_font_size:   float          = 20


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CDF CURVE LAYER  (continuous)
# ─────────────────────────────────────────────────────────────────────────────

class CDFCurveLayer(VGroup):
    """
    Smooth monotone CDF curve for a continuous distribution.
    Builds a dense sequence of Line3D segments from left to right,
    a filled polygon below, and horizontal reference lines at y=0 and y=1.
    """

    def __init__(
        self,
        axes:   StatsAxes3D,
        result: DistributionResult,
        cfg:    CDFCurveConfig,
        theme:  StatsColorPalette,
        mode:   CDFDisplayMode = CDFDisplayMode.CDF,
    ) -> None:
        super().__init__()
        self._axes  = axes
        self._theme = theme
        self._cfg   = cfg
        self._build(result, mode)

    def _build(
        self,
        result: DistributionResult,
        mode:   CDFDisplayMode,
    ) -> None:
        cfg   = self._cfg
        t     = self._theme
        axes  = self._axes
        c     = cfg.color or t.primary

        x  = result.x
        y  = self._select_y(result, mode)

        # ── main curve ────────────────────────────────────────────────────
        valid = np.isfinite(y)
        xv, yv = x[valid], y[valid]

        if len(xv) >= 2:
            pts = [axes.c2p(float(xi), float(yi)) for xi, yi in zip(xv, yv)]
            for i in range(len(pts) - 1):
                seg = Line3D(pts[i], pts[i + 1],
                             color=c,
                             thickness=cfg.stroke_width * 0.006)
                seg.set_opacity(cfg.stroke_opacity)
                self.add(seg)

            # ── fill polygon ──────────────────────────────────────────────
            if cfg.show_fill:
                fc = cfg.fill_color or c
                pts_bot = [
                    axes.c2p(float(xv[-1]), 0.0),
                    axes.c2p(float(xv[0]),  0.0),
                ]
                poly = Polygon(*(pts + pts_bot))
                poly.set_fill(fc, opacity=cfg.fill_opacity)
                poly.set_stroke(width=0)
                self.add(poly)

        # ── y = 1 reference line ──────────────────────────────────────────
        if cfg.show_y1_line and mode in (CDFDisplayMode.CDF,
                                          CDFDisplayMode.SURVIVAL,
                                          CDFDisplayMode.BOTH):
            y_ref = 1.0
            xlo, xhi = axes.x_range[:2]
            rc   = cfg.y1_color or t.neutral
            rl   = DashedLine(
                axes.c2p(xlo, y_ref), axes.c2p(xhi, y_ref),
                color=rc, stroke_width=cfg.y1_width * 1.3,
                dash_length=0.12)
            rl.set_opacity(cfg.y1_opacity)
            self.add(rl)
            lbl = MathTex("1", font_size=18, color=rc)
            lbl.move_to(axes.c2p(xlo - 0.25, y_ref))
            self.add(lbl)

        # ── baseline ──────────────────────────────────────────────────────
        if cfg.show_baseline:
            bc  = cfg.baseline_color or t.neutral
            xlo, xhi = axes.x_range[:2]
            bl  = Line3D(
                axes.c2p(xlo, 0.0), axes.c2p(xhi, 0.0),
                color=bc, thickness=cfg.baseline_width * 0.005)
            bl.set_opacity(cfg.baseline_opacity)
            self.add(bl)

    @staticmethod
    def _select_y(
        result: DistributionResult,
        mode:   CDFDisplayMode,
    ) -> FloatArray:
        if mode == CDFDisplayMode.CDF:
            return result.cdf if result.cdf is not None \
                   else np.zeros_like(result.x)
        elif mode == CDFDisplayMode.SURVIVAL:
            return result.sf if result.sf is not None \
                   else np.zeros_like(result.x)
        elif mode == CDFDisplayMode.LOG_CDF:
            cdf = np.clip(result.cdf, 1e-300, 1.0) \
                  if result.cdf is not None else np.ones_like(result.x)
            return np.log(cdf)
        elif mode == CDFDisplayMode.LOG_SF:
            sf = np.clip(result.sf, 1e-300, 1.0) \
                 if result.sf is not None else np.ones_like(result.x)
            return np.log(sf)
        return result.cdf if result.cdf is not None \
               else np.zeros_like(result.x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STEP FUNCTION 3D  (discrete CDF)
# ─────────────────────────────────────────────────────────────────────────────

class StepFunction3D(VGroup):
    """
    Staircase step-function CDF for discrete distributions.

    Anatomy of one step at x=k with cumulative probability F(k):

        ╌╌╌╌●──────────        ← horizontal segment at height F(k)
             ↑                    closed dot (k is included in ≤k)
        ╌╌╌╌○                  ← open dot at F(k-1) (k not yet reached)
             |
        ─────────               ← previous step at F(k-1)

    The open/closed convention follows the standard CÀDLÀG (right-continuous)
    CDF definition: F(k) = P(X ≤ k).
    """

    def __init__(
        self,
        axes:     StatsAxes3D,
        result:   DistributionResult,
        cfg:      StepConfig,
        theme:    StatsColorPalette,
        x_lo:     Optional[float] = None,
        x_hi:     Optional[float] = None,
    ) -> None:
        super().__init__()
        self._axes  = axes
        self._theme = theme
        self._cfg   = cfg
        self._build(result, x_lo, x_hi)

    def _build(
        self,
        result:  DistributionResult,
        x_lo:    Optional[float],
        x_hi:    Optional[float],
    ) -> None:
        cfg   = self._cfg
        t     = self._theme
        axes  = self._axes

        sc  = cfg.step_color    or t.primary
        jc  = cfg.jump_color    or t.primary
        odc = cfg.open_dot_color   or t.surface
        cdc = cfg.closed_dot_color or t.primary

        # Build integer k sequence and corresponding F(k) values
        x_lo = x_lo if x_lo is not None else axes.x_range[0]
        x_hi = x_hi if x_hi is not None else axes.x_range[1]

        k_vals = np.round(result.x).astype(int)
        seen   = {}
        cdf_v  = result.cdf if result.cdf is not None else np.zeros_like(result.x)
        for k, fk in zip(k_vals, cdf_v):
            ki = int(k)
            if ki not in seen:
                seen[ki] = float(fk)

        k_sorted = sorted(seen.keys())
        # Filter to visible range
        k_sorted = [k for k in k_sorted
                    if x_lo - 1 <= k <= x_hi + 1]

        if not k_sorted:
            return

        # ── segments ──────────────────────────────────────────────────────
        # Step before first k: F = 0 from x_lo to k_sorted[0]
        k_first = k_sorted[0]
        p_before = axes.c2p(x_lo, 0.0)
        p_step0  = axes.c2p(float(k_first), 0.0)
        pre_seg  = Line3D(p_before, p_step0, color=sc,
                          thickness=cfg.step_width * 0.006)
        pre_seg.set_opacity(cfg.step_opacity)
        self.add(pre_seg)

        for idx, k in enumerate(k_sorted):
            fk      = seen[k]
            fk_prev = seen[k_sorted[idx - 1]] if idx > 0 else 0.0
            delta   = fk - fk_prev

            # Horizontal segment: from x=k to x=k+1 (or next k) at height F(k)
            next_k  = k_sorted[idx + 1] if idx + 1 < len(k_sorted) else int(x_hi) + 1
            p_seg_l = axes.c2p(float(k),      fk)
            p_seg_r = axes.c2p(float(next_k), fk)
            seg     = Line3D(p_seg_l, p_seg_r, color=sc,
                             thickness=cfg.step_width * 0.006)
            seg.set_opacity(cfg.step_opacity)
            self.add(seg)

            # Vertical jump from F(k-1) to F(k)
            if idx > 0 or fk_prev > 0:
                p_jump_bot = axes.c2p(float(k), fk_prev)
                p_jump_top = axes.c2p(float(k), fk)

                if cfg.jump_style == StepJumpStyle.DASHED:
                    jump = DashedLine(p_jump_bot, p_jump_top,
                                      color=jc,
                                      stroke_width=cfg.jump_width * 1.3,
                                      dash_length=0.06)
                elif cfg.jump_style == StepJumpStyle.ARROW:
                    jump = Arrow3D(p_jump_bot, p_jump_top, color=jc,
                                   thickness=cfg.jump_width * 0.004,
                                   tip_length=0.10)
                elif cfg.jump_style == StepJumpStyle.SOLID:
                    jump = Line3D(p_jump_bot, p_jump_top, color=jc,
                                  thickness=cfg.jump_width * 0.005)
                else:
                    jump = None

                if jump is not None:
                    jump.set_opacity(cfg.jump_opacity)
                    self.add(jump)

            # Open dot at bottom of jump (F(k-1), exclusive)
            if fk_prev >= 0 and idx >= 0:
                p_open = axes.c2p(float(k), fk_prev)
                open_dot = Dot3D(p_open,
                                  radius=cfg.open_dot_radius,
                                  color=sc)
                # Open = fill with background colour, stroke with step colour
                open_dot.set_color(odc)
                open_dot.set_opacity(0.95)
                # Stroke ring to make it look open
                ring = Dot3D(p_open,
                              radius=cfg.open_dot_radius * 1.35,
                              color=sc)
                ring.set_opacity(0.70)
                self.add(ring, open_dot)

            # Closed dot at top of jump (F(k), inclusive)
            p_closed = axes.c2p(float(k), fk)
            closed_dot = Dot3D(p_closed,
                                radius=cfg.closed_dot_radius,
                                color=cdc)
            closed_dot.set_opacity(0.92)
            self.add(closed_dot)

            # Step label: "F(k) = 0.xxx"
            if cfg.show_step_labels:
                lbl_str = format_stat_value(fk, decimals=cfg.step_label_decimals)
                lbl     = Text(lbl_str,
                               font_size=cfg.step_label_font_size,
                               color=t.text_secondary)
                lbl.move_to(p_seg_r + RIGHT * cfg.step_label_offset
                             + UP * 0.12)
                self.add(lbl)

            # Increment label: "+P(X=k)" on the jump
            if cfg.show_increment_labels and delta > 1e-6:
                pmf_str = f"+{delta:.{cfg.step_label_decimals}f}"
                inc_lbl = Text(pmf_str,
                               font_size=cfg.increment_font_size,
                               color=jc)
                mid_jump = (axes.c2p(float(k), fk_prev) +
                            axes.c2p(float(k), fk)) / 2
                inc_lbl.move_to(mid_jump + RIGHT * 0.35)
                self.add(inc_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CDF ANNOTATION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PLevelConfig:
    """One horizontal p-level annotation line."""
    p:               float
    label:           str   = ""
    color:           Optional[str] = None
    line_width:      float = 1.5
    line_opacity:    float = 0.80
    show_x_bracket:  bool  = True     # drop to x-axis and bracket the quantile
    bracket_color:   Optional[str] = None
    label_font_size: float = 20


class CDFAnnotationSystem(VGroup):
    """
    Horizontal guide lines at p = 0.25, 0.50, 0.75 (quartiles) and any
    user-specified p levels.  Each guide:
        • Horizontal dashed line from y-axis across to the CDF curve
        • Vertical drop from the curve intersection down to the x-axis
        • Bracket label showing the quantile value (x such that F(x) = p)
        • "p = 0.xx" label on the y-axis
    """

    def __init__(
        self,
        axes:     StatsAxes3D,
        result:   DistributionResult,
        dist_fn:  DistributionFunction,
        theme:    StatsColorPalette,
        p_levels: Optional[List[PLevelConfig]] = None,
        show_quartiles: bool = True,
    ) -> None:
        super().__init__()
        self._axes   = axes
        self._theme  = theme
        self._result = result
        self._fn     = dist_fn
        self._annotations: Dict[str, VGroup] = {}

        # Default quartile set
        if show_quartiles:
            default_levels = [
                PLevelConfig(0.25, "Q_1",  color=theme.distribution_palette[2]),
                PLevelConfig(0.50, "Q_2",  color=theme.accent),
                PLevelConfig(0.75, "Q_3",  color=theme.distribution_palette[4]),
            ]
        else:
            default_levels = []

        all_levels = default_levels + (p_levels or [])
        for plc in all_levels:
            grp = self._build_p_level(plc)
            key = f"p_{plc.p:.4f}"
            self._annotations[key] = grp
            self.add(grp)

    # ── public API ────────────────────────────────────────────────────────

    def add_p_level(self, key: str, cfg: PLevelConfig) -> VGroup:
        if key in self._annotations:
            self.remove(self._annotations[key])
        grp = self._build_p_level(cfg)
        self._annotations[key] = grp
        self.add(grp)
        return grp

    def remove_p_level(self, key: str) -> None:
        if key in self._annotations:
            self.remove(self._annotations.pop(key))

    def clear_all(self) -> None:
        for k in list(self._annotations):
            self.remove_p_level(k)

    # ── internal ──────────────────────────────────────────────────────────

    def _build_p_level(self, cfg: PLevelConfig) -> VGroup:
        t    = self._theme
        axes = self._axes
        c    = cfg.color or t.text_secondary
        bc   = cfg.bracket_color or c

        # Find quantile x such that F(x) = p
        try:
            x_q = float(self._fn.ppf(cfg.p))
        except Exception:
            x_q = float(np.interp(
                cfg.p,
                self._result.cdf if self._result.cdf is not None
                else np.linspace(0, 1, len(self._result.x)),
                self._result.x))

        xlo = axes.x_range[0]
        yhi = axes.y_range[1] if axes.y_range[1] <= 1.0 else 1.0

        grp = VGroup()

        # Horizontal guide: from y-axis to curve intersection
        h_line = DashedLine(
            axes.c2p(xlo, cfg.p),
            axes.c2p(x_q, cfg.p),
            color=c,
            stroke_width=cfg.line_width * 1.3,
            dash_length=0.11,
        )
        h_line.set_opacity(cfg.line_opacity)
        grp.add(h_line)

        # p-label on y-axis
        p_lbl_str = cfg.label or f"p={cfg.p:.2f}"
        p_lbl = MathTex(p_lbl_str, font_size=cfg.label_font_size, color=c)
        p_lbl.move_to(axes.c2p(xlo - 0.45, cfg.p))
        grp.add(p_lbl)

        # Vertical drop from (x_q, p) to (x_q, 0)
        v_drop = DashedLine(
            axes.c2p(x_q, cfg.p),
            axes.c2p(x_q, 0.0),
            color=bc,
            stroke_width=cfg.line_width * 1.1,
            dash_length=0.08,
        )
        v_drop.set_opacity(cfg.line_opacity * 0.75)
        grp.add(v_drop)

        # Dot at the intersection on the CDF curve
        curve_dot = Dot3D(axes.c2p(x_q, cfg.p),
                           radius=0.065, color=c)
        curve_dot.set_opacity(0.90)
        grp.add(curve_dot)

        # Bracket label: quantile value below x-axis
        if cfg.show_x_bracket:
            x_str  = format_stat_value(x_q, decimals=3)
            x_lbl  = MathTex(x_str, font_size=cfg.label_font_size - 2,
                              color=bc)
            x_lbl.move_to(axes.c2p(x_q, 0.0) + DOWN * 0.40)
            grp.add(x_lbl)

            # Tiny tick on x-axis
            tick_bot = axes.c2p(x_q, 0.0) + DOWN * 0.10
            tick_top = axes.c2p(x_q, 0.0)
            tick = Line3D(tick_bot, tick_top, color=bc,
                          thickness=0.007)
            grp.add(tick)

        return grp


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PROBABILITY READOUT 3D
# ─────────────────────────────────────────────────────────────────────────────

class ProbabilityReadout3D(VGroup):
    """
    An L-shaped live readout probe on the CDF.

    Two perpendicular lines meet at the curve:
        • Vertical line from x_tracker value on x-axis up to the curve
        • Horizontal line from the curve across to the y-axis
        • Dot at the intersection point on the curve
        • Floating coordinate label "(x, F(x))"

    Both ``x_tracker`` and ``p_tracker`` are public ValueTrackers.
    Animating ``x_tracker`` sweeps the probe along the x-axis;
    animating ``p_tracker`` sweeps along the y-axis (finds quantile).

    The probe is built as an ``always_redraw`` VGroup so it live-updates
    when the tracker values change during a Manim scene.
    """

    def __init__(
        self,
        axes:     StatsAxes3D,
        result:   DistributionResult,
        dist_fn:  DistributionFunction,
        cfg:      ReadoutConfig,
        theme:    StatsColorPalette,
        x_init:   float = 0.0,
    ) -> None:
        super().__init__()
        self._axes   = axes
        self._result = result
        self._fn     = dist_fn
        self._cfg    = cfg
        self._theme  = theme

        # Public ValueTrackers
        self.x_tracker = ValueTracker(x_init)
        self.p_tracker = ValueTracker(
            float(np.interp(x_init, result.x,
                            result.cdf if result.cdf is not None
                            else np.zeros_like(result.x))))

        # Build live-redraw probe
        self._probe = always_redraw(self._make_probe_from_x)
        self.add(self._probe)

    # ── redraw callback ───────────────────────────────────────────────────

    def _make_probe_from_x(self) -> VGroup:
        cfg  = self._cfg
        t    = self._theme
        axes = self._axes
        xc   = cfg.x_line_color or t.accent
        pc   = cfg.p_line_color or t.secondary
        dc   = cfg.dot_color    or t.accent

        x_val = float(self.x_tracker.get_value())
        f_val = float(np.interp(
            x_val, self._result.x,
            self._result.cdf if self._result.cdf is not None
            else np.zeros_like(self._result.x)))
        # Clamp to [0, 1]
        f_val = float(np.clip(f_val, 0.0, 1.0))

        xlo   = axes.x_range[0]
        grp   = VGroup()

        # Vertical line: (x_val, 0) → (x_val, F(x))
        v_line = DashedLine(
            axes.c2p(x_val, 0.0),
            axes.c2p(x_val, f_val),
            color=xc,
            stroke_width=cfg.line_width * 1.3,
            dash_length=0.09,
        )
        v_line.set_opacity(cfg.line_opacity)
        grp.add(v_line)

        # Horizontal line: (xlo, F(x)) → (x_val, F(x))
        h_line = DashedLine(
            axes.c2p(xlo, f_val),
            axes.c2p(x_val, f_val),
            color=pc,
            stroke_width=cfg.line_width * 1.1,
            dash_length=0.09,
        )
        h_line.set_opacity(cfg.line_opacity)
        grp.add(h_line)

        # Dot at intersection
        dot = Dot3D(axes.c2p(x_val, f_val),
                    radius=cfg.dot_radius, color=dc)
        dot.set_opacity(0.95)
        grp.add(dot)

        # x label below x-axis
        if cfg.show_x_label:
            x_str = format_stat_value(x_val, decimals=cfg.label_decimals)
            x_lbl = Text(f"x={x_str}",
                          font_size=cfg.label_font_size, color=xc)
            x_lbl.move_to(axes.c2p(x_val, 0.0) + DOWN * 0.42)
            grp.add(x_lbl)

        # p label on y-axis
        if cfg.show_p_label:
            p_str = format_stat_value(f_val, decimals=cfg.label_decimals)
            p_lbl = Text(f"F={p_str}",
                          font_size=cfg.label_font_size, color=pc)
            p_lbl.move_to(axes.c2p(xlo - 0.55, f_val))
            grp.add(p_lbl)

        # Coordinate floating label
        if cfg.show_coordinates:
            coord_str = (f"({format_stat_value(x_val, 3)},\\ "
                         f"{format_stat_value(f_val, 3)})")
            coord_lbl = MathTex(coord_str,
                                 font_size=cfg.coordinate_font,
                                 color=t.text_primary)
            bg = RoundedRectangle(
                corner_radius=0.09,
                width=coord_lbl.width + 0.25,
                height=coord_lbl.height + 0.18,
            )
            bg.set_fill(t.surface, opacity=0.82)
            bg.set_stroke(t.border, width=0.9)
            bg.move_to(coord_lbl)
            coord_grp = VGroup(bg, coord_lbl)
            coord_grp.move_to(axes.c2p(x_val, f_val) +
                               RIGHT * 0.80 + UP * 0.35)
            grp.add(coord_grp)

        return grp

    # ── sweep animations ──────────────────────────────────────────────────

    def sweep_x(
        self,
        x_start:  float,
        x_end:    float,
        run_time: float = 3.0,
        rate_func: Callable = smooth,
    ) -> Animation:
        """Animate the probe sweeping from x_start to x_end."""
        self.x_tracker.set_value(x_start)
        return self.x_tracker.animate(
    run_time=run_time,
    rate_func=rate_func,
).set_value(x_end)

    def sweep_p(
        self,
        p_start:  float,
        p_end:    float,
        run_time: float = 3.0,
        rate_func: Callable = smooth,
    ) -> Animation:
        """
        Animate the probe by sweeping p (probability level) from
        p_start to p_end.  Internally converts p → x via PPF.
        """
        x_start = float(self._fn.ppf(p_start))
        x_end   = float(self._fn.ppf(p_end))
        return self.sweep_x(x_start, x_end, run_time, rate_func)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CDF COMPARISON LAYER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CDFPair:
    """One CDF to be compared in a ComparisonLayer."""
    result:  DistributionResult
    dist_fn: DistributionFunction
    label:   str = ""
    color:   Optional[str] = None


class CDFComparisonLayer(VGroup):
    """
    Overlays multiple CDFs on the same axes and optionally shows
    the Kolmogorov-Smirnov statistic D = max|F₁(x) - F₂(x)|.

    The layer handles:
        • Drawing each CDF as a coloured curve
        • Shading the region between the first two CDFs
        • A double-headed arrow at the point of maximum divergence
        • A floating panel showing D and its interpretation
    """

    def __init__(
        self,
        axes:    StatsAxes3D,
        pairs:   List[CDFPair],
        cfg:     ComparisonConfig,
        theme:   StatsColorPalette,
    ) -> None:
        super().__init__()
        if not pairs:
            return
        self._axes  = axes
        self._theme = theme
        self._cfg   = cfg

        colors = cfg.colors or [
            theme.primary, theme.secondary,
            theme.positive, theme.accent,
        ]

        # Draw each CDF curve
        for i, pair in enumerate(pairs):
            c   = pair.color or colors[i % len(colors)]
            ccfg = CDFCurveConfig(
                color=c, stroke_width=cfg.line_width,
                stroke_opacity=cfg.line_opacity,
                show_fill=False, show_y1_line=(i == 0),
                show_baseline=(i == 0),
            )
            layer = CDFCurveLayer(axes, pair.result, ccfg, theme)
            self.add(layer)
            # Curve label at the right edge
            if pair.label and pair.result.cdf is not None:
                x_right = axes.x_range[1]
                f_right = float(np.interp(
                    x_right, pair.result.x, pair.result.cdf))
                lbl = MathTex(pair.label, font_size=20, color=c)
                lbl.move_to(axes.c2p(x_right + 0.15, f_right))
                self.add(lbl)

        # KS statistic between first two CDFs
        if cfg.show_ks_stat and len(pairs) >= 2:
            self._add_ks_layer(pairs[0], pairs[1], colors[0], colors[1])

    # ── KS layer ──────────────────────────────────────────────────────────

    def _add_ks_layer(
        self,
        pair_a:  CDFPair,
        pair_b:  CDFPair,
        color_a: str,
        color_b: str,
    ) -> None:
        cfg  = self._cfg
        t    = self._theme
        axes = self._axes

        if pair_a.result.cdf is None or pair_b.result.cdf is None:
            return

        # Common x grid
        x_common = np.linspace(
            max(pair_a.result.x.min(), pair_b.result.x.min()),
            min(pair_a.result.x.max(), pair_b.result.x.max()),
            1000)
        fa = np.interp(x_common, pair_a.result.x, pair_a.result.cdf)
        fb = np.interp(x_common, pair_b.result.x, pair_b.result.cdf)
        diff = np.abs(fa - fb)
        idx  = int(np.argmax(diff))
        D    = float(diff[idx])
        x_D  = float(x_common[idx])
        fa_D = float(fa[idx])
        fb_D = float(fb[idx])

        # Shaded region between CDFs
        if cfg.show_ks_region:
            pts_a = [axes.c2p(float(xi), float(fi))
                     for xi, fi in zip(x_common, fa)]
            pts_b = [axes.c2p(float(xi), float(fi))
                     for xi, fi in reversed(list(zip(x_common, fb)))]
            poly = Polygon(*(pts_a + pts_b))
            poly.set_fill(t.accent, opacity=cfg.ks_region_opacity)
            poly.set_stroke(width=0)
            self.add(poly)

        # Double-headed arrow at maximum divergence
        ac  = cfg.ks_arrow_color or t.accent
        arr = Arrow3D(
            axes.c2p(x_D, min(fa_D, fb_D)),
            axes.c2p(x_D, max(fa_D, fb_D)),
            color=ac,
            thickness=0.014,
            tip_length=0.12,
        )
        self.add(arr)
        # Second arrowhead (reverse direction)
        arr2 = Arrow3D(
            axes.c2p(x_D, max(fa_D, fb_D)),
            axes.c2p(x_D, min(fa_D, fb_D)),
            color=ac, thickness=0.014, tip_length=0.12)
        self.add(arr2)

        # KS label
        ks_lbl = MathTex(
            rf"D = {D:.4f}",
            font_size=cfg.ks_label_font, color=ac)
        ks_lbl.move_to(axes.c2p(x_D, (fa_D + fb_D) / 2) +
                        RIGHT * 0.75)
        bg = RoundedRectangle(
            corner_radius=0.10,
            width=ks_lbl.width + 0.28,
            height=ks_lbl.height + 0.20)
        bg.set_fill(t.surface, opacity=0.82)
        bg.set_stroke(t.border, width=1.0)
        bg.move_to(ks_lbl)
        self.add(VGroup(bg, ks_lbl))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUANTILE FUNCTION 3D
# ─────────────────────────────────────────────────────────────────────────────

class QuantileFunction3D(VGroup):
    """
    The inverse CDF (quantile function / PPF) drawn as a curve on axes
    where x-axis = probability p ∈ [0,1] and y-axis = quantile value.

    Shows:
        • The PPF curve Q(p) = F⁻¹(p)
        • Optional y=x diagonal reference line
        • Tick marks at Q(0.25), Q(0.50), Q(0.75)
        • Relationship annotation: "Q(p) is the p-th quantile"
    """

    def __init__(
        self,
        axes:    StatsAxes3D,
        result:  DistributionResult,
        dist_fn: DistributionFunction,
        cfg:     QuantileFnConfig,
        theme:   StatsColorPalette,
    ) -> None:
        super().__init__()
        self._axes  = axes
        self._theme = theme
        self._build(result, dist_fn, cfg)

    def _build(
        self,
        result:  DistributionResult,
        dist_fn: DistributionFunction,
        cfg:     QuantileFnConfig,
    ) -> None:
        t    = self._theme
        axes = self._axes
        c    = cfg.color or t.secondary

        # p grid for Q(p)
        p_vals = np.linspace(0.001, 0.999, 300)
        q_vals = dist_fn.ppf(p_vals)

        # Map p → x-axis, Q(p) → y-axis
        # Here we reuse axes: x-axis = p ∈ [0,1], y-axis = Q(p) values
        pts = []
        for p, q in zip(p_vals, q_vals):
            if math.isfinite(q):
                pts.append(axes.c2p(float(p), float(q)))

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                seg = Line3D(pts[i], pts[i + 1],
                             color=c,
                             thickness=cfg.stroke_width * 0.006)
                seg.set_opacity(cfg.stroke_opacity)
                self.add(seg)

        # Diagonal y = x reference (only meaningful when axes are square)
        if cfg.show_diagonal:
            dc = cfg.diagonal_color or t.neutral
            xlo, xhi = axes.x_range[:2]
            diag = DashedLine(
                axes.c2p(xlo, xlo),
                axes.c2p(xhi, xhi),
                color=dc,
                stroke_width=1.2,
                dash_length=0.14,
            )
            diag.set_opacity(cfg.diagonal_opacity)
            self.add(diag)

        # Quartile tick marks
        for pq, label in [(0.25, "Q_1"), (0.50, "Q_2"), (0.75, "Q_3")]:
            try:
                qv = float(dist_fn.ppf(pq))
                if math.isfinite(qv):
                    dot = Dot3D(axes.c2p(pq, qv),
                                radius=0.06, color=t.accent)
                    dot.set_opacity(0.85)
                    self.add(dot)
                    lbl = MathTex(label, font_size=16, color=t.accent)
                    lbl.move_to(axes.c2p(pq, qv) + UP * 0.25 + RIGHT * 0.10)
                    self.add(lbl)
            except Exception:
                pass

        # Curve label
        if cfg.label and pts:
            lbl = MathTex(cfg.label, font_size=20, color=c)
            lbl.move_to(pts[len(pts) // 4] + UP * 0.35)
            self.add(lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CDF DECOMPOSITION PANEL  (discrete only)
# ─────────────────────────────────────────────────────────────────────────────

class CDFDecompositionPanel(VGroup):
    """
    A floating panel (positioned beside the axes) that shows how the
    discrete CDF is built as a running sum of PMF values:

        F(k) = P(X≤k) = Σᵢ₌₀ᵏ P(X=i)

    Each term P(X=i) is shown as a small coloured bar inside the panel,
    stacked vertically.  As k increases, bars are highlighted one by one.
    The running total is shown beside each bar.

    This is a standalone VGroup — it does NOT live inside the axes.
    Place it wherever in the scene makes sense.
    """

    def __init__(
        self,
        result:       DistributionResult,
        theme:        StatsColorPalette,
        max_k:        int   = 10,
        panel_pos:    Optional[np.ndarray] = None,
        panel_width:  float = 2.8,
        bar_height:   float = 0.32,
        font_size:    float = 18,
    ) -> None:
        super().__init__()
        self._theme = theme
        pos = panel_pos if panel_pos is not None \
              else np.array([4.5, 0.0, 0.0])

        t   = theme
        pmf = result.pdf if result.pdf is not None \
              else np.zeros_like(result.x)
        x_vals = np.round(result.x).astype(int)

        # Collect unique (k, pmf_k) pairs up to max_k
        seen: Dict[int, float] = {}
        for k, pk in zip(x_vals, pmf):
            ki = int(k)
            if ki not in seen and ki >= 0:
                seen[ki] = float(pk)
        k_sorted = sorted(k for k in seen if k <= max_k)

        if not k_sorted:
            return

        # Background panel
        total_h = len(k_sorted) * (bar_height + 0.06) + 0.40
        bg = RoundedRectangle(
            corner_radius=0.14,
            width=panel_width,
            height=total_h,
        )
        bg.set_fill(t.surface, opacity=0.85)
        bg.set_stroke(t.border, width=1.2, opacity=0.55)
        bg.move_to(pos)
        self.add(bg)

        # Title
        title = MathTex(r"F(k)=\sum_{i=0}^{k}P(X=i)",
                        font_size=font_size + 2,
                        color=t.text_primary)
        title.move_to(pos + UP * (total_h / 2 - 0.28))
        self.add(title)

        # Cumulative running sum
        running = 0.0
        palette = t.distribution_palette
        y_start = total_h / 2 - 0.60

        for row_idx, k in enumerate(k_sorted):
            pk       = seen[k]
            running += pk
            c        = palette[k % len(palette)]

            # Bar representing P(X=k) contribution
            bar_w = max(0.02, pk * (panel_width - 1.4))
            bar   = Rectangle(width=bar_w, height=bar_height * 0.75)
            bar.set_fill(c, opacity=0.75)
            bar.set_stroke(c, width=0.8)
            bar_pos = pos + UP * (y_start - row_idx * (bar_height + 0.06))
            bar.move_to(bar_pos + LEFT * ((panel_width - 1.4 - bar_w) / 2)
                         + LEFT * 0.35)
            self.add(bar)

            # k label
            k_lbl = MathTex(f"k={k}:", font_size=font_size, color=c)
            k_lbl.move_to(bar_pos + LEFT * (panel_width / 2 - 0.28))
            self.add(k_lbl)

            # P(X=k) value
            pk_lbl = Text(f"{pk:.3f}", font_size=font_size - 2,
                           color=t.text_secondary)
            pk_lbl.move_to(bar_pos + RIGHT * 0.15)
            self.add(pk_lbl)

            # Running sum = F(k)
            fk_lbl = Text(f"→ {running:.3f}",
                           font_size=font_size - 2, color=t.accent)
            fk_lbl.move_to(bar_pos + RIGHT * (panel_width / 2 - 0.55))
            self.add(fk_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  EMPIRICAL CDF LAYER
# ─────────────────────────────────────────────────────────────────────────────

class ECDFLayer(VGroup):
    """
    Empirical CDF F̂ₙ(x) = (1/n) Σ 𝟙(Xᵢ ≤ x) drawn as a step function
    from raw data.

    Used to:
        • Overlay on a theoretical CDF to visually assess fit
        • Prepare for KS test visualisation (see CDFComparisonLayer)
        • Demonstrate CLT / LLN: as n grows, ECDF → theoretical CDF

    Steps are drawn as a right-continuous staircase (CÀDLÀG convention).
    Confidence band (Dvoretzky-Kiefer-Wolfowitz) can be shown at any α.
    """

    def __init__(
        self,
        axes:    StatsAxes3D,
        data:    ArrayLike,
        cfg:     ECDFConfig,
        theme:   StatsColorPalette,
        show_dkw_band: bool  = True,
        alpha:         float = 0.05,
    ) -> None:
        super().__init__()
        self._axes  = axes
        self._theme = theme
        x           = np.sort(np.asarray(data, float))
        n           = len(x)
        if n == 0:
            return

        c = cfg.color or theme.positive

        # Step values: F̂(x) = i/n after the i-th sorted point
        f_vals = np.arange(1, n + 1) / n

        # ── step segments ─────────────────────────────────────────────────
        xlo, xhi = axes.x_range[:2]

        # Flat zero before first observation
        seg0 = Line3D(
            axes.c2p(xlo, 0.0),
            axes.c2p(float(x[0]), 0.0),
            color=c, thickness=cfg.stroke_width * 0.006)
        seg0.set_opacity(cfg.stroke_opacity)
        self.add(seg0)

        for i in range(n):
            fi      = float(f_vals[i])
            fi_prev = float(f_vals[i - 1]) if i > 0 else 0.0
            xi      = float(x[i])
            xi_next = float(x[i + 1]) if i + 1 < n else xhi

            # Horizontal segment at height F̂(xᵢ)
            seg_h = Line3D(
                axes.c2p(xi,      fi),
                axes.c2p(xi_next, fi),
                color=c, thickness=cfg.stroke_width * 0.006)
            seg_h.set_opacity(cfg.stroke_opacity)
            self.add(seg_h)

            # Vertical jump at xᵢ
            if fi_prev < fi:
                jump = Line3D(
                    axes.c2p(xi, fi_prev),
                    axes.c2p(xi, fi),
                    color=c, thickness=cfg.stroke_width * 0.004)
                jump.set_opacity(cfg.stroke_opacity * 0.55)
                self.add(jump)

            # Dots at jump tops
            if cfg.show_dots:
                dot = Dot3D(axes.c2p(xi, fi),
                            radius=cfg.dot_radius, color=c)
                dot.set_opacity(0.80)
                self.add(dot)

        # Label
        if cfg.label:
            mid_i = n // 2
            lbl   = MathTex(cfg.label, font_size=cfg.label_font_size,
                             color=c)
            lbl.move_to(axes.c2p(float(x[mid_i]),
                                  float(f_vals[mid_i])) +
                         RIGHT * 0.45 + UP * 0.20)
            self.add(lbl)

        # ── DKW confidence band ───────────────────────────────────────────
        if show_dkw_band and n >= 2:
            eps = math.sqrt(math.log(2.0 / alpha) / (2 * n))
            # Upper band
            x_band  = np.concatenate([[xlo], x, [xhi]])
            f_upper = np.concatenate([[eps], f_vals + eps, [1.0 + eps]])
            f_lower = np.concatenate([[max(0, -eps)],
                                       f_vals - eps,
                                       [1.0 - eps]])
            f_upper = np.clip(f_upper, 0.0, 1.0)
            f_lower = np.clip(f_lower, 0.0, 1.0)

            band_c = theme.positive
            for f_band in [f_upper, f_lower]:
                pts = [axes.c2p(float(xi), float(fi))
                       for xi, fi in zip(x_band, f_band)]
                for i in range(len(pts) - 1):
                    s = DashedLine(pts[i], pts[i + 1],
                                   color=band_c,
                                   stroke_width=0.9,
                                   dash_length=0.08)
                    s.set_opacity(0.40)
                    self.add(s)

            # Band label
            band_lbl = MathTex(
                rf"{int((1-alpha)*100)}\%\;\mathrm{{DKW\;band}}",
                font_size=16, color=band_c)
            band_lbl.move_to(axes.c2p(float(x[n // 3]),
                                       float(f_vals[n // 3]) + eps + 0.06))
            band_lbl.set_opacity(0.70)
            self.add(band_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SURVIVAL FUNCTION LAYER
# ─────────────────────────────────────────────────────────────────────────────

class SurvivalFunctionLayer(VGroup):
    """
    S(x) = 1 - F(x) drawn as a mirrored overlay on the same axes.

    Optional:
        • Joint display with F(x) — shows F + S = 1 at each x visually
        • A vertical bracket at one x value showing F(x) + S(x) = 1
        • Log-survival log(S(x)) mode for exponential / Weibull analyses
    """

    def __init__(
        self,
        axes:         StatsAxes3D,
        result:       DistributionResult,
        theme:        StatsColorPalette,
        color:        Optional[str]  = None,
        stroke_width: float          = 2.5,
        opacity:      float          = 0.75,
        log_scale:    bool           = False,
        show_sum_annotation: bool    = True,
        annotation_x: Optional[float] = None,
    ) -> None:
        super().__init__()
        t   = theme
        c   = color or t.secondary
        axes_ = axes

        if result.sf is None:
            return

        y = result.sf if not log_scale \
            else np.log(np.clip(result.sf, 1e-300, 1.0))

        valid = np.isfinite(y)
        xv, yv = result.x[valid], y[valid]

        if len(xv) < 2:
            return

        # S(x) curve segments
        pts = [axes_.c2p(float(xi), float(yi)) for xi, yi in zip(xv, yv)]
        for i in range(len(pts) - 1):
            seg = Line3D(pts[i], pts[i + 1], color=c,
                         thickness=stroke_width * 0.006)
            seg.set_opacity(opacity)
            self.add(seg)

        # Curve label
        label_str = r"S(x)=1-F(x)" if not log_scale \
                    else r"\log S(x)"
        lbl = MathTex(label_str, font_size=20, color=c)
        lbl.move_to(pts[len(pts) // 3] + UP * 0.30)
        self.add(lbl)

        # F(x) + S(x) = 1 bracket annotation
        if show_sum_annotation and result.cdf is not None:
            ann_x = annotation_x if annotation_x is not None \
                    else float(result.x[len(result.x) // 2])
            fx = float(np.interp(ann_x, result.x, result.cdf))
            sx = 1.0 - fx

            p_bot = axes_.c2p(ann_x + 0.18, 0.0)
            p_mid = axes_.c2p(ann_x + 0.18, fx)
            p_top = axes_.c2p(ann_x + 0.18, 1.0)

            # F(x) region marker
            seg_f = Line3D(p_bot, p_mid, color=t.primary,
                           thickness=0.014)
            seg_f.set_opacity(0.70)
            self.add(seg_f)

            # S(x) region marker
            seg_s = Line3D(p_mid, p_top, color=c, thickness=0.014)
            seg_s.set_opacity(0.70)
            self.add(seg_s)

            lbl_f = MathTex(f"F={fx:.2f}", font_size=16,
                             color=t.primary)
            lbl_f.move_to(p_mid + LEFT * 0.55 + DOWN * 0.12)
            lbl_s = MathTex(f"S={sx:.2f}", font_size=16, color=c)
            lbl_s.move_to(p_mid + LEFT * 0.55 + UP * 0.12)
            self.add(lbl_f, lbl_s)

            # "= 1" brace
            eq_lbl = MathTex(r"F+S=1", font_size=16, color=t.text_secondary)
            eq_lbl.move_to(p_top + RIGHT * 0.50 + DOWN * 0.5)
            self.add(eq_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  CDF VIZ 3D  (master class)
# ─────────────────────────────────────────────────────────────────────────────

class CDFViz3D(StatsObject3D):
    """
    Master CDF visualisation object.

    Owns all seven sub-layers and wires them together.  Works for both
    continuous and discrete DistributionFunction objects.

    Quick start
    -----------
    ::

        axes = StatsAxes3D.for_distribution(
            x_range=(-4, 4, 1), y_range=(0, 1.05, 0.2),
            x_label="x", y_label="F(x)")
        cdf = CDFViz3D(axes,
                       DistributionFunction.normal(0, 1),
                       mode=CDFDisplayMode.CDF)
        scene.add(axes, cdf)
        scene.play(cdf.animate_build())

    Probing
    -------
    ::

        scene.play(cdf.readout.sweep_x(-2.0, 2.0, run_time=4))

    Comparison
    ----------
    ::

        cdf.add_comparison([
            CDFPair(DistributionFunction.normal(0,1).evaluate(), ..., "N(0,1)"),
            CDFPair(DistributionFunction.student_t(5).evaluate(), ..., "t(5)"),
        ])

    Critical values
    ---------------
    ::

        cdf.mark_critical_value(alpha=0.05, tail="both")
        cdf.mark_ci_bounds(alpha=0.05)

    ECDF overlay
    ------------
    ::

        data = np.random.normal(0, 1, 200)
        cdf.add_ecdf(data)
    """

    def __init__(
        self,
        axes:           StatsAxes3D,
        dist_fn:        DistributionFunction,
        mode:           CDFDisplayMode         = CDFDisplayMode.CDF,
        discrete:       bool                   = False,
        curve_cfg:      Optional[CDFCurveConfig]       = None,
        step_cfg:       Optional[StepConfig]           = None,
        readout_cfg:    Optional[ReadoutConfig]         = None,
        show_readout:   bool                   = True,
        show_quartiles: bool                   = True,
        show_survival:  bool                   = False,
        show_quantile_fn: bool                 = False,
        show_decomp:    bool                   = False,  # discrete only
        decomp_max_k:   int                    = 10,
        **kwargs,
    ) -> None:
        self._axes      = axes
        self._dist_fn   = dist_fn
        self._mode      = mode
        self._discrete  = discrete
        self._curve_cfg = curve_cfg  or CDFCurveConfig()
        self._step_cfg  = step_cfg   or StepConfig()
        self._rdout_cfg = readout_cfg or ReadoutConfig()
        self._show_readout    = show_readout
        self._show_quartiles  = show_quartiles
        self._show_survival   = show_survival
        self._show_qfn        = show_quantile_fn
        self._show_decomp     = show_decomp
        self._decomp_max_k    = decomp_max_k

        # Evaluate distribution on axes x-grid
        xlo, xhi = axes.x_range[:2]
        n        = 600 if not discrete else 200
        self._result = dist_fn.evaluate(
            x=np.linspace(xlo, xhi, n))

        # Sub-layer handles (populated in _build_geometry)
        self.curve_layer:    Optional[CDFCurveLayer]        = None
        self.step_layer:     Optional[StepFunction3D]       = None
        self.annotation:     Optional[CDFAnnotationSystem]  = None
        self.readout:        Optional[ProbabilityReadout3D] = None
        self.comparison:     Optional[CDFComparisonLayer]   = None
        self.quantile_fn:    Optional[QuantileFunction3D]   = None
        self.decomp_panel:   Optional[CDFDecompositionPanel] = None
        self.survival_layer: Optional[SurvivalFunctionLayer] = None
        self.ecdf_layer:     Optional[ECDFLayer]             = None

        super().__init__(**kwargs)

    # ── geometry build ────────────────────────────────────────────────────

    def _build_geometry(self) -> None:
        t    = self._palette
        axes = self._axes
        res  = self._result
        mode = self._mode

        # ── primary curve or step ─────────────────────────────────────────
        if self._discrete:
            self.step_layer = StepFunction3D(
                axes, res, self._step_cfg, t)
            self.add(self.step_layer)
        else:
            self.curve_layer = CDFCurveLayer(
                axes, res, self._curve_cfg, t, mode)
            self.add(self.curve_layer)

        # ── quartile / p-level annotations ────────────────────────────────
        self.annotation = CDFAnnotationSystem(
            axes, res, self._dist_fn, t,
            show_quartiles=self._show_quartiles)
        self.add(self.annotation)

        # ── live readout probe ─────────────────────────────────────────────
        if self._show_readout:
            x_init = float(self._dist_fn.ppf(0.50))
            self.readout = ProbabilityReadout3D(
                axes, res, self._dist_fn,
                self._rdout_cfg, t, x_init=x_init)
            self.add(self.readout)

        # ── survival function overlay ──────────────────────────────────────
        if self._show_survival:
            self.survival_layer = SurvivalFunctionLayer(
                axes, res, t, show_sum_annotation=True)
            self.add(self.survival_layer)

        # ── quantile function curve ────────────────────────────────────────
        if self._show_qfn:
            self.quantile_fn = QuantileFunction3D(
                axes, res, self._dist_fn,
                QuantileFnConfig(), t)
            self.add(self.quantile_fn)

        # ── decomposition panel (discrete) ────────────────────────────────
        if self._show_decomp and self._discrete:
            self.decomp_panel = CDFDecompositionPanel(
                res, t, max_k=self._decomp_max_k,
                panel_pos=np.array([5.0, 0.5, 0.0]))
            self.add(self.decomp_panel)

    # ── public API ────────────────────────────────────────────────────────

    def add_comparison(
        self,
        pairs:  List[CDFPair],
        cfg:    Optional[ComparisonConfig] = None,
    ) -> CDFComparisonLayer:
        """Add a multi-CDF comparison layer."""
        if self.comparison is not None:
            self.remove(self.comparison)
        self.comparison = CDFComparisonLayer(
            self._axes, pairs, cfg or ComparisonConfig(), self._palette)
        self.add(self.comparison)
        return self.comparison

    def add_ecdf(
        self,
        data:          ArrayLike,
        cfg:           Optional[ECDFConfig] = None,
        show_dkw_band: bool                 = True,
        alpha:         float                = 0.05,
    ) -> ECDFLayer:
        """Overlay the empirical CDF of *data*."""
        if self.ecdf_layer is not None:
            self.remove(self.ecdf_layer)
        self.ecdf_layer = ECDFLayer(
            self._axes, data,
            cfg or ECDFConfig(), self._palette,
            show_dkw_band=show_dkw_band, alpha=alpha)
        self.add(self.ecdf_layer)
        return self.ecdf_layer

    def add_p_level(self, key: str, p: float, **kwargs) -> VGroup:
        """Add a horizontal p-level annotation line."""
        if self.annotation is None:
            return VGroup()
        cfg = PLevelConfig(p=p, **kwargs)
        return self.annotation.add_p_level(key, cfg)

    def mark_critical_value(
        self,
        alpha: float = 0.05,
        tail:  Literal["left", "right", "both"] = "both",
    ) -> VGroup:
        """
        Mark the critical value(s) x_c where F(x_c) = alpha or 1-alpha.
        Adds p-level annotations and shades the critical region(s).
        """
        grp = VGroup()
        t   = self._palette

        if tail in ("left", "both"):
            x_lo = float(self._dist_fn.ppf(alpha))
            grp.add(self.add_p_level(
                f"crit_lo_{alpha}",
                p=alpha,
                label=rf"\alpha={alpha}",
                color=t.negative,
            ))

        if tail in ("right", "both"):
            x_hi = float(self._dist_fn.ppf(1 - alpha))
            grp.add(self.add_p_level(
                f"crit_hi_{alpha}",
                p=1 - alpha,
                label=rf"1-\alpha={1-alpha}",
                color=t.negative,
            ))
        return grp

    def mark_ci_bounds(
        self,
        alpha: float = 0.05,
        color: Optional[str] = None,
    ) -> VGroup:
        """
        Mark both CI quantile bounds F⁻¹(α/2) and F⁻¹(1-α/2).
        Shades the central (1-α) region on the CDF.
        """
        t   = self._palette
        c   = color or t.positive
        lo  = float(self._dist_fn.ppf(alpha / 2))
        hi  = float(self._dist_fn.ppf(1 - alpha / 2))
        grp = VGroup(
            self.add_p_level(
                f"ci_lo_{alpha}", p=alpha / 2,
                label=rf"\alpha/2",
                color=c),
            self.add_p_level(
                f"ci_hi_{alpha}", p=1 - alpha / 2,
                label=rf"1-\alpha/2",
                color=c),
        )
        return grp

    # ── animations ────────────────────────────────────────────────────────

    def animate_build(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        """
        Layered build sequence:
          1. Baseline / y=1 reference line
          2. CDF curve grows left → right (or steps appear one by one)
          3. Fill fades in beneath curve
          4. Quartile annotations write in
          5. Readout probe materialises
        """
        cfg = cfg or self._anim_cfg
        t   = cfg.run_time

        primary_layer = self.step_layer if self._discrete \
                        else self.curve_layer

        if primary_layer is not None:
            curve_build = self._animate_curve_build(primary_layer, cfg)
        else:
            curve_build = FadeIn(VGroup(), run_time=0.01)

        ann_in  = FadeIn(self.annotation, run_time=t * 0.35,
                         rate_func=smooth) \
                  if self.annotation else FadeIn(VGroup(), run_time=0.01)
        probe_in = FadeIn(self.readout, run_time=t * 0.25,
                          rate_func=smooth) \
                   if self.readout else FadeIn(VGroup(), run_time=0.01)
        surv_in  = FadeIn(self.survival_layer, run_time=t * 0.30,
                          rate_func=smooth) \
                   if self.survival_layer else FadeIn(VGroup(), run_time=0.01)

        return Succession(
            curve_build,
            AnimationGroup(ann_in, probe_in, surv_in, lag_ratio=0.25),
        )

    def _animate_curve_build(
        self,
        layer: VGroup,
        cfg:   AnimationConfig,
    ) -> Animation:
        """
        For continuous: segments appear left → right in tight sequence.
        For discrete: each step + jump appears with a short delay.
        """
        segs = [m for m in layer.submobjects
                if isinstance(m, (Line3D, DashedLine))]
        if not segs:
            return FadeIn(layer, run_time=cfg.run_time * 0.5)

        if self._discrete:
            # Steps appear in groups of 3 (jump + horizontal + dot)
            return AnimationGroup(
                *[FadeIn(m, run_time=cfg.run_time * 0.7 / max(len(segs), 1))
                  for m in segs],
                lag_ratio=0.08,
            )
        else:
            # Continuous: Create each segment in sequence
            return AnimationGroup(
                *[Create(s, run_time=cfg.run_time * 0.55 / max(len(segs), 1))
                  for s in segs],
                lag_ratio=1.0 / max(len(segs), 1),
            )

    def animate_update(
        self, new_data: Any, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        old = self.copy()
        if isinstance(new_data, DistributionFunction):
            self._dist_fn = new_data
            self._result  = new_data.evaluate(
                x=np.linspace(*self._axes.x_range[:2], 600))
        self.submobjects.clear()
        self._build_geometry()
        return Transform(old, self,
                         run_time=cfg.run_time, rate_func=cfg.rate_func)

    def animate_highlight(
        self,
        style: HighlightStyle = HighlightStyle.GLOW,
        cfg:   Optional[AnimationConfig] = None,
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        target = self.curve_layer or self.step_layer or self
        return HighlightSystem.glow(target, cfg)

    def animate_exit(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return FadeOut(self, run_time=cfg.run_time * 0.6, rate_func=smooth)

    # ── mode switch ───────────────────────────────────────────────────────

    def set_mode(
        self, mode: CDFDisplayMode
    ) -> "CDFViz3D":
        """Switch display mode immediately."""
        self._mode = mode
        self.submobjects.clear()
        self._build_geometry()
        return self

    def animate_mode_switch(
        self,
        new_mode: CDFDisplayMode,
        cfg:      Optional[AnimationConfig] = None,
    ) -> Animation:
        cfg      = cfg or self._anim_cfg
        old_copy = self.copy()
        self.set_mode(new_mode)
        return Transform(old_copy, self,
                         run_time=cfg.run_time, rate_func=cfg.rate_func)

    # ── property accessors ────────────────────────────────────────────────

    @property
    def result(self) -> DistributionResult:
        return self._result

    @property
    def mode(self) -> CDFDisplayMode:
        return self._mode

    def __repr__(self) -> str:
        return (f"CDFViz3D(dist={self._result.name!r}, "
                f"mode={self._mode.name}, discrete={self._discrete})")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enumerations
    "CDFDisplayMode",
    "StepJumpStyle",
    # Config dataclasses
    "CDFCurveConfig",
    "StepConfig",
    "ReadoutConfig",
    "ComparisonConfig",
    "QuantileFnConfig",
    "ECDFConfig",
    "PLevelConfig",
    "CDFPair",
    # Sub-layers
    "CDFCurveLayer",
    "StepFunction3D",
    "CDFAnnotationSystem",
    "ProbabilityReadout3D",
    "CDFComparisonLayer",
    "QuantileFunction3D",
    "CDFDecompositionPanel",
    "ECDFLayer",
    "SurvivalFunctionLayer",
    # Master class
    "CDFViz3D",
]