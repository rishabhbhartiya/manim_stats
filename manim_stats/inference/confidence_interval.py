"""
manim_stats/inference/confidence_interval.py
============================================
ConfidenceInterval3D — Highly detailed, statistically rich confidence
interval visualisation for Manim statistics animations.

Primary use cases
-----------------
  Frequentist CI interpretation  — repeated sampling coverage demos
  Z-interval / t-interval        — known vs unknown σ
  CLT demonstration              — how CI width shrinks with n
  Capture probability            — does the CI contain the true parameter?
  Comparison of confidence levels — 90 / 95 / 99% side-by-side

Design goals
------------
Single CI object
  * Rounded gradient bar         — thick central bar with a bright centre
                                   fading to darker ends (probability mass
                                   feels denser near the point estimate).
  * End-cap brackets             — real ruler-bracket style: vertical serif
                                   with a tiny inward tick, not plain lines.
  * Uncertainty fog              — semi-transparent tapered "halo" region
                                   extending slightly beyond each end cap,
                                   visualising that the boundary is not a
                                   hard cliff.
  * Center estimate marker       — multi-layer: outer ring → filled disc →
                                   specular dot.  Two styles: round (x̄) and
                                   diamond (μ̂).
  * True-parameter line          — optional dashed vertical line for μ₀,
                                   coloured green/red based on capture.
  * Capture flash region         — when the CI captures the true value a
                                   subtle filled region between bar and axis
                                   pulses green; red otherwise.
  * Floating badge               — "95% CI" text badge with leader line and
                                   a small rounded rect background.
  * Number line                  — clean axis with tick marks, labels, and
                                   an arrow tip; the CI sits on this line.

CIStack
  * Vertical stack of N intervals  — each row is one CI from a repeated
                                     sample.  Green = captures μ, red = misses.
  * Coverage counter               — live "k / N captured (p%)" text badge
                                     that updates as CIs are added.
  * True-parameter column          — vertical dashed line running the full
                                     height of the stack.

Animations
----------
  BuildCI         — bar grows symmetrically from point estimate outward,
                    end caps materialise, badge fades in.
  RevealCapture   — true-param line drops from above, interval flashes
                    green (captured) or red (missed), badge updates.
  SweepCI         — CI slides horizontally along the number line,
                    showing how a different sample shifts the interval.
  NarrowCI        — CI shrinks from current width to a new (narrower)
                    width; models increasing n or lowering α.
  StackCIs        — animate N CIs dropping onto the stack one by one,
                    counter incrementing, coverage proportion converging.

Dependencies
------------
  manim (CE or GL), numpy, scipy (optional — for t-distribution quantiles)

Usage
-----
    from manim_stats.inference.confidence_interval import (
        ConfidenceInterval3D, CIStack, BuildCI, RevealCapture, StackCIs
    )

    class CoverageDemo(ThreeDScene):
        def construct(self):
            ax = NumberLine(x_range=[60, 100, 5])
            ci = ConfidenceInterval3D(
                center=80.0, half_width=6.0,
                true_param=82.0,
                confidence=0.95,
                axis=ax,
            )
            self.play(BuildCI(ci))
            self.play(RevealCapture(ci))
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Literal, List, Tuple

from manim import (
    VGroup,
    Rectangle, RoundedRectangle, Circle, Annulus, Dot,
    Line, DashedLine, Arrow, DoubleArrow,
    Polygon, Arc,
    Text, MathTex, Tex,
    NumberLine,
    Animation, Succession, AnimationGroup,
    ApplyMethod, FadeIn, FadeOut, GrowFromCenter,
    Flash, Indicate,
    Rotate,
    Create, Write, Uncreate,
    always_redraw,
    interpolate_color,
    color_to_rgb,
    WHITE, BLACK,
    GREY, GREY_A, GREY_B, GREY_C, GREY_D, LIGHT_GREY,
    RED,    RED_A,    RED_B,    RED_C,    RED_D,    RED_E,
    GREEN,  GREEN_A,  GREEN_B,  GREEN_C,  GREEN_D,  GREEN_E,
    BLUE,   BLUE_A,   BLUE_B,   BLUE_C,   BLUE_D,   BLUE_E,
    YELLOW, YELLOW_A, YELLOW_B, YELLOW_E,
    ORANGE, TEAL, TEAL_A, TEAL_B, TEAL_C,
    GOLD,   GOLD_A,   GOLD_B,   GOLD_C,   GOLD_D,
    MAROON, PURPLE_A,
    UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
    PI, TAU,
    rate_functions,
    ManimColor,
    ValueTracker,
)

# ──────────────────────────────────────────────────────────────────────────────
# Colour themes
# ──────────────────────────────────────────────────────────────────────────────

CI_THEMES: dict[str, dict] = {
    # Classic blue interval — neutral / unknown capture status
    "blue": {
        "bar_center":   "#4A90D9",
        "bar_edge":     "#1B4F8A",
        "fog":          "#90C8F8",
        "cap":          "#1B4F8A",
        "cap_tick":     "#4A90D9",
        "marker_outer": "#1B4F8A",
        "marker_inner": "#4A90D9",
        "marker_spec":  "#A8D4F8",
        "badge_bg":     "#1B4F8A",
        "badge_fg":     WHITE,
        "axis":         GREY_B,
        "axis_label":   GREY_A,
    },
    # Green — CI captures the true parameter
    "captured": {
        "bar_center":   "#2A9D6A",
        "bar_edge":     "#155E3E",
        "fog":          "#80DBA8",
        "cap":          "#155E3E",
        "cap_tick":     "#2A9D6A",
        "marker_outer": "#155E3E",
        "marker_inner": "#2A9D6A",
        "marker_spec":  "#A8ECC8",
        "badge_bg":     "#155E3E",
        "badge_fg":     WHITE,
        "axis":         GREY_B,
        "axis_label":   GREY_A,
    },
    # Red — CI misses the true parameter
    "missed": {
        "bar_center":   "#D94A4A",
        "bar_edge":     "#8A1B1B",
        "fog":          "#F8A0A0",
        "cap":          "#8A1B1B",
        "cap_tick":     "#D94A4A",
        "marker_outer": "#8A1B1B",
        "marker_inner": "#D94A4A",
        "marker_spec":  "#F8C8C8",
        "badge_bg":     "#8A1B1B",
        "badge_fg":     WHITE,
        "axis":         GREY_B,
        "axis_label":   GREY_A,
    },
    # Gold — for population parameter / prior highlight
    "gold": {
        "bar_center":   "#D4AF37",
        "bar_edge":     "#8B6914",
        "fog":          "#F8E090",
        "cap":          "#8B6914",
        "cap_tick":     "#D4AF37",
        "marker_outer": "#8B6914",
        "marker_inner": "#D4AF37",
        "marker_spec":  "#FFF0A0",
        "badge_bg":     "#8B6914",
        "badge_fg":     WHITE,
        "axis":         GREY_B,
        "axis_label":   GREY_A,
    },
    # Dark mode
    "dark": {
        "bar_center":   "#6A6AFF",
        "bar_edge":     "#2A2A80",
        "fog":          "#9090FF",
        "cap":          "#2A2A80",
        "cap_tick":     "#6A6AFF",
        "marker_outer": "#2A2A80",
        "marker_inner": "#6A6AFF",
        "marker_spec":  "#C0C0FF",
        "badge_bg":     "#1A1A50",
        "badge_fg":     WHITE,
        "axis":         GREY_C,
        "axis_label":   GREY_B,
    },
}

TRUE_PARAM_COLORS = {
    "unknown":  "#D4AF37",   # gold — not yet revealed
    "captured": "#2A9D6A",   # green — CI contains it
    "missed":   "#D94A4A",   # red  — CI misses it
}

# ──────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────────────────────

def _z_half_width(
    std_err: float,
    confidence: float = 0.95,
) -> float:
    """Half-width of a Z-interval: z* · SE."""
    try:
        from scipy.stats import norm
        z_star = norm.ppf((1 + confidence) / 2)
    except ImportError:
        # Fallback table for common levels
        _table = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960,
                  0.99: 2.576, 0.999: 3.291}
        z_star = _table.get(round(confidence, 3), 1.960)
    return z_star * std_err


def _t_half_width(
    std_err: float,
    df: int,
    confidence: float = 0.95,
) -> float:
    """Half-width of a t-interval: t*(df) · SE."""
    try:
        from scipy.stats import t as t_dist
        t_star = t_dist.ppf((1 + confidence) / 2, df=df)
    except ImportError:
        t_star = _z_half_width(1.0, confidence) * (1 + 1.0 / df)
    return t_star * std_err


def _captures(lower: float, upper: float, true_param: float) -> bool:
    return lower <= true_param <= upper


# ──────────────────────────────────────────────────────────────────────────────
# Sub-components
# ──────────────────────────────────────────────────────────────────────────────

class _CIBar(VGroup):
    """
    The main horizontal interval bar.

    Layers (back → front):
      1. fog_left / fog_right  — tapered semi-transparent halo extending
                                 beyond each end cap
      2. bar_base              — solid rounded rectangle, edge colour
      3. bar_center_highlight  — brighter inner rectangle fading at ends
                                 (gradient approximation)
      4. center_line           — thin bright centre stripe (lit surface)

    Parameters
    ----------
    half_width   : float — CI half-width in axis units
    bar_height   : float — visual bar thickness
    unit_scale   : float — pixels (Manim units) per axis unit
    theme        : dict
    y            : float — vertical position
    """

    def __init__(
        self,
        half_width: float,
        bar_height: float,
        unit_scale: float,
        theme: dict,
        y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hw     = half_width
        self._bh     = bar_height
        self._us     = unit_scale
        self._theme  = theme
        self._y      = y

        w_px  = 2 * half_width * unit_scale   # bar width in Manim units
        fog_w = w_px * 0.12                   # fog extends 12% beyond each end

        # ── 1. Fog halos ──────────────────────────────────────────────
        for sign in [-1, 1]:
            fog = self._make_fog(
                cx=sign * (w_px / 2 + fog_w * 0.45),
                fog_w=fog_w,
                bar_h=bar_height,
                color=theme["fog"],
            )
            fog.move_to([sign * (w_px / 2 + fog_w * 0.45), y, -0.001])
            self.add(fog)

        # ── 2. Bar base ───────────────────────────────────────────────
        bar = RoundedRectangle(
            width=w_px,
            height=bar_height,
            corner_radius=bar_height * 0.40,
            fill_color=theme["bar_edge"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        bar.move_to([0, y, 0])
        self.add(bar)

        # ── 3. Center highlight (gradient approx: 4 overlapping rects) ──
        n_layers = 5
        for i in range(n_layers):
            frac      = i / (n_layers - 1)   # 0 = edge, 1 = centre
            w_layer   = w_px * (0.30 + 0.70 * frac)
            alpha_val = 0.08 + 0.32 * frac
            hl = RoundedRectangle(
                width=w_layer,
                height=bar_height * (0.55 + 0.35 * frac),
                corner_radius=bar_height * 0.30,
                fill_color=theme["bar_center"],
                fill_opacity=alpha_val,
                stroke_width=0,
            )
            hl.move_to([0, y, 0.001 + i * 0.0005])
            self.add(hl)

        # ── 4. Thin bright centre spine ───────────────────────────────
        spine = Line(
            start=[-w_px * 0.42, y, 0.003],
            end  =[ w_px * 0.42, y, 0.003],
            stroke_color=interpolate_color(
                theme["bar_center"], WHITE, 0.50
            ),
            stroke_width=1.0,
            stroke_opacity=0.55,
        )
        self.add(spine)

    @staticmethod
    def _make_fog(
        cx: float,
        fog_w: float,
        bar_h: float,
        color: str,
    ) -> VGroup:
        """Tapered ellipse halo for the fog effect at each end."""
        g = VGroup()
        for i in range(4):
            alpha_val = 0.18 - i * 0.04
            scale_x   = 1.0 - i * 0.18
            scale_y   = 1.0 - i * 0.10
            ell = Circle(
                radius=fog_w * 0.60,
                fill_color=color,
                fill_opacity=max(0, alpha_val),
                stroke_width=0,
            )
            ell.scale([scale_x, scale_y * bar_h / (fog_w * 0.60), 1])
            g.add(ell)
        return g


class _EndCap(VGroup):
    """
    Bracket-style end cap at one end of the CI bar.

    Structure (from outside in):
      • outer_serif  — full-height vertical line (the bracket arm)
      • inner_tick   — short inward tick at the centre height
      • base_disc    — small filled circle where the bracket meets the bar

    Parameters
    ----------
    side     : +1 (right end) or -1 (left end)
    bar_h    : float — bar visual height
    theme    : dict
    y        : float — vertical centre
    """

    def __init__(
        self,
        side: int,
        bar_h: float,
        theme: dict,
        y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        z0     = 0.004
        arm_h  = bar_h * 2.20    # total bracket height
        tick_l = bar_h * 0.55    # inward tick length
        sw     = 2.8             # stroke width

        # ── Vertical arm ──────────────────────────────────────────────
        arm = Line(
            start=[0, y - arm_h / 2, z0],
            end  =[0, y + arm_h / 2, z0],
            stroke_color=theme["cap"],
            stroke_width=sw,
        )
        self.add(arm)

        # ── Inward tick ───────────────────────────────────────────────
        tick = Line(
            start=[0,             y, z0 + 0.001],
            end  =[-side * tick_l, y, z0 + 0.001],
            stroke_color=theme["cap_tick"],
            stroke_width=sw * 0.75,
        )
        self.add(tick)

        # ── Top and bottom serifs (horizontal feet) ───────────────────
        serif_w = bar_h * 0.30
        for y_end in [y - arm_h / 2, y + arm_h / 2]:
            serif = Line(
                start=[-serif_w * 0.4, y_end, z0],
                end  =[ serif_w * 0.4, y_end, z0],
                stroke_color=theme["cap"],
                stroke_width=sw * 0.6,
            )
            self.add(serif)

        # ── Base disc ─────────────────────────────────────────────────
        disc = Circle(
            radius=bar_h * 0.28,
            fill_color=theme["cap"],
            fill_opacity=1.0,
            stroke_width=0,
        )
        disc.move_to([0, y, z0 + 0.001])
        self.add(disc)


class _PointEstimateMarker(VGroup):
    """
    Multi-layer point estimate marker.

    Styles
    ------
    "circle"  — classic ● (sample mean x̄)
    "diamond" — ◆ rotated square (MLE estimate)
    "cross"   — ✕ (population parameter μ)

    Layers:
      outer_ring  — thin annulus
      body        — filled shape
      inner_disc  — slightly brighter inner fill
      specular    — bright offset dot
      label       — optional symbol (x̄, μ, etc.)
    """

    def __init__(
        self,
        style: Literal["circle", "diamond", "cross"] = "circle",
        radius: float = 0.12,
        theme: dict = None,
        label: Optional[str] = None,
        y: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if theme is None:
            theme = CI_THEMES["blue"]
        z0 = 0.008

        mo = theme["marker_outer"]
        mi = theme["marker_inner"]
        ms = theme["marker_spec"]

        if style == "circle":
            # Outer ring
            ring = Annulus(
                inner_radius=radius * 0.55,
                outer_radius=radius,
                fill_color=mo,
                fill_opacity=1.0,
                stroke_width=0,
            )
            ring.move_to([0, y, z0])
            self.add(ring)
            # Inner disc
            inner = Circle(
                radius=radius * 0.52,
                fill_color=mi,
                fill_opacity=1.0,
                stroke_width=0,
            )
            inner.move_to([0, y, z0 + 0.001])
            self.add(inner)

        elif style == "diamond":
            d = radius * 1.30
            diamond = Polygon(
                [0,  d,  z0], [ d, 0, z0],
                [0, -d,  z0], [-d, 0, z0],
                fill_color=mo, fill_opacity=1.0, stroke_width=0,
            )
            diamond.move_to([0, y, z0])
            self.add(diamond)
            inner_d = Polygon(
                [0,  d*0.55, z0+0.001], [ d*0.55, 0, z0+0.001],
                [0, -d*0.55, z0+0.001], [-d*0.55, 0, z0+0.001],
                fill_color=mi, fill_opacity=1.0, stroke_width=0,
            )
            inner_d.move_to([0, y, z0 + 0.001])
            self.add(inner_d)

        elif style == "cross":
            sw = radius * 0.40
            for ang in [0, PI / 2]:
                arm = Line(
                    start=[-radius * np.cos(ang), -radius * np.sin(ang) + y, z0],
                    end  =[ radius * np.cos(ang),  radius * np.sin(ang) + y, z0],
                    stroke_color=mo,
                    stroke_width=sw * 18,
                )
                self.add(arm)

        # Specular highlight
        spec = Circle(
            radius=radius * 0.22,
            fill_color=ms,
            fill_opacity=0.75,
            stroke_width=0,
        )
        spec.move_to([-radius * 0.25, y + radius * 0.26, z0 + 0.003])
        self.add(spec)

        # Label
        if label is not None:
            try:
                lbl = MathTex(label, font_size=radius * 80,
                              color=WHITE)
            except Exception:
                lbl = Text(label, font_size=radius * 60,
                           color=WHITE)
            lbl.move_to([0, y, z0 + 0.005])
            self.add(lbl)


class _CIBadge(VGroup):
    """
    Floating badge showing "95% CI" or custom text, with a rounded-rect
    background and a leader line connecting to the CI bar.

    Parameters
    ----------
    text         : str   — badge content, e.g. "95% CI"
    bar_y        : float — y position of the CI bar
    badge_y      : float — y position of the badge (above bar)
    x_offset     : float — horizontal offset of badge centre from 0
    theme        : dict
    """

    def __init__(
        self,
        text: str,
        bar_y: float,
        badge_y: float,
        x_offset: float = 0.0,
        theme: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if theme is None:
            theme = CI_THEMES["blue"]
        z0 = 0.010

        # Text
        lbl = Text(
            text,
            font_size=20,
            color=theme["badge_fg"],
            font="sans-serif",
        )
        lbl.move_to([x_offset, badge_y, z0 + 0.002])

        # Background pill
        pad_x, pad_y = 0.18, 0.10
        bg = RoundedRectangle(
            width=lbl.width + pad_x * 2,
            height=lbl.height + pad_y * 2,
            corner_radius=0.08,
            fill_color=theme["badge_bg"],
            fill_opacity=0.90,
            stroke_width=0,
        )
        bg.move_to([x_offset, badge_y, z0])

        # Leader line
        leader = Line(
            start=[x_offset, badge_y - lbl.height / 2 - pad_y, z0],
            end  =[x_offset, bar_y   + 0.06, z0],
            stroke_color=theme["badge_bg"],
            stroke_width=1.2,
            stroke_opacity=0.70,
        )

        self.add(leader, bg, lbl)


class _TrueParamLine(VGroup):
    """
    Dashed vertical line marking the true population parameter μ.
    Colour reflects capture status.

    Parameters
    ----------
    x          : float — x position in Manim units
    y_bottom   : float — bottom of the line
    y_top      : float — top of the line
    status     : "unknown" | "captured" | "missed"
    label      : str | None — e.g. "μ₀"
    """

    def __init__(
        self,
        x: float,
        y_bottom: float,
        y_top: float,
        status: Literal["unknown", "captured", "missed"] = "unknown",
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._status = status
        col = TRUE_PARAM_COLORS[status]
        z0  = 0.006

        # Dashed line
        dline = DashedLine(
            start=[x, y_bottom, z0],
            end  =[x, y_top,    z0],
            stroke_color=col,
            stroke_width=2.2,
            dash_length=0.12,
            dashed_ratio=0.55,
        )
        self.add(dline)

        # Label
        if label is not None:
            try:
                lbl = MathTex(label, font_size=22, color=col)
            except Exception:
                lbl = Text(label, font_size=18, color=col)
            lbl.move_to([x, y_top + 0.20, z0])
            self.add(lbl)

        self._x      = x
        self._y_top  = y_top
        self._col    = col

    def set_status(
        self,
        status: Literal["unknown", "captured", "missed"],
    ):
        """Re-colour to reflect capture result (call before animating)."""
        col = TRUE_PARAM_COLORS[status]
        for mob in self.submobjects:
            mob.set_stroke(col)
            mob.set_color(col)
        self._status = status


class _NumberLineAxis(VGroup):
    """
    A clean number line to host CI objects.

    Features:
      * Arrow tips at both ends
      * Major ticks with numeric labels
      * Minor ticks (no labels)
      * Axis label (e.g. "x" or "μ")
    """

    def __init__(
        self,
        x_range: tuple[float, float, float],   # (min, max, step)
        length: float = 8.0,
        minor_ticks: int = 4,
        theme: dict = None,
        axis_label: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if theme is None:
            theme = CI_THEMES["blue"]

        x_min, x_max, x_step = x_range
        self._x_min   = x_min
        self._x_max   = x_max
        self._length  = length
        self._unit_px = length / (x_max - x_min)

        # ── Main axis line ────────────────────────────────────────────
        ax_line = Arrow(
            start=[-length / 2 - 0.3, 0, 0],
            end  =[ length / 2 + 0.3, 0, 0],
            stroke_color=theme["axis"],
            stroke_width=2.0,
            tip_length=0.20,
            buff=0,
        )
        self.add(ax_line)

        # ── Major ticks and labels ────────────────────────────────────
        ticks_grp = VGroup()
        val = x_min
        while val <= x_max + 1e-9:
            px  = self.to_px(val)
            tk  = Line(
                start=[px, -0.15, 0.001],
                end  =[px,  0.15, 0.001],
                stroke_color=theme["axis"],
                stroke_width=1.8,
            )
            ticks_grp.add(tk)
            # Minor ticks between major
            if minor_ticks > 0 and val + x_step <= x_max + 1e-9:
                for mi in range(1, minor_ticks + 1):
                    m_val = val + mi * x_step / (minor_ticks + 1)
                    mpx   = self.to_px(m_val)
                    mtk   = Line(
                        start=[mpx, -0.07, 0.001],
                        end  =[mpx,  0.07, 0.001],
                        stroke_color=theme["axis"],
                        stroke_width=0.9,
                    )
                    ticks_grp.add(mtk)
            # Label
            lbl_str = (f"{val:.0f}" if x_step >= 1
                       else f"{val:.2f}")
            lbl = Text(
                lbl_str,
                font_size=16,
                color=theme["axis_label"],
            )
            lbl.move_to([px, -0.38, 0])
            ticks_grp.add(lbl)
            val = round(val + x_step, 10)
        self.add(ticks_grp)

        # ── Axis label ────────────────────────────────────────────────
        if axis_label:
            try:
                ax_lbl = MathTex(axis_label, font_size=26,
                                 color=theme["axis_label"])
            except Exception:
                ax_lbl = Text(axis_label, font_size=22,
                              color=theme["axis_label"])
            ax_lbl.move_to([length / 2 + 0.55, -0.38, 0])
            self.add(ax_lbl)

    def to_px(self, value: float) -> float:
        """Convert an axis value to a Manim x-coordinate."""
        return (value - self._x_min) / (
            self._x_max - self._x_min
        ) * self._length - self._length / 2


# ──────────────────────────────────────────────────────────────────────────────
# ConfidenceInterval3D  ──  the main export
# ──────────────────────────────────────────────────────────────────────────────

class ConfidenceInterval3D(VGroup):
    """
    A detailed confidence interval visualisation object.

    Parameters
    ----------
    center : float
        Point estimate (sample mean x̄ or MLE).
    half_width : float
        CI half-width in axis units.  E.g. for 95% Z-interval:
        ``half_width = 1.96 * std_err``.
        Alternatively supply ``std_err`` and ``confidence``; then
        ``half_width`` is computed automatically.
    true_param : float | None
        True population parameter μ.  If supplied, a dashed vertical
        line is drawn and capture status is tracked.
    confidence : float
        Nominal confidence level (0–1), e.g. ``0.95``.  Used for
        badge label and auto-computing half_width.
    std_err : float | None
        Standard error.  If provided (and half_width not), half_width
        is computed from the confidence level.
    df : int | None
        Degrees of freedom.  If provided, a t-interval is used
        instead of a Z-interval.
    axis : _NumberLineAxis | None
        Axis object to anchor the CI to.  If None, a default axis
        is created.
    x_range : tuple
        (min, max, step) for the auto-created axis.
    axis_length : float
        Length of the auto-created axis (Manim units).  Default 9.0.
    y_offset : float
        Vertical shift of the CI bar above the axis.  Default 0.0.
    bar_height : float
        Thickness of the CI bar (Manim units).  Default 0.22.
    theme : str
        Named colour theme.  Default ``"blue"``.
    custom_theme : dict | None
        Override individual theme keys.
    marker_style : "circle" | "diamond" | "cross"
        Point estimate marker shape.  Default ``"circle"``.
    marker_label : str | None
        LaTeX label on the point estimate marker (e.g. ``r"\bar{x}"``).
    show_badge : bool
        Show the floating CI badge.  Default ``True``.
    show_true_param : bool
        Show the true parameter line if ``true_param`` is provided.
    badge_text : str | None
        Override badge text (auto: "95% CI").
    badge_y_offset : float
        Upward shift of the badge from the CI bar.  Default 0.55.

    Attributes
    ----------
    center       : float
    lower        : float
    upper        : float
    half_width   : float
    captures     : bool | None   — True/False/None (not yet revealed)
    ci_bar       : _CIBar
    left_cap     : _EndCap
    right_cap    : _EndCap
    marker       : _PointEstimateMarker
    true_line    : _TrueParamLine | None
    badge        : _CIBadge | None
    axis         : _NumberLineAxis
    unit_scale   : float   — Manim units per axis unit
    """

    def __init__(
        self,
        center: float,
        half_width: Optional[float] = None,
        true_param: Optional[float] = None,
        confidence: float = 0.95,
        std_err: Optional[float] = None,
        df: Optional[int] = None,
        axis: Optional[_NumberLineAxis] = None,
        x_range: tuple = (60.0, 100.0, 5.0),
        axis_length: float = 9.0,
        y_offset: float = 0.0,
        bar_height: float = 0.22,
        theme: str = "blue",
        custom_theme: Optional[dict] = None,
        marker_style: Literal["circle", "diamond", "cross"] = "circle",
        marker_label: Optional[str] = None,
        show_badge: bool = True,
        show_true_param: bool = True,
        badge_text: Optional[str] = None,
        badge_y_offset: float = 0.55,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ── Resolve palette ───────────────────────────────────────────
        th = dict(CI_THEMES.get(theme, CI_THEMES["blue"]))
        if custom_theme:
            th.update(custom_theme)

        # ── Compute half-width ────────────────────────────────────────
        if half_width is None:
            if std_err is None:
                raise ValueError(
                    "Provide either half_width or std_err."
                )
            if df is not None:
                half_width = _t_half_width(std_err, df, confidence)
            else:
                half_width = _z_half_width(std_err, confidence)

        self._center     = center
        self._half_width = half_width
        self._lower      = center - half_width
        self._upper      = center + half_width
        self._confidence = confidence
        self._true_param = true_param
        self._captures   = (
            _captures(self._lower, self._upper, true_param)
            if true_param is not None else None
        )
        self._y_offset   = y_offset
        self._bar_height = bar_height

        # ── Axis ──────────────────────────────────────────────────────
        if axis is None:
            axis = _NumberLineAxis(
                x_range=x_range,
                length=axis_length,
                theme=th,
            )
        self.axis = axis
        self._unit_scale = axis._unit_px
        self.add(axis)

        # ── Convert CI bounds to Manim coords ─────────────────────────
        cx_px  = axis.to_px(center)
        hw_px  = half_width * self._unit_scale

        # ── CI Bar ────────────────────────────────────────────────────
        self.ci_bar = _CIBar(
            half_width=half_width,
            bar_height=bar_height,
            unit_scale=self._unit_scale,
            theme=th,
            y=y_offset,
        )
        self.ci_bar.move_to([cx_px, y_offset, 0])
        self.add(self.ci_bar)

        # ── End caps ──────────────────────────────────────────────────
        self.left_cap = _EndCap(
            side=-1, bar_h=bar_height, theme=th, y=y_offset
        )
        self.left_cap.move_to([cx_px - hw_px, y_offset, 0])
        self.add(self.left_cap)

        self.right_cap = _EndCap(
            side=+1, bar_h=bar_height, theme=th, y=y_offset
        )
        self.right_cap.move_to([cx_px + hw_px, y_offset, 0])
        self.add(self.right_cap)

        # ── Point estimate marker ─────────────────────────────────────
        self.marker = _PointEstimateMarker(
            style=marker_style,
            radius=bar_height * 0.70,
            theme=th,
            label=marker_label,
            y=y_offset,
        )
        self.marker.move_to([cx_px, y_offset, 0])
        self.add(self.marker)

        # ── True parameter line ───────────────────────────────────────
        self.true_line = None
        if true_param is not None and show_true_param:
            tp_px    = axis.to_px(true_param)
            self.true_line = _TrueParamLine(
                x=tp_px,
                y_bottom=y_offset - bar_height * 3.5,
                y_top=y_offset + bar_height * 4.0,
                status="unknown",
                label=r"\mu_0",
            )
            self.add(self.true_line)

        # ── Badge ─────────────────────────────────────────────────────
        self.badge = None
        if show_badge:
            if badge_text is None:
                badge_text = f"{int(confidence * 100)}% CI"
            self.badge = _CIBadge(
                text=badge_text,
                bar_y=y_offset + bar_height / 2,
                badge_y=y_offset + badge_y_offset,
                x_offset=cx_px,
                theme=th,
            )
            self.add(self.badge)

        self._theme    = th
        self._cx_px    = cx_px
        self._hw_px    = hw_px

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def center(self) -> float:
        return self._center

    @property
    def lower(self) -> float:
        return self._lower

    @property
    def upper(self) -> float:
        return self._upper

    @property
    def half_width(self) -> float:
        return self._half_width

    @property
    def captures(self) -> Optional[bool]:
        return self._captures

    @property
    def unit_scale(self) -> float:
        return self._unit_scale

    def reveal_capture_status(self):
        """
        Update the true-parameter line colour based on capture result.
        Call this before ``RevealCapture`` animation.
        """
        if self.true_line is None or self._captures is None:
            return
        status = "captured" if self._captures else "missed"
        self.true_line.set_status(status)

    def get_capture_color(self) -> str:
        if self._captures is True:
            return TRUE_PARAM_COLORS["captured"]
        elif self._captures is False:
            return TRUE_PARAM_COLORS["missed"]
        return TRUE_PARAM_COLORS["unknown"]

    def rescale_to_axis(self, new_axis: _NumberLineAxis):
        """Re-position all components after replacing the axis."""
        self.axis = new_axis
        self._unit_scale = new_axis._unit_px
        # (Full rebuild; subclasses can override)


# ──────────────────────────────────────────────────────────────────────────────
# CIStack  ──  repeated-sampling coverage demonstration
# ──────────────────────────────────────────────────────────────────────────────

class CIStack(VGroup):
    """
    A vertical stack of confidence intervals from repeated samples,
    demonstrating the frequentist coverage interpretation.

    Parameters
    ----------
    true_param     : float   — the fixed population parameter μ
    ci_list        : list of (center, half_width) tuples
    confidence     : float   — nominal coverage (for badge / label)
    x_range        : tuple   — (min, max, step) for the shared axis
    axis_length    : float
    row_spacing    : float   — vertical gap between CI rows
    bar_height     : float
    theme_captured : str     — theme for capturing CIs
    theme_missed   : str     — theme for missing CIs
    show_counter   : bool    — show live "k/N captured" badge
    show_true_col  : bool    — show full-height true-param column

    Attributes
    ----------
    cis           : list[ConfidenceInterval3D]
    n_captured    : int
    n_total       : int
    coverage_rate : float
    counter_badge : VGroup
    """

    def __init__(
        self,
        true_param: float,
        ci_list: list[tuple[float, float]],
        confidence: float = 0.95,
        x_range: tuple = (60.0, 100.0, 5.0),
        axis_length: float = 9.0,
        row_spacing: float = 0.42,
        bar_height: float = 0.18,
        theme_captured: str = "captured",
        theme_missed: str = "missed",
        show_counter: bool = True,
        show_true_col: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._true_param    = true_param
        self._confidence    = confidence
        self._row_spacing   = row_spacing
        self._bar_height    = bar_height
        self.cis: list[ConfidenceInterval3D] = []

        # ── Shared axis (drawn once at the top) ───────────────────────
        th_base = CI_THEMES["blue"]
        self.shared_axis = _NumberLineAxis(
            x_range=x_range,
            length=axis_length,
            theme=th_base,
        )
        self.add(self.shared_axis)

        # Extract unit scale for subsequent CIs
        us = self.shared_axis._unit_px

        # ── True-parameter column (full height) ───────────────────────
        self.true_col = None
        self._tp_px   = self.shared_axis.to_px(true_param)
        if show_true_col:
            n_rows       = len(ci_list)
            col_h        = n_rows * (bar_height + row_spacing) + 0.5
            self.true_col = DashedLine(
                start=[self._tp_px,  0.3,    0.002],
                end  =[self._tp_px, -col_h,  0.002],
                stroke_color=TRUE_PARAM_COLORS["unknown"],
                stroke_width=2.0,
                dash_length=0.10,
                dashed_ratio=0.5,
            )
            self.add(self.true_col)

        # ── Populate CI rows ──────────────────────────────────────────
        n_cap = 0
        for i, (ctr, hw) in enumerate(ci_list):
            y_row = -(i + 1) * (bar_height + row_spacing)
            cap   = _captures(ctr - hw, ctr + hw, true_param)
            n_cap += int(cap)
            th_name = theme_captured if cap else theme_missed
            th      = CI_THEMES.get(th_name, CI_THEMES["blue"])

            # Bar
            bar = _CIBar(
                half_width=hw,
                bar_height=bar_height,
                unit_scale=us,
                theme=th,
                y=y_row,
            )
            bar.move_to([self.shared_axis.to_px(ctr), y_row, 0])

            # End caps
            cx_px = self.shared_axis.to_px(ctr)
            hw_px = hw * us
            lcap  = _EndCap(side=-1, bar_h=bar_height,
                            theme=th, y=y_row)
            lcap.move_to([cx_px - hw_px, y_row, 0])
            rcap  = _EndCap(side=+1, bar_h=bar_height,
                            theme=th, y=y_row)
            rcap.move_to([cx_px + hw_px, y_row, 0])

            # Marker
            mk = _PointEstimateMarker(
                style="circle",
                radius=bar_height * 0.60,
                theme=th,
                y=y_row,
            )
            mk.move_to([cx_px, y_row, 0])

            row_grp = VGroup(bar, lcap, rcap, mk)
            self.add(row_grp)

            # Store as minimal CI record
            self.cis.append((ctr - hw, ctr + hw, cap))

        self.n_captured    = n_cap
        self.n_total       = len(ci_list)
        self.coverage_rate = n_cap / max(self.n_total, 1)

        # ── Coverage counter badge ────────────────────────────────────
        self.counter_badge = None
        if show_counter and len(ci_list) > 0:
            self.counter_badge = self._make_counter_badge(
                n_cap, len(ci_list), confidence, th_base
            )
            self.add(self.counter_badge)

    @staticmethod
    def _make_counter_badge(
        n_cap: int,
        n_total: int,
        confidence: float,
        theme: dict,
    ) -> VGroup:
        """Build the "k / N captured (p%)" badge."""
        pct      = n_cap / n_total * 100
        nom_pct  = confidence * 100
        diff     = pct - nom_pct
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"

        txt = Text(
            f"{n_cap}/{n_total} captured  ({pct:.1f}%  nominal {nom_pct:.0f}%  {diff_str})",
            font_size=18,
            color=WHITE,
        )
        bg  = RoundedRectangle(
            width=txt.width + 0.32,
            height=txt.height + 0.18,
            corner_radius=0.07,
            fill_color="#1A1A2E",
            fill_opacity=0.88,
            stroke_color=theme.get("axis", GREY_B),
            stroke_width=0.8,
        )
        bg.move_to([0, 0.85, 0])
        txt.move_to([0, 0.85, 0.001])
        return VGroup(bg, txt)

    @classmethod
    def from_simulation(
        cls,
        true_param: float,
        n_samples: int,
        sample_size: int,
        population_std: float,
        confidence: float = 0.95,
        seed: int = 42,
        **kwargs,
    ) -> "CIStack":
        """
        Build a CIStack by simulating ``n_samples`` draws of size
        ``sample_size`` from N(true_param, population_std²).

        Parameters
        ----------
        true_param     : population mean μ
        n_samples      : number of CI rows
        sample_size    : n per sample
        population_std : σ (assumed known → Z-interval)
        confidence     : nominal level
        seed           : RNG seed

        Returns
        -------
        CIStack ready for ``StackCIs`` animation.
        """
        rng = np.random.default_rng(seed)
        se  = population_std / np.sqrt(sample_size)
        hw  = _z_half_width(se, confidence)

        ci_list = []
        for _ in range(n_samples):
            sample_mean = rng.normal(true_param, se)
            ci_list.append((sample_mean, hw))

        return cls(
            true_param=true_param,
            ci_list=ci_list,
            confidence=confidence,
            **kwargs,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Animations
# ──────────────────────────────────────────────────────────────────────────────

class BuildCI(Animation):
    """
    The CI bar grows symmetrically outward from the point estimate.

    Sequence:
      α ∈ [0.00, 0.60] — bar expands from 0 width to full width
      α ∈ [0.45, 0.75] — end caps materialise (fade in)
      α ∈ [0.70, 1.00] — badge fades in

    Parameters
    ----------
    ci       : ConfidenceInterval3D
    run_time : float
    """

    def __init__(
        self,
        ci: ConfidenceInterval3D,
        **kwargs,
    ):
        self.ci       = ci
        self._cx_px   = ci._cx_px
        self._hw_px   = ci._hw_px
        kwargs.setdefault("run_time", 1.8)
        kwargs.setdefault("rate_func", rate_functions.ease_out_cubic)
        super().__init__(ci, **kwargs)

    def interpolate_mobject(self, alpha: float):
        self.ci.become(self.starting_mobject.copy())

        # Phase 1: bar width expansion [0 → 0.60]
        bar_alpha = min(1.0, alpha / 0.60)
        scale_x   = bar_alpha
        self.ci.ci_bar.scale(
            [scale_x if scale_x > 0 else 1e-6, 1, 1],
            about_point=[self._cx_px, self.ci._y_offset, 0],
        )

        # Phase 2: end caps opacity [0.45 → 0.75]
        cap_alpha = np.clip((alpha - 0.45) / 0.30, 0, 1)
        self.ci.left_cap.set_opacity(cap_alpha)
        self.ci.right_cap.set_opacity(cap_alpha)
        self.ci.marker.set_opacity(cap_alpha)

        # Phase 3: badge opacity [0.70 → 1.00]
        badge_alpha = np.clip((alpha - 0.70) / 0.30, 0, 1)
        if self.ci.badge is not None:
            self.ci.badge.set_opacity(badge_alpha)
        if self.ci.true_line is not None:
            self.ci.true_line.set_opacity(badge_alpha)


class RevealCapture(Succession):
    """
    Drop the true-parameter line into view and flash the CI
    green (captured) or red (missed).

    Sequence:
      1. True-param line drops from above (FadeIn with UP shift)
      2. Line re-colours to capture/miss colour
      3. CI bar flashes (Indicate) in the capture colour
      4. Badge text fades to update status

    Parameters
    ----------
    ci       : ConfidenceInterval3D
    run_time : float
    """

    def __init__(
        self,
        ci: ConfidenceInterval3D,
        **kwargs,
    ):
        ci.reveal_capture_status()
        cap_col = ci.get_capture_color()

        anims = []

        if ci.true_line is not None:
            anims.append(
                FadeIn(ci.true_line, shift=DOWN * 0.5, run_time=0.6)
            )

        # Flash the CI bar
        anims.append(
            Indicate(
                ci.ci_bar,
                color=cap_col,
                scale_factor=1.04,
                run_time=0.7,
            )
        )

        # Flash the end caps
        anims.append(
            AnimationGroup(
                Indicate(ci.left_cap,  color=cap_col,
                         scale_factor=1.06, run_time=0.5),
                Indicate(ci.right_cap, color=cap_col,
                         scale_factor=1.06, run_time=0.5),
            )
        )

        kwargs.setdefault("run_time",
                          sum(getattr(a, "run_time", 0.7) for a in anims))
        super().__init__(*anims, **kwargs)


class SweepCI(Animation):
    """
    Slide the CI bar horizontally to a new center value, keeping
    the true-param line and axis stationary.

    Parameters
    ----------
    ci          : ConfidenceInterval3D
    new_center  : float   — new center value in axis units
    run_time    : float
    """

    def __init__(
        self,
        ci: ConfidenceInterval3D,
        new_center: float,
        **kwargs,
    ):
        self.ci         = ci
        self.old_cx_px  = ci._cx_px
        self.new_cx_px  = ci.axis.to_px(new_center)
        self._new_center = new_center
        kwargs.setdefault("run_time", 1.2)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        # Animate only the mobile components (not axis, not true_line)
        grp = VGroup(
            ci.ci_bar, ci.left_cap, ci.right_cap,
            ci.marker,
            *([ci.badge] if ci.badge else []),
        )
        super().__init__(grp, **kwargs)

    def interpolate_mobject(self, alpha: float):
        dx = (self.new_cx_px - self.old_cx_px) * alpha
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.shift([dx, 0, 0])
        if alpha >= 1.0:
            self.ci._cx_px   = self.new_cx_px
            self.ci._center  = self._new_center
            self.ci._lower   = self._new_center - self.ci._half_width
            self.ci._upper   = self._new_center + self.ci._half_width
            if self.ci._true_param is not None:
                self.ci._captures = _captures(
                    self.ci._lower, self.ci._upper, self.ci._true_param
                )


class NarrowCI(Animation):
    """
    Shrink the CI to a new (narrower or wider) half_width.
    Simulates increasing sample size n or changing confidence level.

    Parameters
    ----------
    ci            : ConfidenceInterval3D
    new_half_width: float   — new half-width in axis units
    run_time      : float
    """

    def __init__(
        self,
        ci: ConfidenceInterval3D,
        new_half_width: float,
        **kwargs,
    ):
        self.ci             = ci
        self.old_hw_px      = ci._hw_px
        self.new_hw_px      = new_half_width * ci._unit_scale
        self._new_half_width = new_half_width
        kwargs.setdefault("run_time", 1.4)
        kwargs.setdefault("rate_func", rate_functions.ease_in_out_cubic)
        grp = VGroup(
            ci.ci_bar, ci.left_cap, ci.right_cap,
            *([ci.badge] if ci.badge else []),
        )
        super().__init__(grp, **kwargs)

    def interpolate_mobject(self, alpha: float):
        hw_now = self.old_hw_px + (self.new_hw_px - self.old_hw_px) * alpha
        scale  = hw_now / max(self.old_hw_px, 1e-6)

        self.mobject.become(self.starting_mobject.copy())
        cx = self.ci._cx_px
        self.ci.ci_bar.scale(
            [scale, 1, 1],
            about_point=[cx, self.ci._y_offset, 0],
        )
        # Shift end caps
        old_l = cx - self.old_hw_px
        old_r = cx + self.old_hw_px
        new_l = cx - hw_now
        new_r = cx + hw_now
        self.ci.left_cap.shift( [new_l - old_l, 0, 0])
        self.ci.right_cap.shift([new_r - old_r, 0, 0])

        if alpha >= 1.0:
            self.ci._half_width = self._new_half_width
            self.ci._hw_px      = self.new_hw_px
            self.ci._lower      = self.ci._center - self._new_half_width
            self.ci._upper      = self.ci._center + self._new_half_width


class StackCIs(Succession):
    """
    Animate N CI rows being added to a CIStack one by one,
    with the coverage counter badge updating after each addition.

    The CIStack should be created empty or with ``ci_list=[]``;
    this animation populates it in real time.

    Since fully dynamic CIStack building is complex, this animation
    works with a pre-built CIStack and reveals each row sequentially
    using FadeIn with a downward shift.

    Parameters
    ----------
    stack     : CIStack    — fully built stack (all rows present but hidden)
    run_time  : float      — total animation time
    per_ci_rt : float      — time per CI row
    stagger   : float      — pause between rows
    """

    def __init__(
        self,
        stack: CIStack,
        per_ci_rt: float = 0.15,
        stagger: float = 0.05,
        **kwargs,
    ):
        # Hide all CI rows initially
        ci_rows = [
            mob for mob in stack.submobjects
            if mob not in (stack.shared_axis, stack.true_col,
                           stack.counter_badge)
            and mob is not None
        ]
        for row in ci_rows:
            row.set_opacity(0)

        # Animate rows appearing one by one
        anims = []
        for i, row in enumerate(ci_rows):
            anims.append(
                FadeIn(row, shift=RIGHT * 0.15, run_time=per_ci_rt)
            )

        # Reveal counter badge at the end
        if stack.counter_badge is not None:
            stack.counter_badge.set_opacity(0)
            anims.append(
                FadeIn(stack.counter_badge, run_time=0.5)
            )

        kwargs.setdefault(
            "run_time",
            len(anims) * (per_ci_rt + stagger),
        )
        super().__init__(*anims, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factories
# ──────────────────────────────────────────────────────────────────────────────

def make_ci_comparison(
    centers: list[float],
    half_widths: list[float],
    confidence_levels: list[float],
    true_param: Optional[float] = None,
    x_range: tuple = (60.0, 100.0, 5.0),
    axis_length: float = 9.0,
    y_spacing: float = 0.9,
    themes: Optional[list[str]] = None,
) -> VGroup:
    """
    Stack multiple CIs for side-by-side comparison — e.g. 90/95/99%
    intervals from the same sample, or same level from different samples.

    Parameters
    ----------
    centers, half_widths, confidence_levels : parallel lists
    true_param    : optional true parameter (shared)
    y_spacing     : vertical gap between CIs
    themes        : one theme name per CI; cycles if short

    Returns
    -------
    VGroup containing a shared axis + N ConfidenceInterval3D objects.
    """
    if themes is None:
        themes = ["blue", "gold", "dark"]
    n = len(centers)

    # Shared axis at y=0
    th_base = CI_THEMES["blue"]
    ax = _NumberLineAxis(
        x_range=x_range,
        length=axis_length,
        theme=th_base,
    )
    group = VGroup(ax)

    for i in range(n):
        y_off = -(i + 1) * y_spacing
        ci = ConfidenceInterval3D(
            center=centers[i],
            half_width=half_widths[i],
            true_param=true_param,
            confidence=confidence_levels[i],
            axis=ax,
            y_offset=y_off,
            bar_height=0.20,
            theme=themes[i % len(themes)],
            show_true_param=(i == n - 1),   # only bottom CI shows true line
            badge_y_offset=0.42,
        )
        group.add(ci)

    return group


def make_clt_ci_stack(
    true_mean: float = 80.0,
    true_std: float = 10.0,
    sample_size: int = 30,
    n_samples: int = 20,
    confidence: float = 0.95,
    x_range: tuple = (60.0, 100.0, 5.0),
    seed: int = 42,
    **stack_kwargs,
) -> CIStack:
    """
    Convenience wrapper: simulate ``n_samples`` of size ``sample_size``
    from N(true_mean, true_std²) and build a CIStack.

    Parameters
    ----------
    true_mean, true_std : population parameters
    sample_size         : n per draw
    n_samples           : number of CI rows
    confidence          : nominal level
    seed                : RNG seed

    Returns
    -------
    CIStack ready for StackCIs animation.
    """
    return CIStack.from_simulation(
        true_param=true_mean,
        n_samples=n_samples,
        sample_size=sample_size,
        population_std=true_std,
        confidence=confidence,
        x_range=x_range,
        seed=seed,
        **stack_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demo Scene  (run with: manim -pql confidence_interval.py CIDemo)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from manim import ThreeDScene, Scene, DEGREES, Wait

    class CIDemo(Scene):
        """Showcase scene for ConfidenceInterval3D, CIStack, and animations."""

        def construct(self):

            # ── Single 95% Z-interval ─────────────────────────────────
            ci = ConfidenceInterval3D(
                center=80.0,
                std_err=3.2,
                confidence=0.95,
                true_param=82.0,
                x_range=(60.0, 100.0, 5.0),
                axis_length=9.0,
                theme="blue",
                marker_label=r"\bar{x}",
                badge_y_offset=0.60,
            )
            self.play(BuildCI(ci, run_time=2.0))
            self.wait(0.4)
            self.play(RevealCapture(ci))
            self.wait(0.5)

            # ── Sweep to a new sample ─────────────────────────────────
            self.play(SweepCI(ci, new_center=76.5, run_time=1.3))
            self.wait(0.3)
            ci.reveal_capture_status()
            self.play(RevealCapture(ci))
            self.wait(0.5)

            # ── Narrow the CI (larger n) ──────────────────────────────
            self.play(NarrowCI(ci, new_half_width=3.5, run_time=1.4))
            self.wait(0.5)

            self.play(FadeOut(ci))
            self.wait(0.3)

            # ── 90 / 95 / 99 % comparison ────────────────────────────
            comparison = make_ci_comparison(
                centers=[80.0, 80.0, 80.0],
                half_widths=[
                    _z_half_width(3.2, 0.90),
                    _z_half_width(3.2, 0.95),
                    _z_half_width(3.2, 0.99),
                ],
                confidence_levels=[0.90, 0.95, 0.99],
                true_param=82.0,
                x_range=(60.0, 100.0, 5.0),
                y_spacing=0.85,
                themes=["blue", "gold", "dark"],
            )
            comparison.scale(0.80).center()
            for mob in comparison.submobjects[1:]:
                self.play(BuildCI(mob, run_time=0.9))
            self.wait(1.0)
            self.play(FadeOut(comparison))

            # ── CIStack coverage demonstration ────────────────────────
            stack = make_clt_ci_stack(
                true_mean=80.0,
                true_std=10.0,
                sample_size=30,
                n_samples=15,
                confidence=0.95,
                x_range=(60.0, 100.0, 5.0),
                row_spacing=0.35,
                bar_height=0.16,
            )
            stack.scale(0.85).center()
            self.play(FadeIn(stack.shared_axis), run_time=0.5)
            self.play(StackCIs(stack, per_ci_rt=0.18, stagger=0.06))
            self.wait(2.0)

except ImportError:
    pass