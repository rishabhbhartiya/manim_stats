"""
manim_stats/ui/ticker.py
=========================
Animated statistical value counters ("tickers") for Manim scenes.

A Ticker3D is a self-contained badge mobject that displays a numerical
statistic with a label, optional unit, optional delta indicator, and an
animated highlight ring.  It can count smoothly from one value to another,
roll digits odometer-style, flash when a value crosses a significance
threshold, and morph its colour based on the statistic's magnitude.

Architecture
------------
  Layer A  Pure-Python number formatting & easing
               TickerFormat, format_value, auto_precision,
               easing functions, threshold_color_key

  Layer B  Configuration dataclasses
               TickerStyle, ThresholdMap

  Layer C  Core Manim mobject
               Ticker3D — badge with label · value · delta · flash ring

  Layer D  Specialised subclasses
               PValueTicker3D   — significance stars + threshold colouring
               StatsCounter3D   — integer odometer for n, df, k
               CorrelationTicker3D — r ∈ [-1,1] with diverging colour ramp

  Layer E  Multi-ticker dashboard
               TickerGroup3D — row / column / grid layout of Ticker3D objects

  Layer F  Pre-built dashboard factories
               regression_dashboard, correlation_dashboard,
               hypothesis_dashboard, distribution_dashboard

Design notes
------------
* All number formatting is pure Python — no Manim dependency.
  The Ticker3D mobject uses Manim's ``DecimalNumber`` for the
  value field so Manim's built-in ValueTracker integration works
  seamlessly for smooth counting animations.

* ``count_to()`` uses Manim's ``ChangeDecimalToValue`` animation
  for the smooth path.  ``odometer_to()`` uses a custom updater that
  rebuilds the MathTex digit-by-digit at each frame.

* ``flash_change()`` combines a ``ChangeDecimalToValue`` with a
  simultaneous colour flash on the badge ring, giving the viewer an
  immediate visual cue that something changed.

* Threshold colouring: callers pass a ``ThresholdMap`` that maps
  upper-bound values to colours.  The badge background and value text
  colour update automatically when ``set_value()`` or any animated
  setter is called.

* No hard Manim dependency at import time — Layers A and B are pure
  Python.  Layer C-F raise ``ImportError`` clearly if Manim is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Graceful Manim import
# ---------------------------------------------------------------------------
try:
    import manim as mn
    from manim import (
        VGroup, VMobject,
        DecimalNumber, Integer, MathTex, Text,
        RoundedRectangle, Rectangle,
        Line, DashedLine,
        Dot, Circle,
        ManimColor, WHITE, BLACK, GRAY, DARK_GRAY,
        RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE,
        UP, DOWN, LEFT, RIGHT, ORIGIN,
        TAU, PI,
        Write, Create, FadeIn, FadeOut,
        Transform, ReplacementTransform,
        Flash, Indicate, Circumscribe,
        AnimationGroup, Succession, LaggedStart,
        ChangeDecimalToValue,
        rate_functions,
        interpolate_color,
        ValueTracker,
    )
    _MANIM_AVAILABLE = True
except ImportError:
    _MANIM_AVAILABLE = False
    VGroup = VMobject = object  # type: ignore

# ---------------------------------------------------------------------------
# Project imports (graceful)
# ---------------------------------------------------------------------------
try:
    from manim_stats.core.colors import (
        REGRESSION_FAMILY, NORMAL_FAMILY, INFERENCE_FAMILY,
        DISCRETE_FAMILY, NEUTRAL_FAMILY, INFORMATION_FAMILY,
        PROBABILITY_FAMILY,
        TEAL_600, TEAL_200, TEAL_100,
        CORAL_600, CORAL_200, CORAL_100,
        PURPLE_600, PURPLE_200,
        AMBER_600, AMBER_200, AMBER_100,
        BLUE_600, BLUE_200,
        GRAY_600, GRAY_400, GRAY_200, GRAY_100,
        BG_DARK, BG_MEDIUM,
        StatColor, ColorFamily,
        interpolate_colors,
    )
    _COLORS_AVAILABLE = True
except ImportError:
    _COLORS_AVAILABLE = False

try:
    from manim_stats.core.tex_utils import TexFormula
    _TEX_AVAILABLE = True
except ImportError:
    _TEX_AVAILABLE = False


def _require_manim(name: str) -> None:
    if not _MANIM_AVAILABLE:
        raise ImportError(
            f"{name} requires Manim. Install with: pip install manim"
        )


# ===========================================================================
# LAYER A — Pure-Python number formatting and easing
# No Manim dependency. All functions are deterministic and testable.
# ===========================================================================

class TickerFormat(Enum):
    """
    Number display format for a Ticker3D.

    AUTO
        Use scientific notation when |value| >= 1e6 or 0 < |value| < 1e-3;
        otherwise use fixed-point.  This is the most readable default.
    FLOAT
        Fixed-point decimal, always.  e.g. ``3.1416``.
    SCI
        Scientific notation.  e.g. ``3.142e+00``.
    PERCENT
        Multiply by 100 and append ``%``.  e.g. ``31.42%``.
    INT
        Round to nearest integer with comma thousands-separator.  e.g. ``1,234``.
    PVALUE
        Fixed-point with at most 4 significant figures.  Values < 0.001
        displayed as ``< 0.001``.
    CORRELATION
        Fixed-point with 4 decimal places, always signed (``+0.7740``).
    """
    AUTO        = "auto"
    FLOAT       = "float"
    SCI         = "sci"
    PERCENT     = "percent"
    INT         = "int"
    PVALUE      = "pvalue"
    CORRELATION = "correlation"


def auto_precision(value: float, n_sig: int = 4) -> int:
    """
    Return the number of decimal places that gives ``n_sig`` significant
    figures for *value*.

    Examples
    --------
    >>> auto_precision(3.14159)   # → 3  (3.142 has 4 sig figs)
    >>> auto_precision(0.00123)   # → 5  (0.00123 has 3 sig figs after decimal)
    >>> auto_precision(1234.5)    # → 0  (already 5 digits before decimal)
    """
    if value == 0:
        return n_sig
    mag = math.floor(math.log10(abs(value)))
    dp  = max(0, n_sig - 1 - mag)
    return dp


def format_value(
    value:      float,
    fmt:        TickerFormat = TickerFormat.AUTO,
    precision:  int          = 4,
) -> str:
    """
    Format *value* as a string according to *fmt* and *precision*.

    Parameters
    ----------
    value : float
        The numeric value to format.
    fmt : TickerFormat
        Display format strategy.
    precision : int
        Decimal places (FLOAT, SCI) or significant figures (AUTO, PVALUE).

    Returns
    -------
    str
        Human-readable string (no LaTeX markup).
    """
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+∞" if value > 0 else "-∞"

    if fmt == TickerFormat.INT:
        return f"{round(value):,}"

    if fmt == TickerFormat.PERCENT:
        return f"{value * 100:.{max(0, precision - 2)}f}%"

    if fmt == TickerFormat.PVALUE:
        if abs(value) < 0.001:
            return "< 0.001"
        return f"{value:.{precision}f}"

    if fmt == TickerFormat.CORRELATION:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.{precision}f}"

    if fmt == TickerFormat.SCI:
        return f"{value:.{precision - 1}e}"

    if fmt == TickerFormat.FLOAT:
        return f"{value:.{precision}f}"

    # AUTO
    if abs(value) != 0 and (abs(value) >= 10 ** precision or abs(value) < 1e-3):
        return f"{value:.{precision - 1}e}"
    return f"{value:.{precision}f}"


def significance_stars(p_value: float) -> str:
    """
    Return APA-style significance stars for a p-value.

    ``***`` p < 0.001, ``**`` p < 0.01, ``*`` p < 0.05, ``†`` p < 0.10,
    empty string otherwise.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "†"
    return ""


def delta_string(old: float, new: float, precision: int = 4) -> str:
    """
    Format the signed change from *old* to *new*.
    Returns e.g. ``+0.0042`` or ``-1.23e-05``.
    """
    d = new - old
    if d == 0:
        return "±0"
    sign   = "▲" if d > 0 else "▼"
    dp     = auto_precision(abs(d), n_sig=3)
    abs_s  = f"{abs(d):.{dp}f}" if abs(d) >= 1e-4 else f"{abs(d):.2e}"
    return f"{sign}{abs_s}"


# ---------------------------------------------------------------------------
# Easing functions (pure Python, 0→1 input/output)
# ---------------------------------------------------------------------------

def _ease_linear(t: float) -> float:
    return t

def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3

def _ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t ** 3
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0

def _ease_out_elastic(t: float) -> float:
    """Slight overshoot for a satisfying "snap" on count arrival."""
    c4 = (2.0 * math.pi) / 3.0
    if t == 0 or t == 1:
        return t
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0

def _ease_out_back(t: float) -> float:
    """Overshoot and settle — good for an odometer digit landing."""
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


_EASING_MAP: Dict[str, Callable[[float], float]] = {
    "linear":        _ease_linear,
    "out_cubic":     _ease_out_cubic,
    "in_out_cubic":  _ease_in_out_cubic,
    "out_elastic":   _ease_out_elastic,
    "out_back":      _ease_out_back,
}


# ---------------------------------------------------------------------------
# Threshold logic — returns a key string for a value given threshold buckets
# ---------------------------------------------------------------------------

class ThresholdMap:
    """
    Maps a numeric value to a named tier by comparing against sorted
    upper-bound thresholds.

    Parameters
    ----------
    thresholds : dict[float, str]
        Mapping of upper-bound → tier name, e.g.
        ``{0.001: 'critical', 0.01: 'significant', 0.05: 'marginal'}``.
        Values above all thresholds fall into the ``default_tier``.
    default_tier : str
        Tier name when the value exceeds every threshold.

    Examples
    --------
    >>> tm = ThresholdMap({0.001: 'critical', 0.05: 'sig'}, 'ns')
    >>> tm.tier(0.0005)   # 'critical'
    >>> tm.tier(0.03)     # 'sig'
    >>> tm.tier(0.08)     # 'ns'
    """

    def __init__(
        self,
        thresholds:   Dict[float, str],
        default_tier: str = "normal",
    ) -> None:
        self._sorted = sorted(thresholds.items())   # [(upper_bound, tier), ...]
        self.default_tier = default_tier

    def tier(self, value: float) -> str:
        """Return the tier name for *value*."""
        for upper_bound, tier_name in self._sorted:
            if value <= upper_bound:
                return tier_name
        return self.default_tier

    def is_significant(self, value: float, alpha: float = 0.05) -> bool:
        """Return True if *value* falls in a tier with upper_bound <= *alpha*."""
        for upper_bound, _ in self._sorted:
            if upper_bound <= alpha and value <= upper_bound:
                return True
        return False


#: Standard p-value threshold map (APA convention)
P_VALUE_THRESHOLDS = ThresholdMap(
    {0.001: "very_significant", 0.01: "significant", 0.05: "marginal"},
    default_tier="not_significant",
)

#: Correlation strength threshold map (Cohen 1988)
CORRELATION_THRESHOLDS = ThresholdMap(
    {0.10: "negligible", 0.30: "small", 0.50: "medium"},
    default_tier="large",
)


# ===========================================================================
# LAYER B — TickerStyle & TickerConfig dataclasses
# No Manim dependency.  These are pure configuration objects.
# ===========================================================================

@dataclass
class TickerStyle:
    """
    Visual styling parameters for a Ticker3D badge.

    Attributes
    ----------
    font_size : int
        Font size for the value text.  Label is ``font_size * 0.7``.
    badge_padding_x, badge_padding_y : float
        Internal horizontal / vertical padding inside the badge rectangle.
    badge_corner_radius : float
        Rounding radius of the badge rectangle.
    badge_opacity : float
        Fill opacity of the badge background.
    stroke_width : float
        Stroke width of the badge border and flash ring.
    delta_font_scale : float
        Scale of the delta label relative to value font size.
    label_gap : float
        Horizontal gap between label and value texts.
    unit_gap : float
        Horizontal gap between value and unit texts.
    show_badge : bool
        Whether to draw the badge background at all.
    """
    font_size:           int   = 36
    badge_padding_x:     float = 0.20
    badge_padding_y:     float = 0.12
    badge_corner_radius: float = 0.10
    badge_opacity:       float = 0.88
    stroke_width:        float = 1.2
    delta_font_scale:    float = 0.55
    label_gap:           float = 0.08
    unit_gap:            float = 0.06
    show_badge:          bool  = True

    @property
    def label_font_size(self) -> int:
        return max(12, round(self.font_size * 0.70))

    @property
    def unit_font_size(self) -> int:
        return max(10, round(self.font_size * 0.60))

    @property
    def delta_font_size(self) -> int:
        return max(10, round(self.font_size * self.delta_font_scale))


#: Compact badge for dense dashboards
COMPACT_STYLE = TickerStyle(
    font_size=24, badge_padding_x=0.14, badge_padding_y=0.08,
    badge_corner_radius=0.07, delta_font_scale=0.50,
)

#: Large badge for prominent single-statistic display
LARGE_STYLE = TickerStyle(
    font_size=52, badge_padding_x=0.28, badge_padding_y=0.16,
    badge_corner_radius=0.14, delta_font_scale=0.48,
)

#: Minimal — value only, no badge
MINIMAL_STYLE = TickerStyle(
    font_size=32, show_badge=False, badge_opacity=0,
)


@dataclass
class TickerColors:
    """
    Color scheme for a Ticker3D.  All stored as hex strings so they work
    with or without Manim's color system.

    Attributes
    ----------
    value_color : str
        Main color of the value text.
    label_color : str
        Color of the label/unit text.
    badge_bg : str
        Badge background fill color.
    badge_border : str
        Badge border stroke color.
    flash_ring : str
        Color of the animated highlight ring.
    delta_up : str
        Color of the delta indicator when value increased.
    delta_down : str
        Color of the delta indicator when value decreased.
    """
    value_color:   str = "#afa9ec"   # PURPLE_200
    label_color:   str = "#888780"   # GRAY_600
    badge_bg:      str = "#161620"   # BG_MEDIUM
    badge_border:  str = "#3e3490"   # PURPLE_700
    flash_ring:    str = "#FFD700"   # gold
    delta_up:      str = "#1d9d74"   # TEAL_400
    delta_down:    str = "#d95b31"   # CORAL_400

    def resolve(self, key: str):
        """Return the hex string for attribute *key*."""
        return getattr(self, key)


#: Default dark-theme colors (matches DARK_THEME)
DARK_TICKER_COLORS = TickerColors()

#: Light-theme ticker colors
LIGHT_TICKER_COLORS = TickerColors(
    value_color="#3e3490",
    label_color="#5f5e5a",
    badge_bg="#f4f2ec",
    badge_border="#afa9ec",
    flash_ring="#534ab7",
    delta_up="#0f6e56",
    delta_down="#993d1d",
)

#: p-value specific colors (coral tones)
PVALUE_TICKER_COLORS = TickerColors(
    value_color="#f5c4b3",
    label_color="#c09070",
    badge_bg="#1a0d08",
    badge_border="#b84830",
    flash_ring="#f5c4b3",
    delta_up="#1d9d74",
    delta_down="#d95b31",
)

#: Correlation-specific colors (purple family)
CORR_TICKER_COLORS = TickerColors(
    value_color="#afa9ec",
    label_color="#7f77dd",
    badge_bg="#161224",
    badge_border="#534ab7",
    flash_ring="#afa9ec",
    delta_up="#1d9d74",
    delta_down="#d95b31",
)


# ===========================================================================
# LAYER C — Ticker3D
# The core animated counter Manim VGroup.
# ===========================================================================

class Ticker3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    An animated statistical value badge for Manim scenes.

    A Ticker3D displays a single statistic as a styled badge containing:
    * An optional label (e.g. ``R² =``).
    * An animated numeric value (``DecimalNumber``).
    * An optional unit string (e.g. ``%``).
    * An optional delta indicator showing the last change (``▲0.042``).
    * A coloured badge background rectangle.
    * A gold highlight ring that flashes during value changes.

    Parameters
    ----------
    value : float
        Initial numeric value.
    label : str
        Label string (plain text or LaTeX).  Displayed left of the value.
        Use a LaTeX string like ``r"R^2 ="`` for math symbols.
        Default ``""`` (no label).
    unit : str
        Unit string appended right of the value (plain text).
        Default ``""`` (no unit).
    fmt : TickerFormat
        Number display format.  Default ``TickerFormat.AUTO``.
    precision : int
        Decimal places (or significant figures for AUTO).  Default 4.
    style : TickerStyle
        Visual styling parameters.
    colors : TickerColors
        Color scheme.
    threshold_map : ThresholdMap, optional
        When provided, the badge colour updates automatically to reflect
        whether the current value is significant / critical / etc.
    tier_colors : dict[str, str], optional
        Mapping from threshold tier name → badge border hex color.
        E.g. ``{"critical": "#993c1d", "not_significant": "#1d6e56"}``.
    show_delta : bool
        Whether to show the delta indicator.  Default False.
    use_math_label : bool
        If True, render the label as ``MathTex``; otherwise as ``Text``.

    Key sub-mobjects
    ----------------
    .badge_bg       : RoundedRectangle
    .flash_ring     : RoundedRectangle (hidden by default)
    .label_mob      : Text or MathTex
    .value_mob      : DecimalNumber
    .unit_mob       : Text
    .delta_mob      : Text (hidden when show_delta=False)

    Instant setters (no animation)
    -------------------------------
    .set_value(v)                   update value and recolour badge if thresholds set
    .set_label(s)                   replace label text
    .set_unit(s)                    replace unit text
    .highlight()                    show flash_ring
    .unhighlight()                  hide flash_ring
    .show_delta_indicator(show)     toggle delta text visibility

    Animated methods (return Manim Animation objects)
    -------------------------------------------------
    .count_to(target, run_time, rate_func)
        Smoothly count from current value to *target* using
        ``ChangeDecimalToValue`` with a simultaneous badge recolour.
    .odometer_to(target, run_time)
        Digit-by-digit rolling odometer animation using a custom
        ``ValueTracker`` updater.
    .flash_change(target, flash_color, run_time)
        Simultaneously count and flash the highlight ring gold.
    .pulse(run_time)
        Scale-pulse the badge once (attention-getting).
    .write_in(run_time)
        FadeIn the entire badge (first appearance).
    .fade_out(run_time)
        FadeOut the entire badge.

    Dashboard helpers
    -----------------
    .update_from_stat(value, label, run_time)
        Update value and label together (for live scene updates).
    .clone()
        Return a deep-copy at the same position.
    """

    def __init__(
        self,
        value:          float                   = 0.0,
        label:          str                     = "",
        unit:           str                     = "",
        fmt:            TickerFormat            = TickerFormat.AUTO,
        precision:      int                     = 4,
        style:          TickerStyle             = None,
        colors:         TickerColors            = None,
        threshold_map:  Optional[ThresholdMap]  = None,
        tier_colors:    Optional[Dict[str, str]] = None,
        show_delta:     bool                    = False,
        use_math_label: bool                    = True,
        **kwargs,
    ) -> None:
        _require_manim("Ticker3D.__init__")
        super().__init__(**kwargs)

        self._value        = float(value)
        self._prev_value   = float(value)
        self._label_str    = label
        self._unit_str     = unit
        self._fmt          = fmt
        self._precision    = precision
        self._style        = style or TickerStyle()
        self._colors       = colors or DARK_TICKER_COLORS
        self._threshold_map= threshold_map
        self._tier_colors  = tier_colors or {}
        self._show_delta   = show_delta
        self._use_math     = use_math_label

        self._build()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Construct all sub-mobjects from scratch."""
        s = self._style
        c = self._colors

        value_color  = ManimColor(c.value_color)
        label_color  = ManimColor(c.label_color)
        badge_color  = ManimColor(c.badge_bg)
        border_color = ManimColor(c.badge_border)
        ring_color   = ManimColor(c.flash_ring)
        delta_up     = ManimColor(c.delta_up)
        delta_down   = ManimColor(c.delta_down)

        # ---- Label ----
        if self._label_str:
            if self._use_math:
                self.label_mob = MathTex(
                    self._label_str,
                    font_size = s.label_font_size,
                    color     = label_color,
                )
            else:
                self.label_mob = Text(
                    self._label_str,
                    font_size = s.label_font_size,
                    color     = label_color,
                )
        else:
            # Invisible placeholder so layout is consistent
            self.label_mob = Text("", font_size=s.label_font_size)

        # ---- Value ----
        dp = precision_for_format(self._fmt, self._precision, self._value)
        self.value_mob = DecimalNumber(
            self._value,
            num_decimal_places = dp,
            include_sign       = (self._fmt == TickerFormat.CORRELATION),
            font_size          = s.font_size,
            color              = value_color,
        )

        # ---- Unit ----
        self.unit_mob = Text(
            self._unit_str,
            font_size = s.unit_font_size,
            color     = label_color,
        )

        # ---- Delta indicator ----
        delta_str = delta_string(self._prev_value, self._value, self._precision)
        d_color   = delta_up if self._value >= self._prev_value else delta_down
        self.delta_mob = Text(
            delta_str,
            font_size = s.delta_font_size,
            color     = d_color,
        )
        if not self._show_delta:
            self.delta_mob.set_opacity(0)

        # ---- Arrange horizontally ----
        mobs = []
        if self._label_str:
            mobs.append(self.label_mob)
        mobs.append(self.value_mob)
        if self._unit_str:
            mobs.append(self.unit_mob)

        # Position: label | gap | value | gap | unit
        x_cursor = 0.0
        for i, mob in enumerate(mobs):
            gap = s.label_gap if i > 0 else 0.0
            mob.move_to([x_cursor + gap + mob.width / 2, 0, 0])
            x_cursor += gap + mob.width

        # Delta sits above the value
        self.delta_mob.next_to(self.value_mob, UP, buff=0.04)

        # ---- Badge background ----
        content_width  = x_cursor
        content_height = max(
            (self.label_mob.height if self._label_str else 0),
            self.value_mob.height,
            (self.unit_mob.height if self._unit_str else 0),
        )
        bw = content_width  + 2 * s.badge_padding_x
        bh = content_height + 2 * s.badge_padding_y

        # Shift all content mobs so the badge is centred at origin
        shift_x = -content_width / 2
        for mob in [self.label_mob, self.value_mob, self.unit_mob, self.delta_mob]:
            mob.shift([shift_x, 0, 0])

        if s.show_badge:
            self.badge_bg = RoundedRectangle(
                width         = bw,
                height        = bh,
                corner_radius = s.badge_corner_radius,
                fill_color    = badge_color,
                fill_opacity  = s.badge_opacity,
                stroke_color  = border_color,
                stroke_width  = s.stroke_width,
            )
        else:
            self.badge_bg = RoundedRectangle(
                width=bw, height=bh,
                fill_opacity=0, stroke_width=0,
            )

        # ---- Flash ring (hidden initially) ----
        self.flash_ring = RoundedRectangle(
            width         = bw  + 0.10,
            height        = bh  + 0.10,
            corner_radius = s.badge_corner_radius + 0.05,
            fill_opacity  = 0,
            stroke_color  = ring_color,
            stroke_width  = s.stroke_width * 2.2,
        )
        self.flash_ring.set_opacity(0)

        # ---- Assemble ----
        self.add(self.badge_bg)
        self.add(self.flash_ring)
        if self._label_str:
            self.add(self.label_mob)
        self.add(self.value_mob)
        if self._unit_str:
            self.add(self.unit_mob)
        self.add(self.delta_mob)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_value(self) -> float:
        return self._value

    @property
    def label_text(self) -> str:
        return self._label_str

    # ------------------------------------------------------------------
    # Instant setters
    # ------------------------------------------------------------------

    def set_value(self, value: float) -> "Ticker3D":
        """
        Instantly update the displayed value (no animation).

        Also updates the delta indicator and applies threshold colouring.
        """
        self._prev_value = self._value
        self._value      = float(value)
        self.value_mob.set_value(value)
        self._update_threshold_color()
        self._refresh_delta()
        return self

    def set_label(self, label: str) -> "Ticker3D":
        """Replace the label string (no animation)."""
        self._label_str = label
        if self._label_str and self._use_math:
            self.label_mob.become(
                MathTex(label, font_size=self._style.label_font_size,
                        color=ManimColor(self._colors.label_color))
            )
        elif self._label_str:
            self.label_mob.become(
                Text(label, font_size=self._style.label_font_size,
                     color=ManimColor(self._colors.label_color))
            )
        return self

    def set_unit(self, unit: str) -> "Ticker3D":
        """Replace the unit string (no animation)."""
        self._unit_str = unit
        self.unit_mob.become(
            Text(unit, font_size=self._style.unit_font_size,
                 color=ManimColor(self._colors.label_color))
        )
        return self

    def highlight(self) -> "Ticker3D":
        """Show the gold flash ring."""
        self.flash_ring.set_opacity(1)
        return self

    def unhighlight(self) -> "Ticker3D":
        """Hide the flash ring."""
        self.flash_ring.set_opacity(0)
        return self

    def show_delta_indicator(self, show: bool = True) -> "Ticker3D":
        """Toggle the delta indicator visibility."""
        self._show_delta = show
        self.delta_mob.set_opacity(1.0 if show else 0.0)
        return self

    # ------------------------------------------------------------------
    # Internal colour helpers
    # ------------------------------------------------------------------

    def _update_threshold_color(self) -> None:
        """Recolour the badge border if a threshold_map and tier_colors are set."""
        if self._threshold_map is None or not self._tier_colors:
            return
        tier  = self._threshold_map.tier(self._value)
        color = self._tier_colors.get(tier)
        if color and hasattr(self, "badge_bg"):
            self.badge_bg.set_stroke(color=ManimColor(color))

    def _refresh_delta(self) -> None:
        """Update the delta text and colour."""
        if not self._show_delta:
            return
        delta_str = delta_string(self._prev_value, self._value, self._precision)
        going_up  = self._value >= self._prev_value
        d_color   = ManimColor(self._colors.delta_up if going_up else self._colors.delta_down)
        self.delta_mob.become(
            Text(delta_str, font_size=self._style.delta_font_size, color=d_color)
            .next_to(self.value_mob, UP, buff=0.04)
        )

    # ------------------------------------------------------------------
    # Animated methods
    # ------------------------------------------------------------------

    def count_to(
        self,
        target:    float,
        run_time:  float = 1.2,
        rate_func          = None,
    ) -> "mn.Animation":
        """
        Smoothly count from the current value to *target*.

        Uses ``ChangeDecimalToValue`` so Manim interpolates the displayed
        digits at every rendered frame.  The badge border colour updates at
        the end of the animation via a simultaneous ``ApplyMethod``.

        Parameters
        ----------
        target : float
        run_time : float
        rate_func : callable, optional
            Manim rate function.  Default ``rate_functions.ease_out_cubic``.

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("count_to")
        rf = rate_func or rate_functions.ease_out_cubic

        self._prev_value = self._value
        self._value      = float(target)

        count_anim = ChangeDecimalToValue(
            self.value_mob, target,
            run_time  = run_time,
            rate_func = rf,
        )
        # Flash ring brightens then fades during the count
        ring_anim = Succession(
            self.flash_ring.animate(run_time=run_time * 0.35)
            .set_opacity(0.8),
            self.flash_ring.animate(run_time=run_time * 0.65)
            .set_opacity(0),
        )
        return AnimationGroup(count_anim, ring_anim)

    def odometer_to(
        self,
        target:   float,
        run_time: float = 1.5,
    ) -> "mn.Animation":
        """
        Digit-by-digit "odometer" roll from the current value to *target*.

        Implementation
        --------------
        A ``ValueTracker`` is driven from the current value to *target*.
        An ``add_updater`` callback rebuilds the ``DecimalNumber`` text from
        the tracker's current value at every frame.  The tracker is then
        animated with ``tracker.animate.set_value(target)``, producing the
        classic rolling-digits effect.

        Parameters
        ----------
        target : float
        run_time : float

        Returns
        -------
        manim.Succession
        """
        _require_manim("odometer_to")
        tracker = ValueTracker(self._value)

        def _updater(mob):
            mob.set_value(tracker.get_value())

        self.value_mob.add_updater(_updater)

        self._prev_value = self._value
        self._value      = float(target)

        def _cleanup(_):
            self.value_mob.remove_updater(_updater)
            self._update_threshold_color()
            self._refresh_delta()

        roll_anim = tracker.animate(
            run_time  = run_time,
            rate_func = rate_functions.ease_in_out_cubic,
        ).set_value(target)

        return Succession(
            roll_anim,
            mn.UpdateFromFunc(self.value_mob, _cleanup, run_time=0.001),
        )

    def flash_change(
        self,
        target:      float,
        flash_color             = None,
        run_time:    float      = 1.0,
    ) -> "mn.AnimationGroup":
        """
        Count to *target* while simultaneously flashing the ring.

        The ring pulses gold (or *flash_color*) as the number rolls in,
        then fades back to invisible.  A ``Flash`` burst radiates from the
        badge on arrival.

        Parameters
        ----------
        target : float
        flash_color : ManimColor, optional
            Override the ring colour (default: badge's flash_ring color).
        run_time : float

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("flash_change")
        fc = flash_color or ManimColor(self._colors.flash_ring)
        self.flash_ring.set_stroke(color=fc)

        count_anim = ChangeDecimalToValue(
            self.value_mob, target,
            run_time  = run_time * 0.75,
            rate_func = rate_functions.ease_out_cubic,
        )
        ring_in    = self.flash_ring.animate(run_time=run_time * 0.30).set_opacity(1.0)
        ring_out   = self.flash_ring.animate(run_time=run_time * 0.35).set_opacity(0.0)
        burst      = Flash(
            self.badge_bg,
            color        = fc,
            flash_radius = max(self.badge_bg.width, self.badge_bg.height) * 0.65,
            line_length  = 0.14,
            num_lines    = 10,
            run_time     = run_time * 0.40,
        )

        self._prev_value = self._value
        self._value      = float(target)
        self._update_threshold_color()
        self._refresh_delta()

        return Succession(
            AnimationGroup(count_anim, ring_in),
            AnimationGroup(ring_out, burst),
        )

    def pulse(self, run_time: float = 0.5) -> "mn.Animation":
        """
        Scale-pulse the entire badge once for attention.

        Grows to 1.08× then returns to normal size.
        """
        _require_manim("pulse")
        return Succession(
            self.animate(run_time=run_time * 0.45,
                         rate_func=rate_functions.ease_out_cubic).scale(1.08),
            self.animate(run_time=run_time * 0.55,
                         rate_func=rate_functions.ease_out_back).scale(1 / 1.08),
        )

    def write_in(self, run_time: float = 0.8) -> "mn.Animation":
        """FadeIn the badge from zero opacity."""
        _require_manim("write_in")
        self.set_opacity(0)
        return FadeIn(self, run_time=run_time, scale=1.15)

    def fade_out(self, run_time: float = 0.6) -> "mn.Animation":
        """FadeOut the badge."""
        _require_manim("fade_out")
        return FadeOut(self, run_time=run_time)

    def update_from_stat(
        self,
        value:    float,
        label:    Optional[str] = None,
        run_time: float         = 0.8,
    ) -> "mn.Animation":
        """
        Convenience: update value (animated) and optionally set a new label
        (instant) in one call.

        Returns the ``count_to`` animation for the value.
        """
        _require_manim("update_from_stat")
        if label is not None:
            self.set_label(label)
        return self.count_to(value, run_time=run_time)

    def clone(self) -> "Ticker3D":
        """Return an independent copy at the same position."""
        _require_manim("clone")
        t = Ticker3D(
            value         = self._value,
            label         = self._label_str,
            unit          = self._unit_str,
            fmt           = self._fmt,
            precision     = self._precision,
            style         = self._style,
            colors        = self._colors,
            threshold_map = self._threshold_map,
            tier_colors   = self._tier_colors,
            show_delta    = self._show_delta,
            use_math_label= self._use_math,
        )
        t.move_to(self.get_center())
        return t

    def __repr__(self) -> str:
        val_str = format_value(self._value, self._fmt, self._precision)
        return f"Ticker3D({self._label_str!r} = {val_str})"


# ---------------------------------------------------------------------------
# Formatting helper used inside _build
# ---------------------------------------------------------------------------

def precision_for_format(
    fmt:       TickerFormat,
    precision: int,
    value:     float,
) -> int:
    """
    Return the ``num_decimal_places`` argument for ``DecimalNumber``.

    For scientific and auto formats, DecimalNumber shows fixed decimal places
    in its numeric value; we adjust based on the format so the badge looks
    right even before the first animation.
    """
    if fmt in (TickerFormat.INT,):
        return 0
    if fmt == TickerFormat.PVALUE:
        return 4
    if fmt == TickerFormat.CORRELATION:
        return 4
    if fmt == TickerFormat.PERCENT:
        return max(0, precision - 2)
    return precision


# ===========================================================================
# LAYER D — Specialised Ticker3D subclasses
# ===========================================================================

class PValueTicker3D(Ticker3D):
    """
    A Ticker3D pre-configured for p-value display.

    Automatically:
    * Formats values as p-values (``< 0.001`` for very small values).
    * Shows APA significance stars (``*``, ``**``, ``***``).
    * Colours the badge border according to significance level:
        - ``p < 0.001``  coral (very significant)
        - ``p < 0.01``   amber (significant)
        - ``p < 0.05``   amber-light (marginal)
        - ``p ≥ 0.05``   teal (not significant)

    Parameters
    ----------
    value : float
        Initial p-value.
    label : str
        Default ``r"p ="``
    alpha : float
        Significance level used to determine the flash trigger.
        Default 0.05.
    show_stars : bool
        If True, render significance stars next to the value.  Default True.

    Additional animations
    ---------------------
    .flash_significant(run_time)
        Flash the ring if the current p-value < alpha.
    .set_alpha_level(alpha)
        Update the significance level without changing the value.
    """

    def __init__(
        self,
        value:      float = 1.0,
        label:      str   = r"p =",
        alpha:      float = 0.05,
        show_stars: bool  = True,
        **kwargs,
    ) -> None:
        self._alpha      = alpha
        self._show_stars = show_stars

        # Pre-build tier colors
        if _COLORS_AVAILABLE:
            tier_colors = {
                "very_significant": CORAL_600.hex,
                "significant":      AMBER_600.hex,
                "marginal":         AMBER_200.hex,
                "not_significant":  TEAL_200.hex,
            }
        else:
            tier_colors = {}

        super().__init__(
            value         = value,
            label         = label,
            fmt           = TickerFormat.PVALUE,
            precision     = 4,
            colors        = PVALUE_TICKER_COLORS,
            threshold_map = P_VALUE_THRESHOLDS,
            tier_colors   = tier_colors,
            **kwargs,
        )

        # Build stars mob
        if show_stars and _MANIM_AVAILABLE:
            stars_str  = significance_stars(value)
            star_color = (ManimColor(CORAL_600.hex) if _COLORS_AVAILABLE else RED)
            self.stars_mob = Text(
                stars_str,
                font_size = self._style.unit_font_size,
                color     = star_color,
            )
            self.stars_mob.next_to(self.value_mob, UR if False else RIGHT, buff=0.06)
            self.add(self.stars_mob)

    def set_value(self, value: float) -> "PValueTicker3D":
        super().set_value(value)
        if self._show_stars and hasattr(self, "stars_mob"):
            stars_str = significance_stars(value)
            self.stars_mob.become(
                Text(
                    stars_str,
                    font_size = self._style.unit_font_size,
                    color     = (ManimColor(CORAL_600.hex) if _COLORS_AVAILABLE else RED),
                ).next_to(self.value_mob, RIGHT, buff=0.06)
            )
        return self

    def flash_significant(self, run_time: float = 0.8) -> "mn.Animation":
        """Flash the ring if p < alpha."""
        _require_manim("flash_significant")
        if self._value < self._alpha:
            fc = ManimColor(CORAL_600.hex) if _COLORS_AVAILABLE else RED
            return self.pulse(run_time=run_time)
        return AnimationGroup()

    def set_alpha_level(self, alpha: float) -> "PValueTicker3D":
        """Update significance threshold (no visual change)."""
        self._alpha = alpha
        return self

    def count_to(self, target: float, run_time: float = 1.0, **kwargs) -> "mn.Animation":
        anim = super().count_to(target, run_time=run_time, **kwargs)
        # Also animate stars
        if self._show_stars and hasattr(self, "stars_mob") and _MANIM_AVAILABLE:
            new_stars = significance_stars(target)
            stars_anim = Succession(
                FadeOut(self.stars_mob, run_time=run_time * 0.30),
                mn.UpdateFromFunc(
                    self.stars_mob,
                    lambda m: m.become(
                        Text(new_stars, font_size=self._style.unit_font_size,
                             color=(ManimColor(CORAL_600.hex) if _COLORS_AVAILABLE else RED))
                        .next_to(self.value_mob, RIGHT, buff=0.06)
                    ),
                    run_time=0.001,
                ),
                FadeIn(self.stars_mob, run_time=run_time * 0.30),
            )
            return AnimationGroup(anim, stars_anim)
        return anim


class StatsCounter3D(Ticker3D):
    """
    A Ticker3D specialised for non-negative integer statistics: sample size n,
    degrees of freedom df, number of predictors k, iteration count, etc.

    Integers are displayed with comma thousands-separators (``1,234``).
    The default animation is the odometer roll.

    Parameters
    ----------
    value : int
        Initial integer value.
    label : str
        Default ``"n ="``
    min_value, max_value : int, optional
        Clamp range for ``increment`` / ``decrement``.

    Additional methods
    ------------------
    .increment(by, run_time)   — add *by* with odometer animation
    .decrement(by, run_time)   — subtract *by* with odometer animation
    .set_int(v)                — instant update (alias for set_value)
    """

    def __init__(
        self,
        value:     int  = 0,
        label:     str  = r"n =",
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._min = min_value
        self._max = max_value
        super().__init__(
            value     = float(value),
            label     = label,
            fmt       = TickerFormat.INT,
            precision = 0,
            **kwargs,
        )

    def _clamp(self, v: float) -> float:
        if self._min is not None:
            v = max(float(self._min), v)
        if self._max is not None:
            v = min(float(self._max), v)
        return v

    def increment(self, by: int = 1, run_time: float = 0.6) -> "mn.Animation":
        """Add *by* with an odometer roll."""
        _require_manim("increment")
        target = self._clamp(self._value + by)
        return self.odometer_to(target, run_time=run_time)

    def decrement(self, by: int = 1, run_time: float = 0.6) -> "mn.Animation":
        """Subtract *by* with an odometer roll."""
        _require_manim("decrement")
        target = self._clamp(self._value - by)
        return self.odometer_to(target, run_time=run_time)

    def set_int(self, value: int) -> "StatsCounter3D":
        """Alias for set_value with integer input."""
        return self.set_value(float(value))


class CorrelationTicker3D(Ticker3D):
    """
    A Ticker3D for displaying a correlation coefficient r ∈ [-1, 1].

    Color ramp: strong positive (teal) → zero (neutral gray) → strong negative (coral).
    The badge border and value text colour interpolate continuously as r changes.

    An optional CI bracket annotation can be shown below the value.

    Parameters
    ----------
    value : float
        Initial r value (clamped to [-1, 1]).
    label : str
        Default ``r"r ="``
    show_ci : bool
        Show a ``[lo, hi]`` confidence interval annotation.  Default False.
    ci_low, ci_high : float
        Initial CI bounds (used only when show_ci=True).

    Additional methods
    ------------------
    .morph_r(target, run_time)
        Count + recolour.
    .show_ci_annotation(lo, hi, run_time)
        Fade in the CI text below the badge.
    .hide_ci_annotation(run_time)
        Fade out the CI text.
    """

    def __init__(
        self,
        value:    float = 0.0,
        label:    str   = r"r =",
        show_ci:  bool  = False,
        ci_low:   float = -1.0,
        ci_high:  float = 1.0,
        **kwargs,
    ) -> None:
        self._show_ci  = show_ci
        self._ci_low   = ci_low
        self._ci_high  = ci_high

        super().__init__(
            value     = max(-1.0, min(1.0, value)),
            label     = label,
            fmt       = TickerFormat.CORRELATION,
            precision = 4,
            colors    = CORR_TICKER_COLORS,
            **kwargs,
        )

        # CI annotation
        if show_ci and _MANIM_AVAILABLE:
            self._build_ci_mob(ci_low, ci_high)
        elif _MANIM_AVAILABLE:
            self.ci_mob = Text("", font_size=12)

        self._recolour_for_r(value)

    def _recolour_for_r(self, r: float) -> None:
        """Interpolate badge border and value text colour based on r."""
        if not _MANIM_AVAILABLE or not _COLORS_AVAILABLE:
            return
        r = max(-1.0, min(1.0, r))
        if r >= 0:
            t     = r
            color = interpolate_color(
                ManimColor(GRAY_400.hex),
                ManimColor(TEAL_600.hex),
                t,
            )
        else:
            t     = -r
            color = interpolate_color(
                ManimColor(GRAY_400.hex),
                ManimColor(CORAL_600.hex),
                t,
            )
        self.badge_bg.set_stroke(color=color)
        self.value_mob.set_color(color)

    def _build_ci_mob(self, lo: float, hi: float) -> None:
        ci_str = f"95% CI [{lo:+.3f}, {hi:+.3f}]"
        if _COLORS_AVAILABLE:
            ci_color = ManimColor(GRAY_400.hex)
        else:
            ci_color = GRAY
        self.ci_mob = Text(ci_str, font_size=14, color=ci_color)
        self.ci_mob.next_to(self.badge_bg, DOWN, buff=0.10)
        self.add(self.ci_mob)

    def set_value(self, value: float) -> "CorrelationTicker3D":
        value = max(-1.0, min(1.0, value))
        super().set_value(value)
        self._recolour_for_r(value)
        return self

    def morph_r(self, target: float, run_time: float = 1.2) -> "mn.AnimationGroup":
        """Smooth count + live colour interpolation."""
        _require_manim("morph_r")
        target = max(-1.0, min(1.0, target))
        start  = self._value

        tracker = ValueTracker(start)

        def _updater(mob):
            r = tracker.get_value()
            mob.set_value(r)
            self._recolour_for_r(r)

        self.value_mob.add_updater(_updater)
        self._value = target

        def _cleanup(_):
            self.value_mob.remove_updater(_updater)

        return Succession(
            tracker.animate(
                run_time  = run_time,
                rate_func = rate_functions.ease_in_out_sine,
            ).set_value(target),
            mn.UpdateFromFunc(self.value_mob, _cleanup, run_time=0.001),
        )

    def show_ci_annotation(
        self,
        lo:       float,
        hi:       float,
        run_time: float = 0.6,
    ) -> "mn.Animation":
        """Fade in a CI annotation text below the badge."""
        _require_manim("show_ci_annotation")
        self._ci_low  = lo
        self._ci_high = hi
        ci_str = f"95% CI [{lo:+.3f}, {hi:+.3f}]"
        ci_color = ManimColor(GRAY_400.hex) if _COLORS_AVAILABLE else GRAY
        self.ci_mob.become(
            Text(ci_str, font_size=14, color=ci_color)
            .next_to(self.badge_bg, DOWN, buff=0.10)
        )
        return FadeIn(self.ci_mob, run_time=run_time)

    def hide_ci_annotation(self, run_time: float = 0.4) -> "mn.Animation":
        """Fade out the CI annotation."""
        _require_manim("hide_ci_annotation")
        return FadeOut(self.ci_mob, run_time=run_time)


# ===========================================================================
# LAYER E — TickerGroup3D
# ===========================================================================

class TickerGroup3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    A dashboard of multiple ``Ticker3D`` objects arranged in a row, column,
    or grid layout.

    Parameters
    ----------
    tickers : list[Ticker3D]
        Pre-built ticker objects to arrange.
    layout : {'row', 'column', 'grid'}
        Spatial arrangement.
    n_cols : int
        Number of columns when ``layout='grid'``.  Ignored otherwise.
    spacing : float
        Gap between ticker badges.

    Key sub-mobjects
    ----------------
    .tickers : list[Ticker3D]
        The individual ticker objects (same list as passed in).

    Methods
    -------
    .update_all(values, run_time)
        Animate all tickers to new values simultaneously.
        *values* is a list aligned with *tickers*.
    .update_by_name(mapping, run_time)
        Animate tickers by name.  Ticker must have been named via
        ``ticker._name = "r_squared"`` etc.  *mapping* is ``{name: value}``.
    .flash_significant(alpha, run_time)
        Call ``flash_significant`` on every ``PValueTicker3D`` in the group.
    .highlight_stat(index_or_name, run_time)
        Pulse a specific ticker to draw attention.
    .write_in_sequence(run_time, stagger)
        LaggedStart write-in for all tickers.
    .align_labels()
        Right-align all label texts for a clean column layout.

    Class methods (factories)
    -------------------------
    .from_regression_result(result)
        Build a dashboard for a ``RegressionResult`` with tickers for
        R², Adj R², RMSE, F-stat, p(F).
    .from_correlation_result(corr_result)
        Build a dashboard for a ``CorrelationResult`` with tickers for r,
        p-value, n, 95% CI.
    """

    def __init__(
        self,
        tickers:  List[Ticker3D],
        layout:   str   = "row",
        n_cols:   int   = 3,
        spacing:  float = 0.30,
        **kwargs,
    ) -> None:
        _require_manim("TickerGroup3D.__init__")
        super().__init__(**kwargs)

        self.tickers  = list(tickers)
        self._layout  = layout
        self._n_cols  = n_cols
        self._spacing = spacing
        self._names:  Dict[str, Ticker3D] = {}

        self._arrange(layout, n_cols, spacing)
        for t in self.tickers:
            self.add(t)

    def _arrange(self, layout: str, n_cols: int, spacing: float) -> None:
        """Position all tickers according to the layout spec."""
        if layout == "row":
            x = 0.0
            for t in self.tickers:
                t.move_to([x + t.width / 2, 0, 0])
                x += t.width + spacing
            # Centre the whole row
            total_w = x - spacing
            for t in self.tickers:
                t.shift([-total_w / 2, 0, 0])

        elif layout == "column":
            y = 0.0
            for t in self.tickers:
                t.move_to([0, -y, 0])
                y += t.height + spacing
            total_h = y - spacing
            for t in self.tickers:
                t.shift([0, total_h / 2, 0])

        elif layout == "grid":
            n     = len(self.tickers)
            n_r   = math.ceil(n / n_cols)
            for idx, t in enumerate(self.tickers):
                col = idx % n_cols
                row = idx // n_cols
                # Use a uniform cell size
                cell_w = max(t.width for t in self.tickers) + spacing
                cell_h = max(t.height for t in self.tickers) + spacing
                x = (col - (n_cols - 1) / 2) * cell_w
                y = ((n_r - 1) / 2 - row) * cell_h
                t.move_to([x, y, 0])

    def _by_name(self, name: str) -> Optional[Ticker3D]:
        return self._names.get(name)

    def register_name(self, ticker: Ticker3D, name: str) -> "TickerGroup3D":
        """Register a name→ticker mapping for ``update_by_name``."""
        self._names[name] = ticker
        return self

    # ------------------------------------------------------------------
    # Batch update methods
    # ------------------------------------------------------------------

    def update_all(
        self,
        values:   List[float],
        run_time: float = 1.0,
    ) -> "mn.AnimationGroup":
        """
        Animate all tickers to new values simultaneously.

        Parameters
        ----------
        values : list[float]
            One value per ticker, in the same order as ``self.tickers``.
        run_time : float

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("update_all")
        if len(values) != len(self.tickers):
            raise ValueError(
                f"len(values)={len(values)} must equal "
                f"len(tickers)={len(self.tickers)}."
            )
        anims = [
            t.count_to(v, run_time=run_time)
            for t, v in zip(self.tickers, values)
        ]
        return AnimationGroup(*anims)

    def update_by_name(
        self,
        mapping:  Dict[str, float],
        run_time: float = 1.0,
    ) -> "mn.AnimationGroup":
        """
        Animate tickers by registered name.

        Parameters
        ----------
        mapping : dict[str, float]
            ``{registered_name: target_value}`` pairs.
        run_time : float

        Returns
        -------
        manim.AnimationGroup
        """
        _require_manim("update_by_name")
        anims = []
        for name, value in mapping.items():
            t = self._by_name(name)
            if t is None:
                raise KeyError(
                    f"No ticker registered with name {name!r}. "
                    f"Available: {list(self._names)}"
                )
            anims.append(t.count_to(value, run_time=run_time))
        return AnimationGroup(*anims)

    def flash_significant(
        self,
        alpha:    float = 0.05,
        run_time: float = 0.8,
    ) -> "mn.AnimationGroup":
        """
        Call ``flash_significant`` on every ``PValueTicker3D`` whose
        current value is below *alpha*.
        """
        _require_manim("flash_significant")
        anims = []
        for t in self.tickers:
            if isinstance(t, PValueTicker3D):
                anims.append(t.flash_significant(run_time=run_time))
        return AnimationGroup(*anims) if anims else AnimationGroup()

    def highlight_stat(
        self,
        index_or_name: Union[int, str],
        run_time:      float = 0.6,
    ) -> "mn.Animation":
        """Pulse the specified ticker to draw viewer attention."""
        _require_manim("highlight_stat")
        if isinstance(index_or_name, str):
            t = self._by_name(index_or_name)
            if t is None:
                return AnimationGroup()
        else:
            if index_or_name < 0 or index_or_name >= len(self.tickers):
                return AnimationGroup()
            t = self.tickers[index_or_name]
        return t.pulse(run_time=run_time)

    def write_in_sequence(
        self,
        run_time: float = 1.6,
        stagger:  float = 0.12,
    ) -> "mn.Animation":
        """Staggered write-in of all tickers."""
        _require_manim("write_in_sequence")
        return LaggedStart(
            *[t.write_in(run_time=run_time * 0.50) for t in self.tickers],
            lag_ratio=stagger / run_time,
        )

    def align_labels(self) -> "TickerGroup3D":
        """
        Right-align all label texts for a clean multi-row layout.
        Only meaningful for column/grid layouts.
        """
        if not self.tickers:
            return self
        max_label_w = max(t.label_mob.width for t in self.tickers)
        for t in self.tickers:
            delta = max_label_w - t.label_mob.width
            t.value_mob.shift([delta, 0, 0])
            if t._unit_str:
                t.unit_mob.shift([delta, 0, 0])
        return self

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_regression_result(
        cls,
        result,
        style:   TickerStyle = None,
        layout:  str         = "row",
        spacing: float       = 0.35,
    ) -> "TickerGroup3D":
        """
        Build a standard regression dashboard from a ``RegressionResult``.

        Tickers created:
          R²  Adj R²  RMSE  F-stat  p(F)  n  k

        Parameters
        ----------
        result : RegressionResult
            A fitted OLS model from ``correlation.ols_fit()``.
        style : TickerStyle, optional
        layout, spacing : str, float

        Returns
        -------
        TickerGroup3D
        """
        _require_manim("from_regression_result")
        s = style or COMPACT_STYLE

        tickers = [
            Ticker3D(result.r_squared,      label=r"R^2 =",        style=s),
            Ticker3D(result.adj_r_squared,  label=r"\bar{R}^2 =",  style=s),
            Ticker3D(result.rmse,           label=r"\hat{\sigma} =",style=s),
            Ticker3D(result.f_stat,         label=r"F =",           style=s),
            PValueTicker3D(result.f_p_value, label=r"p(F) =",       style=s),
            StatsCounter3D(result.n,        label=r"n =",           style=s),
            StatsCounter3D(result.k,        label=r"k =",           style=s),
        ]
        group = cls(tickers, layout=layout, spacing=spacing)
        for name, ticker in zip(
            ["r_sq", "adj_r_sq", "rmse", "f_stat", "f_pval", "n", "k"],
            tickers,
        ):
            group.register_name(ticker, name)
        return group

    @classmethod
    def from_correlation_result(
        cls,
        corr_result,
        style:   TickerStyle = None,
        layout:  str         = "row",
        spacing: float       = 0.35,
    ) -> "TickerGroup3D":
        """
        Build a correlation dashboard from a ``CorrelationResult``.

        Tickers: r / rho  p-value  n  95% CI low  95% CI high

        Parameters
        ----------
        corr_result : CorrelationResult
            From ``correlation.pearson()``, ``spearman()``, etc.
        style, layout, spacing : as above

        Returns
        -------
        TickerGroup3D
        """
        _require_manim("from_correlation_result")
        s = style or COMPACT_STYLE

        symbol = {
            "pearson":        r"r =",
            "spearman":       r"\rho =",
            "kendall":        r"\tau =",
            "point_biserial": r"r_{pb} =",
            "phi":            r"\phi =",
            "cramers_v":      r"V =",
            "partial":        r"r_p =",
        }.get(corr_result.method.value, r"r =")

        tickers = [
            CorrelationTicker3D(corr_result.r,       label=symbol,    style=s),
            PValueTicker3D(corr_result.p_value or 1.0, label=r"p =", style=s),
            StatsCounter3D(corr_result.n,              label=r"n =",  style=s),
        ]
        if corr_result.ci_low is not None:
            tickers += [
                CorrelationTicker3D(corr_result.ci_low,  label=r"CI_{lo} =", style=s),
                CorrelationTicker3D(corr_result.ci_high, label=r"CI_{hi} =", style=s),
            ]

        group = cls(tickers, layout=layout, spacing=spacing)
        for name, ticker in zip(
            ["r", "p_value", "n", "ci_low", "ci_high"],
            tickers,
        ):
            group.register_name(ticker, name)
        return group

    def __repr__(self) -> str:
        return f"TickerGroup3D({len(self.tickers)} tickers, layout={self._layout!r})"


# ===========================================================================
# LAYER F — Pre-built dashboard factories (scene-level)
# ===========================================================================

def regression_dashboard(
    result,
    position    = None,
    style:  TickerStyle = None,
    layout: str         = "grid",
    n_cols: int         = 4,
    spacing: float      = 0.30,
) -> "TickerGroup3D":
    """
    Build a complete regression statistics dashboard.

    Seven tickers arranged in a grid: R², Adj R², σ̂, F, p(F), n, k.

    Parameters
    ----------
    result : RegressionResult
    position : array-like, optional
        Where to place the dashboard centre.
    style : TickerStyle, optional
    layout : str
        Grid layout.  Default ``'grid'``.
    n_cols : int
        Columns in grid.  Default 4.
    spacing : float

    Returns
    -------
    TickerGroup3D
    """
    _require_manim("regression_dashboard")
    group = TickerGroup3D.from_regression_result(
        result, style=style, layout=layout, spacing=spacing,
    )
    if layout == "grid":
        group._arrange("grid", n_cols, spacing)
    if position is not None:
        import numpy as np
        group.move_to(np.asarray(position, dtype=float))
    return group


def correlation_dashboard(
    corr_result,
    position    = None,
    style:  TickerStyle = None,
    spacing: float      = 0.28,
) -> "TickerGroup3D":
    """
    Build a correlation statistics dashboard.

    Tickers: r, p-value (with stars), n, 95% CI if available.

    Parameters
    ----------
    corr_result : CorrelationResult
    position : array-like, optional
    style : TickerStyle, optional
    spacing : float

    Returns
    -------
    TickerGroup3D
    """
    _require_manim("correlation_dashboard")
    group = TickerGroup3D.from_correlation_result(
        corr_result, style=style, layout="row", spacing=spacing,
    )
    if position is not None:
        import numpy as np
        group.move_to(np.asarray(position, dtype=float))
    return group


def hypothesis_dashboard(
    test_stat:  float,
    p_value:    float,
    df:         int,
    alpha:      float = 0.05,
    stat_label: str   = r"t =",
    style:  TickerStyle = None,
    spacing: float      = 0.28,
) -> "TickerGroup3D":
    """
    Build a hypothesis test dashboard: test statistic, p-value, df.

    Parameters
    ----------
    test_stat : float
    p_value : float
    df : int
    alpha : float
    stat_label : str
        LaTeX label for the test statistic, e.g. ``r"t ="`` or ``r"\\chi^2 ="``.
    style, spacing

    Returns
    -------
    TickerGroup3D
    """
    _require_manim("hypothesis_dashboard")
    s = style or COMPACT_STYLE
    tickers = [
        Ticker3D(test_stat, label=stat_label, fmt=TickerFormat.FLOAT,
                 precision=3, style=s),
        PValueTicker3D(p_value, label=r"p =", alpha=alpha, style=s),
        StatsCounter3D(df, label=r"df =", style=s),
    ]
    return TickerGroup3D(tickers, layout="row", spacing=spacing)


def distribution_dashboard(
    mean:     float,
    std:      float,
    skewness: float = 0.0,
    kurtosis: float = 0.0,
    n:        int   = 0,
    style:    TickerStyle = None,
    spacing:  float       = 0.28,
) -> "TickerGroup3D":
    """
    Build a descriptive-statistics dashboard for a distribution.

    Tickers: μ, σ, skewness, kurtosis, n.

    Parameters
    ----------
    mean, std, skewness, kurtosis : float
    n : int
    style, spacing

    Returns
    -------
    TickerGroup3D
    """
    _require_manim("distribution_dashboard")
    s = style or COMPACT_STYLE
    tickers = [
        Ticker3D(mean,     label=r"\mu =",     fmt=TickerFormat.FLOAT, style=s),
        Ticker3D(std,      label=r"\sigma =",  fmt=TickerFormat.FLOAT, style=s),
        Ticker3D(skewness, label=r"\gamma_1 =",fmt=TickerFormat.FLOAT,
                 precision=3, style=s),
        Ticker3D(kurtosis, label=r"\gamma_2 =",fmt=TickerFormat.FLOAT,
                 precision=3, style=s),
    ]
    if n > 0:
        tickers.append(StatsCounter3D(n, label=r"n =", style=s))
    return TickerGroup3D(tickers, layout="row", spacing=spacing)


# ===========================================================================
# __all__
# ===========================================================================

__all__ = [
    # Layer A
    "TickerFormat",
    "format_value",
    "auto_precision",
    "significance_stars",
    "delta_string",
    "ThresholdMap",
    "P_VALUE_THRESHOLDS",
    "CORRELATION_THRESHOLDS",
    # Easing functions
    "_ease_linear",
    "_ease_out_cubic",
    "_ease_in_out_cubic",
    "_ease_out_elastic",
    "_ease_out_back",

    # Layer B
    "TickerStyle",
    "TickerColors",
    "COMPACT_STYLE",
    "LARGE_STYLE",
    "MINIMAL_STYLE",
    "DARK_TICKER_COLORS",
    "LIGHT_TICKER_COLORS",
    "PVALUE_TICKER_COLORS",
    "CORR_TICKER_COLORS",
    "precision_for_format",

    # Layer C
    "Ticker3D",

    # Layer D
    "PValueTicker3D",
    "StatsCounter3D",
    "CorrelationTicker3D",

    # Layer E
    "TickerGroup3D",

    # Layer F
    "regression_dashboard",
    "correlation_dashboard",
    "hypothesis_dashboard",
    "distribution_dashboard",
]