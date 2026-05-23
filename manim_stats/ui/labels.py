"""
manim_stats/ui/labels.py
==========================
Production-quality floating annotation and UI overlay objects for Manim.
Every text/label primitive needed to annotate the charts, distributions,
and probability objects across the full manim_stats module lives here.

Objects
-------

Shared utilities
    ``_auto_contrast_color(bg)``
        Returns white or near-black for readable text on any background,
        computed from WCAG perceived luminance.
    ``_significance_stars(p)``
        Returns ("***", color), ("**", color), ("*", color), or ("n.s.", color)
        for a given p-value.
    ``_place_clear_of(pos, existing, min_dist)``
        Returns a nudged position that avoids crowding existing label positions.
    ``_format_value(v, fmt, sci_thresh)``
        Smart formatter: scientific notation for |v| < 0.001 or > 9999,
        percentage for fmt="%", otherwise configurable decimal places.

LabelConfig  (dataclass)
    Centralised visual parameters shared across all label objects.

StatLabel3D  (VGroup)
    The workhorse floating badge.  Combines name + value + optional unit
    on a shaded 3D card prism.  Color-coded by statistic type.
    Supports: delta indicator (Δ with ↑↓ arrows), significance stars,
    [lo, hi] confidence bound line, and a thin connector stem to the
    annotated point.

AnnotationArrow3D  (VGroup)
    Directional Arrow3D from a source label position to a target point.
    Tip styles: filled, open, circle, dot.  Optional pulsing animation.
    Label text at the tail end.  Auto-dodge against a list of placed
    arrow positions.

FormulaPanel3D  (VGroup)
    Floating panel holding one or more MathTex lines.
    Background: shaded prism (same idiom as rest of module).
    Per-term color coding via a color_map dict.
    ``add_line(tex, color)`` adds lines dynamically.
    ``animate_collapse()`` / ``animate_expand()`` fold to a title bar.
    ``animate_write_lines()`` writes lines one by one.

LegendPanel3D  (VGroup)
    Floating legend with one entry per series/category.
    Glyph styles: sphere, square, line, dashed.
    Auto-layout: vertical stacking with consistent spacing.
    ``from_series(colors, labels, style)`` classmethod.

DataCallout3D  (VGroup)
    Speech-bubble annotation pointing at a specific data point.
    Background panel + triangular pointer aimed at the data point.
    Shows (x, y, z) coordinate values and optional rank string.

Ticker3D  (VGroup)
    Animated DecimalNumber that counts from start → target with easing.
    Prefix/suffix text.  ``animate_count(target)`` returns an animation.

AxisLabel3D  (VGroup)
    Enhanced axis label set: tick marks + numeric labels + axis title +
    optional unit badge.  Supports linear, log, percentage, categorical.

TooltipBadge3D  (VGroup)
    Compact multi-field key-value popup badge.
    ``animate_show()`` / ``animate_hide()`` for reveal/dismiss.

StatSummaryBox3D  (VGroup)
    Five-number summary (min, Q1, median, Q3, max) in a structured panel.
    IQR bracket on the side.

HighlightRing3D  (VGroup)
    Pulsing ring orbiting a 3D point — draws the eye to key features.
    ``animate_pulse()`` expands/contracts the ring.
    ``animate_orbit(angle)`` rotates around the target point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from manim import (
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    BLUE, BLUE_E, RED, RED_E, GREEN, GREEN_E, YELLOW, ORANGE, PURPLE,
    UP, DOWN, LEFT, RIGHT,
    PI, TAU,
    VGroup, VMobject,
    Line3D, Polygon, Sphere, Dot3D, Arrow3D, Annulus,
    Text, MathTex, DecimalNumber,
    ParametricFunction,
    FadeIn, FadeOut, GrowFromCenter, GrowFromPoint, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    Rotate, Flash,
    interpolate_color, ValueTracker, always_redraw,
    ManimColor,
    DEGREES,
    rate_functions,
)


# ---------------------------------------------------------------------------
# Semantic color constants
# ---------------------------------------------------------------------------

# Statistic-type colors
STAT_MEAN_COLOR    = ManimColor("#E040FB")   # violet
STAT_MEDIAN_COLOR  = ManimColor("#00BCD4")   # cyan
STAT_STD_COLOR     = ManimColor("#FFD600")   # yellow
STAT_VAR_COLOR     = ManimColor("#FF9800")   # orange
STAT_IQR_COLOR     = ManimColor("#8BC34A")   # light green
STAT_PVAL_SIG      = ManimColor("#00E676")   # green   – p < 0.05
STAT_PVAL_INSIG    = ManimColor("#FF5252")   # red     – p ≥ 0.05
STAT_CORR_POS      = ManimColor("#2979FF")   # blue    – r > 0
STAT_CORR_NEG      = ManimColor("#F44336")   # red     – r < 0
STAT_CI_COLOR      = ManimColor("#80CBC4")   # teal    – confidence interval
STAT_N_COLOR       = ManimColor("#B0BEC5")   # slate   – sample size

# Delta indicators
DELTA_UP_COLOR     = ManimColor("#00E676")   # green   – increase
DELTA_DOWN_COLOR   = ManimColor("#FF5252")   # red     – decrease
DELTA_FLAT_COLOR   = GRAY_C

# Panel backgrounds
PANEL_BG_COLOR     = ManimColor("#1A2530")   # dark blue-grey
PANEL_BORDER_COLOR = ManimColor("#37474F")   # medium slate
PANEL_TITLE_COLOR  = ManimColor("#78909C")   # lighter slate
LABEL_TEXT_COLOR   = ManimColor("#ECEFF1")   # near-white

# Significance star colors
SIG_3STAR = ManimColor("#00E676")
SIG_2STAR = ManimColor("#69F0AE")
SIG_1STAR = ManimColor("#FFD600")
SIG_NS    = GRAY_C

FACE_DARKEN_SIDE  = 0.38
FACE_DARKEN_RIGHT = 0.55


def _dk(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lt(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def _auto_contrast_color(bg: ManimColor) -> ManimColor:
    """WCAG perceived luminance — returns white or near-black for readability."""
    r, g, b = [x / 255.0 for x in bg.to_rgb()]

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return ManimColor("#F8F8F8") if lum < 0.45 else ManimColor("#1A1A1A")


def _significance_stars(p: float) -> tuple[str, ManimColor]:
    """Return (stars_string, color) for a given p-value.

    Returns one of: ``("***", green)``, ``("**", lt-green)``,
    ``("*", yellow)``, ``("n.s.", grey)``.
    """
    if p < 0.001:
        return "***", SIG_3STAR
    elif p < 0.01:
        return "**",  SIG_2STAR
    elif p < 0.05:
        return "*",   SIG_1STAR
    else:
        return "n.s.", SIG_NS


def _place_clear_of(
    desired:   np.ndarray,
    existing:  list[np.ndarray],
    min_dist:  float = 0.40,
    max_tries: int   = 8,
) -> np.ndarray:
    """Return a position close to ``desired`` that is at least
    ``min_dist`` away from all ``existing`` positions.

    Attempts successive offsets in cardinal directions before giving up
    and returning the desired position as-is.
    """
    offsets = [
        np.array([0, 0, 0]),
        np.array([min_dist, 0, 0]),
        np.array([-min_dist, 0, 0]),
        np.array([0, min_dist, 0]),
        np.array([0, -min_dist, 0]),
        np.array([min_dist,  min_dist, 0]),
        np.array([-min_dist, min_dist, 0]),
        np.array([min_dist, -min_dist, 0]),
    ]
    for offset in offsets[:max_tries]:
        candidate = desired + offset
        if all(np.linalg.norm(candidate - ex) >= min_dist for ex in existing):
            return candidate
    return desired


def _format_value(
    v:          float,
    decimals:   int   = 3,
    fmt:        str   = "f",
    sci_thresh: float = 0.001,
) -> str:
    """Smart numeric formatter.

    Parameters
    ----------
    v : float
        The value to format.
    decimals : int
        Decimal places for normal-range values.
    fmt : str
        ``"f"`` → decimal  ``"%"`` → percentage  ``"e"`` → always scientific
    sci_thresh : float
        Values with |v| < sci_thresh use scientific notation.
    """
    if fmt == "%":
        return f"{v * 100:.{max(decimals-2, 0)}f}%"
    if fmt == "e" or (abs(v) < sci_thresh and v != 0) or abs(v) > 9999:
        return f"{v:.{decimals}e}"
    return f"{v:.{decimals}f}"


def _stat_color(stat_type: str) -> ManimColor:
    """Return the canonical color for a named statistic type."""
    return {
        "mean":    STAT_MEAN_COLOR,
        "median":  STAT_MEDIAN_COLOR,
        "std":     STAT_STD_COLOR,
        "sd":      STAT_STD_COLOR,
        "var":     STAT_VAR_COLOR,
        "iqr":     STAT_IQR_COLOR,
        "p":       STAT_PVAL_SIG,
        "pval":    STAT_PVAL_SIG,
        "r":       STAT_CORR_POS,
        "corr":    STAT_CORR_POS,
        "ci":      STAT_CI_COLOR,
        "n":       STAT_N_COLOR,
    }.get(stat_type.lower(), LABEL_TEXT_COLOR)


# ---------------------------------------------------------------------------
# LabelConfig
# ---------------------------------------------------------------------------

@dataclass
class LabelConfig:
    """Centralised visual parameters shared across all label objects.

    Card geometry
    -------------
    card_padding_x : float
        Horizontal padding inside the card background prism.
    card_padding_y : float
        Vertical padding inside the card background prism.
    card_depth : float
        Z depth of the card background prism.
    card_height : float
        Y height of the card background prism (not the text height).
    card_opacity : float
        Fill opacity of the card face.
    card_border_width : float
        Stroke width of the card edges.
    card_border_opacity : float
        Opacity of card edge strokes.

    Typography
    ----------
    name_font_size : int
        Font size for the statistic name / label.
    value_font_size : int
        Font size for the numeric value.
    unit_font_size : int
        Font size for the unit string.
    delta_font_size : int
        Font size for the delta indicator.
    stars_font_size : int
        Font size for significance star strings.

    Arrow / connector
    -----------------
    stem_stroke_width : float
        Width of the thin connector line from card to annotation point.
    stem_opacity : float
        Opacity of the connector stem.
    arrow_tip_length : float
        Length of arrowhead tips.

    Legend
    ------
    legend_glyph_radius : float
        Radius of sphere/square glyphs in legend entries.
    legend_row_spacing : float
        Vertical spacing between legend entries.
    legend_col_width : float
        Width allocated per legend column.

    Panel
    -----
    panel_line_spacing : float
        Vertical spacing between formula/text lines in panels.
    panel_title_font_size : int
        Font size for panel title bars.
    panel_content_font_size : int
        Font size for panel body text.
    panel_min_width : float
        Minimum panel width in Manim units.

    Ticker
    ------
    ticker_font_size : int
        Font size for Ticker3D numbers.
    ticker_decimals : int
        Decimal places displayed by Ticker3D.

    Ring
    ----
    ring_stroke_width : float
        Stroke width of HighlightRing3D.
    ring_pulse_scale : float
        Scale factor for ring pulse animation.
    """

    # ---- card geometry ----
    card_padding_x:     float = 0.18
    card_padding_y:     float = 0.12
    card_depth:         float = 0.12
    card_height:        float = 0.08
    card_opacity:       float = 0.88
    card_border_width:  float = 0.80
    card_border_opacity: float = 0.60

    # ---- typography ----
    name_font_size:   int = 22
    value_font_size:  int = 28
    unit_font_size:   int = 18
    delta_font_size:  int = 20
    stars_font_size:  int = 20

    # ---- arrow / connector ----
    stem_stroke_width: float = 0.90
    stem_opacity:      float = 0.55
    arrow_tip_length:  float = 0.12

    # ---- legend ----
    legend_glyph_radius: float = 0.10
    legend_row_spacing:  float = 0.42
    legend_col_width:    float = 1.80

    # ---- panel ----
    panel_line_spacing:    float = 0.40
    panel_title_font_size: int   = 24
    panel_content_font_size: int = 20
    panel_min_width:       float = 2.50

    # ---- ticker ----
    ticker_font_size: int   = 34
    ticker_decimals:  int   = 2

    # ---- ring ----
    ring_stroke_width: float = 2.00
    ring_pulse_scale:  float = 1.30


# ---------------------------------------------------------------------------
# _CardPrism  —  shared 3-face background card used by multiple objects
# ---------------------------------------------------------------------------

class _CardPrism(VGroup):
    """Thin 3D prism used as a floating label background card.

    Same shaded-face idiom (top/front/right) as all other prisms in
    the module.  Width and height are set from content dimensions.
    """

    def __init__(
        self,
        x0:     float,
        z0:     float,
        width:  float,
        height: float,
        depth:  float,
        color:  ManimColor = PANEL_BG_COLOR,
        cfg:    LabelConfig | None = None,
    ):
        super().__init__()
        cfg = cfg or LabelConfig()
        h   = max(height, 0.005)

        top_color   = color
        front_color = _dk(color, FACE_DARKEN_SIDE)
        right_color = _dk(color, FACE_DARKEN_RIGHT)
        op          = cfg.card_opacity
        bw          = cfg.card_border_width
        bo          = cfg.card_border_opacity

        AFL = np.array([x0,         0,     z0      ])
        AFR = np.array([x0 + width, 0,     z0      ])
        ABL = np.array([x0,         0,     z0+depth])
        ABR = np.array([x0 + width, 0,     z0+depth])
        TFL = np.array([x0,         h,     z0      ])
        TFR = np.array([x0 + width, h,     z0      ])
        TBL = np.array([x0,         h,     z0+depth])
        TBR = np.array([x0 + width, h,     z0+depth])

        def _face(pts, col):
            p = Polygon(*pts, color=col)
            p.set_fill(color=col, opacity=op)
            p.set_stroke(color=_dk(col, 0.50), width=bw, opacity=bo)
            return p

        self.top_face   = _face([TFL, TFR, TBR, TBL], top_color)
        self.front_face = _face([AFL, AFR, TFR, TFL], front_color)
        self.right_face = _face([AFR, ABR, TBR, TFR], right_color)
        self.add(self.front_face, self.right_face, self.top_face)

        self._top_center = np.array([
            x0 + width / 2, h, z0 + depth / 2
        ])

    @property
    def top_center(self) -> np.ndarray:
        return self._top_center.copy()


# ---------------------------------------------------------------------------
# StatLabel3D
# ---------------------------------------------------------------------------

class StatLabel3D(VGroup):
    """Floating annotation badge showing a named statistic value.

    Layout (top view of the card):
        ┌──────────────────────────────────┐
        │  NAME      VALUE  unit  [stars]  │
        │            [Δ indicator]         │
        │            [lo ──── hi]          │
        └──────────────────────────────────┘

    Parameters
    ----------
    name : str
        Statistic label, e.g. ``"mean"``, ``"σ"``, ``"r"``.
    value : float
        Numeric value to display.
    pos : np.ndarray
        World position of the card centre.
    stat_type : str
        Controls automatic card color.  One of: ``"mean"``, ``"median"``,
        ``"std"``, ``"var"``, ``"iqr"``, ``"p"``, ``"r"``, ``"ci"``,
        ``"n"``, or any string (uses default text color).
    unit : str
        Optional unit string appended after the value.
    p_value : float | None
        If set, appends significance stars.
    ci_lo, ci_hi : float | None
        If both set, appends a [lo, hi] confidence bound line.
    delta : float | None
        If set, shows a Δ indicator: the change from a reference value.
        Positive → green ↑, negative → red ↓, zero → grey →.
    use_math_tex : bool
        If True, render name and value with MathTex instead of Text.
    anchor_point : np.ndarray | None
        If set, a thin Line3D connector stem extends from the card to
        this world position.
    config : LabelConfig | None
    """

    def __init__(
        self,
        name:         str,
        value:        float,
        pos:          np.ndarray,
        stat_type:    str        = "",
        unit:         str        = "",
        p_value:      float | None = None,
        ci_lo:        float | None = None,
        ci_hi:        float | None = None,
        delta:        float | None = None,
        use_math_tex: bool         = False,
        anchor_point: np.ndarray | None = None,
        decimals:     int          = 3,
        fmt:          str          = "f",
        config:       LabelConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg = config or LabelConfig()

        # ---- color from stat type ----
        color = _stat_color(stat_type) if stat_type else LABEL_TEXT_COLOR
        self._color = color

        # ---- text elements (built first, card sized around them) ----
        Tx = MathTex if use_math_tex else Text

        name_obj = Tx(name, color=_lt(color, 0.25),
                      font_size=cfg.name_font_size)
        val_str  = _format_value(value, decimals=decimals, fmt=fmt)
        val_obj  = Tx(val_str, color=color,
                      font_size=cfg.value_font_size, weight="BOLD")

        # Arrange name above value on the card top face
        # We place everything at pos + Y offset after card is built
        text_width = max(name_obj.width, val_obj.width)
        extra_w    = 0.0

        # Unit string
        unit_obj = None
        if unit:
            unit_obj = Text(unit, color=_lt(color, 0.40),
                            font_size=cfg.unit_font_size)
            extra_w += unit_obj.width + 0.12

        # Significance stars
        stars_obj = None
        if p_value is not None:
            stars_str, stars_color = _significance_stars(p_value)
            stars_obj = Text(stars_str, color=stars_color,
                             font_size=cfg.stars_font_size)
            extra_w += stars_obj.width + 0.10

        card_w = max(text_width + extra_w + cfg.card_padding_x * 2,
                     cfg.panel_min_width * 0.6)
        card_h = cfg.card_height

        # ---- background card ----
        px, _, pz = float(pos[0]), float(pos[1]), float(pos[2])
        card = _CardPrism(
            x0     = px - card_w / 2,
            z0     = pz - cfg.card_depth / 2,
            width  = card_w,
            height = card_h,
            depth  = cfg.card_depth,
            color  = PANEL_BG_COLOR,
            cfg    = cfg,
        )
        self.add(card)
        self.card = card

        y_top = float(pos[1]) + card_h + cfg.card_padding_y

        # ---- place text on top of card ----
        name_obj.move_to(np.array([px, y_top + 0.24, pz]))
        val_obj.move_to( np.array([px, y_top + 0.52, pz]))
        self.add(name_obj, val_obj)
        self.name_obj = name_obj
        self.val_obj  = val_obj

        # Unit and stars beside value
        x_cursor = px + val_obj.width / 2 + 0.12
        if unit_obj:
            unit_obj.move_to(np.array([x_cursor + unit_obj.width / 2,
                                       y_top + 0.52, pz]))
            self.add(unit_obj)
            x_cursor += unit_obj.width + 0.10

        if stars_obj:
            stars_obj.move_to(np.array([x_cursor + stars_obj.width / 2,
                                        y_top + 0.52, pz]))
            self.add(stars_obj)
            self.stars_obj = stars_obj

        # ---- delta indicator ----
        if delta is not None:
            if delta > 1e-9:
                arrow_char = "↑"
                dcolor = DELTA_UP_COLOR
            elif delta < -1e-9:
                arrow_char = "↓"
                dcolor = DELTA_DOWN_COLOR
            else:
                arrow_char = "→"
                dcolor = DELTA_FLAT_COLOR

            delta_txt = Text(
                f"{arrow_char} {_format_value(abs(delta), decimals=decimals, fmt=fmt)}",
                color=dcolor,
                font_size=cfg.delta_font_size,
            )
            delta_txt.move_to(np.array([px, y_top + 0.80, pz]))
            self.add(delta_txt)
            self.delta_obj = delta_txt

        # ---- confidence interval line ----
        if ci_lo is not None and ci_hi is not None:
            # Horizontal line [lo ─── hi] below the main value
            ci_y  = y_top + 0.08
            span  = max(card_w * 0.60, 0.30)
            lo_x  = px - span / 2
            hi_x  = px + span / 2

            ci_line = Line3D(
                start=np.array([lo_x, ci_y, pz]),
                end  =np.array([hi_x, ci_y, pz]),
                color=STAT_CI_COLOR, stroke_width=1.5,
            )
            ci_line.set_opacity(0.80)
            for tick_x in [lo_x, hi_x]:
                tick = Line3D(
                    start=np.array([tick_x, ci_y - 0.06, pz]),
                    end  =np.array([tick_x, ci_y + 0.06, pz]),
                    color=STAT_CI_COLOR, stroke_width=1.2,
                )
                self.add(tick)
            ci_lo_lbl = Text(
                _format_value(ci_lo, decimals=decimals-1),
                color=STAT_CI_COLOR, font_size=cfg.unit_font_size - 2,
            )
            ci_hi_lbl = Text(
                _format_value(ci_hi, decimals=decimals-1),
                color=STAT_CI_COLOR, font_size=cfg.unit_font_size - 2,
            )
            ci_lo_lbl.move_to(np.array([lo_x, ci_y - 0.22, pz]))
            ci_hi_lbl.move_to(np.array([hi_x, ci_y - 0.22, pz]))
            self.add(ci_line, ci_lo_lbl, ci_hi_lbl)

        # ---- connector stem ----
        if anchor_point is not None:
            stem = Line3D(
                start=np.array([px, float(pos[1]), pz]),
                end  =anchor_point,
                color=color,
                stroke_width=cfg.stem_stroke_width,
            )
            stem.set_opacity(cfg.stem_opacity)
            self.add(stem)
            self.stem = stem

    # ------------------------------------------------------------------

    def animate_appear(self, run_time: float = 0.6) -> GrowFromCenter:
        """Scale the label in from zero."""
        return GrowFromCenter(self, run_time=run_time)

    def animate_update_value(
        self,
        new_value: float,
        decimals:  int   = 3,
        fmt:       str   = "f",
        run_time:  float = 0.5,
    ) -> Transform:
        """Morph the displayed value to ``new_value``.

        Builds a target label and transforms into it.
        """
        new_val_str = _format_value(new_value, decimals=decimals, fmt=fmt)
        target_obj  = Text(new_val_str, color=self._color,
                           font_size=self.cfg.value_font_size, weight="BOLD")
        target_obj.move_to(self.val_obj.get_center())
        return Transform(self.val_obj, target_obj, run_time=run_time)

    def animate_pulse(
        self,
        scale_factor: float = 1.20,
        run_time:     float = 0.45,
    ) -> Succession:
        """Briefly scale up and return — draws attention."""
        return Succession(
            self.animate(run_time=run_time / 2).scale(scale_factor),
            self.animate(run_time=run_time / 2).scale(1 / scale_factor),
        )

    # ---- Convenience factory classmethods ----

    @classmethod
    def mean_label(
        cls,
        value: float, pos: np.ndarray,
        anchor_point: np.ndarray | None = None,
        **kw,
    ) -> "StatLabel3D":
        return cls(r"\mu", value, pos, stat_type="mean",
                   use_math_tex=True, anchor_point=anchor_point, **kw)

    @classmethod
    def median_label(
        cls,
        value: float, pos: np.ndarray,
        anchor_point: np.ndarray | None = None,
        **kw,
    ) -> "StatLabel3D":
        return cls(r"\tilde{x}", value, pos, stat_type="median",
                   use_math_tex=True, anchor_point=anchor_point, **kw)

    @classmethod
    def std_label(
        cls,
        value: float, pos: np.ndarray,
        anchor_point: np.ndarray | None = None,
        **kw,
    ) -> "StatLabel3D":
        return cls(r"\sigma", value, pos, stat_type="std",
                   use_math_tex=True, anchor_point=anchor_point, **kw)

    @classmethod
    def pval_label(
        cls,
        p_value: float, pos: np.ndarray,
        anchor_point: np.ndarray | None = None,
        **kw,
    ) -> "StatLabel3D":
        color = STAT_PVAL_SIG if p_value < 0.05 else STAT_PVAL_INSIG
        return cls("p-value", p_value, pos, stat_type="p",
                   p_value=p_value, anchor_point=anchor_point,
                   fmt="e", decimals=4, **kw)

    @classmethod
    def corr_label(
        cls,
        r_value: float, p_value: float | None,
        pos: np.ndarray,
        anchor_point: np.ndarray | None = None,
        **kw,
    ) -> "StatLabel3D":
        color = STAT_CORR_POS if r_value >= 0 else STAT_CORR_NEG
        return cls("r", r_value, pos, stat_type="r",
                   p_value=p_value, anchor_point=anchor_point,
                   decimals=3, **kw)

    @classmethod
    def sample_size_label(
        cls,
        n: int, pos: np.ndarray, **kw,
    ) -> "StatLabel3D":
        return cls("n", float(n), pos, stat_type="n",
                   decimals=0, **kw)


# ---------------------------------------------------------------------------
# AnnotationArrow3D
# ---------------------------------------------------------------------------

class AnnotationArrow3D(VGroup):
    """Directional Arrow3D from a source label position to a target point.

    Parameters
    ----------
    source : np.ndarray
        Tail position of the arrow (where the label sits).
    target : np.ndarray
        Tip position of the arrow (what it points to).
    label_text : str
        Text at the tail end (pass ``""`` to omit).
    use_math_tex : bool
        Render label with MathTex instead of Text.
    tip_style : str
        ``"filled"`` – standard filled arrowhead (default)
        ``"open"``   – no tip fill
        ``"dot"``    – small sphere at the tip
        ``"none"``   – plain line, no tip
    color : ManimColor
    label_font_size : int
    config : LabelConfig | None
    """

    def __init__(
        self,
        source:         np.ndarray,
        target:         np.ndarray,
        label_text:     str  = "",
        use_math_tex:   bool = False,
        tip_style:      str  = "filled",
        color:          ManimColor = LABEL_TEXT_COLOR,
        label_font_size: int = 22,
        config:         LabelConfig | None = None,
    ):
        super().__init__()
        cfg = config or LabelConfig()
        self._source = source
        self._target = target
        self._color  = color

        # ---- arrow body ----
        if tip_style in ("filled", "open"):
            arrow = Arrow3D(
                start=source,
                end  =target,
                color=color,
                stroke_width=cfg.stem_stroke_width * 1.5,
                tip_length=cfg.arrow_tip_length,
            )
            arrow.set_opacity(0.88)
            self.add(arrow)
            self.arrow = arrow
        elif tip_style == "none":
            line = Line3D(
                start=source, end=target,
                color=color,
                stroke_width=cfg.stem_stroke_width * 1.5,
            )
            line.set_opacity(0.88)
            self.add(line)
            self.arrow = line
        elif tip_style == "dot":
            line = Line3D(
                start=source, end=target,
                color=color, stroke_width=cfg.stem_stroke_width * 1.5,
            )
            dot = Dot3D(point=target, radius=0.06, color=color)
            dot.set_opacity(0.95)
            self.add(line, dot)
            self.arrow = line

        # ---- label at source ----
        if label_text:
            Tx   = MathTex if use_math_tex else Text
            lbl  = Tx(label_text, color=color, font_size=label_font_size)
            # Offset the label away from the arrow direction
            direction = target - source
            norm_d    = np.linalg.norm(direction)
            if norm_d > 1e-6:
                perp = np.array([-direction[2], 0, direction[0]]) / norm_d
            else:
                perp = np.array([0, 0, 1])
            lbl.move_to(source + perp * 0.35)
            self.add(lbl)
            self.label = lbl

    # ------------------------------------------------------------------

    def animate_draw(self, run_time: float = 0.8) -> Create:
        """Trace the arrow from source to target."""
        return Create(self.arrow, run_time=run_time)

    def animate_pulse(
        self,
        n_pulses: int   = 2,
        run_time: float = 0.8,
    ) -> LaggedStart:
        """Briefly brighten and fade the arrow N times."""
        pulse_rt = run_time / (n_pulses * 2)
        return LaggedStart(
            *[Succession(
                self.arrow.animate(run_time=pulse_rt).set_opacity(1.0),
                self.arrow.animate(run_time=pulse_rt).set_opacity(0.50),
            ) for _ in range(n_pulses)],
            lag_ratio=0.0,
            run_time=run_time,
        )

    def animate_redirect(
        self,
        new_target: np.ndarray,
        run_time:   float = 0.6,
    ) -> AnimationGroup:
        """Move the arrow tip to a new position."""
        new_arrow = Arrow3D(
            start=self._source,
            end  =new_target,
            color=self._color,
            stroke_width=1.5,
        )
        return AnimationGroup(
            Transform(self.arrow, new_arrow, run_time=run_time)
        )


# ---------------------------------------------------------------------------
# FormulaPanel3D
# ---------------------------------------------------------------------------

class FormulaPanel3D(VGroup):
    """Floating 3D panel holding multi-line MathTex formula content.

    The panel background is a shaded card prism.  Lines are stacked
    vertically inside.  Terms within each line can be independently
    colored via a ``color_map`` dict passed to ``add_line()``.

    Parameters
    ----------
    pos : np.ndarray
        World position of the panel centre (top-face level).
    title : str
        Optional title displayed at the top of the panel.
    config : LabelConfig | None
    """

    def __init__(
        self,
        pos:    np.ndarray,
        title:  str = "",
        config: LabelConfig | None = None,
    ):
        super().__init__()
        self.cfg    = config or LabelConfig()
        self._pos   = pos
        self._title = title
        self._lines: list[VGroup] = []
        self._y_cursor = float(pos[1])

        # Build the title bar (the card is extended downward as lines are added)
        if title:
            title_obj = Text(title, color=PANEL_TITLE_COLOR,
                             font_size=self.cfg.panel_title_font_size,
                             weight="BOLD")
            title_obj.move_to(pos + np.array([0, 0.22, 0]))
            self.add(title_obj)
            self.title_obj = title_obj
            self._y_cursor -= self.cfg.panel_line_spacing

        # Card built lazily in _rebuild_card()
        self._card: _CardPrism | None = None
        self._card_width  = self.cfg.panel_min_width
        self._card_height = 0.08

    def _rebuild_card(self) -> None:
        """Rebuild the background card to fit current content."""
        if self._card is not None:
            self.remove(self._card)

        n_lines  = len(self._lines) + (1 if self._title else 0)
        card_h   = max(
            n_lines * self.cfg.panel_line_spacing + 0.25,
            0.08
        )
        # Compute required width from widest line
        max_w = self._card_width
        for ln in self._lines:
            max_w = max(max_w, ln.width + self.cfg.card_padding_x * 2)

        px, py, pz = float(self._pos[0]), float(self._pos[1]), float(self._pos[2])
        self._card = _CardPrism(
            x0     = px - max_w / 2,
            z0     = pz - self.cfg.card_depth / 2,
            width  = max_w,
            height = self.cfg.card_height,
            depth  = self.cfg.card_depth,
            color  = PANEL_BG_COLOR,
            cfg    = self.cfg,
        )
        self.add(self._card)
        self._card_width  = max_w
        self._card_height = card_h

    def add_line(
        self,
        tex:       str,
        color:     ManimColor | None = None,
        color_map: dict[str, ManimColor] | None = None,
    ) -> MathTex:
        """Add one line of MathTex to the panel.

        Parameters
        ----------
        tex : str
            LaTeX string for this line.
        color : ManimColor | None
            Uniform color for the whole line; overridden by ``color_map``.
        color_map : dict | None
            Map from substring → ManimColor for per-term coloring.

        Returns the MathTex object for external manipulation.
        """
        col   = color or LABEL_TEXT_COLOR
        obj   = MathTex(tex, color=col,
                        font_size=self.cfg.panel_content_font_size)

        if color_map:
            for substr, c in color_map.items():
                for subobj in obj:
                    if hasattr(subobj, "get_tex_string"):
                        if substr in subobj.get_tex_string():
                            subobj.set_color(c)

        px, pz = float(self._pos[0]), float(self._pos[2])
        self._y_cursor -= self.cfg.panel_line_spacing
        obj.move_to(np.array([px, self._y_cursor + self.cfg.card_height + 0.22, pz]))

        line_grp = VGroup(obj)
        self._lines.append(line_grp)
        self.add(line_grp)
        self._rebuild_card()
        return obj

    def animate_write_lines(
        self,
        lag_ratio: float = 0.30,
        run_time:  float = 2.5,
    ) -> LaggedStart:
        """Write lines one after another using the Write animation."""
        return LaggedStart(
            *[Write(ln, run_time=run_time * 0.55) for ln in self._lines],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_collapse(
        self,
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Shrink content lines to invisible, leaving only the title bar."""
        anims = [
            ln.animate(run_time=run_time).scale(0.01).set_opacity(0)
            for ln in self._lines
        ]
        return AnimationGroup(*anims) if anims else AnimationGroup()

    def animate_expand(
        self,
        run_time: float = 0.6,
    ) -> LaggedStart:
        """Restore collapsed content lines."""
        return LaggedStart(
            *[FadeIn(ln, run_time=run_time * 0.5) for ln in self._lines],
            lag_ratio=0.15,
            run_time=run_time,
        )


# ---------------------------------------------------------------------------
# LegendPanel3D
# ---------------------------------------------------------------------------

class LegendPanel3D(VGroup):
    """Floating legend panel with one row per series or category.

    Glyph styles
    ------------
    ``"sphere"``  – small ``Dot3D`` sphere
    ``"square"``  – flat square ``Polygon``
    ``"line"``    – short solid ``Line3D``
    ``"dashed"``  – two short dashes simulating a dashed line

    Parameters
    ----------
    entries : list[tuple[ManimColor, str]]
        (color, label) pairs, one per legend row.
    pos : np.ndarray
        World position of the top-left corner of the legend.
    glyph_style : str
        One of ``"sphere"``, ``"square"``, ``"line"``, ``"dashed"``.
    n_cols : int
        Number of columns.  Entries wrap left to right.
    title : str
        Optional title above the legend.
    config : LabelConfig | None
    """

    def __init__(
        self,
        entries:     list[tuple[ManimColor, str]],
        pos:         np.ndarray,
        glyph_style: str  = "sphere",
        n_cols:      int  = 1,
        title:       str  = "",
        config:      LabelConfig | None = None,
    ):
        super().__init__()
        self.cfg  = cfg = config or LabelConfig()
        self._entries = entries

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        y_cursor   = py

        # Title
        if title:
            t_obj = Text(title, color=PANEL_TITLE_COLOR,
                         font_size=cfg.panel_title_font_size, weight="BOLD")
            t_obj.move_to(np.array([px, y_cursor, pz]))
            self.add(t_obj)
            y_cursor -= cfg.legend_row_spacing * 0.8

        # Entries
        self._glyphs: list[VGroup] = []
        for k, (color, label) in enumerate(entries):
            col_idx = k % n_cols
            row_idx = k // n_cols
            ex  = px + col_idx * cfg.legend_col_width
            ey  = y_cursor - row_idx * cfg.legend_row_spacing

            # Glyph
            glyph_grp = VGroup()
            gr = cfg.legend_glyph_radius
            if glyph_style == "sphere":
                g = Dot3D(point=np.array([ex, ey, pz]),
                          radius=gr, color=color)
                g.set_opacity(0.92)
                glyph_grp.add(g)
            elif glyph_style == "square":
                sq = Polygon(
                    np.array([ex - gr, ey - gr, pz]),
                    np.array([ex + gr, ey - gr, pz]),
                    np.array([ex + gr, ey + gr, pz]),
                    np.array([ex - gr, ey + gr, pz]),
                    color=color,
                )
                sq.set_fill(color=color, opacity=0.88)
                sq.set_stroke(color=_dk(color, 0.4), width=0.6)
                glyph_grp.add(sq)
            elif glyph_style == "line":
                ln = Line3D(
                    start=np.array([ex - gr * 1.8, ey, pz]),
                    end  =np.array([ex + gr * 1.8, ey, pz]),
                    color=color, stroke_width=2.2,
                )
                glyph_grp.add(ln)
            elif glyph_style == "dashed":
                for dx in [-gr * 1.2, gr * 0.2]:
                    seg = Line3D(
                        start=np.array([ex + dx,          ey, pz]),
                        end  =np.array([ex + dx + gr * 0.9, ey, pz]),
                        color=color, stroke_width=2.0,
                    )
                    glyph_grp.add(seg)

            self._glyphs.append(glyph_grp)
            self.add(glyph_grp)

            # Label
            lbl = Text(label, color=_lt(color, 0.20),
                       font_size=cfg.name_font_size)
            lbl.move_to(np.array([ex + gr * 2.2 + lbl.width / 2, ey, pz]))
            self.add(lbl)

        # Background card sized to fit all entries
        n_rows    = (len(entries) + n_cols - 1) // n_cols
        card_h_total = n_rows * cfg.legend_row_spacing + 0.30
        card_w_total = n_cols * cfg.legend_col_width + 0.25

        card = _CardPrism(
            x0    = px - 0.12,
            z0    = pz - cfg.card_depth / 2,
            width = card_w_total,
            height= cfg.card_height,
            depth = cfg.card_depth,
            cfg   = cfg,
        )
        self.add(card)

    # ------------------------------------------------------------------

    def animate_appear(
        self,
        lag_ratio: float = 0.15,
        run_time:  float = 1.2,
    ) -> LaggedStart:
        """Fade in glyphs and labels staggered top to bottom."""
        return LaggedStart(
            *[FadeIn(g, run_time=run_time * 0.4) for g in self._glyphs],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    @classmethod
    def from_series(
        cls,
        colors:      list[ManimColor],
        labels:      list[str],
        pos:         np.ndarray,
        glyph_style: str  = "sphere",
        title:       str  = "",
        config:      LabelConfig | None = None,
    ) -> "LegendPanel3D":
        """Build from parallel color and label lists."""
        entries = list(zip(colors, labels))
        return cls(entries, pos, glyph_style=glyph_style,
                   title=title, config=config)


# ---------------------------------------------------------------------------
# DataCallout3D
# ---------------------------------------------------------------------------

class DataCallout3D(VGroup):
    """Speech-bubble callout pointing at a specific data point.

    Shows coordinate values (and optional rank) in a compact card
    with a triangular pointer aimed at the annotated point.

    Parameters
    ----------
    data_point : np.ndarray
        World position of the data point being annotated.
    values : dict[str, float]
        Key-value pairs to display, e.g. ``{"x": 2.3, "y": 5.1}``.
    rank_text : str
        Optional rank string, e.g. ``"3rd highest"``.
    bubble_pos : np.ndarray | None
        Explicit position for the bubble center.  Auto-placed above the
        data point if None.
    color : ManimColor
    config : LabelConfig | None
    """

    def __init__(
        self,
        data_point:  np.ndarray,
        values:      dict[str, float],
        rank_text:   str = "",
        bubble_pos:  np.ndarray | None = None,
        color:       ManimColor = LABEL_TEXT_COLOR,
        decimals:    int = 3,
        config:      LabelConfig | None = None,
    ):
        super().__init__()
        cfg = config or LabelConfig()
        self._color = color

        # Auto-place bubble above the data point
        if bubble_pos is None:
            bubble_pos = data_point + np.array([0.0, 0.70, 0.0])

        bp = bubble_pos
        dp = data_point

        # ---- Bubble card ----
        lines = [f"{k}: {_format_value(v, decimals=decimals)}"
                 for k, v in values.items()]
        if rank_text:
            lines.insert(0, rank_text)

        max_line_len = max(len(ln) for ln in lines) * 0.10
        card_w  = max(max_line_len, cfg.panel_min_width * 0.55)
        card    = _CardPrism(
            x0    = float(bp[0]) - card_w / 2,
            z0    = float(bp[2]) - cfg.card_depth / 2,
            width = card_w,
            height= cfg.card_height,
            depth = cfg.card_depth,
            color = PANEL_BG_COLOR,
            cfg   = cfg,
        )
        self.add(card)

        # Text lines inside bubble
        y_txt = float(bp[1]) + cfg.card_height + 0.18
        for i, line in enumerate(lines):
            tc = color if i > 0 else _lt(color, 0.20)
            lbl = Text(line, color=tc, font_size=cfg.name_font_size)
            lbl.move_to(np.array([float(bp[0]), y_txt + i * 0.32, float(bp[2])]))
            self.add(lbl)

        # ---- Pointer triangle ----
        direction = dp - bp
        norm_d    = np.linalg.norm(direction)
        if norm_d > 1e-6:
            unit_d = direction / norm_d
        else:
            unit_d = np.array([0, -1, 0])

        perp  = np.array([-unit_d[2], 0, unit_d[0]])
        tip_w = 0.08
        ptr_base1 = bp + perp * tip_w
        ptr_base2 = bp - perp * tip_w
        ptr_tip   = bp + unit_d * min(norm_d, 0.40)

        pointer = Polygon(ptr_base1, ptr_base2, ptr_tip, color=color)
        pointer.set_fill(color=PANEL_BG_COLOR, opacity=cfg.card_opacity)
        pointer.set_stroke(color=PANEL_BORDER_COLOR, width=cfg.card_border_width)
        self.add(pointer)

        # Dot at the data point
        dot = Dot3D(point=dp, radius=0.055, color=color)
        dot.set_opacity(0.90)
        self.add(dot)

    # ------------------------------------------------------------------

    def animate_appear(self, run_time: float = 0.7) -> GrowFromPoint:
        """Grow the callout from the data point upward."""
        return GrowFromPoint(self, point=self.submobjects[-1].get_center(),
                             run_time=run_time)


# ---------------------------------------------------------------------------
# Ticker3D
# ---------------------------------------------------------------------------

class Ticker3D(VGroup):
    """Animated DecimalNumber that counts to a target value with easing.

    Parameters
    ----------
    initial_value : float
        Starting value displayed before animation.
    pos : np.ndarray
        World position of the ticker centre.
    prefix : str
        Text prepended to the number, e.g. ``"n = "``.
    suffix : str
        Text appended, e.g. ``"%"``.
    color : ManimColor
    config : LabelConfig | None
    """

    def __init__(
        self,
        initial_value: float,
        pos:           np.ndarray,
        prefix:        str = "",
        suffix:        str = "",
        color:         ManimColor = LABEL_TEXT_COLOR,
        config:        LabelConfig | None = None,
    ):
        super().__init__()
        cfg   = config or LabelConfig()
        self._pos   = pos
        self._color = color
        self._cfg   = cfg

        self._tracker = ValueTracker(initial_value)

        # DecimalNumber that follows the tracker
        self._number = DecimalNumber(
            initial_value,
            num_decimal_places=cfg.ticker_decimals,
            color=color,
            font_size=cfg.ticker_font_size,
        )
        self._number.move_to(pos)

        # Prefix and suffix
        self._prefix_obj = None
        self._suffix_obj = None

        if prefix:
            self._prefix_obj = Text(prefix, color=_lt(color, 0.30),
                                    font_size=cfg.ticker_font_size - 4)
        if suffix:
            self._suffix_obj = Text(suffix, color=_lt(color, 0.30),
                                    font_size=cfg.ticker_font_size - 4)

        self._layout()
        self.add(self._number)
        if self._prefix_obj: self.add(self._prefix_obj)
        if self._suffix_obj: self.add(self._suffix_obj)

        # Hook DecimalNumber to tracker
        self._number.add_updater(
            lambda m: m.set_value(self._tracker.get_value())
        )

    def _layout(self) -> None:
        """Position prefix / suffix beside the number."""
        px, py, pz = self._pos
        self._number.move_to(np.array([px, py, pz]))
        nw = self._number.width
        if self._prefix_obj:
            pw = self._prefix_obj.width
            self._prefix_obj.move_to(
                np.array([px - nw / 2 - pw / 2 - 0.08, py, pz])
            )
        if self._suffix_obj:
            sw = self._suffix_obj.width
            self._suffix_obj.move_to(
                np.array([px + nw / 2 + sw / 2 + 0.08, py, pz])
            )

    # ------------------------------------------------------------------

    def animate_count(
        self,
        target:    float,
        run_time:  float = 2.0,
        rate_func: Callable | None = None,
    ) -> AnimationGroup:
        """Animate the displayed number from current value to ``target``.

        Parameters
        ----------
        target : float
            The value to count to.
        run_time : float
        rate_func : callable | None
            Manim rate function.  Defaults to ``smooth`` (ease in-out).
        """
        from manim import smooth
        rf = rate_func or smooth
        return AnimationGroup(
            self._tracker.animate(run_time=run_time,
                                  rate_func=rf).set_value(target)
        )

    def animate_flash_on_change(self, run_time: float = 0.3) -> Succession:
        """Briefly scale the number when it changes."""
        return Succession(
            self._number.animate(run_time=run_time / 2).scale(1.25),
            self._number.animate(run_time=run_time / 2).scale(1 / 1.25),
        )


# ---------------------------------------------------------------------------
# AxisLabel3D
# ---------------------------------------------------------------------------

class AxisLabel3D(VGroup):
    """Enhanced axis annotation set: ticks + labels + title + optional unit.

    Supports four scale modes:
    ``"linear"``      – evenly spaced numeric labels
    ``"log"``         – 10⁰, 10¹, 10², … labels
    ``"percent"``     – 0%, 25%, 50%, … labels
    ``"categorical"`` – string labels at fixed positions

    Parameters
    ----------
    positions : list[float]
        World-space positions of tick marks along the axis.
    values : list[float | str]
        Label values corresponding to each position.
    axis : str
        ``"x"``, ``"y"``, or ``"z"`` — controls tick direction.
    origin : np.ndarray
        World position of the axis zero point.
    scale_mode : str
    tick_length : float
    title : str
        Axis title text.
    unit : str
        Unit badge appended after the title (e.g. ``"[ms]"``).
    color : ManimColor
    font_size : int
    title_font_size : int
    config : LabelConfig | None
    """

    def __init__(
        self,
        positions:      list[float],
        values:         list[float | str],
        axis:           str = "x",
        origin:         np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale_mode:     str = "linear",
        tick_length:    float = 0.12,
        title:          str   = "",
        unit:           str   = "",
        color:          ManimColor = GRAY_B,
        font_size:      int   = 18,
        title_font_size: int  = 22,
        rotate_labels:  float = 0.0,
        config:         LabelConfig | None = None,
    ):
        super().__init__()
        cfg = config or LabelConfig()

        self._ticks: list[Line3D] = []
        self._tick_labels: list[Text | MathTex] = []

        for pos, val in zip(positions, values):
            # Tick mark direction based on axis
            if axis == "x":
                pt_on_axis  = origin + np.array([pos, 0, 0])
                tick_end    = pt_on_axis + np.array([0, -tick_length, 0])
                lbl_offset  = np.array([0, -tick_length - 0.22, 0])
            elif axis == "y":
                pt_on_axis  = origin + np.array([0, pos, 0])
                tick_end    = pt_on_axis + np.array([-tick_length, 0, 0])
                lbl_offset  = np.array([-tick_length - 0.28, 0, 0])
            else:  # z
                pt_on_axis  = origin + np.array([0, 0, pos])
                tick_end    = pt_on_axis + np.array([0, -tick_length, 0])
                lbl_offset  = np.array([0, -tick_length - 0.22, 0])

            tick = Line3D(start=pt_on_axis, end=tick_end,
                          color=color, stroke_width=0.8)
            self.add(tick)
            self._ticks.append(tick)

            # Format label
            if scale_mode == "log":
                exp = int(round(float(val)))
                lbl = MathTex(rf"10^{{{exp}}}", color=color, font_size=font_size)
            elif scale_mode == "percent":
                lbl = Text(f"{int(float(val) * 100)}%", color=color,
                           font_size=font_size)
            elif scale_mode == "categorical":
                lbl = Text(str(val), color=color, font_size=font_size)
            else:
                lbl = Text(_format_value(float(val), decimals=1),
                           color=color, font_size=font_size)

            if rotate_labels:
                lbl.rotate(rotate_labels * DEGREES,
                           axis=np.array([0, 0, 1]))
            lbl.move_to(pt_on_axis + lbl_offset)
            self.add(lbl)
            self._tick_labels.append(lbl)

        # Title (with optional unit badge)
        if title:
            title_str = f"{title} [{unit}]" if unit else title
            title_obj = Text(title_str, color=color,
                             font_size=title_font_size)
            # Position at midpoint of the axis, offset outward
            if positions:
                mid_pos = (positions[0] + positions[-1]) / 2
                if axis == "x":
                    title_obj.move_to(origin + np.array([mid_pos, -0.65, 0]))
                elif axis == "y":
                    title_obj.move_to(origin + np.array([-0.75, mid_pos, 0]))
                    title_obj.rotate(90 * DEGREES, axis=np.array([0, 0, 1]))
                else:
                    title_obj.move_to(origin + np.array([0, -0.65, mid_pos]))
            self.add(title_obj)
            self.title_obj = title_obj

    # ------------------------------------------------------------------

    def animate_appear(
        self,
        lag_ratio: float = 0.06,
        run_time:  float = 1.2,
    ) -> LaggedStart:
        """Ticks and labels appear staggered along the axis."""
        pairs = list(zip(self._ticks, self._tick_labels))
        return LaggedStart(
            *[AnimationGroup(Create(t), FadeIn(l), run_time=run_time * 0.4)
              for t, l in pairs],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )


# ---------------------------------------------------------------------------
# TooltipBadge3D
# ---------------------------------------------------------------------------

class TooltipBadge3D(VGroup):
    """Compact multi-field key-value popup.

    Appears beside a highlighted element showing selected attributes.

    Parameters
    ----------
    fields : dict[str, str | float]
        Ordered key-value pairs to display.
    pos : np.ndarray
        World position of the tooltip.
    color : ManimColor
        Accent color for keys.
    config : LabelConfig | None
    """

    def __init__(
        self,
        fields:  dict[str, str | float],
        pos:     np.ndarray,
        color:   ManimColor = LABEL_TEXT_COLOR,
        decimals: int = 3,
        config:  LabelConfig | None = None,
    ):
        super().__init__()
        cfg   = config or LabelConfig()
        self._visible = False
        self.set_opacity(0)   # start hidden

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        n_fields   = len(fields)
        row_h      = 0.30
        card_h_val = n_fields * row_h + 0.20
        card_w_val = cfg.panel_min_width

        card = _CardPrism(
            x0    = px - card_w_val / 2,
            z0    = pz - cfg.card_depth / 2,
            width = card_w_val,
            height= cfg.card_height,
            depth = cfg.card_depth,
            color = PANEL_BG_COLOR,
            cfg   = cfg,
        )
        self.add(card)

        y_cursor = py + cfg.card_height + 0.15
        for k, (key, val) in enumerate(fields.items()):
            val_str = (_format_value(float(val), decimals=decimals)
                       if isinstance(val, (int, float)) else str(val))

            key_obj = Text(f"{key}:", color=_lt(color, 0.25),
                           font_size=cfg.name_font_size - 2)
            val_obj = Text(val_str, color=LABEL_TEXT_COLOR,
                           font_size=cfg.name_font_size - 2)

            y_pos = y_cursor + k * row_h
            key_obj.move_to(np.array([px - card_w_val * 0.22, y_pos, pz]))
            val_obj.move_to(np.array([px + card_w_val * 0.20, y_pos, pz]))
            self.add(key_obj, val_obj)

    # ------------------------------------------------------------------

    def animate_show(self, run_time: float = 0.30) -> FadeIn:
        """Fade the tooltip into view."""
        self._visible = True
        return FadeIn(self, run_time=run_time)

    def animate_hide(self, run_time: float = 0.25) -> FadeOut:
        """Fade the tooltip out."""
        self._visible = False
        return FadeOut(self, run_time=run_time)


# ---------------------------------------------------------------------------
# StatSummaryBox3D
# ---------------------------------------------------------------------------

class StatSummaryBox3D(VGroup):
    """Five-number summary panel: min, Q1, median, Q3, max.

    Parameters
    ----------
    data : array-like
        1-D observations.  Statistics computed internally.
    pos : np.ndarray
        World position of the panel centre.
    color : ManimColor
        Accent color for the IQR band.
    config : LabelConfig | None
    """

    def __init__(
        self,
        data:   Sequence[float] | np.ndarray,
        pos:    np.ndarray,
        color:  ManimColor = STAT_IQR_COLOR,
        config: LabelConfig | None = None,
    ):
        super().__init__()
        cfg  = config or LabelConfig()
        arr  = np.asarray(data, dtype=float)

        stats = {
            "Min":    float(arr.min()),
            "Q1":     float(np.percentile(arr, 25)),
            "Median": float(np.median(arr)),
            "Q3":     float(np.percentile(arr, 75)),
            "Max":    float(arr.max()),
            "Mean":   float(arr.mean()),
            "Std":    float(arr.std()),
            "N":      float(len(arr)),
        }
        stat_colors = {
            "Min":    GRAY_C,
            "Q1":     STAT_IQR_COLOR,
            "Median": STAT_MEDIAN_COLOR,
            "Q3":     STAT_IQR_COLOR,
            "Max":    GRAY_C,
            "Mean":   STAT_MEAN_COLOR,
            "Std":    STAT_STD_COLOR,
            "N":      STAT_N_COLOR,
        }

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        row_h = 0.36
        card_w = cfg.panel_min_width * 1.10

        # Background card
        card = _CardPrism(
            x0    = px - card_w / 2,
            z0    = pz - cfg.card_depth / 2,
            width = card_w,
            height= cfg.card_height,
            depth = cfg.card_depth,
            color = PANEL_BG_COLOR,
            cfg   = cfg,
        )
        self.add(card)

        # Title
        title = Text("Summary", color=PANEL_TITLE_COLOR,
                     font_size=cfg.panel_title_font_size, weight="BOLD")
        title.move_to(np.array([px, py + cfg.card_height + 0.22, pz]))
        self.add(title)

        # Stat rows
        self._stat_labels: list[VGroup] = []
        for k, (name, val) in enumerate(stats.items()):
            sc    = stat_colors.get(name, LABEL_TEXT_COLOR)
            y_pos = py + cfg.card_height + 0.55 + k * row_h

            name_obj = Text(f"{name}:", color=_lt(sc, 0.20),
                            font_size=cfg.name_font_size)
            val_obj  = Text(
                _format_value(val, decimals=0 if name == "N" else 3),
                color=sc,
                font_size=cfg.value_font_size - 4,
                weight="BOLD" if name in ("Median", "Mean") else "NORMAL",
            )
            name_obj.move_to(np.array([px - card_w * 0.22, y_pos, pz]))
            val_obj.move_to( np.array([px + card_w * 0.20, y_pos, pz]))
            row_grp = VGroup(name_obj, val_obj)
            self._stat_labels.append(row_grp)
            self.add(row_grp)

        # IQR bracket line on the right side
        q1_y = py + cfg.card_height + 0.55 + 1 * row_h
        q3_y = py + cfg.card_height + 0.55 + 3 * row_h
        bx   = px + card_w / 2 + 0.12

        iqr_line = Line3D(
            start=np.array([bx, q1_y, pz]),
            end  =np.array([bx, q3_y, pz]),
            color=STAT_IQR_COLOR, stroke_width=2.0,
        )
        for yt in [q1_y, q3_y]:
            cap = Line3D(
                start=np.array([bx - 0.08, yt, pz]),
                end  =np.array([bx + 0.08, yt, pz]),
                color=STAT_IQR_COLOR, stroke_width=1.5,
            )
            self.add(cap)
        iqr_lbl = Text("IQR", color=STAT_IQR_COLOR,
                        font_size=cfg.unit_font_size - 2)
        iqr_lbl.move_to(np.array([bx + 0.28, (q1_y + q3_y) / 2, pz]))
        self.add(iqr_line, iqr_lbl)

    # ------------------------------------------------------------------

    def animate_reveal(
        self,
        lag_ratio: float = 0.10,
        run_time:  float = 1.8,
    ) -> LaggedStart:
        """Rows appear from top to bottom."""
        return LaggedStart(
            *[FadeIn(row, run_time=run_time * 0.4)
              for row in self._stat_labels],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )


# ---------------------------------------------------------------------------
# HighlightRing3D
# ---------------------------------------------------------------------------

class HighlightRing3D(VGroup):
    """A pulsing ring that draws the eye to a 3D point.

    The ring is a ``ParametricFunction`` circle floating at a given
    height around the target point.  It can pulse in size and orbit
    around the vertical axis.

    Parameters
    ----------
    center : np.ndarray
        World position being highlighted.
    radius : float
        Starting ring radius.
    y_offset : float
        How high above ``center`` the ring floats.
    color : ManimColor
    config : LabelConfig | None
    """

    def __init__(
        self,
        center:    np.ndarray,
        radius:    float = 0.25,
        y_offset:  float = 0.08,
        color:     ManimColor = YELLOW,
        config:    LabelConfig | None = None,
    ):
        super().__init__()
        cfg = config or LabelConfig()
        self._center  = center
        self._radius  = radius
        self._y_off   = y_offset
        self._color   = color
        self._cfg     = cfg

        self._ring = self._make_ring(radius, y_offset, color, cfg)
        self.add(self._ring)

    def _make_ring(
        self,
        r:    float,
        y_off: float,
        color: ManimColor,
        cfg:  LabelConfig,
    ) -> ParametricFunction:
        cx, cy, cz = self._center
        ring = ParametricFunction(
            lambda t: np.array([
                cx + r * np.cos(t * TAU),
                cy + y_off,
                cz + r * np.sin(t * TAU),
            ]),
            t_range=[0, 1, 1 / 80],
            color=color,
            stroke_width=cfg.ring_stroke_width,
        )
        ring.set_opacity(0.85)
        return ring

    # ------------------------------------------------------------------

    def animate_pulse(
        self,
        n_pulses: int   = 2,
        run_time: float = 1.0,
    ) -> LaggedStart:
        """Expand and contract the ring N times."""
        pulse_rt  = run_time / (n_pulses * 2)
        scale     = self._cfg.ring_pulse_scale
        return LaggedStart(
            *[Succession(
                self._ring.animate(run_time=pulse_rt).scale(scale),
                self._ring.animate(run_time=pulse_rt).scale(1 / scale),
            ) for _ in range(n_pulses)],
            lag_ratio=0.0,
            run_time=run_time,
        )

    def animate_orbit(
        self,
        angle_deg: float = 360.0,
        run_time:  float = 2.0,
    ) -> Rotate:
        """Rotate the ring around the vertical axis through center."""
        return Rotate(
            self._ring,
            angle=np.radians(angle_deg),
            axis=np.array([0, 1, 0]),
            about_point=self._center,
            run_time=run_time,
        )

    def animate_appear(self, run_time: float = 0.4) -> GrowFromCenter:
        """Scale the ring in from zero."""
        return GrowFromCenter(self._ring, run_time=run_time)

    def set_target(self, new_center: np.ndarray) -> None:
        """Move the ring to orbit a different point (instant, no animation)."""
        self._center = new_center
        new_ring = self._make_ring(
            self._radius, self._y_off, self._color, self._cfg
        )
        self.remove(self._ring)
        self._ring = new_ring
        self.add(self._ring)