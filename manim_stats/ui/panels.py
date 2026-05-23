"""
manim_stats/ui/panels.py
==========================
Production-quality floating panel and dashboard overlay objects for Manim.
Where ``labels.py`` provides individual annotation badges, this module
provides full multi-section dashboard panels — the difference between a
sticky note and a poster.

Objects
-------

Shared infrastructure
    ``PanelConfig``        – centralised visual parameters for all panels
    ``_PanelBackground``   – shaded 3D card body with accent header edge
    ``_SectionDivider``    – thin Line3D ruled between panel sections
    ``_StepBadge``         – numbered circle badge for step-by-step panels
    ``_ValueCell``         – colored value cell for comparison / matrix grids
    ``_DecisionBanner``    – bold reject/fail-to-reject banner with border

InfoPanel3D  (VGroup)
    General-purpose information panel: header bar + body sections + footer.
    Sections are flexible key-value grids (1- or 2-column).  Can be docked
    to scene corners (TL, TR, BL, BR) or placed at explicit 3D coordinates.
    ``add_section(title, items)`` extends the panel dynamically.
    ``animate_build()``             – header slides in, sections cascade
    ``animate_update_section(i)``   – morph one section's content in-place

StepPanel3D  (VGroup)
    Numbered step-by-step derivation/proof panel.
    Steps have three visual states: pending (grey), active (bright),
    done (dimmed + checkmark).
    ``animate_activate_step(i)``    – transition step i to active
    ``animate_complete_step(i)``    – mark done with checkmark flash
    ``animate_walk_steps()``        – activate each step in sequence

ComparisonPanel3D  (VGroup)
    Side-by-side comparison of N items across M metrics.
    Direction-aware winner highlighting (higher-is-better or lower-is-better
    per row).  Column headers with color accents.
    ``animate_build_columns()``     – columns grow left to right
    ``animate_highlight_winner(r)`` – flash the winning cell in row r
    ``animate_reveal_all()``        – full staggered reveal

DistributionInfoPanel3D  (VGroup)
    Specialised panel for probability distributions.
    Shows: distribution name, PDF/PMF MathTex formula, parameter table
    (name / symbol / value / range), and key statistics section
    (mean, variance, skewness, kurtosis with formulas).
    ``animate_reveal()``            – formula writes, then params cascade

HypothesisPanel3D  (VGroup)
    Hypothesis test result panel.
    Displays H₀ / H₁ statements, test statistic with observed + critical
    values, p-value with significance stars, effect size badge, and a
    large decision banner: "Reject H₀" (red) or "Fail to Reject H₀" (green).
    ``animate_build()``             – build up to decision banner
    ``animate_reveal_decision()``   – banner flashes in with colored glow

FormulaDerivationPanel3D  (VGroup)
    Multi-step formula derivation with annotation labels.
    Each step is a MathTex equation + optional right-side annotation
    (e.g. "← definition", "← substitution").  Vertical connector lines
    link consecutive steps.  States: neutral, active (highlighted), derived.
    ``animate_derive(i)``           – highlight, annotate, advance
    ``animate_full_derivation()``   – walk all steps automatically

MatrixPanel3D  (VGroup)
    Floating matrix display with row/column headers and color-coded cells.
    Same palette engine as ``heat_map3d.py``.  Supports sparse mode
    (only render non-zero cells).
    ``animate_reveal_by_row()``     – rows appear top to bottom
    ``animate_reveal_by_col()``     – columns appear left to right
    ``animate_highlight_cell(i,j)`` – flash one cell
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
    Line3D, Polygon, Sphere, Dot3D, Arrow3D,
    Text, MathTex, DecimalNumber,
    FadeIn, FadeOut, GrowFromCenter, GrowFromPoint, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    Flash,
    interpolate_color,
    ManimColor,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Semantic palette (consistent with labels.py)
# ---------------------------------------------------------------------------

PANEL_BG        = ManimColor("#131E26")   # very dark navy
PANEL_BG_ALT    = ManimColor("#1A2B36")   # slightly lighter
PANEL_HEADER_BG = ManimColor("#1E3040")   # header band
PANEL_BORDER    = ManimColor("#37474F")   # border lines
PANEL_DIVIDER   = ManimColor("#2C3E4A")   # section dividers
PANEL_TITLE_COL = ManimColor("#78909C")   # section titles
PANEL_TEXT      = ManimColor("#ECEFF1")   # body text

STEP_PENDING    = ManimColor("#37474F")   # grey – not yet reached
STEP_ACTIVE     = ManimColor("#FFD600")   # gold – currently active
STEP_DONE       = ManimColor("#455A64")   # dimmed – completed

DECISION_REJECT = ManimColor("#C62828")   # dark red  – reject H₀
DECISION_FAIL   = ManimColor("#2E7D32")   # dark green – fail to reject
DECISION_BORDER = ManimColor("#FF5252")   # bright red border (reject)
DECISION_BORDER_FAIL = ManimColor("#00E676")  # bright green border (fail)

WIN_HIGHLIGHT   = ManimColor("#FFD600")   # gold – comparison winner
WIN_BORDER      = ManimColor("#F9A825")

FORMULA_ACTIVE  = ManimColor("#E040FB")   # violet – active derivation step
FORMULA_DONE    = ManimColor("#546E7A")   # muted – derived step

MATRIX_COLD     = ManimColor("#1565C0")
MATRIX_HOT      = ManimColor("#B71C1C")

FACE_DARKEN_SIDE  = 0.38
FACE_DARKEN_RIGHT = 0.55
FACE_DARKEN_ACCENT = 0.20


def _dk(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lt(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)

def _palette_t(t: float, cold: ManimColor, hot: ManimColor) -> ManimColor:
    return interpolate_color(cold, hot, float(np.clip(t, 0.0, 1.0)))

def _format_v(v: float, d: int = 3) -> str:
    if abs(v) < 0.001 and v != 0.0 or abs(v) > 9999:
        return f"{v:.{d}e}"
    return f"{v:.{d}f}"

def _sig_stars(p: float) -> tuple[str, ManimColor]:
    if p < 0.001: return "***", ManimColor("#00E676")
    if p < 0.01:  return "**",  ManimColor("#69F0AE")
    if p < 0.05:  return "*",   ManimColor("#FFD600")
    return "n.s.", GRAY_C


# ---------------------------------------------------------------------------
# PanelConfig
# ---------------------------------------------------------------------------

@dataclass
class PanelConfig:
    """Centralised visual parameters for all panel objects.

    Background & geometry
    ---------------------
    panel_width : float
        Default panel width in Manim units.
    panel_depth : float
        Z depth of the panel prism body.
    panel_bg_height : float
        Y height of the flat background face prism.
    header_height : float
        Height of the accent header bar prism (slightly taller than body).
    section_row_height : float
        Height of each key-value row inside a body section.
    section_gap : float
        Vertical gap between sections.
    cell_padding_x : float
        Horizontal padding inside value cells.
    corner_radius : float
        Currently unused; reserved for future rounded-corner support.

    Typography
    ----------
    header_font_size : int
    subtitle_font_size : int
    section_title_font_size : int
    body_font_size : int
    value_font_size : int
    formula_font_size : int
    annotation_font_size : int
    footer_font_size : int
    step_badge_font_size : int

    Step panel
    ----------
    step_badge_radius : float
        Radius of the numbered circle badges.
    step_row_height : float
        Height per step row.
    step_connector_width : float
        Stroke width of the vertical line linking step badges.

    Comparison panel
    ----------------
    comparison_col_width : float
        Width per comparison column.
    comparison_row_height : float
        Height per comparison row.
    winner_scale : float
        Scale boost for the winning cell.

    Matrix panel
    ------------
    matrix_cell_size : float
        Width/depth of each matrix cell.
    matrix_cell_gap : float
        Gap between matrix cells.
    matrix_max_height : float
        Height encoding for maximum absolute value.

    Decision banner
    ---------------
    banner_height : float
        Height of the decision banner prism.
    banner_border_width : float
        Stroke width of the banner border.

    Opacity
    -------
    bg_opacity : float
    header_opacity : float
    cell_opacity : float
    divider_opacity : float
    """

    # ---- background ----
    panel_width:        float = 5.00
    panel_depth:        float = 0.14
    panel_bg_height:    float = 0.06
    header_height:      float = 0.10
    section_row_height: float = 0.38
    section_gap:        float = 0.22
    cell_padding_x:     float = 0.15
    corner_radius:      float = 0.00

    # ---- typography ----
    header_font_size:        int = 28
    subtitle_font_size:      int = 20
    section_title_font_size: int = 22
    body_font_size:          int = 19
    value_font_size:         int = 21
    formula_font_size:       int = 24
    annotation_font_size:    int = 17
    footer_font_size:        int = 15
    step_badge_font_size:    int = 20

    # ---- step panel ----
    step_badge_radius:    float = 0.18
    step_row_height:      float = 0.50
    step_connector_width: float = 0.70

    # ---- comparison panel ----
    comparison_col_width: float = 1.70
    comparison_row_height: float = 0.40
    winner_scale:         float = 1.08

    # ---- matrix panel ----
    matrix_cell_size:   float = 0.55
    matrix_cell_gap:    float = 0.04
    matrix_max_height:  float = 0.30

    # ---- decision banner ----
    banner_height:       float = 0.22
    banner_border_width: float = 2.50

    # ---- opacity ----
    bg_opacity:      float = 0.90
    header_opacity:  float = 0.95
    cell_opacity:    float = 0.88
    divider_opacity: float = 0.40


# ---------------------------------------------------------------------------
# _PanelBackground  —  shaded 3D card body
# ---------------------------------------------------------------------------

class _PanelBackground(VGroup):
    """Background prism with a colored accent bar along the top (header) edge.

    Layout (side view):
        ┌──────── accent bar (header_height) ────────┐  ← top face (accent color)
        │                                             │
        │          body face (panel_bg_height)        │
        │                                             │
        └─────────────────────────────────────────────┘

    The accent bar is a separate thin prism sitting on top of the main body
    prism, colored with ``accent_color``.  The body uses ``PANEL_BG``.
    Both share the front/right darkening idiom.
    """

    def __init__(
        self,
        x0:           float,
        z0:           float,
        width:        float,
        total_height: float,
        cfg:          PanelConfig,
        accent_color: ManimColor = BLUE_E,
        y0:           float = 0.0,
    ):
        super().__init__()
        d  = cfg.panel_depth
        ah = cfg.header_height
        op = cfg.bg_opacity

        # ---- Body prism ----
        body_h = max(total_height - ah, cfg.panel_bg_height)
        body_color = PANEL_BG
        self._make_prism(x0, y0, z0, width, body_h, d,
                         body_color, op, suffix="body")

        # ---- Accent header prism (sits on top of body) ----
        acc_color = accent_color
        self._make_prism(x0, y0 + body_h, z0, width, ah, d,
                         acc_color, cfg.header_opacity, suffix="header")

        self._x0    = x0
        self._z0    = z0
        self._width = width
        self._total_height = total_height
        self._y0    = y0

    def _make_prism(
        self,
        x0: float, y0: float, z0: float,
        w: float, h: float, d: float,
        color: ManimColor, opacity: float,
        suffix: str = "",
    ) -> None:
        fc = _dk(color, FACE_DARKEN_SIDE)
        rc = _dk(color, FACE_DARKEN_RIGHT)

        AFL = np.array([x0,     y0,     z0    ])
        AFR = np.array([x0 + w, y0,     z0    ])
        ABL = np.array([x0,     y0,     z0 + d])
        ABR = np.array([x0 + w, y0,     z0 + d])
        TFL = np.array([x0,     y0 + h, z0    ])
        TFR = np.array([x0 + w, y0 + h, z0    ])
        TBL = np.array([x0,     y0 + h, z0 + d])
        TBR = np.array([x0 + w, y0 + h, z0 + d])

        def _face(pts, col):
            p = Polygon(*pts, color=col)
            p.set_fill(color=col, opacity=opacity)
            p.set_stroke(color=_dk(col, 0.50), width=0.55, opacity=0.40)
            return p

        top   = _face([TFL, TFR, TBR, TBL], color)
        front = _face([AFL, AFR, TFR, TFL], fc)
        right = _face([AFR, ABR, TBR, TFR], rc)
        self.add(front, right, top)

        if suffix == "header":
            self.header_top = top
        elif suffix == "body":
            self.body_top = top


# ---------------------------------------------------------------------------
# _SectionDivider  —  ruled line between panel sections
# ---------------------------------------------------------------------------

class _SectionDivider(VGroup):
    def __init__(
        self,
        x0:    float,
        x1:    float,
        y:     float,
        z:     float,
        cfg:   PanelConfig,
        color: ManimColor = PANEL_DIVIDER,
    ):
        super().__init__()
        line = Line3D(
            start=np.array([x0, y, z]),
            end  =np.array([x1, y, z]),
            color=color, stroke_width=0.70,
        )
        line.set_opacity(cfg.divider_opacity)
        self.add(line)


# ---------------------------------------------------------------------------
# _StepBadge  —  numbered circle for StepPanel3D
# ---------------------------------------------------------------------------

class _StepBadge(VGroup):
    """Filled circle with a step number, color-coded by state."""

    def __init__(
        self,
        number:   int,
        pos:      np.ndarray,
        state:    str = "pending",    # "pending" | "active" | "done"
        cfg:      PanelConfig | None = None,
    ):
        super().__init__()
        cfg   = cfg or PanelConfig()
        r     = cfg.step_badge_radius
        color = {
            "pending": STEP_PENDING,
            "active":  STEP_ACTIVE,
            "done":    STEP_DONE,
        }.get(state, STEP_PENDING)

        # Circle (approximated as a flat polygon with many sides)
        n_pts  = 24
        pts    = [
            np.array([pos[0] + r * np.cos(k / n_pts * TAU),
                      pos[1],
                      pos[2] + r * np.sin(k / n_pts * TAU)])
            for k in range(n_pts)
        ]
        circle = Polygon(*pts, color=color)
        circle.set_fill(color=color, opacity=0.95)
        circle.set_stroke(color=_dk(color, 0.35), width=0.70)
        self.add(circle)
        self.circle = circle

        # Number text
        txt_color = _lt(color, 0.45) if state != "active" else BLACK
        num_obj = Text(str(number), color=txt_color,
                       font_size=cfg.step_badge_font_size, weight="BOLD")
        num_obj.move_to(pos)
        self.add(num_obj)
        self.num_obj  = num_obj
        self._color   = color
        self._pos     = pos
        self._state   = state
        self._cfg     = cfg

    def set_state(self, state: str) -> None:
        """Update color in-place (not animated; use Transform for animation)."""
        color = {
            "pending": STEP_PENDING,
            "active":  STEP_ACTIVE,
            "done":    STEP_DONE,
        }.get(state, STEP_PENDING)
        self.circle.set_fill(color=color)
        self.circle.set_stroke(color=_dk(color, 0.35))
        self._state = state
        self._color = color


# ---------------------------------------------------------------------------
# _ValueCell  —  colored cell for comparison / matrix grids
# ---------------------------------------------------------------------------

class _ValueCell(VGroup):
    """A flat rectangular cell displaying a numeric or text value.

    Used as a building block inside ComparisonPanel3D and MatrixPanel3D.
    """

    def __init__(
        self,
        x0:     float,
        z0:     float,
        width:  float,
        depth:  float,
        height: float,
        color:  ManimColor,
        text:   str,
        y0:     float = 0.0,
        font_size: int = 19,
        bold:   bool = False,
        opacity: float = 0.88,
    ):
        super().__init__()
        fc = _dk(color, FACE_DARKEN_SIDE)
        rc = _dk(color, FACE_DARKEN_RIGHT)
        h  = max(height, 0.004)

        AFL = np.array([x0,         y0,     z0       ])
        AFR = np.array([x0 + width, y0,     z0       ])
        ABL = np.array([x0,         y0,     z0 + depth])
        ABR = np.array([x0 + width, y0,     z0 + depth])
        TFL = np.array([x0,         y0 + h, z0       ])
        TFR = np.array([x0 + width, y0 + h, z0       ])
        TBL = np.array([x0,         y0 + h, z0 + depth])
        TBR = np.array([x0 + width, y0 + h, z0 + depth])

        def _face(pts, col):
            p = Polygon(*pts, color=col)
            p.set_fill(color=col, opacity=opacity)
            p.set_stroke(color=PANEL_BORDER, width=0.40, opacity=0.45)
            return p

        self.top_face   = _face([TFL, TFR, TBR, TBL], color)
        self.front_face = _face([AFL, AFR, TFR, TFL], fc)
        self.right_face = _face([AFR, ABR, TBR, TFR], rc)
        self.add(self.front_face, self.right_face, self.top_face)

        text_color = _lt(color, 0.35)
        lbl = Text(text, color=text_color, font_size=font_size,
                   weight="BOLD" if bold else "NORMAL")
        lbl.move_to(np.array([
            x0 + width / 2,
            y0 + h + 0.18,
            z0 + depth / 2,
        ]))
        self.add(lbl)
        self.label    = lbl
        self._color   = color
        self._y_top   = y0 + h
        self._x_center = x0 + width / 2
        self._z_center = z0 + depth / 2

    @property
    def top_center(self) -> np.ndarray:
        return np.array([self._x_center, self._y_top, self._z_center])

    @property
    def floor_center(self) -> np.ndarray:
        return np.array([self._x_center, 0.0, self._z_center])


# ---------------------------------------------------------------------------
# _DecisionBanner  —  Reject / Fail-to-Reject bold panel
# ---------------------------------------------------------------------------

class _DecisionBanner(VGroup):
    """Bold colored banner for hypothesis test decisions.

    A wide flat prism with a thick border stroke and large centered text.
    """

    def __init__(
        self,
        x0:      float,
        z0:      float,
        width:   float,
        depth:   float,
        reject:  bool,
        y0:      float = 0.0,
        cfg:     PanelConfig | None = None,
    ):
        super().__init__()
        cfg = cfg or PanelConfig()
        h   = cfg.banner_height

        color       = DECISION_REJECT if reject else DECISION_FAIL
        border_col  = DECISION_BORDER if reject else DECISION_BORDER_FAIL
        verdict_txt = "Reject H\u2080" if reject else "Fail to Reject H\u2080"

        # Background face
        bg = Polygon(
            np.array([x0,         y0 + h, z0        ]),
            np.array([x0 + width, y0 + h, z0        ]),
            np.array([x0 + width, y0 + h, z0 + depth]),
            np.array([x0,         y0 + h, z0 + depth]),
            color=color,
        )
        bg.set_fill(color=color, opacity=0.92)
        bg.set_stroke(color=border_col, width=cfg.banner_border_width,
                      opacity=0.95)
        self.add(bg)
        self.bg = bg

        # Front face
        front = Polygon(
            np.array([x0,         y0,     z0]),
            np.array([x0 + width, y0,     z0]),
            np.array([x0 + width, y0 + h, z0]),
            np.array([x0,         y0 + h, z0]),
            color=_dk(color, FACE_DARKEN_SIDE),
        )
        front.set_fill(color=_dk(color, FACE_DARKEN_SIDE), opacity=0.90)
        front.set_stroke(color=border_col, width=cfg.banner_border_width * 0.7)
        self.add(front)

        # Verdict text
        lbl = Text(verdict_txt, color=WHITE,
                   font_size=cfg.header_font_size + 4, weight="BOLD")
        lbl.move_to(np.array([x0 + width / 2, y0 + h + 0.28, z0 + depth / 2]))
        self.add(lbl)
        self.verdict_label = lbl
        self._color = color
        self._border_color = border_col


# ---------------------------------------------------------------------------
# InfoPanel3D
# ---------------------------------------------------------------------------

class InfoPanel3D(VGroup):
    """General-purpose floating information panel.

    Layout (top view, looking down the Y axis):
        ┌─────────── Header bar (accent color) ───────────┐
        │  TITLE                           SUBTITLE        │
        ├─────────────────────────────────────────────────┤
        │  Section 1 title                                 │
        │  key1   value1   key2   value2                   │
        ├─────────────────────────────────────────────────┤
        │  Section 2 title                                 │
        │  key1   value1                                   │
        ├─────────────────────────────────────────────────┤
        │  footer text                                     │
        └─────────────────────────────────────────────────┘

    Parameters
    ----------
    title : str
        Panel header text.
    subtitle : str
        Optional smaller text on the right of the header.
    pos : np.ndarray
        World position of the top-left corner of the panel.
    accent_color : ManimColor
        Header bar accent color.
    footer : str
        Optional footer row text.
    config : PanelConfig | None
    """

    def __init__(
        self,
        title:        str,
        subtitle:     str = "",
        pos:          np.ndarray = np.array([0.0, 0.0, 0.0]),
        accent_color: ManimColor = BLUE_E,
        footer:       str = "",
        config:       PanelConfig | None = None,
    ):
        super().__init__()
        self.cfg    = cfg = config or PanelConfig()
        self._pos   = pos.copy()
        self._accent = accent_color
        self._footer = footer

        self._sections:        list[VGroup] = []
        self._section_items:   list[list]   = []
        self._section_titles:  list[str]    = []
        self._dividers:        list[_SectionDivider] = []

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        w  = cfg.panel_width
        d  = cfg.panel_depth
        ah = cfg.header_height

        # ---- Header bar ----
        self._hdr_bg = _PanelBackground(
            x0=px, z0=pz, width=w,
            total_height=ah + cfg.panel_bg_height,
            cfg=cfg, accent_color=accent_color, y0=py,
        )
        self.add(self._hdr_bg)

        y_hdr_top = py + ah + cfg.panel_bg_height
        title_obj = Text(title, color=WHITE,
                         font_size=cfg.header_font_size, weight="BOLD")
        title_obj.move_to(np.array([px + w * 0.30, y_hdr_top + 0.18, pz + d / 2]))
        self.add(title_obj)
        self.title_obj = title_obj

        if subtitle:
            sub_obj = Text(subtitle, color=_lt(accent_color, 0.35),
                           font_size=cfg.subtitle_font_size)
            sub_obj.move_to(np.array([px + w * 0.78, y_hdr_top + 0.18, pz + d / 2]))
            self.add(sub_obj)
            self.subtitle_obj = sub_obj

        # y cursor starts just below the header
        self._y_cursor = y_hdr_top + 0.05

        # Footer placeholder (positioned after sections are added)
        self._footer_obj = None
        if footer:
            self._footer_txt = footer
        else:
            self._footer_txt = ""

    # ------------------------------------------------------------------

    def add_section(
        self,
        title:   str,
        items:   list[tuple[str, str]],
        n_cols:  int = 2,
    ) -> VGroup:
        """Add a body section with a title and key-value item rows.

        Parameters
        ----------
        title : str
            Section title (displayed above the items).
        items : list of (key, value) string pairs.
        n_cols : int
            Number of key-value columns per row (1 or 2).

        Returns the VGroup for this section.
        """
        cfg = self.cfg
        px, py, pz = float(self._pos[0]), float(self._pos[1]), float(self._pos[2])
        w  = cfg.panel_width
        d  = cfg.panel_depth
        y  = self._y_cursor

        section_grp = VGroup()

        # Section divider above title (skip for first section)
        if self._sections:
            div = _SectionDivider(px, px + w, y, pz + d / 2, cfg)
            self._dividers.append(div)
            self.add(div)
            y += cfg.section_gap * 0.3

        # Section title
        if title:
            sec_title = Text(title, color=PANEL_TITLE_COL,
                             font_size=cfg.section_title_font_size,
                             weight="BOLD")
            sec_title.move_to(np.array([px + w * 0.08, y + 0.18, pz + d / 2]))
            section_grp.add(sec_title)
            y += cfg.section_row_height * 0.75

        # Item rows
        n_per_row = n_cols
        for row_start in range(0, len(items), n_per_row):
            row_items = items[row_start: row_start + n_per_row]
            col_width = w / n_per_row

            for col_idx, (key, val) in enumerate(row_items):
                cx = px + col_idx * col_width + cfg.cell_padding_x

                key_obj = Text(key + ":", color=PANEL_TITLE_COL,
                               font_size=cfg.body_font_size)
                val_obj = Text(val, color=PANEL_TEXT,
                               font_size=cfg.value_font_size, weight="BOLD")

                key_obj.move_to(np.array([cx + key_obj.width / 2, y, pz + d / 2]))
                val_obj.move_to(np.array([
                    cx + key_obj.width + 0.12 + val_obj.width / 2,
                    y, pz + d / 2
                ]))
                section_grp.add(key_obj, val_obj)

            y += cfg.section_row_height

        self._sections.append(section_grp)
        self._section_items.append(items)
        self._section_titles.append(title)
        self._y_cursor = y + cfg.section_gap * 0.5
        self.add(section_grp)

        # Rebuild background to cover current content
        self._rebuild_background()

        # Update footer position
        if self._footer_txt:
            self._rebuild_footer()

        return section_grp

    def _rebuild_background(self) -> None:
        """Extend the background prism to cover all current sections."""
        if hasattr(self, "_body_bg"):
            self.remove(self._body_bg)
        cfg    = self.cfg
        px, py, pz = float(self._pos[0]), float(self._pos[1]), float(self._pos[2])
        body_h = max(
            self._y_cursor - py - cfg.header_height - cfg.panel_bg_height,
            cfg.panel_bg_height,
        )
        body_bg = Polygon(
            np.array([px,              py, pz              ]),
            np.array([px + cfg.panel_width, py, pz              ]),
            np.array([px + cfg.panel_width, py, pz + cfg.panel_depth]),
            np.array([px,              py, pz + cfg.panel_depth]),
            color=PANEL_BG,
        )
        body_bg.set_fill(color=PANEL_BG, opacity=cfg.bg_opacity)
        body_bg.set_stroke(color=PANEL_BORDER, width=0.55, opacity=0.45)
        self._body_bg = body_bg
        self.add(body_bg)

    def _rebuild_footer(self) -> None:
        if self._footer_obj is not None:
            self.remove(self._footer_obj)
        cfg = self.cfg
        px, pz = float(self._pos[0]), float(self._pos[2])
        y_f    = self._y_cursor + 0.10
        foot   = Text(self._footer_txt, color=GRAY_C,
                      font_size=cfg.footer_font_size)
        foot.move_to(np.array([px + cfg.panel_width / 2, y_f, pz + cfg.panel_depth / 2]))
        self._footer_obj = foot
        self.add(foot)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_build(
        self,
        lag_ratio: float = 0.30,
        run_time:  float = 2.5,
    ) -> Succession:
        """Header slides in from above, then sections cascade downward."""
        header_anim = GrowFromPoint(
            self._hdr_bg,
            point=self._hdr_bg.get_top(),
            run_time=run_time * 0.35,
        )
        section_anims = [
            FadeIn(sec, shift=DOWN * 0.08, run_time=run_time * 0.25)
            for sec in self._sections
        ]
        divider_anims = [
            Create(div, run_time=run_time * 0.15)
            for div in self._dividers
        ]
        body_anim = LaggedStart(
            *[a for pair in zip(divider_anims + [None] * len(section_anims),
                                section_anims)
              for a in pair if a is not None],
            lag_ratio=lag_ratio,
            run_time=run_time * 0.65,
        )
        return Succession(header_anim, body_anim)

    def animate_update_section(
        self,
        section_index: int,
        new_items:     list[tuple[str, str]],
        n_cols:        int  = 2,
        run_time:      float = 0.8,
    ) -> AnimationGroup:
        """Morph one section's key-value content to new values.

        Fades out the old section content and fades in new Text objects.
        """
        if section_index >= len(self._sections):
            return AnimationGroup()
        old_sec = self._sections[section_index]
        fade_out = FadeOut(old_sec, run_time=run_time * 0.4)
        # Build new section group at the same position
        # (simplified: just rebuild the text objects)
        cfg    = self.cfg
        px, pz = float(self._pos[0]), float(self._pos[2])
        d      = cfg.panel_depth
        # Approximate y position by counting sections above
        y_approx = (float(self._pos[1])
                    + cfg.header_height + cfg.panel_bg_height
                    + section_index * (cfg.section_row_height * 2 + cfg.section_gap))
        new_grp = VGroup()
        n_per_row = n_cols
        for row_start in range(0, len(new_items), n_per_row):
            row_items = new_items[row_start: row_start + n_per_row]
            col_width = cfg.panel_width / n_per_row
            for col_idx, (key, val) in enumerate(row_items):
                cx = px + col_idx * col_width + cfg.cell_padding_x
                y  = y_approx + row_start // n_per_row * cfg.section_row_height
                val_obj = Text(val, color=PANEL_TEXT,
                               font_size=cfg.value_font_size, weight="BOLD")
                val_obj.move_to(np.array([cx + 0.60, y, pz + d / 2]))
                new_grp.add(val_obj)

        self.add(new_grp)
        self._sections[section_index] = new_grp
        fade_in = FadeIn(new_grp, run_time=run_time * 0.6)
        return AnimationGroup(fade_out, Succession(fade_out, fade_in))

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def docked(
        cls,
        title:        str,
        corner:       str = "TR",
        scene_width:  float = 14.0,
        scene_height: float = 8.0,
        accent_color: ManimColor = BLUE_E,
        config:       PanelConfig | None = None,
    ) -> "InfoPanel3D":
        """Create a panel docked to a scene corner.

        Parameters
        ----------
        corner : str
            One of ``"TL"``, ``"TR"``, ``"BL"``, ``"BR"``.
        scene_width, scene_height : float
            Approximate Manim scene dimensions in world units.
        """
        cfg   = config or PanelConfig()
        hw    = scene_width / 2
        hh    = scene_height / 2
        pw    = cfg.panel_width
        pad   = 0.20

        positions = {
            "TL": np.array([-hw + pad,           hh - pad - 0.5, 0.0]),
            "TR": np.array([ hw - pw - pad,       hh - pad - 0.5, 0.0]),
            "BL": np.array([-hw + pad,           -hh + pad,       0.0]),
            "BR": np.array([ hw - pw - pad,      -hh + pad,       0.0]),
        }
        pos = positions.get(corner.upper(), np.array([0.0, 0.0, 0.0]))
        return cls(title=title, pos=pos, accent_color=accent_color, config=cfg)


# ---------------------------------------------------------------------------
# StepPanel3D
# ---------------------------------------------------------------------------

class StepPanel3D(VGroup):
    """Numbered step-by-step derivation/proof panel.

    Each step has a circular badge (number) + text + optional sub-text.
    Steps transition through three states:
        ``"pending"`` – grey, not yet reached
        ``"active"``  – gold, currently being shown
        ``"done"``    – dimmed slate, already covered

    Parameters
    ----------
    steps : list[tuple[str, str]]
        Each entry is (main_text, sub_text).  ``sub_text`` may be empty.
    pos : np.ndarray
        World position of the top of the panel.
    title : str
        Optional panel header.
    accent_color : ManimColor
    config : PanelConfig | None
    """

    def __init__(
        self,
        steps:        list[tuple[str, str]],
        pos:          np.ndarray = np.array([0.0, 0.0, 0.0]),
        title:        str = "",
        accent_color: ManimColor = STEP_ACTIVE,
        config:       PanelConfig | None = None,
    ):
        super().__init__()
        self.cfg    = cfg = config or PanelConfig()
        self._steps = steps
        self._pos   = pos.copy()
        self._current_step = -1

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        w  = cfg.panel_width
        d  = cfg.panel_depth
        rh = cfg.step_row_height
        br = cfg.step_badge_radius

        # Background
        total_h = (len(steps) + 1) * rh + 0.30
        bg = _PanelBackground(
            x0=px, z0=pz, width=w,
            total_height=total_h,
            cfg=cfg, accent_color=accent_color, y0=py,
        )
        self.add(bg)
        self.background = bg

        y_start = py + total_h + 0.20

        # Title
        if title:
            t_obj = Text(title, color=WHITE,
                         font_size=cfg.header_font_size, weight="BOLD")
            t_obj.move_to(np.array([px + w / 2, y_start, pz + d / 2]))
            self.add(t_obj)
            y_start -= rh * 0.60

        # Step rows
        self._badges:     list[_StepBadge] = []
        self._step_texts: list[VGroup]      = []
        self._connectors: list[Line3D]      = []

        for i, (main_txt, sub_txt) in enumerate(steps):
            y_row = y_start - i * rh

            badge_pos = np.array([px + br + 0.18, y_row, pz + d / 2])
            badge     = _StepBadge(i + 1, badge_pos, state="pending", cfg=cfg)
            self._badges.append(badge)
            self.add(badge)

            # Connector line from this badge down to next
            if i < len(steps) - 1:
                conn = Line3D(
                    start=np.array([badge_pos[0], y_row - br - 0.04, badge_pos[2]]),
                    end  =np.array([badge_pos[0], y_row - rh + br + 0.04, badge_pos[2]]),
                    color=STEP_PENDING, stroke_width=cfg.step_connector_width,
                )
                conn.set_opacity(0.45)
                self._connectors.append(conn)
                self.add(conn)

            # Main text and sub-text
            txt_x = px + br * 2 + 0.45
            row_grp = VGroup()

            main_obj = Text(main_txt, color=PANEL_TEXT,
                            font_size=cfg.body_font_size)
            main_obj.move_to(np.array([txt_x + main_obj.width / 2, y_row, pz + d / 2]))
            row_grp.add(main_obj)

            if sub_txt:
                sub_obj = Text(sub_txt, color=GRAY_C,
                               font_size=cfg.annotation_font_size)
                sub_obj.move_to(np.array([txt_x + sub_obj.width / 2,
                                          y_row - 0.22, pz + d / 2]))
                row_grp.add(sub_obj)

            self._step_texts.append(row_grp)
            self.add(row_grp)

    # ------------------------------------------------------------------

    def animate_activate_step(
        self,
        step_index: int,
        run_time:   float = 0.55,
    ) -> AnimationGroup:
        """Transition step ``step_index`` to the active state.

        The previous active step is moved to done.  All pending steps
        ahead remain grey.
        """
        anims = []
        if self._current_step >= 0:
            old_badge  = self._badges[self._current_step]
            done_badge = _StepBadge(
                self._current_step + 1,
                old_badge._pos, state="done", cfg=self.cfg,
            )
            anims.append(Transform(old_badge, done_badge, run_time=run_time))
            anims.append(
                self._step_texts[self._current_step]
                .animate(run_time=run_time)
                .set_opacity(0.45)
            )

        new_badge = _StepBadge(
            step_index + 1,
            self._badges[step_index]._pos, state="active", cfg=self.cfg,
        )
        anims.append(Transform(self._badges[step_index], new_badge,
                               run_time=run_time))
        anims.append(
            self._step_texts[step_index]
            .animate(run_time=run_time)
            .set_opacity(1.0)
            .set_color(PANEL_TEXT)
        )
        self._current_step = step_index
        return AnimationGroup(*anims)

    def animate_complete_step(
        self,
        step_index: int,
        run_time:   float = 0.45,
    ) -> Succession:
        """Mark step ``step_index`` as done with a brief checkmark flash."""
        badge   = self._badges[step_index]
        done_b  = _StepBadge(step_index + 1, badge._pos,
                              state="done", cfg=self.cfg)

        # Flash gold → done
        flash   = badge.animate(run_time=run_time * 0.30).scale(1.20)
        morph   = Transform(badge, done_b, run_time=run_time * 0.70)
        return Succession(flash, morph)

    def animate_walk_steps(
        self,
        pause_between: float = 0.20,
        run_time_each: float = 0.55,
    ) -> Succession:
        """Activate each step in sequence, then mark them all done."""
        anims = []
        for i in range(len(self._steps)):
            anims.append(self.animate_activate_step(i, run_time=run_time_each))
        return Succession(*anims)

    def animate_reveal_all(
        self,
        lag_ratio: float = 0.15,
        run_time:  float = 2.0,
    ) -> LaggedStart:
        """Fade in all step rows simultaneously (no state changes)."""
        items = []
        for badge, txt in zip(self._badges, self._step_texts):
            items.append(FadeIn(badge, run_time=run_time * 0.4))
            items.append(FadeIn(txt,   run_time=run_time * 0.4))
        return LaggedStart(*items, lag_ratio=lag_ratio, run_time=run_time)


# ---------------------------------------------------------------------------
# ComparisonPanel3D
# ---------------------------------------------------------------------------

class ComparisonPanel3D(VGroup):
    """Side-by-side comparison of N items across M metrics.

    Parameters
    ----------
    items : list[str]
        Column headers (e.g. model names, method names).
    metrics : list[dict]
        Each dict has keys:
          ``"name"``    (str)   – row label
          ``"values"``  (list[float | str]) – one per item column
          ``"higher_is_better"`` (bool) – controls winner highlighting
          ``"format"``  (str, optional) – ``"f"``, ``"%"``, ``"e"``
          ``"decimals"`` (int, optional)
    pos : np.ndarray
    item_colors : list[ManimColor] | None
    config : PanelConfig | None
    """

    def __init__(
        self,
        items:        list[str],
        metrics:      list[dict],
        pos:          np.ndarray = np.array([0.0, 0.0, 0.0]),
        item_colors:  list[ManimColor] | None = None,
        config:       PanelConfig | None = None,
    ):
        super().__init__()
        self.cfg     = cfg  = config or PanelConfig()
        self._items  = items
        self._metrics = metrics
        n_cols       = len(items)
        n_rows       = len(metrics)

        default_palette = [
            ManimColor("#1565C0"), ManimColor("#B71C1C"),
            ManimColor("#2E7D32"), ManimColor("#E65100"),
            ManimColor("#6A1B9A"), ManimColor("#00838F"),
        ]
        colors = item_colors or default_palette[:n_cols]

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        cw  = cfg.comparison_col_width
        rh  = cfg.comparison_row_height
        d   = cfg.panel_depth
        total_w = cw * (n_cols + 1)   # +1 for the metric-name column

        # ---- Header row ----
        self._header_cells: list[_ValueCell] = []
        # Metric label column header (empty)
        hdr_bg = _ValueCell(
            x0=px, z0=pz,
            width=cw, depth=d, height=cfg.panel_bg_height,
            color=PANEL_HEADER_BG, text="",
            y0=py + n_rows * rh, font_size=cfg.section_title_font_size,
        )
        self.add(hdr_bg)

        for j, (item, color) in enumerate(zip(items, colors)):
            hdr = _ValueCell(
                x0=px + (j + 1) * cw, z0=pz,
                width=cw, depth=d, height=cfg.panel_bg_height,
                color=_dk(color, 0.25), text=item, bold=True,
                y0=py + n_rows * rh,
                font_size=cfg.section_title_font_size,
            )
            self._header_cells.append(hdr)
            self.add(hdr)

        # ---- Metric rows ----
        self._row_cells:  list[list[_ValueCell]] = []
        self._metric_labels: list[Text]           = []

        for i, metric in enumerate(metrics):
            name      = metric.get("name", f"M{i}")
            values    = metric.get("values", [0] * n_cols)
            hib       = metric.get("higher_is_better", True)
            fmt       = metric.get("format", "f")
            dec       = metric.get("decimals", 3)
            y_row     = py + (n_rows - 1 - i) * rh

            # Metric name cell
            metric_lbl = Text(name, color=PANEL_TITLE_COL,
                              font_size=cfg.body_font_size)
            metric_lbl.move_to(np.array([px + cw / 2, y_row + rh / 2, pz + d / 2]))
            self._metric_labels.append(metric_lbl)
            self.add(metric_lbl)

            # Determine winner
            numeric_vals = []
            for v in values:
                try:
                    numeric_vals.append(float(v))
                except (TypeError, ValueError):
                    numeric_vals.append(None)

            valid_vals = [v for v in numeric_vals if v is not None]
            if valid_vals:
                winner_val = max(valid_vals) if hib else min(valid_vals)
            else:
                winner_val = None

            row_cells = []
            for j, (val, color) in enumerate(zip(values, colors)):
                is_winner = (
                    winner_val is not None
                    and isinstance(val, (int, float))
                    and abs(float(val) - winner_val) < 1e-10
                )
                cell_color  = _lt(color, 0.15) if is_winner else _dk(color, 0.30)
                try:
                    val_str = _format_v(float(val), dec)
                except (TypeError, ValueError):
                    val_str = str(val)

                cell = _ValueCell(
                    x0=px + (j + 1) * cw, z0=pz,
                    width=cw, depth=d, height=cfg.panel_bg_height,
                    color=cell_color, text=val_str,
                    y0=y_row, bold=is_winner,
                    font_size=cfg.value_font_size,
                    opacity=0.92 if is_winner else 0.70,
                )
                row_cells.append(cell)
                self.add(cell)

            self._row_cells.append(row_cells)

    # ------------------------------------------------------------------

    def animate_build_columns(
        self,
        lag_ratio: float = 0.20,
        run_time:  float = 2.0,
    ) -> LaggedStart:
        """Columns grow left to right: header + all its data cells."""
        col_groups = [VGroup(h) for h in self._header_cells]
        for i, row in enumerate(self._row_cells):
            for j, cell in enumerate(row):
                col_groups[j].add(cell)

        return LaggedStart(
            *[GrowFromPoint(grp, grp.get_bottom(), run_time=run_time * 0.55)
              for grp in col_groups],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_highlight_winner(
        self,
        row_index: int,
        run_time:  float = 0.55,
    ) -> Succession:
        """Flash-scale the winning cell in a given metric row."""
        if row_index >= len(self._row_cells):
            return Succession(FadeIn(VGroup(), run_time=0.1))
        row    = self._row_cells[row_index]
        metric = self._metrics[row_index]
        hib    = metric.get("higher_is_better", True)
        values = metric.get("values", [])
        try:
            numeric = [float(v) for v in values]
            winner_val = max(numeric) if hib else min(numeric)
            winner_idx = numeric.index(winner_val)
        except (TypeError, ValueError):
            return Succession(FadeIn(VGroup(), run_time=0.1))

        cell   = row[winner_idx]
        scale  = cfg = self.cfg
        flash  = cell.animate(run_time=run_time / 2).scale(cfg.winner_scale)
        restore = cell.animate(run_time=run_time / 2).scale(1 / cfg.winner_scale)
        return Succession(flash, restore)

    def animate_reveal_all(
        self,
        lag_ratio: float = 0.04,
        run_time:  float = 2.5,
    ) -> LaggedStart:
        """All cells fade in row by row."""
        all_items = list(self._header_cells)
        for row in self._row_cells:
            all_items.extend(row)
        return LaggedStart(
            *[FadeIn(c, run_time=run_time * 0.35) for c in all_items],
            lag_ratio=lag_ratio, run_time=run_time,
        )


# ---------------------------------------------------------------------------
# DistributionInfoPanel3D
# ---------------------------------------------------------------------------

class DistributionInfoPanel3D(VGroup):
    """Specialised information panel for probability distributions.

    Shows: name, PDF/PMF formula, parameter table, key statistics.

    Parameters
    ----------
    name : str
        Distribution name, e.g. ``"Normal"``.
    formula_tex : str
        LaTeX for the PDF/PMF, e.g. ``r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{...}"``.
    parameters : list[dict]
        Each dict: ``{"symbol": str, "name": str, "value": float, "range": str}``.
    stats : dict[str, str]
        Key statistics as TeX strings, e.g.
        ``{"Mean": r"\mu", "Variance": r"\sigma^2"}``.
    pos : np.ndarray
    config : PanelConfig | None
    """

    def __init__(
        self,
        name:         str,
        formula_tex:  str,
        parameters:   list[dict],
        stats:        dict[str, str],
        pos:          np.ndarray = np.array([0.0, 0.0, 0.0]),
        accent_color: ManimColor = ManimColor("#1565C0"),
        config:       PanelConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg = config or PanelConfig()
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        w  = cfg.panel_width
        d  = cfg.panel_depth
        y  = py

        # ---- Header ----
        name_obj = Text(name, color=WHITE,
                        font_size=cfg.header_font_size + 2, weight="BOLD")
        name_obj.move_to(np.array([px + w / 2, y + 0.35, pz + d / 2]))
        self.add(name_obj)
        y += 0.65

        # Accent divider
        div = Line3D(
            start=np.array([px + 0.10, y, pz + d / 2]),
            end  =np.array([px + w - 0.10, y, pz + d / 2]),
            color=accent_color, stroke_width=1.5,
        )
        div.set_opacity(0.70)
        self.add(div)
        y += 0.20

        # ---- Formula ----
        formula_obj = MathTex(formula_tex, color=_lt(accent_color, 0.20),
                              font_size=cfg.formula_font_size)
        formula_obj.move_to(np.array([px + w / 2, y + 0.25, pz + d / 2]))
        self.add(formula_obj)
        self.formula_obj = formula_obj
        y += 0.65

        # ---- Parameters table ----
        sec_div = _SectionDivider(px, px + w, y, pz + d / 2, cfg)
        self.add(sec_div)
        y += 0.18

        param_title = Text("Parameters", color=PANEL_TITLE_COL,
                           font_size=cfg.section_title_font_size, weight="BOLD")
        param_title.move_to(np.array([px + 0.25, y, pz + d / 2]))
        self.add(param_title)
        y += 0.35

        # Header row: Symbol | Name | Value | Range
        hdr_cols = ["Symbol", "Name", "Value", "Range"]
        col_w    = w / len(hdr_cols)
        for j, hdr in enumerate(hdr_cols):
            h_obj = Text(hdr, color=PANEL_TITLE_COL,
                         font_size=cfg.annotation_font_size, weight="BOLD")
            h_obj.move_to(np.array([px + (j + 0.5) * col_w, y, pz + d / 2]))
            self.add(h_obj)
        y += 0.32

        self._param_rows: list[VGroup] = []
        for param in parameters:
            row_grp = VGroup()
            sym_color = _lt(accent_color, 0.20)
            for j, (key, fc) in enumerate([
                ("symbol", sym_color),
                ("name",   PANEL_TEXT),
                ("value",  PANEL_TEXT),
                ("range",  GRAY_C),
            ]):
                val_txt = str(param.get(key, ""))
                if key == "value":
                    try:
                        val_txt = _format_v(float(val_txt), 3)
                    except (TypeError, ValueError):
                        pass
                obj = MathTex(val_txt, color=fc,
                              font_size=cfg.body_font_size) \
                      if key in ("symbol", "range") else \
                      Text(val_txt, color=fc, font_size=cfg.body_font_size)
                obj.move_to(np.array([px + (j + 0.5) * col_w, y, pz + d / 2]))
                row_grp.add(obj)
            self._param_rows.append(row_grp)
            self.add(row_grp)
            y += 0.33

        # ---- Statistics section ----
        y += 0.12
        sec_div2 = _SectionDivider(px, px + w, y, pz + d / 2, cfg)
        self.add(sec_div2)
        y += 0.18

        stats_title = Text("Key Statistics", color=PANEL_TITLE_COL,
                           font_size=cfg.section_title_font_size, weight="BOLD")
        stats_title.move_to(np.array([px + 0.25, y, pz + d / 2]))
        self.add(stats_title)
        y += 0.35

        self._stat_rows: list[VGroup] = []
        n_cols = 2
        stat_items = list(stats.items())
        for row_start in range(0, len(stat_items), n_cols):
            row_items = stat_items[row_start: row_start + n_cols]
            row_grp   = VGroup()
            for j, (stat_name, stat_tex) in enumerate(row_items):
                cx = px + j * (w / n_cols) + 0.12
                nm = Text(stat_name + ":", color=PANEL_TITLE_COL,
                          font_size=cfg.body_font_size)
                fm = MathTex(stat_tex, color=PANEL_TEXT,
                             font_size=cfg.body_font_size)
                nm.move_to(np.array([cx + nm.width / 2, y, pz + d / 2]))
                fm.move_to(np.array([cx + nm.width + 0.12 + fm.width / 2,
                                     y, pz + d / 2]))
                row_grp.add(nm, fm)
            self._stat_rows.append(row_grp)
            self.add(row_grp)
            y += 0.34

        # Background (built last, covers full height)
        total_h = y - py + 0.25
        bg = _PanelBackground(
            x0=px, z0=pz, width=w,
            total_height=total_h,
            cfg=cfg, accent_color=accent_color, y0=py,
        )
        self.add(bg)

    # ------------------------------------------------------------------

    def animate_reveal(
        self,
        run_time: float = 3.0,
    ) -> Succession:
        """Formula writes in, then parameter rows cascade, then stats."""
        write_formula = Write(self.formula_obj, run_time=run_time * 0.30)
        param_cascade = LaggedStart(
            *[FadeIn(row, run_time=run_time * 0.20)
              for row in self._param_rows],
            lag_ratio=0.15, run_time=run_time * 0.40,
        )
        stat_cascade = LaggedStart(
            *[FadeIn(row, run_time=run_time * 0.20)
              for row in self._stat_rows],
            lag_ratio=0.18, run_time=run_time * 0.30,
        )
        return Succession(write_formula, param_cascade, stat_cascade)


# ---------------------------------------------------------------------------
# HypothesisPanel3D
# ---------------------------------------------------------------------------

class HypothesisPanel3D(VGroup):
    """Hypothesis test result panel.

    Parameters
    ----------
    h0 : str
        H₀ statement (plain text or LaTeX).
    h1 : str
        H₁ statement.
    test_name : str
        Test statistic name, e.g. ``"t"``, ``"z"``, ``"F"``, ``"chi^2"``.
    observed : float
        Observed test statistic value.
    critical : float
        Critical value at the given significance level.
    p_value : float
    alpha : float
        Significance level (default 0.05).
    effect_size : float | None
        Optional effect size (Cohen's d, eta-squared, etc.).
    effect_name : str
        Name for the effect size measure, e.g. ``"Cohen's d"``.
    pos : np.ndarray
    config : PanelConfig | None
    """

    def __init__(
        self,
        h0:          str,
        h1:          str,
        test_name:   str,
        observed:    float,
        critical:    float,
        p_value:     float,
        alpha:       float = 0.05,
        effect_size: float | None = None,
        effect_name: str   = "Effect size",
        pos:         np.ndarray = np.array([0.0, 0.0, 0.0]),
        config:      PanelConfig | None = None,
    ):
        super().__init__()
        self.cfg    = cfg = config or PanelConfig()
        self._reject = p_value < alpha

        reject_color = ManimColor("#C62828") if self._reject else ManimColor("#2E7D32")
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        w  = cfg.panel_width
        d  = cfg.panel_depth
        y  = py

        # ---- Hypotheses ----
        for label, stmt, color in [
            ("H\u2080:", h0, GRAY_C),
            ("H\u2081:", h1, _lt(ManimColor("#1565C0"), 0.25)),
        ]:
            lbl = Text(label, color=PANEL_TITLE_COL,
                       font_size=cfg.body_font_size, weight="BOLD")
            txt = Text(stmt, color=color, font_size=cfg.body_font_size)
            lbl.move_to(np.array([px + 0.22, y + 0.22, pz + d / 2]))
            txt.move_to(np.array([px + 0.22 + lbl.width + txt.width / 2 + 0.10,
                                   y + 0.22, pz + d / 2]))
            self.add(lbl, txt)
            y += 0.38

        # ---- Test statistic row ----
        y += 0.12
        div1 = _SectionDivider(px, px + w, y, pz + d / 2, cfg)
        self.add(div1)
        y += 0.22

        stars_str, stars_col = _sig_stars(p_value)
        stat_items = [
            (rf"{test_name}_{{obs}}", _format_v(observed, 3)),
            (rf"{test_name}_{{crit}}", _format_v(critical, 3)),
            (r"p\text{-value}", f"{p_value:.4f} {stars_str}"),
        ]
        if effect_size is not None:
            stat_items.append((effect_name, _format_v(effect_size, 3)))

        for tex, val in stat_items:
            lbl = MathTex(tex, color=PANEL_TITLE_COL,
                          font_size=cfg.body_font_size)
            eq  = Text("=", color=GRAY_C, font_size=cfg.body_font_size)
            val_obj = Text(val, color=PANEL_TEXT,
                           font_size=cfg.value_font_size, weight="BOLD")
            lbl.move_to(np.array([px + 0.30, y, pz + d / 2]))
            eq.move_to( np.array([px + 0.30 + lbl.width + 0.14, y, pz + d / 2]))
            val_obj.move_to(np.array([px + 0.30 + lbl.width + 0.32 + val_obj.width / 2,
                                       y, pz + d / 2]))
            self.add(lbl, eq, val_obj)
            y += 0.36

        # ---- Decision banner ----
        y += 0.18
        self._banner = _DecisionBanner(
            x0=px + 0.10, z0=pz,
            width=w - 0.20, depth=d,
            reject=self._reject,
            y0=y, cfg=cfg,
        )
        self.add(self._banner)
        y += cfg.banner_height + 0.60

        # Background
        bg = _PanelBackground(
            x0=px, z0=pz, width=w,
            total_height=y - py,
            cfg=cfg, accent_color=reject_color, y0=py,
        )
        self.add(bg)

    # ------------------------------------------------------------------

    def animate_build(
        self,
        run_time: float = 2.0,
    ) -> LaggedStart:
        """Reveal hypothesis statement + test stats before the decision."""
        # Exclude the banner from the initial build
        non_banner = VGroup(*[m for m in self.submobjects
                               if m is not self._banner])
        return LaggedStart(
            FadeIn(non_banner, run_time=run_time * 0.65),
            lag_ratio=0.0, run_time=run_time * 0.65,
        )

    def animate_reveal_decision(
        self,
        run_time: float = 1.2,
    ) -> Succession:
        """Flash the decision banner in with a colored border glow."""
        grow   = GrowFromPoint(
            self._banner,
            point=self._banner.get_center(),
            run_time=run_time * 0.60,
        )
        pulse1 = self._banner.animate(run_time=run_time * 0.20).scale(1.04)
        pulse2 = self._banner.animate(run_time=run_time * 0.20).scale(1 / 1.04)
        return Succession(grow, pulse1, pulse2)


# ---------------------------------------------------------------------------
# FormulaDerivationPanel3D
# ---------------------------------------------------------------------------

class FormulaDerivationPanel3D(VGroup):
    """Multi-step formula derivation with right-side annotation labels.

    Each step: MathTex equation + optional annotation ("← definition",
    "← algebra").  States: neutral, active, derived.

    Parameters
    ----------
    steps : list[dict]
        Each dict: ``{"tex": str, "annotation": str}``.
        ``annotation`` may be empty.
    pos : np.ndarray
    title : str
    config : PanelConfig | None
    """

    def __init__(
        self,
        steps:        list[dict],
        pos:          np.ndarray = np.array([0.0, 0.0, 0.0]),
        title:        str = "",
        accent_color: ManimColor = FORMULA_ACTIVE,
        config:       PanelConfig | None = None,
    ):
        super().__init__()
        self.cfg    = cfg = config or PanelConfig()
        self._steps = steps
        self._current = -1

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        w  = cfg.panel_width
        d  = cfg.panel_depth
        rh = cfg.step_row_height
        y  = py

        if title:
            t_obj = Text(title, color=WHITE,
                         font_size=cfg.header_font_size, weight="BOLD")
            t_obj.move_to(np.array([px + w / 2, y + 0.25, pz + d / 2]))
            self.add(t_obj)
            y += 0.55

        self._formula_objs:     list[MathTex] = []
        self._annotation_objs:  list[Text]    = []
        self._connectors:       list[Line3D]  = []

        for i, step in enumerate(steps):
            tex  = step.get("tex", "")
            ann  = step.get("annotation", "")

            # Formula (initially in neutral color)
            f_obj = MathTex(tex, color=GRAY_C,
                            font_size=cfg.formula_font_size)
            f_obj.move_to(np.array([px + w * 0.35, y, pz + d / 2]))
            self._formula_objs.append(f_obj)
            self.add(f_obj)

            # Annotation
            if ann:
                a_obj = Text(ann, color=GRAY_D,
                             font_size=cfg.annotation_font_size)
                a_obj.move_to(np.array([px + w * 0.78, y, pz + d / 2]))
                self._annotation_objs.append(a_obj)
                self.add(a_obj)
            else:
                self._annotation_objs.append(None)

            # Connector to next step
            if i < len(steps) - 1:
                conn = Line3D(
                    start=np.array([px + w * 0.35, y - rh * 0.25, pz + d / 2]),
                    end  =np.array([px + w * 0.35, y - rh * 0.75, pz + d / 2]),
                    color=GRAY_D, stroke_width=0.70,
                )
                conn.set_opacity(0.35)
                self._connectors.append(conn)
                self.add(conn)

            y -= rh

        # Background
        total_h = abs(y - py) + 0.30
        bg = _PanelBackground(
            x0=px, z0=pz, width=w,
            total_height=total_h,
            cfg=cfg, accent_color=accent_color, y0=py - total_h,
        )
        self.add(bg)

    # ------------------------------------------------------------------

    def animate_derive(
        self,
        step_index: int,
        run_time:   float = 0.80,
    ) -> Succession:
        """Highlight step ``step_index``, show its annotation, advance."""
        f_obj  = self._formula_objs[step_index]
        a_obj  = self._annotation_objs[step_index]

        # Dim previous active step
        anims_pre = []
        if self._current >= 0:
            prev = self._formula_objs[self._current]
            anims_pre.append(
                prev.animate(run_time=run_time * 0.25).set_color(FORMULA_DONE)
            )

        self._current = step_index
        highlight = f_obj.animate(run_time=run_time * 0.40).set_color(FORMULA_ACTIVE)
        show_ann  = FadeIn(a_obj, run_time=run_time * 0.35) \
                    if a_obj is not None else FadeIn(VGroup(), run_time=0.1)

        if anims_pre:
            return Succession(
                AnimationGroup(*anims_pre),
                AnimationGroup(highlight, show_ann),
            )
        return Succession(AnimationGroup(highlight, show_ann))

    def animate_full_derivation(
        self,
        pause_between: float = 0.15,
        run_time_each: float = 0.80,
    ) -> Succession:
        """Walk through all derivation steps automatically."""
        return Succession(
            *[self.animate_derive(i, run_time=run_time_each)
              for i in range(len(self._steps))]
        )

    def animate_reveal_all(
        self,
        lag_ratio: float = 0.12,
        run_time:  float = 2.0,
    ) -> LaggedStart:
        """Fade in all formula lines simultaneously (no state changes)."""
        return LaggedStart(
            *[FadeIn(f, run_time=run_time * 0.4)
              for f in self._formula_objs],
            lag_ratio=lag_ratio, run_time=run_time,
        )


# ---------------------------------------------------------------------------
# MatrixPanel3D
# ---------------------------------------------------------------------------

class MatrixPanel3D(VGroup):
    """Floating matrix display with row/column headers and color-coded cells.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of values to display.
    row_labels : list[str]
    col_labels : list[str]
    pos : np.ndarray
    palette_cold, palette_hot : ManimColor
        Color ends for the value-magnitude gradient.
    sparse : bool
        If True, only render cells with |value| > sparse_threshold.
    sparse_threshold : float
    title : str
    config : PanelConfig | None
    """

    def __init__(
        self,
        matrix:          np.ndarray,
        row_labels:      list[str] | None = None,
        col_labels:      list[str] | None = None,
        pos:             np.ndarray = np.array([0.0, 0.0, 0.0]),
        palette_cold:    ManimColor = MATRIX_COLD,
        palette_hot:     ManimColor = MATRIX_HOT,
        sparse:          bool = False,
        sparse_threshold: float = 1e-6,
        title:           str   = "",
        config:          PanelConfig | None = None,
    ):
        super().__init__()
        self.cfg    = cfg = config or PanelConfig()
        matrix      = np.asarray(matrix, dtype=float)
        n_rows, n_cols = matrix.shape

        row_labels = row_labels or [str(i) for i in range(n_rows)]
        col_labels = col_labels or [str(j) for j in range(n_cols)]

        cs  = cfg.matrix_cell_size
        cg  = cfg.matrix_cell_gap
        step = cs + cg
        d   = cfg.panel_depth * 0.6

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        y_base = py

        # Color mapping
        v_abs_max = np.abs(matrix).max()
        if v_abs_max < 1e-10:
            v_abs_max = 1.0

        def _cell_color(v: float) -> ManimColor:
            t = float(np.clip(abs(v) / v_abs_max, 0.0, 1.0))
            return _palette_t(t, palette_cold, palette_hot)

        # ---- Title ----
        if title:
            t_obj = Text(title, color=PANEL_TEXT,
                         font_size=cfg.header_font_size, weight="BOLD")
            t_obj.move_to(np.array([
                px + n_cols * step / 2,
                y_base + n_rows * step + 0.40,
                pz + d / 2,
            ]))
            self.add(t_obj)

        # ---- Column labels ----
        for j, cl in enumerate(col_labels):
            lbl = Text(cl, color=PANEL_TITLE_COL,
                       font_size=cfg.annotation_font_size)
            lbl.move_to(np.array([
                px + j * step + cs / 2,
                y_base + n_rows * step + 0.20,
                pz + d / 2,
            ]))
            self.add(lbl)

        # ---- Cells and row labels ----
        self._cells:    list[list[_ValueCell | None]] = []
        self._cell_map: dict[tuple[int, int], _ValueCell] = {}

        for i in range(n_rows):
            row_cells: list[_ValueCell | None] = []
            y_row = y_base + (n_rows - 1 - i) * step

            # Row label
            rl = Text(row_labels[i], color=PANEL_TITLE_COL,
                      font_size=cfg.annotation_font_size)
            rl.move_to(np.array([px - rl.width / 2 - 0.15, y_row + cs / 2, pz + d / 2]))
            self.add(rl)

            for j in range(n_cols):
                v = float(matrix[i, j])

                if sparse and abs(v) < sparse_threshold:
                    row_cells.append(None)
                    continue

                color  = _cell_color(v)
                height = cfg.matrix_max_height * abs(v) / v_abs_max
                cell   = _ValueCell(
                    x0=px + j * step,
                    z0=pz,
                    width=cs, depth=d, height=max(height, 0.004),
                    color=color,
                    text=_format_v(v, 2),
                    y0=y_row,
                    font_size=cfg.annotation_font_size,
                    opacity=cfg.cell_opacity,
                )
                row_cells.append(cell)
                self._cell_map[(i, j)] = cell
                self.add(cell)

            self._cells.append(row_cells)

        self._n_rows = n_rows
        self._n_cols = n_cols

    # ------------------------------------------------------------------

    def animate_reveal_by_row(
        self,
        lag_ratio: float = 0.18,
        run_time:  float = 2.0,
    ) -> LaggedStart:
        """Rows appear from top to bottom."""
        row_groups = [
            VGroup(*[c for c in row if c is not None])
            for row in self._cells
        ]
        return LaggedStart(
            *[FadeIn(rg, run_time=run_time * 0.45) for rg in row_groups],
            lag_ratio=lag_ratio, run_time=run_time,
        )

    def animate_reveal_by_col(
        self,
        lag_ratio: float = 0.18,
        run_time:  float = 2.0,
    ) -> LaggedStart:
        """Columns appear from left to right."""
        col_groups = []
        for j in range(self._n_cols):
            grp = VGroup(*[
                self._cells[i][j]
                for i in range(self._n_rows)
                if self._cells[i][j] is not None
            ])
            col_groups.append(grp)
        return LaggedStart(
            *[FadeIn(cg, run_time=run_time * 0.45) for cg in col_groups],
            lag_ratio=lag_ratio, run_time=run_time,
        )

    def animate_highlight_cell(
        self,
        row:          int,
        col:          int,
        scale_factor: float = 1.18,
        run_time:     float = 0.50,
    ) -> Succession:
        """Flash-scale one matrix cell."""
        cell = self._cell_map.get((row, col))
        if cell is None:
            return Succession(FadeIn(VGroup(), run_time=0.1))
        flash   = cell.animate(run_time=run_time / 2).scale(scale_factor)
        restore = cell.animate(run_time=run_time / 2).scale(1 / scale_factor)
        return Succession(flash, restore)