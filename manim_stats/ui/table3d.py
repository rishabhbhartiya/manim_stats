"""
manim_stats/ui/table3d.py
===========================
Production-quality floating 3D data table for Manim.

A full-featured interactive table — not a flat grid overlay, not a
simple matrix panel.  Every cell is a shaded 3D prism.  The table
supports typed columns, zebra striping, sort indicators, filter masks,
row/col/cell selection highlights, sparkline cells, badge cells,
a summary row, and pagination.

Architecture
------------

TableConfig  (dataclass)
    Every visual and behavioural parameter in one place.

Cell geometry primitives
    ``_TableCell``      – body cell: shaded prism + auto-contrast text
    ``_HeaderCell``     – header cell: taller accent prism + sort indicator
    ``_SparklineCell``  – cell containing a mini inline bar chart
    ``_BadgeCell``      – cell with a colored rounded-rectangle pill label

Engines
    ``TableSorter``   – computes sorted row permutations
    ``TableFilter``   – computes visible row masks from predicates
    ``TableSelection``– tracks selected rows/cols/cells; returns highlight groups

DataTable3D  (VGroup)
    Main class.  Builds and manages the full table.

    Structural features:
      - Header row (taller, accent color, sort arrows, filter funnels)
      - Body rows with zebra striping (even/odd row colors)
      - Frozen first column (row-label column stays left-anchored)
      - Auto-width columns: each column fits its widest content + padding
      - Column type system: "text", "numeric", "percent", "bool",
        "badge", "sparkline"  — each with sensible default alignment
        and formatting
      - Optional summary/footer row (sum, mean, min, max, count per col)
      - Pagination: show N rows at a time

    Animation suite:
      ``animate_build()``              – rows stagger in top-to-bottom
      ``animate_sort(col)``            – cells glide to new positions
      ``animate_filter(col, pred)``    – non-matching rows shrink/fade
      ``animate_clear_filter()``       – filtered rows restore
      ``animate_select_row(i)``        – sweep highlight across a row
      ``animate_select_col(j)``        – sweep highlight down a column
      ``animate_highlight_cell(i, j)`` – flash one cell
      ``animate_update_cell(i,j,val)`` – morph cell content to new value
      ``animate_next_page()``          – slide to next page of rows
      ``animate_prev_page()``          – slide to previous page
      ``animate_reveal_summary()``     – summary row slides up from below
      ``animate_reveal_header()``      – header row drops into place

    Factory classmethods:
      ``from_dataframe(df)``           – from dict-of-lists / pandas-like
      ``from_numpy(arr, rl, cl)``      – from numpy matrix
      ``stats_summary(named_arrays)``  – descriptive stats table
      ``comparison_results(…)``        – model comparison table
      ``frequency_table(cats, counts)``– freq + rel.freq + cumulative
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence
import warnings

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
    interpolate_color,
    ManimColor,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

CELL_HEADER    = ManimColor("#1E3040")   # header row background
CELL_EVEN      = ManimColor("#192631")   # even body row
CELL_ODD       = ManimColor("#131E26")   # odd body row
CELL_SELECTED  = ManimColor("#1A3A5C")   # selected cell/row/col
CELL_HOVER     = ManimColor("#223347")   # highlighted (hover) cell
CELL_FROZEN    = ManimColor("#1C2D3A")   # frozen column background
CELL_SUMMARY   = ManimColor("#0E1B24")   # summary/footer row
CELL_FILTER_DIM = ManimColor("#0A1218")  # filtered-out row (very dim)

HEADER_ACCENT  = ManimColor("#37474F")   # header top-face tint
SORT_UP_COLOR  = ManimColor("#FFD600")   # ascending sort indicator
SORT_DOWN_COLOR = ManimColor("#00BCD4")  # descending sort indicator
FILTER_ACTIVE  = ManimColor("#FF9800")   # filter funnel active color

BORDER_COLOR   = ManimColor("#263238")   # cell border
TEXT_PRIMARY   = ManimColor("#ECEFF1")   # main text
TEXT_SECONDARY = ManimColor("#78909C")   # secondary / dim text
TEXT_ACCENT    = ManimColor("#FFD600")   # bold / highlighted text

SPARK_COLD     = ManimColor("#1565C0")
SPARK_HOT      = ManimColor("#B71C1C")

BADGE_PALETTE: list[ManimColor] = [
    ManimColor("#1565C0"),
    ManimColor("#B71C1C"),
    ManimColor("#2E7D32"),
    ManimColor("#E65100"),
    ManimColor("#6A1B9A"),
    ManimColor("#00838F"),
    ManimColor("#880E4F"),
    ManimColor("#37474F"),
]

FACE_DARKEN_SIDE  = 0.36
FACE_DARKEN_RIGHT = 0.52

BOOL_TRUE_COLOR  = ManimColor("#00E676")
BOOL_FALSE_COLOR = ManimColor("#FF5252")


def _dk(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lt(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)

def _contrast(bg: ManimColor) -> ManimColor:
    r, g, b = [x / 255.0 for x in bg.to_rgb()]
    def _l(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * _l(r) + 0.7152 * _l(g) + 0.0722 * _l(b)
    return ManimColor("#F8F8F8") if lum < 0.45 else ManimColor("#1A1A1A")

def _fmt(v: float, decimals: int = 2, col_type: str = "numeric") -> str:
    if col_type == "percent":
        return f"{v * 100:.{max(decimals - 2, 0)}f}%"
    if col_type == "bool":
        return "✓" if bool(v) else "✗"
    if abs(v) < 0.001 and v != 0.0 or abs(v) > 99999:
        return f"{v:.{decimals}e}"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# TableConfig
# ---------------------------------------------------------------------------

@dataclass
class TableConfig:
    """All visual and behavioural parameters for DataTable3D.

    Geometry
    --------
    cell_depth : float
        Z depth of all cell prisms.
    cell_height : float
        Y height of body cell top faces.
    header_height : float
        Y height of header cell top faces (taller than body).
    summary_height : float
        Y height of the summary/footer row.
    min_col_width : float
        Minimum column width in Manim units.
    max_col_width : float
        Maximum column width before text truncation.
    col_padding : float
        Horizontal padding inside each cell.
    row_spacing : float
        Additional vertical gap between rows (0 = flush).
    frozen_col_z_boost : float
        How much higher the frozen column sits (subtle 3D separation).

    Typography
    ----------
    header_font_size : int
    body_font_size : int
    summary_font_size : int
    badge_font_size : int
    sparkline_font_size : int
    max_text_len : int
        Truncate cell text to this many characters.

    Column types
    ------------
    numeric_decimals : int
        Default decimal places for "numeric" columns.
    percent_decimals : int
        Decimal places for "percent" columns.
    bool_true_text : str
        Text for True values in "bool" columns.
    bool_false_text : str

    Sparkline
    ---------
    sparkline_bar_width_ratio : float
        Width of each mini bar relative to the available cell width.
    sparkline_max_height : float
        Max height of sparkline bars (in cell height units).

    Pagination
    ----------
    rows_per_page : int
        How many body rows to show at a time (0 = show all).
    page_slide_direction : str
        ``"up"`` or ``"right"`` for page transition direction.

    Summary row
    -----------
    show_summary : bool
        Render the summary footer row.
    summary_functions : dict[str, str]
        Maps column name → aggregate: ``"sum"``, ``"mean"``,
        ``"min"``, ``"max"``, ``"count"``, ``"std"``, ``""`` (blank).

    Visual style
    ------------
    zebra_striping : bool
        Alternate even/odd row colors.
    show_grid_lines : bool
        Render thin grid lines at cell boundaries.
    grid_line_opacity : float
    cell_opacity : float
    header_opacity : float
    sort_arrow_size : float
    filter_funnel_size : float
    """

    # ---- geometry ----
    cell_depth:            float = 0.16
    cell_height:           float = 0.06
    header_height:         float = 0.10
    summary_height:        float = 0.08
    min_col_width:         float = 0.70
    max_col_width:         float = 3.50
    col_padding:           float = 0.18
    row_spacing:           float = 0.00
    frozen_col_z_boost:    float = 0.02

    # ---- typography ----
    header_font_size:      int   = 21
    body_font_size:        int   = 18
    summary_font_size:     int   = 17
    badge_font_size:       int   = 16
    sparkline_font_size:   int   = 13
    max_text_len:          int   = 18

    # ---- column types ----
    numeric_decimals:      int   = 3
    percent_decimals:      int   = 1
    bool_true_text:        str   = "✓"
    bool_false_text:       str   = "✗"

    # ---- sparkline ----
    sparkline_bar_width_ratio: float = 0.75
    sparkline_max_height:      float = 0.80

    # ---- pagination ----
    rows_per_page:         int   = 0       # 0 = show all
    page_slide_direction:  str   = "up"

    # ---- summary ----
    show_summary:          bool  = False
    summary_functions:     dict  = field(default_factory=dict)

    # ---- visual style ----
    zebra_striping:        bool  = True
    show_grid_lines:       bool  = True
    grid_line_opacity:     float = 0.22
    cell_opacity:          float = 0.90
    header_opacity:        float = 0.95
    sort_arrow_size:       float = 0.10
    filter_funnel_size:    float = 0.09


# ---------------------------------------------------------------------------
# Column specification
# ---------------------------------------------------------------------------

@dataclass
class TableColumn:
    """Specification for one column.

    Parameters
    ----------
    name : str
        Column header label.
    col_type : str
        One of ``"text"``, ``"numeric"``, ``"percent"``, ``"bool"``,
        ``"badge"``, ``"sparkline"``.
    width : float | None
        Explicit column width.  Auto-sized from content if None.
    align : str
        ``"left"``, ``"center"``, ``"right"``.  Defaults by type.
    decimals : int | None
        Override numeric decimal places.
    badge_palette : list[ManimColor] | None
        Color palette for badge cells.
    color : ManimColor | None
        Optional accent color for this column's header.
    """
    name:          str
    col_type:      str = "text"
    width:         float | None = None
    align:         str | None   = None
    decimals:      int | None   = None
    badge_palette: list[ManimColor] | None = None
    color:         ManimColor | None       = None

    def effective_align(self) -> str:
        if self.align:
            return self.align
        return {
            "text":      "left",
            "numeric":   "right",
            "percent":   "right",
            "bool":      "center",
            "badge":     "center",
            "sparkline": "center",
        }.get(self.col_type, "left")


# ---------------------------------------------------------------------------
# _TableCell  —  standard body cell
# ---------------------------------------------------------------------------

class _TableCell(VGroup):
    """One body cell: shaded prism + formatted text content.

    Parameters
    ----------
    x0, z0 : float
        Bottom-left world coordinates of the cell footprint.
    width, depth, height : float
        Cell dimensions.
    bg_color : ManimColor
        Fill color for the top face.
    text : str
        Content string (already formatted).
    align : str
    y0 : float
        Base Y level.
    cfg : TableConfig
    bold : bool
    text_color : ManimColor | None
        Override auto-contrast text color.
    """

    def __init__(
        self,
        x0:     float,
        z0:     float,
        width:  float,
        depth:  float,
        height: float,
        bg_color: ManimColor,
        text:   str,
        align:  str = "center",
        y0:     float = 0.0,
        cfg:    TableConfig | None = None,
        bold:   bool = False,
        text_color: ManimColor | None = None,
        font_size:  int | None = None,
    ):
        super().__init__()
        cfg  = cfg or TableConfig()
        h    = max(height, 0.003)
        tc   = text_color or _contrast(bg_color)

        fc = _dk(bg_color, FACE_DARKEN_SIDE)
        rc = _dk(bg_color, FACE_DARKEN_RIGHT)
        op = cfg.cell_opacity

        AFL = np.array([x0,         y0,     z0       ])
        AFR = np.array([x0 + width, y0,     z0       ])
        ABL = np.array([x0,         y0,     z0 + depth])
        ABR = np.array([x0 + width, y0,     z0 + depth])
        TFL = np.array([x0,         y0 + h, z0       ])
        TFR = np.array([x0 + width, y0 + h, z0       ])
        TBL = np.array([x0,         y0 + h, z0 + depth])
        TBR = np.array([x0 + width, y0 + h, z0 + depth])

        def _face(pts, col, face_op=None):
            p = Polygon(*pts, color=col)
            p.set_fill(color=col, opacity=face_op or op)
            p.set_stroke(color=BORDER_COLOR, width=0.40, opacity=cfg.grid_line_opacity)
            return p

        self.top_face   = _face([TFL, TFR, TBR, TBL], bg_color)
        self.front_face = _face([AFL, AFR, TFR, TFL], fc)
        self.right_face = _face([AFR, ABR, TBR, TFR], rc)
        self.add(self.front_face, self.right_face, self.top_face)

        # Text placement
        fs   = font_size or cfg.body_font_size
        txt  = text[:cfg.max_text_len] + ("…" if len(text) > cfg.max_text_len else "")
        lbl  = Text(txt, color=tc, font_size=fs,
                    weight="BOLD" if bold else "NORMAL")

        cx   = x0 + width / 2
        if align == "left":
            cx = x0 + cfg.col_padding + lbl.width / 2
        elif align == "right":
            cx = x0 + width - cfg.col_padding - lbl.width / 2

        lbl.move_to(np.array([cx, y0 + h + 0.16, z0 + depth / 2]))
        self.add(lbl)
        self.label = lbl

        # Store geometry for animation use
        self._x0     = x0
        self._y0     = y0
        self._z0     = z0
        self._width  = width
        self._depth  = depth
        self._height = h
        self._bg_color = bg_color

    @property
    def top_center(self) -> np.ndarray:
        return np.array([
            self._x0 + self._width / 2,
            self._y0 + self._height,
            self._z0 + self._depth / 2,
        ])

    @property
    def floor_center(self) -> np.ndarray:
        c = self.top_center.copy()
        c[1] = self._y0
        return c


# ---------------------------------------------------------------------------
# _HeaderCell  —  header row cell with sort indicator
# ---------------------------------------------------------------------------

class _HeaderCell(VGroup):
    """Taller accent-colored header cell with sort arrow and filter indicator."""

    def __init__(
        self,
        x0:       float,
        z0:       float,
        width:    float,
        depth:    float,
        col:      TableColumn,
        y0:       float = 0.0,
        sort_dir: str   = "",      # "" | "asc" | "desc"
        filtered: bool  = False,
        cfg:      TableConfig | None = None,
    ):
        super().__init__()
        cfg   = cfg or TableConfig()
        h     = cfg.header_height
        color = col.color or CELL_HEADER
        op    = cfg.header_opacity

        fc = _dk(color, FACE_DARKEN_SIDE)
        rc = _dk(color, FACE_DARKEN_RIGHT)

        AFL = np.array([x0,         y0,     z0       ])
        AFR = np.array([x0 + width, y0,     z0       ])
        ABL = np.array([x0,         y0,     z0 + depth])
        ABR = np.array([x0 + width, y0,     z0 + depth])
        TFL = np.array([x0,         y0 + h, z0       ])
        TFR = np.array([x0 + width, y0 + h, z0       ])
        TBL = np.array([x0,         y0 + h, z0 + depth])
        TBR = np.array([x0 + width, y0 + h, z0 + depth])

        def _face(pts, col_f, face_op=None):
            p = Polygon(*pts, color=col_f)
            p.set_fill(color=col_f, opacity=face_op or op)
            p.set_stroke(color=BORDER_COLOR, width=0.50, opacity=0.35)
            return p

        # Use HEADER_ACCENT for the top face to distinguish from body
        top_col = _lt(color, 0.08)
        self.add(_face([AFL, AFR, TFR, TFL], fc))
        self.add(_face([AFR, ABR, TBR, TFR], rc))
        self.top_face = _face([TFL, TFR, TBR, TBL], top_col)
        self.add(self.top_face)

        # Column name
        name_txt = col.name[:cfg.max_text_len]
        lbl = Text(name_txt, color=TEXT_PRIMARY,
                   font_size=cfg.header_font_size, weight="BOLD")
        cx  = x0 + width / 2
        lbl.move_to(np.array([cx, y0 + h + 0.18, z0 + depth / 2]))
        self.add(lbl)
        self.name_label = lbl

        # Sort arrow
        self.sort_arrow: VGroup | None = None
        if sort_dir:
            arrow_color  = SORT_UP_COLOR if sort_dir == "asc" else SORT_DOWN_COLOR
            arrow_char   = "▲" if sort_dir == "asc" else "▼"
            arrow_obj    = Text(arrow_char, color=arrow_color,
                                font_size=int(cfg.header_font_size * 0.7))
            arrow_obj.move_to(np.array([
                x0 + width - cfg.col_padding - 0.12,
                y0 + h + 0.18,
                z0 + depth / 2,
            ]))
            self.add(arrow_obj)
            self.sort_arrow = arrow_obj

        # Filter funnel indicator
        if filtered:
            funnel = Text("⊽", color=FILTER_ACTIVE,
                          font_size=int(cfg.header_font_size * 0.65))
            funnel.move_to(np.array([
                x0 + cfg.col_padding + 0.12,
                y0 + h + 0.18,
                z0 + depth / 2,
            ]))
            self.add(funnel)

        self._x0  = x0
        self._z0  = z0
        self._width = width
        self._depth = depth
        self._h   = h
        self._y0  = y0


# ---------------------------------------------------------------------------
# _SparklineCell  —  mini inline bar chart inside a cell
# ---------------------------------------------------------------------------

class _SparklineCell(VGroup):
    """A cell containing a tiny bar chart showing a trend series."""

    def __init__(
        self,
        x0:     float,
        z0:     float,
        width:  float,
        depth:  float,
        values: list[float],
        y0:     float = 0.0,
        cfg:    TableConfig | None = None,
        bg_color: ManimColor = CELL_EVEN,
    ):
        super().__init__()
        cfg = cfg or TableConfig()
        h   = cfg.cell_height

        # Background prism (flat — sparkline is the visual)
        bg = Polygon(
            np.array([x0,         y0 + h, z0       ]),
            np.array([x0 + width, y0 + h, z0       ]),
            np.array([x0 + width, y0 + h, z0 + depth]),
            np.array([x0,         y0 + h, z0 + depth]),
            color=bg_color,
        )
        bg.set_fill(color=bg_color, opacity=cfg.cell_opacity)
        bg.set_stroke(color=BORDER_COLOR, width=0.35, opacity=0.25)
        self.add(bg)

        # Front face
        fc = _dk(bg_color, FACE_DARKEN_SIDE)
        front = Polygon(
            np.array([x0,         y0,     z0]),
            np.array([x0 + width, y0,     z0]),
            np.array([x0 + width, y0 + h, z0]),
            np.array([x0,         y0 + h, z0]),
            color=fc,
        )
        front.set_fill(color=fc, opacity=cfg.cell_opacity * 0.85)
        front.set_stroke(color=BORDER_COLOR, width=0.35, opacity=0.20)
        self.add(front)

        # Mini bars
        if not values:
            return
        n      = len(values)
        v_max  = max(abs(v) for v in values) or 1.0
        avail_w = width - 2 * cfg.col_padding
        bar_w  = (avail_w / n) * cfg.sparkline_bar_width_ratio
        gap    = (avail_w / n) * (1 - cfg.sparkline_bar_width_ratio)
        max_bh = h * cfg.sparkline_max_height

        for k, v in enumerate(values):
            t      = abs(v) / v_max
            color  = interpolate_color(SPARK_COLD, SPARK_HOT, t)
            bar_h  = max(max_bh * t, 0.006)
            bx0    = x0 + cfg.col_padding + k * (bar_w + gap)
            bz     = z0 + depth / 2 - bar_w / 2

            bar_top = Polygon(
                np.array([bx0,          y0 + h,        bz        ]),
                np.array([bx0 + bar_w,  y0 + h,        bz        ]),
                np.array([bx0 + bar_w,  y0 + h,        bz + bar_w]),
                np.array([bx0,          y0 + h,        bz + bar_w]),
                color=color,
            )
            bar_top.set_fill(color=color, opacity=0.95)
            bar_top.set_stroke(width=0)
            self.add(bar_top)

            # Front face of mini bar
            bar_front = Polygon(
                np.array([bx0,         y0 + h - bar_h, bz]),
                np.array([bx0 + bar_w, y0 + h - bar_h, bz]),
                np.array([bx0 + bar_w, y0 + h,         bz]),
                np.array([bx0,         y0 + h,         bz]),
                color=_dk(color, 0.30),
            )
            bar_front.set_fill(color=_dk(color, 0.30), opacity=0.90)
            bar_front.set_stroke(width=0)
            self.add(bar_front)

        self._x0 = x0; self._y0 = y0; self._z0 = z0
        self._w  = width; self._d = depth; self._h = h


# ---------------------------------------------------------------------------
# _BadgeCell  —  colored pill label cell
# ---------------------------------------------------------------------------

class _BadgeCell(VGroup):
    """A cell with a colored rounded-rectangle badge for categorical values."""

    # Badge category → color index map (populated on first use)
    _category_colors: dict[str, ManimColor] = {}
    _color_counter:   int = 0

    def __init__(
        self,
        x0:       float,
        z0:       float,
        width:    float,
        depth:    float,
        value:    str,
        y0:       float = 0.0,
        palette:  list[ManimColor] | None = None,
        cfg:      TableConfig | None = None,
        bg_color: ManimColor = CELL_EVEN,
    ):
        super().__init__()
        cfg   = cfg or TableConfig()
        h     = cfg.cell_height
        pal   = palette or BADGE_PALETTE

        # Cell background
        bg = Polygon(
            np.array([x0,         y0 + h, z0       ]),
            np.array([x0 + width, y0 + h, z0       ]),
            np.array([x0 + width, y0 + h, z0 + depth]),
            np.array([x0,         y0 + h, z0 + depth]),
            color=bg_color,
        )
        bg.set_fill(color=bg_color, opacity=cfg.cell_opacity)
        bg.set_stroke(color=BORDER_COLOR, width=0.35, opacity=0.22)
        self.add(bg)

        # Front face of background
        fc = _dk(bg_color, FACE_DARKEN_SIDE)
        front = Polygon(
            np.array([x0,         y0,     z0]),
            np.array([x0 + width, y0,     z0]),
            np.array([x0 + width, y0 + h, z0]),
            np.array([x0,         y0 + h, z0]),
            color=fc,
        )
        front.set_fill(color=fc, opacity=cfg.cell_opacity * 0.80)
        front.set_stroke(color=BORDER_COLOR, width=0.35, opacity=0.18)
        self.add(front)

        # Assign a consistent color to this category
        val_key = str(value)
        if val_key not in _BadgeCell._category_colors:
            idx  = _BadgeCell._color_counter % len(pal)
            _BadgeCell._category_colors[val_key] = pal[idx]
            _BadgeCell._color_counter += 1
        badge_color = _BadgeCell._category_colors[val_key]

        # Badge pill (flat polygon on top face)
        lbl_obj = Text(val_key[:12], color=_contrast(badge_color),
                       font_size=cfg.badge_font_size)
        pill_w  = lbl_obj.width + 0.22
        pill_h  = 0.18
        pill_z  = z0 + depth / 2
        pill_x0 = x0 + width / 2 - pill_w / 2
        y_pill  = y0 + h + 0.01

        pill = Polygon(
            np.array([pill_x0,          y_pill,           pill_z - 0.06]),
            np.array([pill_x0 + pill_w, y_pill,           pill_z - 0.06]),
            np.array([pill_x0 + pill_w, y_pill,           pill_z + 0.06]),
            np.array([pill_x0,          y_pill,           pill_z + 0.06]),
            color=badge_color,
        )
        pill.set_fill(color=badge_color, opacity=0.95)
        pill.set_stroke(color=_dk(badge_color, 0.40), width=0.55)
        self.add(pill)

        lbl_obj.move_to(np.array([x0 + width / 2, y_pill + 0.15, pill_z]))
        self.add(lbl_obj)

        self._x0 = x0; self._y0 = y0; self._z0 = z0
        self._w  = width; self._d = depth; self._h = h


# ---------------------------------------------------------------------------
# TableSorter
# ---------------------------------------------------------------------------

class TableSorter:
    """Compute sorted row orderings for a table column.

    Parameters
    ----------
    rows : list[list]
        The full data rows (excluding header).
    """

    def __init__(self, rows: list[list]):
        self._rows = rows

    def sort_by(
        self,
        col_index: int,
        ascending: bool = True,
    ) -> list[int]:
        """Return a list of row indices in sorted order.

        Non-numeric values sort lexicographically; numeric values sort
        by value.  ``None`` values are placed at the end.
        """
        n = len(self._rows)

        def _key(i: int):
            v = self._rows[i][col_index] if col_index < len(self._rows[i]) else None
            if v is None:
                return (1, 0, "")
            if isinstance(v, (int, float)):
                return (0, float(v), "")
            try:
                return (0, float(v), "")
            except (TypeError, ValueError):
                return (0, 0, str(v))

        indices = sorted(range(n), key=_key, reverse=not ascending)
        return indices


# ---------------------------------------------------------------------------
# TableFilter
# ---------------------------------------------------------------------------

class TableFilter:
    """Compute visible row masks from predicates.

    Parameters
    ----------
    rows : list[list]
        The full data rows (excluding header).
    """

    def __init__(self, rows: list[list]):
        self._rows  = rows
        self._masks: list[np.ndarray] = []   # stack of filter masks

    @property
    def visible_mask(self) -> np.ndarray:
        """Boolean mask of currently visible rows (all filters combined)."""
        n = len(self._rows)
        if not self._masks:
            return np.ones(n, dtype=bool)
        combined = self._masks[0].copy()
        for m in self._masks[1:]:
            combined &= m
        return combined

    def filter_by(
        self,
        col_index: int,
        predicate: Callable[[object], bool],
    ) -> np.ndarray:
        """Apply a filter and return the new visible mask.

        Parameters
        ----------
        col_index : int
        predicate : callable
            A function that accepts a cell value and returns True to
            keep the row.
        """
        n    = len(self._rows)
        mask = np.zeros(n, dtype=bool)
        for i, row in enumerate(self._rows):
            v = row[col_index] if col_index < len(row) else None
            try:
                mask[i] = bool(predicate(v))
            except Exception:
                mask[i] = False
        self._masks.append(mask)
        return self.visible_mask

    def clear(self) -> None:
        """Remove all active filters."""
        self._masks.clear()

    def pop_last(self) -> None:
        """Remove the most recently applied filter."""
        if self._masks:
            self._masks.pop()


# ---------------------------------------------------------------------------
# TableSelection
# ---------------------------------------------------------------------------

class TableSelection:
    """Track selected rows, columns, and cells.

    Maintains selection state without modifying Manim objects directly;
    returns highlight VGroup objects for use in animations.
    """

    def __init__(self):
        self._selected_rows:  set[int]            = set()
        self._selected_cols:  set[int]             = set()
        self._selected_cells: set[tuple[int, int]] = set()

    def select_row(self, i: int) -> None:
        self._selected_rows.add(i)

    def deselect_row(self, i: int) -> None:
        self._selected_rows.discard(i)

    def select_col(self, j: int) -> None:
        self._selected_cols.add(j)

    def select_cell(self, i: int, j: int) -> None:
        self._selected_cells.add((i, j))

    def clear(self) -> None:
        self._selected_rows.clear()
        self._selected_cols.clear()
        self._selected_cells.clear()

    def is_selected(self, row: int, col: int) -> bool:
        return (
            row in self._selected_rows
            or col in self._selected_cols
            or (row, col) in self._selected_cells
        )


# ---------------------------------------------------------------------------
# Main DataTable3D
# ---------------------------------------------------------------------------

class DataTable3D(VGroup):
    """A full-featured floating 3D data table for Manim.

    Each cell is a shaded 3D prism.  Supports typed columns, zebra
    striping, sorting, filtering, selection, sparklines, badges,
    a summary row, and pagination.

    Basic usage
    -----------
    >>> from manim import *
    >>> from manim_stats.ui.table3d import DataTable3D, TableColumn, TableConfig
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         tbl = DataTable3D.from_dataframe(
    ...             {
    ...                 "Model":    ["LR", "RF", "SVM", "XGB"],
    ...                 "Accuracy": [0.82, 0.91, 0.88, 0.94],
    ...                 "F1":       [0.80, 0.90, 0.87, 0.93],
    ...             },
    ...             col_types=["text", "percent", "percent"],
    ...         )
    ...         self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
    ...         self.play(tbl.animate_build())

    Parameters
    ----------
    columns : list[TableColumn]
        Column specifications.
    rows : list[list]
        Data rows.  Each row must have the same number of elements as
        ``columns``.
    pos : np.ndarray
        World position of the top-left corner of the table.
    config : TableConfig | None
    """

    def __init__(
        self,
        columns:  list[TableColumn],
        rows:     list[list],
        pos:      np.ndarray = np.array([0.0, 0.0, 0.0]),
        config:   TableConfig | None = None,
    ):
        super().__init__()
        self.cfg      = cfg  = config or TableConfig()
        self._columns = columns
        self._rows    = rows
        self._pos     = pos.copy()

        self._sorter   = TableSorter(rows)
        self._filter   = TableFilter(rows)
        self._selection = TableSelection()

        self._sort_col: int  = -1
        self._sort_asc: bool = True

        # Row visibility (from filter)
        self._visible  = np.ones(len(rows), dtype=bool)

        # Pagination
        self._page         = 0
        self._rows_per_page = cfg.rows_per_page or len(rows)

        # Computed column widths (filled in _compute_layout)
        self._col_widths: list[float] = []
        self._col_x:      list[float] = []

        # Cell registry  [row_i][col_j]
        self._body_cells:   list[list[_TableCell | _SparklineCell | _BadgeCell]] = []
        self._header_cells: list[_HeaderCell] = []
        self._summary_cells: list[_TableCell] = []

        self._compute_layout()
        self._build()

    # ------------------------------------------------------------------
    # Layout computation
    # ------------------------------------------------------------------

    def _compute_layout(self) -> None:
        """Compute column widths from content, then assign X positions."""
        cfg  = self.cfg
        n_c  = len(self._columns)

        widths: list[float] = []
        for j, col in enumerate(self._columns):
            if col.width is not None:
                widths.append(col.width)
                continue

            # Auto-size: measure header + all cell values
            max_len = len(col.name)
            for row in self._rows:
                v = row[j] if j < len(row) else ""
                if col.col_type in ("sparkline",):
                    max_len = max(max_len, 8)
                elif col.col_type == "badge":
                    max_len = max(max_len, len(str(v)[:12]))
                else:
                    max_len = max(max_len, len(str(v)[:cfg.max_text_len]))

            # Approximate: each character ≈ 0.11 Manim units at font 18
            estimated = max_len * 0.105 + cfg.col_padding * 2
            widths.append(float(np.clip(estimated, cfg.min_col_width, cfg.max_col_width)))

        self._col_widths = widths
        px = float(self._pos[0])
        self._col_x = []
        cursor = px
        for w in widths:
            self._col_x.append(cursor)
            cursor += w

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg  = self.cfg
        d    = cfg.cell_depth
        rh   = cfg.cell_height + cfg.row_spacing
        px, py, pz = float(self._pos[0]), float(self._pos[1]), float(self._pos[2])
        n_c  = len(self._columns)

        # ---- How many rows to show on first page ----
        page_rows = self._get_page_rows()

        # ---- Header row (sits above body rows) ----
        n_visible  = len(page_rows)
        y_header   = py + n_visible * rh

        self._header_cells = []
        for j, col in enumerate(self._columns):
            sort_dir = ""
            if j == self._sort_col:
                sort_dir = "asc" if self._sort_asc else "desc"
            hcell = _HeaderCell(
                x0=self._col_x[j],
                z0=pz,
                width=self._col_widths[j],
                depth=d,
                col=col,
                y0=y_header,
                sort_dir=sort_dir,
                cfg=cfg,
            )
            self._header_cells.append(hcell)
            self.add(hcell)

        # ---- Body rows ----
        self._body_cells = []
        for display_row_idx, data_row_idx in enumerate(page_rows):
            row_data = self._rows[data_row_idx]
            y_row    = py + (n_visible - 1 - display_row_idx) * rh
            is_even  = display_row_idx % 2 == 0
            base_bg  = CELL_EVEN if is_even else CELL_ODD

            row_cells: list = []
            for j, col in enumerate(self._columns):
                v    = row_data[j] if j < len(row_data) else ""
                bg   = CELL_FROZEN if j == 0 else base_bg
                w    = self._col_widths[j]
                xj   = self._col_x[j]

                # Dispatch by column type
                if col.col_type == "sparkline" and isinstance(v, (list, np.ndarray)):
                    cell = _SparklineCell(
                        x0=xj, z0=pz, width=w, depth=d,
                        values=list(v), y0=y_row, cfg=cfg, bg_color=bg,
                    )
                elif col.col_type == "badge":
                    cell = _BadgeCell(
                        x0=xj, z0=pz, width=w, depth=d,
                        value=str(v), y0=y_row,
                        palette=col.badge_palette,
                        cfg=cfg, bg_color=bg,
                    )
                else:
                    # Format value
                    dec = col.decimals if col.decimals is not None \
                          else cfg.numeric_decimals
                    if col.col_type == "bool":
                        txt   = cfg.bool_true_text if v else cfg.bool_false_text
                        tc    = BOOL_TRUE_COLOR if v else BOOL_FALSE_COLOR
                    elif col.col_type == "percent" and isinstance(v, (int, float)):
                        txt   = _fmt(float(v), dec, "percent")
                        tc    = None
                    elif col.col_type == "numeric" and isinstance(v, (int, float)):
                        txt   = _fmt(float(v), dec, "numeric")
                        tc    = None
                    else:
                        txt   = str(v)
                        tc    = None

                    # Selection highlight
                    if self._selection.is_selected(display_row_idx, j):
                        bg = CELL_SELECTED

                    cell = _TableCell(
                        x0=xj, z0=pz, width=w, depth=d,
                        height=cfg.cell_height,
                        bg_color=bg, text=txt,
                        align=col.effective_align(),
                        y0=y_row, cfg=cfg,
                        bold=(j == 0),
                        text_color=tc,
                    )

                row_cells.append(cell)
                self.add(cell)

            self._body_cells.append(row_cells)

        # ---- Grid lines ----
        if cfg.show_grid_lines:
            self._build_grid_lines(py, y_header, n_visible, d, pz)

        # ---- Summary row ----
        self._summary_cells = []
        if cfg.show_summary:
            self._build_summary_row(py - cfg.cell_height - cfg.row_spacing, d, pz)

    def _get_page_rows(self) -> list[int]:
        """Return visible data row indices for the current page."""
        visible_indices = [i for i in range(len(self._rows))
                           if self._visible[i]]
        rpp   = self._rows_per_page
        start = self._page * rpp
        return visible_indices[start: start + rpp]

    def _build_grid_lines(
        self,
        y_bottom: float,
        y_top:    float,
        n_rows:   int,
        depth:    float,
        pz:       float,
    ) -> None:
        """Thin ruling lines between rows and columns."""
        cfg = self.cfg
        op  = cfg.grid_line_opacity

        # Horizontal lines (row boundaries)
        rh = cfg.cell_height + cfg.row_spacing
        x_left  = self._col_x[0]
        x_right = self._col_x[-1] + self._col_widths[-1]

        for k in range(n_rows + 1):
            yv = y_bottom + k * rh
            self.add(Line3D(
                start=np.array([x_left,  yv, pz         ]),
                end  =np.array([x_right, yv, pz         ]),
                color=BORDER_COLOR, stroke_width=0.35,
            ).set_opacity(op))

        # Vertical lines (column boundaries)
        for j in range(len(self._columns) + 1):
            xv = self._col_x[j] if j < len(self._col_x) \
                 else self._col_x[-1] + self._col_widths[-1]
            self.add(Line3D(
                start=np.array([xv, y_bottom, pz]),
                end  =np.array([xv, y_top + cfg.header_height, pz]),
                color=BORDER_COLOR, stroke_width=0.35,
            ).set_opacity(op))

    def _build_summary_row(
        self,
        y0:    float,
        depth: float,
        pz:    float,
    ) -> None:
        """Append a footer row showing column aggregates."""
        cfg = self.cfg
        for j, col in enumerate(self._columns):
            func  = cfg.summary_functions.get(col.name, "")
            xj    = self._col_x[j]
            w     = self._col_widths[j]
            txt   = self._compute_aggregate(j, func)
            cell  = _TableCell(
                x0=xj, z0=pz, width=w, depth=depth,
                height=cfg.summary_height,
                bg_color=CELL_SUMMARY, text=txt,
                align=col.effective_align(),
                y0=y0, cfg=cfg, bold=True,
                font_size=cfg.summary_font_size,
            )
            self._summary_cells.append(cell)
            self.add(cell)

    def _compute_aggregate(self, col_index: int, func: str) -> str:
        """Compute a single aggregate string for a column."""
        if not func:
            return ""
        col  = self._columns[col_index]
        vals: list[float] = []
        for row in self._rows:
            v = row[col_index] if col_index < len(row) else None
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return ""
        dec  = col.decimals if col.decimals is not None else self.cfg.numeric_decimals
        agg_map = {
            "sum":   sum(vals),
            "mean":  float(np.mean(vals)),
            "min":   min(vals),
            "max":   max(vals),
            "count": float(len(vals)),
            "std":   float(np.std(vals)),
        }
        v = agg_map.get(func)
        if v is None:
            return ""
        return _fmt(v, dec, col.col_type)

    # ------------------------------------------------------------------
    # Animation API
    # ------------------------------------------------------------------

    def animate_build(
        self,
        lag_ratio: float = 0.04,
        run_time:  float = 2.5,
    ) -> Succession:
        """Header drops in, then body rows stagger top-to-bottom.

        ::

            self.play(table.animate_build())
        """
        header_grp = VGroup(*self._header_cells)
        header_anim = GrowFromPoint(
            header_grp,
            point=header_grp.get_top(),
            run_time=run_time * 0.30,
        )
        row_anims = []
        for row_cells in self._body_cells:
            row_grp = VGroup(*[c for c in row_cells if c is not None])
            row_anims.append(
                FadeIn(row_grp, shift=DOWN * 0.06, run_time=run_time * 0.25)
            )
        body_anim = LaggedStart(*row_anims, lag_ratio=lag_ratio,
                                run_time=run_time * 0.70)
        return Succession(header_anim, body_anim)

    def animate_reveal_header(self, run_time: float = 0.60) -> GrowFromPoint:
        """Header row drops in from above."""
        grp = VGroup(*self._header_cells)
        return GrowFromPoint(grp, point=grp.get_top(), run_time=run_time)

    def animate_sort(
        self,
        col_index: int,
        ascending: bool = True,
        run_time:  float = 1.5,
    ) -> AnimationGroup:
        """Sort the table by a column and animate cells gliding to new rows.

        Each body cell moves to its new Y position.  The header's sort
        arrow updates simultaneously.

        Parameters
        ----------
        col_index : int
            0-based index of the column to sort by.
        ascending : bool
        """
        new_order = self._sorter.sort_by(col_index, ascending)
        self._sort_col = col_index
        self._sort_asc = ascending

        cfg  = self.cfg
        rh   = cfg.cell_height + cfg.row_spacing
        py   = float(self._pos[1])
        n_v  = len(self._body_cells)

        anims = []
        for new_display_idx, old_data_idx in enumerate(new_order[:n_v]):
            # Find which display row currently holds this data index
            page_rows = self._get_page_rows()
            if old_data_idx not in page_rows:
                continue
            old_display_idx = page_rows.index(old_data_idx)
            if old_display_idx >= len(self._body_cells):
                continue

            old_row_cells = self._body_cells[old_display_idx]
            new_y = py + (n_v - 1 - new_display_idx) * rh

            for cell in old_row_cells:
                if cell is None:
                    continue
                delta_y = new_y - cell._y0
                anims.append(
                    cell.animate(run_time=run_time)
                        .shift(np.array([0, delta_y, 0]))
                )

        # Update header sort arrow
        if col_index < len(self._header_cells):
            old_hdr  = self._header_cells[col_index]
            new_hdr  = _HeaderCell(
                x0=old_hdr._x0, z0=old_hdr._z0,
                width=old_hdr._width, depth=old_hdr._depth,
                col=self._columns[col_index],
                y0=old_hdr._y0,
                sort_dir="asc" if ascending else "desc",
                cfg=cfg,
            )
            anims.append(Transform(old_hdr, new_hdr, run_time=run_time * 0.4))

        return AnimationGroup(*anims)

    def animate_filter(
        self,
        col_index: int,
        predicate: Callable[[object], bool],
        run_time:  float = 1.0,
    ) -> AnimationGroup:
        """Apply a filter: non-matching rows shrink and dim.

        Parameters
        ----------
        col_index : int
        predicate : callable
            Returns True to keep a row.
        """
        new_mask = self._filter.filter_by(col_index, predicate)
        self._visible = new_mask

        page_rows = self._get_page_rows()
        anims     = []

        for display_idx, data_idx in enumerate(self._get_page_rows()):
            row_cells = self._body_cells[display_idx] \
                if display_idx < len(self._body_cells) else []
            keep = new_mask[data_idx]
            for cell in row_cells:
                if cell is None:
                    continue
                if keep:
                    anims.append(
                        cell.animate(run_time=run_time * 0.5).set_opacity(0.90)
                    )
                else:
                    anims.append(
                        cell.animate(run_time=run_time)
                            .scale(0.95)
                            .set_opacity(0.12)
                    )

        # Update filter indicator on header
        if col_index < len(self._header_cells):
            old_hdr = self._header_cells[col_index]
            new_hdr = _HeaderCell(
                x0=old_hdr._x0, z0=old_hdr._z0,
                width=old_hdr._width, depth=old_hdr._depth,
                col=self._columns[col_index],
                y0=old_hdr._y0,
                filtered=True, cfg=self.cfg,
            )
            anims.append(Transform(old_hdr, new_hdr, run_time=run_time * 0.4))

        return AnimationGroup(*anims)

    def animate_clear_filter(self, run_time: float = 0.7) -> AnimationGroup:
        """Remove all filters and restore hidden rows."""
        self._filter.clear()
        self._visible = np.ones(len(self._rows), dtype=bool)
        anims = []
        for row_cells in self._body_cells:
            for cell in row_cells:
                if cell is None:
                    continue
                anims.append(
                    cell.animate(run_time=run_time).set_opacity(self.cfg.cell_opacity)
                )
        return AnimationGroup(*anims)

    def animate_select_row(
        self,
        row_index: int,
        run_time:  float = 0.55,
        lag_ratio: float = 0.08,
    ) -> LaggedStart:
        """Sweep a selection highlight across an entire row left to right."""
        if row_index >= len(self._body_cells):
            return LaggedStart(FadeIn(VGroup()), lag_ratio=0, run_time=0.1)
        row_cells = self._body_cells[row_index]
        anims = []
        for cell in row_cells:
            if cell is None:
                continue
            anims.append(
                cell.animate(run_time=run_time * 0.40)
                    .set_color(CELL_SELECTED)
                    .set_opacity(0.98)
            )
        self._selection.select_row(row_index)
        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)

    def animate_select_col(
        self,
        col_index: int,
        run_time:  float = 0.55,
        lag_ratio: float = 0.10,
    ) -> LaggedStart:
        """Sweep a selection highlight down an entire column top to bottom."""
        anims = []
        for row_cells in self._body_cells:
            if col_index >= len(row_cells):
                continue
            cell = row_cells[col_index]
            if cell is None:
                continue
            anims.append(
                cell.animate(run_time=run_time * 0.40)
                    .set_color(CELL_SELECTED)
                    .set_opacity(0.98)
            )
        self._selection.select_col(col_index)
        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)

    def animate_highlight_cell(
        self,
        row_index:    int,
        col_index:    int,
        scale_factor: float = 1.15,
        run_time:     float = 0.50,
    ) -> Succession:
        """Flash-scale one cell and restore."""
        if (row_index >= len(self._body_cells)
                or col_index >= len(self._body_cells[row_index])):
            return Succession(FadeIn(VGroup(), run_time=0.1))
        cell  = self._body_cells[row_index][col_index]
        if cell is None:
            return Succession(FadeIn(VGroup(), run_time=0.1))
        flash   = cell.animate(run_time=run_time / 2).scale(scale_factor)
        restore = cell.animate(run_time=run_time / 2).scale(1 / scale_factor)
        return Succession(flash, restore)

    def animate_update_cell(
        self,
        row_index: int,
        col_index: int,
        new_value: object,
        run_time:  float = 0.5,
    ) -> AnimationGroup:
        """Morph a cell's text content to a new value.

        Parameters
        ----------
        new_value : object
            The new cell value.  Will be formatted according to the
            column's type specification.
        """
        if (row_index >= len(self._body_cells)
                or col_index >= len(self._body_cells[row_index])):
            return AnimationGroup()

        old_cell = self._body_cells[row_index][col_index]
        if old_cell is None or not isinstance(old_cell, _TableCell):
            return AnimationGroup()

        col  = self._columns[col_index]
        dec  = col.decimals or self.cfg.numeric_decimals
        if col.col_type == "numeric" and isinstance(new_value, (int, float)):
            new_txt = _fmt(float(new_value), dec, "numeric")
        elif col.col_type == "percent" and isinstance(new_value, (int, float)):
            new_txt = _fmt(float(new_value), dec, "percent")
        else:
            new_txt = str(new_value)

        new_lbl = Text(
            new_txt[:self.cfg.max_text_len],
            color=old_cell.label.get_color(),
            font_size=self.cfg.body_font_size,
        )
        new_lbl.move_to(old_cell.label.get_center())
        return AnimationGroup(
            Transform(old_cell.label, new_lbl, run_time=run_time)
        )

    def animate_next_page(self, run_time: float = 0.8) -> Succession:
        """Slide to the next page of rows (if paginated).

        Rows slide upward and the next page slides in from below.
        """
        max_page = max(0,
            (sum(self._visible) - 1) // self._rows_per_page
        )
        if self._page >= max_page:
            return Succession(FadeIn(VGroup(), run_time=0.1))
        self._page += 1
        old_body = VGroup(*[c for row in self._body_cells
                             for c in row if c is not None])
        slide_out = old_body.animate(run_time=run_time * 0.40) \
                            .shift(np.array([0, 0.8, 0])) \
                            .set_opacity(0)
        rebuild_anim = FadeIn(VGroup(), run_time=0.01)   # trigger rebuild
        self._rebuild_body()
        new_body_grp = VGroup(*[c for row in self._body_cells
                                for c in row if c is not None])
        slide_in = FadeIn(new_body_grp, shift=UP * 0.8, run_time=run_time * 0.60)
        return Succession(slide_out, slide_in)

    def animate_prev_page(self, run_time: float = 0.8) -> Succession:
        """Slide to the previous page of rows."""
        if self._page <= 0:
            return Succession(FadeIn(VGroup(), run_time=0.1))
        self._page -= 1
        old_body = VGroup(*[c for row in self._body_cells
                             for c in row if c is not None])
        slide_out = old_body.animate(run_time=run_time * 0.40) \
                            .shift(np.array([0, -0.8, 0])) \
                            .set_opacity(0)
        self._rebuild_body()
        new_body_grp = VGroup(*[c for row in self._body_cells
                                for c in row if c is not None])
        slide_in = FadeIn(new_body_grp, shift=DOWN * 0.8, run_time=run_time * 0.60)
        return Succession(slide_out, slide_in)

    def _rebuild_body(self) -> None:
        """Rebuild body cells for the current page (removes old, adds new)."""
        for row_cells in self._body_cells:
            for cell in row_cells:
                if cell is not None and cell in self.submobjects:
                    self.remove(cell)
        self._body_cells = []
        cfg = self.cfg
        d   = cfg.cell_depth
        rh  = cfg.cell_height + cfg.row_spacing
        px, py, pz = (float(self._pos[0]), float(self._pos[1]),
                      float(self._pos[2]))
        page_rows = self._get_page_rows()
        n_v = len(page_rows)
        for display_row_idx, data_row_idx in enumerate(page_rows):
            row_data = self._rows[data_row_idx]
            y_row    = py + (n_v - 1 - display_row_idx) * rh
            is_even  = display_row_idx % 2 == 0
            base_bg  = CELL_EVEN if is_even else CELL_ODD
            row_cells: list = []
            for j, col in enumerate(self._columns):
                v   = row_data[j] if j < len(row_data) else ""
                bg  = CELL_FROZEN if j == 0 else base_bg
                w   = self._col_widths[j]
                xj  = self._col_x[j]
                dec = col.decimals or cfg.numeric_decimals
                if col.col_type == "sparkline" and isinstance(v, (list, np.ndarray)):
                    cell = _SparklineCell(x0=xj, z0=pz, width=w, depth=d,
                                         values=list(v), y0=y_row, cfg=cfg, bg_color=bg)
                elif col.col_type == "badge":
                    cell = _BadgeCell(x0=xj, z0=pz, width=w, depth=d,
                                      value=str(v), y0=y_row,
                                      palette=col.badge_palette, cfg=cfg, bg_color=bg)
                else:
                    if col.col_type == "bool":
                        txt = cfg.bool_true_text if v else cfg.bool_false_text
                        tc  = BOOL_TRUE_COLOR if v else BOOL_FALSE_COLOR
                    elif col.col_type == "percent" and isinstance(v, (int, float)):
                        txt = _fmt(float(v), dec, "percent"); tc = None
                    elif col.col_type == "numeric" and isinstance(v, (int, float)):
                        txt = _fmt(float(v), dec, "numeric"); tc = None
                    else:
                        txt = str(v); tc = None
                    cell = _TableCell(
                        x0=xj, z0=pz, width=w, depth=d,
                        height=cfg.cell_height, bg_color=bg, text=txt,
                        align=col.effective_align(), y0=y_row, cfg=cfg,
                        bold=(j == 0), text_color=tc,
                    )
                row_cells.append(cell)
                self.add(cell)
            self._body_cells.append(row_cells)

    def animate_reveal_summary(self, run_time: float = 0.70) -> GrowFromPoint:
        """Summary row slides up from below the table."""
        if not self._summary_cells:
            return GrowFromPoint(VGroup(), point=np.array([0, 0, 0]))
        grp = VGroup(*self._summary_cells)
        return GrowFromPoint(grp, point=grp.get_bottom(), run_time=run_time)

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df:         dict[str, list],
        col_types:  list[str] | None = None,
        pos:        np.ndarray = np.array([0.0, 0.0, 0.0]),
        config:     TableConfig | None = None,
    ) -> "DataTable3D":
        """Build from a dict-of-lists (pandas-compatible interface).

        Parameters
        ----------
        df : dict[str, list]
            Keys are column names; values are equal-length lists.
        col_types : list[str] | None
            Column type strings.  Defaults to "numeric" for numeric
            data, "text" otherwise.

        Example
        -------
        >>> tbl = DataTable3D.from_dataframe({
        ...     "Model":    ["LR", "RF", "SVM"],
        ...     "Accuracy": [0.82, 0.91, 0.88],
        ...     "AUC":      [0.85, 0.93, 0.90],
        ... }, col_types=["text", "percent", "percent"])
        """
        col_names = list(df.keys())
        rows_t    = list(zip(*[df[k] for k in col_names]))
        rows      = [list(r) for r in rows_t]

        columns = []
        for i, name in enumerate(col_names):
            ct = (col_types[i] if col_types and i < len(col_types) else None)
            if ct is None:
                sample = df[name][0] if df[name] else ""
                ct = "numeric" if isinstance(sample, (int, float)) else "text"
            columns.append(TableColumn(name=name, col_type=ct))

        return cls(columns=columns, rows=rows, pos=pos, config=config)

    @classmethod
    def from_numpy(
        cls,
        array:      np.ndarray,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        pos:        np.ndarray = np.array([0.0, 0.0, 0.0]),
        config:     TableConfig | None = None,
    ) -> "DataTable3D":
        """Build from a 2D numpy matrix.

        Parameters
        ----------
        array : (R, C) ndarray
        row_labels : list[str] | None
        col_labels : list[str] | None
        """
        array      = np.asarray(array, dtype=float)
        n_rows, n_cols = array.shape
        rl         = row_labels or [str(i) for i in range(n_rows)]
        cl         = col_labels or [str(j) for j in range(n_cols)]

        columns    = [TableColumn(name="", col_type="text")]  # row-label col
        columns   += [TableColumn(name=c, col_type="numeric") for c in cl]
        rows       = [[rl[i]] + list(array[i]) for i in range(n_rows)]

        return cls(columns=columns, rows=rows, pos=pos, config=config)

    @classmethod
    def stats_summary(
        cls,
        named_arrays: dict[str, Sequence[float]],
        pos:          np.ndarray = np.array([0.0, 0.0, 0.0]),
        config:       TableConfig | None = None,
    ) -> "DataTable3D":
        """Descriptive statistics table for one or more 1-D arrays.

        Computes min, Q1, median, mean, Q3, max, std, n for each array.

        Parameters
        ----------
        named_arrays : dict[str, array-like]
            Keys are variable names; values are observation arrays.

        Example
        -------
        >>> tbl = DataTable3D.stats_summary({
        ...     "Height": heights,
        ...     "Weight": weights,
        ... })
        """
        stat_names = ["N", "Min", "Q1", "Median", "Mean", "Q3", "Max", "Std"]
        columns    = [TableColumn(name="Variable", col_type="text", width=1.40)]
        columns   += [TableColumn(name=s, col_type="numeric") for s in stat_names]

        rows = []
        for name, data in named_arrays.items():
            arr = np.asarray(data, dtype=float)
            rows.append([
                name,
                int(len(arr)),
                float(arr.min()),
                float(np.percentile(arr, 25)),
                float(np.median(arr)),
                float(arr.mean()),
                float(np.percentile(arr, 75)),
                float(arr.max()),
                float(arr.std()),
            ])

        return cls(columns=columns, rows=rows, pos=pos, config=config)

    @classmethod
    def comparison_results(
        cls,
        models:   list[str],
        metrics:  dict[str, list[float]],
        col_type: str = "numeric",
        pos:      np.ndarray = np.array([0.0, 0.0, 0.0]),
        config:   TableConfig | None = None,
    ) -> "DataTable3D":
        """Model / method comparison table.

        Parameters
        ----------
        models : list[str]
            Row labels (one per model).
        metrics : dict[str, list[float]]
            Keys are metric names; values are per-model results
            (must match len(models)).

        Example
        -------
        >>> tbl = DataTable3D.comparison_results(
        ...     models=["LR", "RF", "SVM", "XGB"],
        ...     metrics={
        ...         "Accuracy": [0.82, 0.91, 0.88, 0.94],
        ...         "F1":       [0.80, 0.90, 0.87, 0.93],
        ...         "AUC":      [0.85, 0.93, 0.90, 0.96],
        ...     },
        ... )
        """
        columns = [TableColumn(name="Model", col_type="text", width=1.20)]
        columns += [TableColumn(name=m, col_type=col_type)
                    for m in metrics.keys()]

        rows = []
        for i, model in enumerate(models):
            row = [model] + [metrics[m][i] for m in metrics]
            rows.append(row)

        return cls(columns=columns, rows=rows, pos=pos, config=config)

    @classmethod
    def frequency_table(
        cls,
        categories:  list[str],
        counts:      list[int],
        pos:         np.ndarray = np.array([0.0, 0.0, 0.0]),
        config:      TableConfig | None = None,
    ) -> "DataTable3D":
        """Frequency + relative frequency + cumulative frequency table.

        Parameters
        ----------
        categories : list[str]
        counts : list[int]

        Columns produced: Category | Frequency | Rel. Freq (%) | Cum. Freq | Cum. %
        """
        total   = sum(counts)
        cum_n   = 0
        columns = [
            TableColumn(name="Category",  col_type="text",    width=1.40),
            TableColumn(name="Freq",      col_type="numeric", decimals=0),
            TableColumn(name="Rel. Freq", col_type="percent", decimals=1),
            TableColumn(name="Cum. Freq", col_type="numeric", decimals=0),
            TableColumn(name="Cum. %",    col_type="percent", decimals=1),
        ]
        rows = []
        for cat, cnt in zip(categories, counts):
            cum_n += cnt
            rows.append([
                cat,
                float(cnt),
                cnt / total,
                float(cum_n),
                cum_n / total,
            ])

        return cls(columns=columns, rows=rows, pos=pos, config=config)