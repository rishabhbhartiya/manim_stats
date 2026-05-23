"""
manim_stats/charts/heat_map3d.py
==================================
A production-quality 3D heat map for Manim with:

  Cell geometry
  -------------
  - Each cell is a _HeatCell: a 3D prism that rises proportionally to its
    value — dual encoding (height + color) for stronger visual differentiation
  - Three shaded faces per prism: top (primary color), front side (darkened),
    right side (more darkened) — same physically-shaded idiom used across
    the manim_stats module
  - Masked cells (e.g. upper triangle of a correlation matrix) rendered as
    flat hatched polygons, visually distinct from data cells
  - Optional flat mode: disable height encoding, render all cells at uniform
    height (classic flat heatmap look inside a 3D scene)

  Color systems
  -------------
  - Five named built-in palettes:
      "diverging"   : blue → white → red  (great for correlations, z-scores)
      "sequential"  : white → deep blue   (counts, probabilities, 0-bounded)
      "hot"         : black → red → yellow → white  (intensity maps)
      "cool"        : cyan → magenta      (perceptual, uniform lightness)
      "viridis"     : purple → blue → green → yellow  (perceptual, print-safe)
  - Full multi-stop interpolation through the palette — not just two-color lerp
  - Custom palette override: pass any list of ManimColor stops
  - Auto-contrast text: white labels on dark cells, near-black on light cells
    (contrast ratio computed from perceived luminance)

  Annotations
  -----------
  - Floating value label centered above each cell top face
  - Precision and format string configurable
  - Label size auto-scales with cell size so they never overflow
  - Optional row-total and column-total margin labels

  Row / column marginals
  ----------------------
  - Mean-value bar strip along the right edge (column aggregates)
  - Mean-value bar strip along the top edge (row aggregates)
  - Bars are proportional prisms sharing the same color palette

  Dendrogram
  ----------
  - Optional hierarchical clustering (scipy.cluster.hierarchy) displayed
    as Line3D trees above the column axis and beside the row axis
  - Clustering linkage method configurable: "ward", "average", "complete"
  - Leaf order resorting: matrix rows/cols are reordered to match the
    dendrogram leaf order so similar entries are adjacent

  Highlight overlays
  ------------------
  - _HighlightRect: a thin rectangular border raised slightly above the
    cell tops, used for row, column, and cell highlights
  - highlight_row(i), highlight_col(j), highlight_cell(i, j) methods
    return the overlay object for .play(Create(...))

  Correlation matrix mode
  -----------------------
  - from_correlation() classmethod: accepts raw data matrix, computes
    Pearson correlation, sets palette to "diverging", vmin/vmax to -1/1,
    masks the upper triangle, fixes diagonal annotation to "1.00"

  Animation suite
  ---------------
  - animate_grow()            : all cells rise from the floor simultaneously
                                (lag_ratio stagger, left-to-right, row by row)
  - animate_sweep_row(i)      : one row of cells grows left to right
  - animate_sweep_col(j)      : one column grows bottom to top
  - animate_morph_values(M')  : interpolate every cell to a new matrix M'
                                (same shape required; rebuilds cell heights
                                and colors via Transform)
  - animate_highlight_row(i)  : Create a highlight rect across the full row
  - animate_highlight_col(j)  : Create a highlight rect down the full column
  - animate_highlight_cell(i,j): flash + scale a single cell
  - animate_reveal_colorbar() : colorbar gradient draws in from bottom to top
  - animate_reveal_dendro()   : dendrogram lines trace outward from leaves
  - animate_palette_morph(p)  : morph all cell colors to a new named palette
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from scipy import stats as scipy_stats

from manim import (
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    UP, DOWN, LEFT, RIGHT,
    PI, TAU,
    VGroup, VMobject,
    Line3D, Polygon, Dot3D,
    Text, MathTex,
    FadeIn, FadeOut, GrowFromPoint, Transform,
    Create, AnimationGroup, Succession, LaggedStart,
    interpolate_color,
    ManimColor,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Palette definitions  (multi-stop, linearly interpolated)
# ---------------------------------------------------------------------------

PALETTES: dict[str, list[ManimColor]] = {
    "diverging": [
        ManimColor("#053061"),   # dark blue
        ManimColor("#2166AC"),   # blue
        ManimColor("#92C5DE"),   # light blue
        ManimColor("#F7F7F7"),   # near-white (mid / zero)
        ManimColor("#FDDBC7"),   # light red
        ManimColor("#D6604D"),   # red
        ManimColor("#67001F"),   # dark red
    ],
    "sequential": [
        ManimColor("#F7FBFF"),   # almost white
        ManimColor("#C6DBEF"),
        ManimColor("#6BAED6"),
        ManimColor("#2171B5"),
        ManimColor("#08306B"),   # dark navy
    ],
    "hot": [
        ManimColor("#000000"),   # black
        ManimColor("#7F0000"),   # dark red
        ManimColor("#FF0000"),   # red
        ManimColor("#FF7F00"),   # orange
        ManimColor("#FFFF00"),   # yellow
        ManimColor("#FFFFFF"),   # white
    ],
    "cool": [
        ManimColor("#00FFFF"),   # cyan
        ManimColor("#7F00FF"),   # violet
        ManimColor("#FF00FF"),   # magenta
    ],
    "viridis": [
        ManimColor("#440154"),   # dark purple
        ManimColor("#31688E"),   # blue
        ManimColor("#35B779"),   # green
        ManimColor("#FDE725"),   # yellow
    ],
}


def _palette_color(t: float, stops: list[ManimColor]) -> ManimColor:
    """Multi-stop palette interpolation.  t ∈ [0, 1]."""
    t   = float(np.clip(t, 0.0, 1.0))
    n   = len(stops) - 1
    seg = min(int(t * n), n - 1)
    lt  = t * n - seg
    return interpolate_color(stops[seg], stops[seg + 1], lt)


def _perceived_luminance(color: ManimColor) -> float:
    """Approximate WCAG relative luminance for auto-contrast text.

    Returns a value in [0, 1]; 0 = black, 1 = white.
    """
    r, g, b = [x / 255.0 for x in color.to_rgb()]

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast_text_color(bg: ManimColor) -> ManimColor:
    """Return white or near-black depending on which gives better contrast."""
    lum = _perceived_luminance(bg)
    return ManimColor("#F8F8F8") if lum < 0.45 else ManimColor("#1A1A1A")


def _darken(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lighten(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeatMapConfig:
    """All visual and statistical options for HeatMap3D.

    Cell geometry
    -------------
    cell_size : float
        Width and depth of each cell in Manim units (cells are square).
    cell_gap : float
        Gap between adjacent cells (0 = flush, 0.04 = small separation).
    max_height : float
        Maximum prism height in Manim units (tallest cell = this height).
        Set to 0 to use flat mode (no height encoding).
    flat_mode : bool
        Override max_height to 0 and render as a classic flat heatmap.
    face_darken_side : float
        How much to darken the front face of each prism vs the top face.
    face_darken_right : float
        How much to darken the right face of each prism vs the top face.

    Color
    -----
    palette : str | list[ManimColor]
        Named palette key (see PALETTES dict) or a custom list of ManimColor
        stops.  Built-in names: "diverging", "sequential", "hot", "cool",
        "viridis".
    vmin : float | None
        Data value mapped to the cold end of the palette.  Auto-computed
        from the matrix if None.
    vmax : float | None
        Data value mapped to the hot end of the palette.  Auto-computed
        from the matrix if None.
    vcenter : float | None
        If set, forces this data value to map to the exact midpoint of the
        palette (useful for diverging palettes centered on 0).
    clip_values : bool
        If True, values outside [vmin, vmax] are clamped to the palette ends.
        If False, they use the end color without error.

    Annotations
    -----------
    show_values : bool
        Render a floating value label above each (unmasked) cell.
    value_format : str
        Python format string for cell labels, e.g. ``"{:.2f}"`` or
        ``"{:.0%}"``.
    value_font_size : int
        Base font size for cell labels; auto-scaled down for small cells.
    value_z_offset : float
        How far above the cell top face the label floats (Manim units).
    show_row_labels : bool
        Render row name labels on the left side of the matrix.
    show_col_labels : bool
        Render column name labels below the matrix.
    label_font_size : int
        Font size for row / column category labels.
    rotate_col_labels : bool
        Rotate column labels 45° to save horizontal space.
    show_marginal_rows : bool
        Draw a bar strip of row means along the right edge.
    show_marginal_cols : bool
        Draw a bar strip of column means along the top edge.
    marginal_bar_max : float
        Maximum width/height of marginal bars in Manim units.

    Masks
    -----
    mask : np.ndarray | None
        Boolean array of shape (rows, cols).  True = mask this cell
        (render as flat hatch, exclude from color/height encoding).
    mask_color : ManimColor
        Fill color of masked cells.
    mask_hatch_color : ManimColor
        Color of the diagonal hatch lines drawn on masked cells.
    mask_opacity : float
        Opacity of masked cell faces.

    Dendrogram
    ----------
    show_row_dendrogram : bool
        Compute and display hierarchical clustering tree beside the rows.
    show_col_dendrogram : bool
        Compute and display hierarchical clustering tree above the columns.
    reorder_by_dendrogram : bool
        Reorder matrix rows and/or columns to match the dendrogram leaf order.
    linkage_method : str
        Linkage method for ``scipy.cluster.hierarchy.linkage``.
        Options: ``"ward"``, ``"average"``, ``"complete"``, ``"single"``.
    dendro_stroke_width : float
        Stroke width of dendrogram tree lines.
    dendro_color : ManimColor
        Color of dendrogram lines.
    dendro_clearance : float
        Gap between the matrix edge and the start of the dendrogram.
    dendro_extent : float
        Maximum length of the dendrogram arm in Manim units.

    Color bar
    ---------
    show_colorbar : bool
        Render a color-gradient bar legend beside the matrix.
    colorbar_width : float
        Width of the colorbar strip in Manim units.
    colorbar_height : float
        Height of the colorbar strip (equal to the matrix height by default;
        pass 0 to use matrix height automatically).
    colorbar_n_ticks : int
        Number of labeled tick marks on the colorbar.
    colorbar_label : str
        Optional title label above the colorbar.

    Grid lines
    ----------
    show_grid : bool
        Draw thin grid lines at cell boundaries on the floor plane.
    grid_color : ManimColor
        Color of grid lines.
    grid_opacity : float
        Opacity of grid lines.

    Layout
    ------
    origin : np.ndarray
        World position of the bottom-left corner of the matrix.
    """

    # ---- cell geometry ----
    cell_size:          float = 0.70
    cell_gap:           float = 0.03
    max_height:         float = 0.80
    flat_mode:          bool  = False
    face_darken_side:   float = 0.38
    face_darken_right:  float = 0.55

    # ---- color ----
    palette:     str | list[ManimColor] = "viridis"
    vmin:        float | None           = None
    vmax:        float | None           = None
    vcenter:     float | None           = None
    clip_values: bool                   = True

    # ---- annotations ----
    show_values:        bool  = True
    value_format:       str   = "{:.2f}"
    value_font_size:    int   = 16
    value_z_offset:     float = 0.08
    show_row_labels:    bool  = True
    show_col_labels:    bool  = True
    label_font_size:    int   = 20
    rotate_col_labels:  bool  = False
    show_marginal_rows: bool  = False
    show_marginal_cols: bool  = False
    marginal_bar_max:   float = 0.40

    # ---- masks ----
    mask:             np.ndarray | None = None
    mask_color:       ManimColor        = ManimColor("#1C1C1C")
    mask_hatch_color: ManimColor        = ManimColor("#3A3A3A")
    mask_opacity:     float             = 0.60

    # ---- dendrogram ----
    show_row_dendrogram:    bool       = False
    show_col_dendrogram:    bool       = False
    reorder_by_dendrogram:  bool       = True
    linkage_method:         str        = "ward"
    dendro_stroke_width:    float      = 1.2
    dendro_color:           ManimColor = ManimColor("#78909C")
    dendro_clearance:       float      = 0.15
    dendro_extent:          float      = 0.90

    # ---- colorbar ----
    show_colorbar:      bool       = True
    colorbar_width:     float      = 0.25
    colorbar_height:    float      = 0.0     # 0 = match matrix height
    colorbar_n_ticks:   int        = 5
    colorbar_label:     str        = ""

    # ---- grid ----
    show_grid:    bool       = True
    grid_color:   ManimColor = GRAY_D
    grid_opacity: float      = 0.30

    # ---- layout ----
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Value → palette-t  (handles vmin / vmax / vcenter)
# ---------------------------------------------------------------------------

class _ColorMapper:
    """Maps scalar data values to palette t-values in [0, 1]."""

    def __init__(
        self,
        matrix: np.ndarray,
        cfg:    HeatMapConfig,
        mask:   np.ndarray | None,
    ):
        valid = matrix[~mask] if mask is not None else matrix
        valid = valid[np.isfinite(valid)]

        self.vmin    = float(cfg.vmin    if cfg.vmin    is not None else valid.min())
        self.vmax    = float(cfg.vmax    if cfg.vmax    is not None else valid.max())
        self.vcenter = cfg.vcenter
        self.clip    = cfg.clip_values

        if self.vmax <= self.vmin:
            self.vmax = self.vmin + 1.0

    def __call__(self, v: float) -> float:
        """Return t ∈ [0, 1] for value v."""
        if self.vcenter is not None:
            # Piecewise linear: vmin→0, vcenter→0.5, vmax→1
            if v <= self.vcenter:
                lo, hi = self.vmin, self.vcenter
                t_base = 0.0
                t_scale = 0.5
            else:
                lo, hi = self.vcenter, self.vmax
                t_base = 0.5
                t_scale = 0.5
            span = hi - lo if hi > lo else 1.0
            t = t_base + t_scale * (v - lo) / span
        else:
            t = (v - self.vmin) / (self.vmax - self.vmin)

        if self.clip:
            t = np.clip(t, 0.0, 1.0)
        return float(t)


# ---------------------------------------------------------------------------
# _HeatCell  — 3D prism with front / right / top shaded faces
# ---------------------------------------------------------------------------

class _HeatCell(VGroup):
    """A single cell of the heat map rendered as a shaded 3D prism.

    Coordinate layout (right-hand Manim axes):
        x : left → right  (column axis)
        y : bottom → top  (height, encodes value magnitude)
        z : front → back  (row axis, note: row 0 = front)

    The cell occupies:
        x : [x0, x0 + size]
        y : [0,  height   ]
        z : [z0, z0 + size]
    """

    def __init__(
        self,
        x0:          float,
        z0:          float,
        height:      float,
        size:        float,
        top_color:   ManimColor,
        cfg:         HeatMapConfig,
    ):
        super().__init__()
        self.x0     = x0
        self.z0     = z0
        self.height = height
        self.size   = size
        self._cfg   = cfg

        front_color = _darken(top_color, cfg.face_darken_side)
        right_color = _darken(top_color, cfg.face_darken_right)

        s = size
        h = max(height, 0.005)   # never truly flat even in "height=0" mode

        # 8 corners
        AFL = np.array([x0,     0, z0    ])
        AFR = np.array([x0 + s, 0, z0    ])
        ABL = np.array([x0,     0, z0 + s])
        ABR = np.array([x0 + s, 0, z0 + s])
        TFL = np.array([x0,     h, z0    ])
        TFR = np.array([x0 + s, h, z0    ])
        TBL = np.array([x0,     h, z0 + s])
        TBR = np.array([x0 + s, h, z0 + s])

        def _face(pts: list, color: ManimColor, opacity: float = 0.92) -> Polygon:
            p = Polygon(*pts, color=color)
            p.set_fill(color=color, opacity=opacity)
            p.set_stroke(color=_darken(color, 0.50), width=0.5, opacity=0.40)
            return p

        self.top_face   = _face([TFL, TFR, TBR, TBL], top_color)
        self.front_face = _face([AFL, AFR, TFR, TFL], front_color)
        self.right_face = _face([AFR, ABR, TBR, TFR], right_color)

        self.add(self.front_face, self.right_face, self.top_face)

        # Store corners for animation
        self._corners = dict(
            AFL=AFL, AFR=AFR, ABL=ABL, ABR=ABR,
            TFL=TFL, TFR=TFR, TBL=TBL, TBR=TBR,
        )
        self._top_color   = top_color
        self._front_color = front_color
        self._right_color = right_color

    @property
    def top_center(self) -> np.ndarray:
        c = self._corners
        return (c["TFL"] + c["TBR"]) / 2

    def set_colors(
        self,
        top_color: ManimColor,
        cfg:       HeatMapConfig,
    ) -> None:
        """Update face colors (used in animate_palette_morph)."""
        self.top_face.set_fill(color=top_color, opacity=0.92)
        fc = _darken(top_color, cfg.face_darken_side)
        rc = _darken(top_color, cfg.face_darken_right)
        self.front_face.set_fill(color=fc, opacity=0.92)
        self.right_face.set_fill(color=rc, opacity=0.92)


# ---------------------------------------------------------------------------
# _MaskedCell — flat hatched polygon for masked cells
# ---------------------------------------------------------------------------

class _MaskedCell(VGroup):
    """A flat hatched cell indicating masked / NA / upper-triangle positions."""

    def __init__(
        self,
        x0:  float,
        z0:  float,
        size: float,
        cfg: HeatMapConfig,
    ):
        super().__init__()
        s = size

        # Flat top quad (floor level)
        base = Polygon(
            np.array([x0,     0.002, z0    ]),
            np.array([x0 + s, 0.002, z0    ]),
            np.array([x0 + s, 0.002, z0 + s]),
            np.array([x0,     0.002, z0 + s]),
            color=cfg.mask_color,
        )
        base.set_fill(color=cfg.mask_color, opacity=cfg.mask_opacity)
        base.set_stroke(color=cfg.mask_hatch_color, width=0.4, opacity=0.5)
        self.add(base)

        # Diagonal hatch lines (2 diagonals)
        for (pa, pb) in [
            (np.array([x0,     0.006, z0    ]), np.array([x0 + s, 0.006, z0 + s])),
            (np.array([x0 + s, 0.006, z0    ]), np.array([x0,     0.006, z0 + s])),
        ]:
            line = Line3D(start=pa, end=pb,
                          color=cfg.mask_hatch_color, stroke_width=0.6)
            line.set_opacity(cfg.mask_opacity * 0.70)
            self.add(line)


# ---------------------------------------------------------------------------
# _HighlightRect — thin border above a row / col / cell
# ---------------------------------------------------------------------------

class _HighlightRect(VGroup):
    """A rectangular border raised slightly above the cell top faces.

    Used to outline a row, column, or individual cell.
    """

    def __init__(
        self,
        x0:     float,
        z0:     float,
        x1:     float,
        z1:     float,
        y_top:  float,
        color:  ManimColor,
        stroke_width: float = 2.5,
        lift:   float       = 0.04,
    ):
        super().__init__()
        y = y_top + lift
        pts = [
            np.array([x0, y, z0]),
            np.array([x1, y, z0]),
            np.array([x1, y, z1]),
            np.array([x0, y, z1]),
        ]
        rect = Polygon(*pts, color=color)
        rect.set_fill(opacity=0.0)
        rect.set_stroke(color=color, width=stroke_width)
        self.add(rect)


# ---------------------------------------------------------------------------
# _Dendrogram — hierarchical clustering tree as Line3D branches
# ---------------------------------------------------------------------------

class _Dendrogram(VGroup):
    """Hierarchical clustering tree rendered as a set of Line3D segments.

    Supports row (vertical) or column (horizontal) orientation.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Output of ``scipy.cluster.hierarchy.linkage``.
    n_leaves : int
        Number of leaves (matrix rows or columns).
    positions : np.ndarray
        (n_leaves,) world-space positions of the leaf centres on the
        primary axis (X for column dendro, Z for row dendro).
    axis : str
        ``"col"`` — tree grows upward (along Y), leaves on X.
        ``"row"`` — tree grows rightward (along X), leaves on Z.
    origin_y : float
        Y level at which the leaves attach (top of the matrix or bottom).
    clearance : float
        Gap from origin_y before the first branch.
    extent : float
        Maximum arm length in Manim units.
    color : ManimColor
    stroke_width : float
    """

    def __init__(
        self,
        linkage_matrix: np.ndarray,
        n_leaves:       int,
        positions:      np.ndarray,
        axis:           str,
        origin_y:       float,
        clearance:      float,
        extent:         float,
        color:          ManimColor,
        stroke_width:   float,
    ):
        super().__init__()
        self.line_objects: list[Line3D] = []

        Z    = linkage_matrix
        n    = n_leaves
        dmax = float(Z[:, 2].max()) if len(Z) else 1.0

        # Map scipy dendrogram distance to world extent
        def _dist_to_arm(d: float) -> float:
            return clearance + (d / dmax) * extent if dmax > 0 else clearance

        # Cluster centre positions (leaves + internal nodes)
        cluster_pos = np.zeros(2 * n - 1)
        cluster_pos[:n] = positions            # leaf positions

        for k, row in enumerate(Z):
            li, ri = int(row[0]), int(row[1])
            dist   = float(row[2])
            arm    = _dist_to_arm(dist)

            lp = cluster_pos[li]
            rp = cluster_pos[ri]
            cp = (lp + rp) / 2.0
            cluster_pos[n + k] = cp

            l_arm_prev = _dist_to_arm(
                float(Z[li - n, 2]) if li >= n else 0.0
            )
            r_arm_prev = _dist_to_arm(
                float(Z[ri - n, 2]) if ri >= n else 0.0
            )

            if axis == "col":
                # Leaves run along X; tree grows along Y (upward)
                def _seg(p1, p2):
                    return Line3D(
                        start=np.array([p1[0], origin_y + p1[1], 0]),
                        end  =np.array([p2[0], origin_y + p2[1], 0]),
                        color=color, stroke_width=stroke_width,
                    )
                # Vertical stems from child levels up to this merge level
                seg_l = _seg((lp, l_arm_prev), (lp,  arm))
                seg_r = _seg((rp, r_arm_prev), (rp,  arm))
                # Horizontal crossbar
                seg_h = _seg((lp, arm),        (rp,  arm))
            else:
                # axis == "row": leaves along Z; tree grows along -X (leftward)
                def _seg(p1, p2):
                    return Line3D(
                        start=np.array([origin_y - p1[1], 0, p1[0]]),
                        end  =np.array([origin_y - p2[1], 0, p2[0]]),
                        color=color, stroke_width=stroke_width,
                    )
                seg_l = _seg((lp, l_arm_prev), (lp,  arm))
                seg_r = _seg((rp, r_arm_prev), (rp,  arm))
                seg_h = _seg((lp, arm),        (rp,  arm))

            for seg in (seg_l, seg_r, seg_h):
                self.add(seg)
                self.line_objects.append(seg)


# ---------------------------------------------------------------------------
# _ColorBar — gradient legend beside the matrix
# ---------------------------------------------------------------------------

class _ColorBar(VGroup):
    """A vertical gradient strip with labeled tick marks.

    Built as a stack of thin flat quads each filled with the local
    palette color, producing a smooth gradient appearance.
    """

    SEGMENTS = 64   # number of quads in the gradient

    def __init__(
        self,
        x_pos:    float,
        y_bottom: float,
        height:   float,
        width:    float,
        stops:    list[ManimColor],
        vmin:     float,
        vmax:     float,
        n_ticks:  int,
        label:    str,
        font_size: int,
        z_pos:    float = 0.0,
    ):
        super().__init__()
        seg_h = height / self.SEGMENTS

        for k in range(self.SEGMENTS):
            t     = k / (self.SEGMENTS - 1)
            color = _palette_color(t, stops)
            y0    = y_bottom + k * seg_h
            quad  = Polygon(
                np.array([x_pos,         y0,          z_pos]),
                np.array([x_pos + width, y0,          z_pos]),
                np.array([x_pos + width, y0 + seg_h,  z_pos]),
                np.array([x_pos,         y0 + seg_h,  z_pos]),
                color=color,
            )
            quad.set_fill(color=color, opacity=1.0)
            quad.set_stroke(width=0)
            self.add(quad)

        # Outline border
        border = Polygon(
            np.array([x_pos,         y_bottom,          z_pos]),
            np.array([x_pos + width, y_bottom,          z_pos]),
            np.array([x_pos + width, y_bottom + height, z_pos]),
            np.array([x_pos,         y_bottom + height, z_pos]),
            color=GRAY_B,
        )
        border.set_fill(opacity=0.0)
        border.set_stroke(color=GRAY_B, width=0.8)
        self.add(border)

        # Tick marks + labels
        for k in range(n_ticks):
            t     = k / (n_ticks - 1)
            y_t   = y_bottom + t * height
            dv    = vmin + t * (vmax - vmin)
            tick  = Line3D(
                start=np.array([x_pos + width,        y_t, z_pos]),
                end  =np.array([x_pos + width + 0.10, y_t, z_pos]),
                color=GRAY_B, stroke_width=0.8,
            )
            lbl = Text(f"{dv:.2f}", color=GRAY_B, font_size=font_size - 2)
            lbl.move_to(np.array([x_pos + width + 0.35, y_t, z_pos]))
            self.add(tick, lbl)

        # Title
        if label:
            title = Text(label, color=GRAY_B, font_size=font_size)
            title.move_to(np.array([x_pos + width / 2,
                                    y_bottom + height + 0.28, z_pos]))
            self.add(title)


# ---------------------------------------------------------------------------
# _MarginalBar — thin bar strip for row/col aggregate display
# ---------------------------------------------------------------------------

class _MarginalBars(VGroup):
    """A strip of small bars showing row or column aggregate values.

    Parameters
    ----------
    values : (N,) ndarray
        Aggregate values (e.g., row means or column means).
    positions : (N,) ndarray
        World-space centre of each bar along the primary axis.
    orientation : str
        ``"row"`` — bars extend along X (beside the rows).
        ``"col"`` — bars extend along Y (above the columns).
    base_coord : float
        X (row) or Y (col) coordinate of the bar base.
    cell_size : float
        Width of each bar (= cell_size - cell_gap).
    bar_max : float
        Maximum bar extension in Manim units.
    palette_stops : list[ManimColor]
    """

    def __init__(
        self,
        values:        np.ndarray,
        positions:     np.ndarray,
        orientation:   str,
        base_coord:    float,
        cell_size:     float,
        bar_max:       float,
        palette_stops: list[ManimColor],
        vmin:          float,
        vmax:          float,
    ):
        super().__init__()
        vspan  = vmax - vmin if vmax > vmin else 1.0
        hw     = cell_size / 2

        for pos, val in zip(positions, values):
            t      = np.clip((val - vmin) / vspan, 0.0, 1.0)
            color  = _palette_color(t, palette_stops)
            length = t * bar_max

            if orientation == "row":
                # Bar extends along +X from base_coord
                bar = Polygon(
                    np.array([base_coord,          0.002, pos - hw]),
                    np.array([base_coord + length, 0.002, pos - hw]),
                    np.array([base_coord + length, 0.002, pos + hw]),
                    np.array([base_coord,          0.002, pos + hw]),
                    color=color,
                )
            else:
                # Bar extends along +Y from base_coord
                bar = Polygon(
                    np.array([pos - hw, base_coord,          0.002]),
                    np.array([pos + hw, base_coord,          0.002]),
                    np.array([pos + hw, base_coord + length, 0.002]),
                    np.array([pos - hw, base_coord + length, 0.002]),
                    color=color,
                )
            bar.set_fill(color=color, opacity=0.85)
            bar.set_stroke(color=_darken(color, 0.4), width=0.5)
            self.add(bar)


# ---------------------------------------------------------------------------
# Floor grid
# ---------------------------------------------------------------------------

class _FloorGrid(VGroup):
    def __init__(
        self,
        x_range: tuple[float, float],
        z_range: tuple[float, float],
        n_x:     int,
        n_z:     int,
        color:   ManimColor,
        opacity: float,
    ):
        super().__init__()
        x0, x1 = x_range
        z0, z1 = z_range
        for i in range(n_x + 1):
            xv = x0 + i * (x1 - x0) / n_x
            self.add(Line3D(start=np.array([xv, 0, z0]),
                            end  =np.array([xv, 0, z1]),
                            color=color, stroke_width=0.5).set_opacity(opacity))
        for j in range(n_z + 1):
            zv = z0 + j * (z1 - z0) / n_z
            self.add(Line3D(start=np.array([x0, 0, zv]),
                            end  =np.array([x1, 0, zv]),
                            color=color, stroke_width=0.5).set_opacity(opacity))


# ---------------------------------------------------------------------------
# Main HeatMap3D class
# ---------------------------------------------------------------------------

class HeatMap3D(VGroup):
    """A detailed 3D heat map for Manim statistics animations.

    Each cell is a height-encoded 3D prism colored by a multi-stop palette.
    Supports mask layers, dendrograms, marginal bars, colorbar, and a rich
    animation API.

    Basic usage
    -----------
    >>> import numpy as np
    >>> from manim import *
    >>> from manim_stats.charts.heat_map3d import HeatMap3D, HeatMapConfig
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         matrix = np.random.uniform(0, 1, (6, 6))
    ...         cfg    = HeatMapConfig(palette="viridis", show_values=True)
    ...         hm     = HeatMap3D(matrix, config=cfg)
    ...         self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
    ...         self.play(hm.animate_grow())

    Correlation matrix
    ------------------
    >>> data = np.random.randn(100, 5)
    >>> hm   = HeatMap3D.from_correlation(data, col_labels=["A","B","C","D","E"])

    Parameters
    ----------
    matrix : np.ndarray  — shape (rows, cols) of float values
    row_labels : list[str], optional
    col_labels : list[str], optional
    config : HeatMapConfig, optional
    """

    def __init__(
        self,
        matrix:     np.ndarray,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        config:     HeatMapConfig | None = None,
    ):
        super().__init__()
        self.cfg    = config or HeatMapConfig()
        self._orig  = np.asarray(matrix, dtype=float)
        self.matrix = self._orig.copy()
        self.n_rows, self.n_cols = self.matrix.shape

        self.row_labels = row_labels or [str(i) for i in range(self.n_rows)]
        self.col_labels = col_labels or [str(j) for j in range(self.n_cols)]

        self._build()

    # ------------------------------------------------------------------
    # Internal build pipeline
    # ------------------------------------------------------------------

    def _resolve_palette(self) -> list[ManimColor]:
        p = self.cfg.palette
        if isinstance(p, str):
            if p not in PALETTES:
                raise ValueError(
                    f"Unknown palette '{p}'. "
                    f"Choose from: {list(PALETTES.keys())}"
                )
            return PALETTES[p]
        return list(p)

    def _build(self) -> None:
        cfg      = self.cfg
        matrix   = self.matrix
        n_rows   = self.n_rows
        n_cols   = self.n_cols

        # ---- Resolve mask ------------------------------------------
        if cfg.mask is not None:
            mask = np.asarray(cfg.mask, dtype=bool)
            if mask.shape != matrix.shape:
                raise ValueError("mask shape must match matrix shape")
        else:
            mask = np.zeros(matrix.shape, dtype=bool)

        self._mask = mask

        # ---- Palette & color mapper ---------------------------------
        self._stops    = self._resolve_palette()
        self._cmap     = _ColorMapper(matrix, cfg, mask)

        # ---- Dendrogram reordering ----------------------------------
        row_order = np.arange(n_rows)
        col_order = np.arange(n_cols)

        if cfg.reorder_by_dendrogram:
            if cfg.show_row_dendrogram and n_rows > 1:
                Z    = sch.linkage(matrix, method=cfg.linkage_method)
                dend = sch.dendrogram(Z, no_plot=True)
                row_order = np.array(dend["leaves"])
            if cfg.show_col_dendrogram and n_cols > 1:
                Z    = sch.linkage(matrix.T, method=cfg.linkage_method)
                dend = sch.dendrogram(Z, no_plot=True)
                col_order = np.array(dend["leaves"])

        matrix = matrix[np.ix_(row_order, col_order)]
        mask   = mask  [np.ix_(row_order, col_order)]
        self._display_matrix = matrix
        self._display_mask   = mask
        self.row_labels = [self.row_labels[i] for i in row_order]
        self.col_labels = [self.col_labels[j] for j in col_order]

        # ---- Geometry constants ------------------------------------
        step   = cfg.cell_size + cfg.cell_gap
        ox, oy, oz = cfg.origin
        max_h  = 0.0 if cfg.flat_mode else cfg.max_height
        vmin   = self._cmap.vmin
        vmax   = self._cmap.vmax
        vspan  = vmax - vmin if vmax > vmin else 1.0

        # ---- Floor grid --------------------------------------------
        if cfg.show_grid:
            grid = _FloorGrid(
                x_range=(ox - cfg.cell_gap / 2,
                          ox + n_cols * step - cfg.cell_gap / 2),
                z_range=(oz - cfg.cell_gap / 2,
                          oz + n_rows * step - cfg.cell_gap / 2),
                n_x=n_cols, n_z=n_rows,
                color=cfg.grid_color,
                opacity=cfg.grid_opacity,
            )
            self.add(grid)
            self.floor_grid = grid

        # ---- Cells -------------------------------------------------
        # cells[i][j] = _HeatCell or _MaskedCell for row i, col j
        self.cells: list[list[_HeatCell | _MaskedCell]] = []
        self._value_labels: VGroup = VGroup()

        for i in range(n_rows):
            row_cells = []
            for j in range(n_cols):
                x0 = ox + j * step
                z0 = oz + i * step
                v  = float(matrix[i, j])

                if mask[i, j]:
                    cell = _MaskedCell(x0=x0, z0=z0, size=cfg.cell_size, cfg=cfg)
                    self.add(cell)
                    row_cells.append(cell)
                    continue

                t         = self._cmap(v)
                top_color = _palette_color(t, self._stops)
                height    = max_h * t if vmin >= 0 else max_h * abs(t - 0.5) * 2

                if cfg.flat_mode:
                    height = 0.0

                cell = _HeatCell(
                    x0=x0, z0=z0,
                    height=height,
                    size=cfg.cell_size,
                    top_color=top_color,
                    cfg=cfg,
                )
                self.add(cell)
                row_cells.append(cell)

                # Value annotation
                if cfg.show_values:
                    text_color = _contrast_text_color(top_color)
                    fmt_val    = cfg.value_format.format(v)
                    lbl        = Text(fmt_val,
                                      color=text_color,
                                      font_size=cfg.value_font_size)
                    tc = cell.top_center + np.array([0, cfg.value_z_offset, 0])
                    lbl.move_to(tc)
                    self._value_labels.add(lbl)

            self.cells.append(row_cells)

        if cfg.show_values:
            self.add(self._value_labels)

        # ---- Row / col labels --------------------------------------
        self._label_group = VGroup()
        if cfg.show_row_labels:
            for i, lbl_text in enumerate(self.row_labels):
                zc = oz + i * step + cfg.cell_size / 2
                lbl = Text(lbl_text, color=cfg.grid_color,
                           font_size=cfg.label_font_size)
                lbl.move_to(np.array([ox - 0.35, 0.04, zc]))
                self._label_group.add(lbl)

        if cfg.show_col_labels:
            for j, lbl_text in enumerate(self.col_labels):
                xc = ox + j * step + cfg.cell_size / 2
                lbl = Text(lbl_text, color=cfg.grid_color,
                           font_size=cfg.label_font_size)
                if cfg.rotate_col_labels:
                    lbl.rotate(45 * DEGREES, axis=np.array([0, 0, 1]))
                lbl.move_to(np.array([xc, 0.04, oz - 0.35]))
                self._label_group.add(lbl)

        self.add(self._label_group)

        # ---- Marginal bars -----------------------------------------
        if cfg.show_marginal_rows:
            row_means = np.nanmean(matrix, axis=1)
            z_positions = np.array([
                oz + i * step + cfg.cell_size / 2 for i in range(n_rows)
            ])
            x_base = ox + n_cols * step + cfg.cell_gap * 2
            self.marginal_rows = _MarginalBars(
                values=row_means,
                positions=z_positions,
                orientation="row",
                base_coord=x_base,
                cell_size=cfg.cell_size,
                bar_max=cfg.marginal_bar_max,
                palette_stops=self._stops,
                vmin=vmin, vmax=vmax,
            )
            self.add(self.marginal_rows)

        if cfg.show_marginal_cols:
            col_means = np.nanmean(matrix, axis=0)
            x_positions = np.array([
                ox + j * step + cfg.cell_size / 2 for j in range(n_cols)
            ])
            y_base = max_h + cfg.cell_gap * 2
            self.marginal_cols = _MarginalBars(
                values=col_means,
                positions=x_positions,
                orientation="col",
                base_coord=y_base,
                cell_size=cfg.cell_size,
                bar_max=cfg.marginal_bar_max,
                palette_stops=self._stops,
                vmin=vmin, vmax=vmax,
            )
            self.add(self.marginal_cols)

        # ---- Dendrograms -------------------------------------------
        x_positions = np.array([
            ox + j * step + cfg.cell_size / 2 for j in range(n_cols)
        ])
        z_positions = np.array([
            oz + i * step + cfg.cell_size / 2 for i in range(n_rows)
        ])

        if cfg.show_col_dendrogram and n_cols > 1:
            Z_col = sch.linkage(matrix.T, method=cfg.linkage_method)
            self.col_dendro = _Dendrogram(
                linkage_matrix=Z_col,
                n_leaves=n_cols,
                positions=x_positions,
                axis="col",
                origin_y=max_h + cfg.cell_gap,
                clearance=cfg.dendro_clearance,
                extent=cfg.dendro_extent,
                color=cfg.dendro_color,
                stroke_width=cfg.dendro_stroke_width,
            )
            self.add(self.col_dendro)

        if cfg.show_row_dendrogram and n_rows > 1:
            Z_row = sch.linkage(matrix, method=cfg.linkage_method)
            self.row_dendro = _Dendrogram(
                linkage_matrix=Z_row,
                n_leaves=n_rows,
                positions=z_positions,
                axis="row",
                origin_y=ox - cfg.cell_gap,
                clearance=cfg.dendro_clearance,
                extent=cfg.dendro_extent,
                color=cfg.dendro_color,
                stroke_width=cfg.dendro_stroke_width,
            )
            self.add(self.row_dendro)

        # ---- Color bar ---------------------------------------------
        if cfg.show_colorbar:
            matrix_width  = n_cols * step
            matrix_height = n_rows * step
            cb_height = cfg.colorbar_height if cfg.colorbar_height > 0 \
                        else matrix_height
            cb_x  = ox + matrix_width + 0.50
            cb_z  = oz

            self.colorbar = _ColorBar(
                x_pos    = cb_x,
                y_bottom = 0.0,
                height   = cb_height,
                width    = cfg.colorbar_width,
                stops    = self._stops,
                vmin     = vmin,
                vmax     = vmax,
                n_ticks  = cfg.colorbar_n_ticks,
                label    = cfg.colorbar_label,
                font_size= cfg.label_font_size - 2,
                z_pos    = cb_z,
            )
            self.add(self.colorbar)

        # ---- Store geometry metadata for animation -----------------
        self._step     = step
        self._ox       = ox
        self._oz       = oz
        self._max_h    = max_h
        self._vmin     = vmin
        self._vmax     = vmax
        self._vspan    = vspan

    # ------------------------------------------------------------------
    # Highlight helpers
    # ------------------------------------------------------------------

    def _cell_rect(
        self,
        row_lo: int, row_hi: int,
        col_lo: int, col_hi: int,
        color:  ManimColor,
        lift:   float = 0.04,
    ) -> _HighlightRect:
        """Build a highlight rectangle covering the given row/col range."""
        step  = self._step
        ox    = self._ox
        oz    = self._oz
        max_h = self._max_h
        x0 = ox + col_lo * step
        x1 = ox + col_hi * step + self.cfg.cell_size
        z0 = oz + row_lo * step
        z1 = oz + row_hi * step + self.cfg.cell_size
        return _HighlightRect(
            x0=x0, z0=z0, x1=x1, z1=z1,
            y_top=max_h,
            color=color, lift=lift,
        )

    def highlight_row(
        self,
        row_index: int,
        color: ManimColor = ManimColor("#FFD600"),
    ) -> _HighlightRect:
        """Return a highlight rectangle outlining an entire row."""
        return self._cell_rect(
            row_lo=row_index, row_hi=row_index,
            col_lo=0, col_hi=self.n_cols - 1,
            color=color,
        )

    def highlight_col(
        self,
        col_index: int,
        color: ManimColor = ManimColor("#00E5FF"),
    ) -> _HighlightRect:
        """Return a highlight rectangle outlining an entire column."""
        return self._cell_rect(
            row_lo=0, row_hi=self.n_rows - 1,
            col_lo=col_index, col_hi=col_index,
            color=color,
        )

    def highlight_cell(
        self,
        row_index: int,
        col_index: int,
        color: ManimColor = ManimColor("#FF6D00"),
    ) -> _HighlightRect:
        """Return a highlight rectangle outlining a single cell."""
        return self._cell_rect(
            row_lo=row_index, row_hi=row_index,
            col_lo=col_index, col_hi=col_index,
            color=color,
        )

    # ------------------------------------------------------------------
    # Animation API
    # ------------------------------------------------------------------

    def animate_grow(
        self,
        lag_ratio: float = 0.025,
        run_time:  float = 3.0,
    ) -> LaggedStart:
        """Rise all cells from the floor, sweeping left-to-right row by row.

        Masked cells are excluded; they simply fade in.  Data cells
        grow upward from zero height to their final height via
        ``GrowFromPoint``::

            self.play(hm.animate_grow())
        """
        anims = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                cell = self.cells[i][j]
                if self._display_mask[i, j]:
                    anims.append(FadeIn(cell, run_time=run_time * 0.3))
                else:
                    # Grow from the cell's floor centre
                    floor_pt = np.array([
                        self._ox + j * self._step + self.cfg.cell_size / 2,
                        0.0,
                        self._oz + i * self._step + self.cfg.cell_size / 2,
                    ])
                    anims.append(
                        GrowFromPoint(cell, point=floor_pt,
                                      run_time=run_time * 0.55)
                    )
        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)

    def animate_sweep_row(
        self,
        row_index: int,
        run_time:  float = 1.5,
        lag_ratio: float = 0.12,
    ) -> LaggedStart:
        """Grow a single row of cells from left to right.

        Parameters
        ----------
        row_index : int
            0-based row index.
        """
        row   = self.cells[row_index]
        anims = []
        for j, cell in enumerate(row):
            floor_pt = np.array([
                self._ox + j * self._step + self.cfg.cell_size / 2,
                0.0,
                self._oz + row_index * self._step + self.cfg.cell_size / 2,
            ])
            if self._display_mask[row_index, j]:
                anims.append(FadeIn(cell, run_time=run_time * 0.4))
            else:
                anims.append(GrowFromPoint(cell, point=floor_pt,
                                           run_time=run_time * 0.6))
        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)

    def animate_sweep_col(
        self,
        col_index: int,
        run_time:  float = 1.5,
        lag_ratio: float = 0.12,
    ) -> LaggedStart:
        """Grow a single column of cells from bottom to top (row 0 first).

        Parameters
        ----------
        col_index : int
            0-based column index.
        """
        anims = []
        for i in range(self.n_rows):
            cell     = self.cells[i][col_index]
            floor_pt = np.array([
                self._ox + col_index * self._step + self.cfg.cell_size / 2,
                0.0,
                self._oz + i * self._step + self.cfg.cell_size / 2,
            ])
            if self._display_mask[i, col_index]:
                anims.append(FadeIn(cell, run_time=run_time * 0.4))
            else:
                anims.append(GrowFromPoint(cell, point=floor_pt,
                                           run_time=run_time * 0.6))
        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)

    def animate_highlight_row(
        self,
        row_index: int,
        color:     ManimColor = ManimColor("#FFD600"),
        run_time:  float      = 0.5,
    ) -> Create:
        """Draw a highlight border across an entire row.

        Add the returned object to the scene first, then Create it::

            rect = hm.highlight_row(2)
            hm.add(rect)
            self.play(hm.animate_highlight_row(2))
        """
        rect = self.highlight_row(row_index, color=color)
        self.add(rect)
        return Create(rect, run_time=run_time)

    def animate_highlight_col(
        self,
        col_index: int,
        color:     ManimColor = ManimColor("#00E5FF"),
        run_time:  float      = 0.5,
    ) -> Create:
        """Draw a highlight border down an entire column."""
        rect = self.highlight_col(col_index, color=color)
        self.add(rect)
        return Create(rect, run_time=run_time)

    def animate_highlight_cell(
        self,
        row_index:    int,
        col_index:    int,
        color:        ManimColor = ManimColor("#FF6D00"),
        scale_factor: float      = 1.20,
        run_time:     float      = 0.50,
    ) -> Succession:
        """Flash and scale a single cell, then restore.

        A highlight border is also briefly created and removed.
        """
        cell     = self.cells[row_index][col_index]
        rect     = self.highlight_cell(row_index, col_index, color=color)
        self.add(rect)

        scale_up   = AnimationGroup(
            cell.animate(run_time=run_time / 2).scale(scale_factor),
            Create(rect, run_time=run_time / 2),
        )
        scale_down = AnimationGroup(
            cell.animate(run_time=run_time / 2).scale(1 / scale_factor),
            FadeOut(rect, run_time=run_time / 2),
        )
        return Succession(scale_up, scale_down)

    def animate_morph_values(
        self,
        new_matrix: np.ndarray,
        run_time:   float = 2.0,
    ) -> AnimationGroup:
        """Interpolate every cell to values from a new same-shaped matrix.

        Heights, colors, and value labels all transform together.
        Useful for showing how a heatmap changes over time (e.g. a
        confusion matrix evolving during training)::

            self.play(hm.animate_morph_values(new_matrix))

        Parameters
        ----------
        new_matrix : np.ndarray
            Must have the same shape as the original matrix.
        """
        new_matrix = np.asarray(new_matrix, dtype=float)
        if new_matrix.shape != (self.n_rows, self.n_cols):
            raise ValueError(
                f"new_matrix shape {new_matrix.shape} must match "
                f"original {(self.n_rows, self.n_cols)}"
            )
        new_cfg = HeatMapConfig(**{
            **self.cfg.__dict__,
            "vmin": float(new_matrix.min()),
            "vmax": float(new_matrix.max()),
        })
        new_cmap   = _ColorMapper(new_matrix, new_cfg, self._display_mask)
        new_stops  = self._stops

        anims = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self._display_mask[i, j]:
                    continue
                cell  = self.cells[i][j]
                v_new = float(new_matrix[i, j])
                t_new = new_cmap(v_new)
                new_top_color  = _palette_color(t_new, new_stops)
                new_front_color = _darken(new_top_color, self.cfg.face_darken_side)
                new_right_color = _darken(new_top_color, self.cfg.face_darken_right)
                new_h = self._max_h * t_new if self._vmin >= 0 \
                        else self._max_h * abs(t_new - 0.5) * 2
                if self.cfg.flat_mode:
                    new_h = 0.0

                # Rebuild target cell geometry
                target = _HeatCell(
                    x0=self._ox + j * self._step,
                    z0=self._oz + i * self._step,
                    height=new_h,
                    size=self.cfg.cell_size,
                    top_color=new_top_color,
                    cfg=self.cfg,
                )
                anims.append(Transform(cell, target, run_time=run_time))

        return AnimationGroup(*anims)

    def animate_palette_morph(
        self,
        new_palette: str | list[ManimColor],
        run_time:    float = 1.5,
    ) -> AnimationGroup:
        """Recolor all cells by morphing to a different palette.

        Only colors change; heights remain the same::

            self.play(hm.animate_palette_morph("hot"))

        Parameters
        ----------
        new_palette : str | list[ManimColor]
            Named palette or custom stop list.
        """
        if isinstance(new_palette, str):
            new_stops = PALETTES[new_palette]
        else:
            new_stops = list(new_palette)

        anims = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self._display_mask[i, j]:
                    continue
                cell  = self.cells[i][j]
                v     = float(self._display_matrix[i, j])
                t     = self._cmap(v)
                new_tc = _palette_color(t, new_stops)
                new_fc = _darken(new_tc, self.cfg.face_darken_side)
                new_rc = _darken(new_tc, self.cfg.face_darken_right)

                anims.append(AnimationGroup(
                    cell.top_face.animate(run_time=run_time)
                        .set_fill(color=new_tc),
                    cell.front_face.animate(run_time=run_time)
                        .set_fill(color=new_fc),
                    cell.right_face.animate(run_time=run_time)
                        .set_fill(color=new_rc),
                ))
        return AnimationGroup(*anims)

    def animate_reveal_colorbar(
        self,
        run_time: float = 1.2,
    ) -> FadeIn:
        """Fade the colorbar gradient in from transparent."""
        if not hasattr(self, "colorbar"):
            return FadeIn(VGroup(), run_time=0.1)
        return FadeIn(self.colorbar, run_time=run_time)

    def animate_reveal_dendro(
        self,
        run_time:  float = 2.0,
        lag_ratio: float = 0.06,
    ) -> LaggedStart:
        """Trace dendrogram lines outward from leaves to root.

        Works for both row and column dendrograms if both are present.
        """
        lines = []
        if hasattr(self, "col_dendro"):
            lines.extend(self.col_dendro.line_objects)
        if hasattr(self, "row_dendro"):
            lines.extend(self.row_dendro.line_objects)

        if not lines:
            return LaggedStart(FadeIn(VGroup(), run_time=0.1),
                               lag_ratio=0, run_time=0.1)
        return LaggedStart(
            *[Create(ln, run_time=run_time * 0.4) for ln in lines],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_correlation(
        cls,
        data:       np.ndarray,
        col_labels: list[str] | None = None,
        mask_upper: bool = True,
        config:     HeatMapConfig | None = None,
    ) -> "HeatMap3D":
        """Build a correlation matrix heatmap from raw data.

        Computes the Pearson correlation matrix, sets the palette to
        ``"diverging"``, centers the color map at 0, and optionally masks
        the upper triangle.

        Parameters
        ----------
        data : (n_samples, n_features) ndarray
            Raw observations; correlation computed along axis 0.
        col_labels : list[str], optional
            Feature names; used for both row and column labels.
        mask_upper : bool
            If True, mask the upper triangle (excluding diagonal) so only
            the lower triangle is displayed.
        config : HeatMapConfig, optional
            Base config to inherit settings from.  ``palette``, ``vmin``,
            ``vmax``, and ``vcenter`` are always overridden.
        """
        data   = np.asarray(data, dtype=float)
        corr   = np.corrcoef(data.T)
        n      = corr.shape[0]
        labels = col_labels or [f"X{i}" for i in range(n)]

        mask = np.zeros((n, n), dtype=bool)
        if mask_upper:
            mask[np.triu_indices(n, k=1)] = True

        base = config.__dict__.copy() if config else {}
        base.update(
            palette="diverging",
            vmin=-1.0,
            vmax= 1.0,
            vcenter=0.0,
            mask=mask,
        )
        cfg = HeatMapConfig(**base)
        return cls(
            matrix=corr,
            row_labels=labels,
            col_labels=labels,
            config=cfg,
        )

    @classmethod
    def from_confusion(
        cls,
        y_true:     Sequence[int],
        y_pred:     Sequence[int],
        class_names: list[str] | None = None,
        normalize:   bool = False,
        config:      HeatMapConfig | None = None,
    ) -> "HeatMap3D":
        """Build a confusion matrix heatmap from classification results.

        Parameters
        ----------
        y_true, y_pred : sequence of int
            Ground-truth and predicted labels.
        class_names : list[str], optional
            Class name strings for the axes.
        normalize : bool
            If True, normalize rows to sum to 1 (show recall per class).
        """
        from sklearn.metrics import confusion_matrix  # optional dep
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm       = cm / np.where(row_sums > 0, row_sums, 1)

        n      = cm.shape[0]
        labels = class_names or [str(k) for k in range(n)]

        base = config.__dict__.copy() if config else {}
        base.setdefault("palette", "sequential")
        cfg  = HeatMapConfig(**base)
        return cls(matrix=cm.astype(float),
                   row_labels=labels, col_labels=labels,
                   config=cfg)

    @classmethod
    def random_demo(
        cls,
        n_rows:  int = 6,
        n_cols:  int = 8,
        palette: str = "viridis",
        seed:    int = 0,
        config:  HeatMapConfig | None = None,
    ) -> "HeatMap3D":
        """Create a demo heatmap with random values for quick testing."""
        rng    = np.random.default_rng(seed)
        matrix = rng.uniform(0, 1, (n_rows, n_cols))
        base   = config.__dict__.copy() if config else {}
        base["palette"] = palette
        cfg    = HeatMapConfig(**base)
        labels_r = [f"Row {i}" for i in range(n_rows)]
        labels_c = [f"Col {j}" for j in range(n_cols)]
        return cls(matrix=matrix, row_labels=labels_r,
                   col_labels=labels_c, config=cfg)