"""
manim_stats/charts/violin_plot3d.py
=====================================
A production-quality 3D violin plot for Manim with:

  Violin geometry
  ---------------
  - Surface of revolution: KDE profile rotated around the vertical axis via
    Surface, producing a smooth 3D shape — not a flat extruded polygon
  - Three distinct surface zones with independent coloring:
      • Lower tail   (below Q1)
      • IQR body     (Q1 – Q3, the "thick" waist)
      • Upper tail   (above Q3)
  - Half-violin mode: show only the right or left half-surface for
    side-by-side mirrored pair comparisons (split violin)
  - Configurable KDE bandwidth: "scott", "silverman", or an explicit float

  Inner box-plot layer
  --------------------
  - IQR prism: a 3D rectangular prism (front/side/top shading) spanning Q1→Q3
    centered inside the violin
  - Median sphere: Sphere glyph at the median, distinct color from the body
  - Mean marker: diamond-shaped Polygon at the mean value
  - Whisker lines: Line3D extending from IQR box to the Tukey fences (1.5×IQR)
  - Outlier dots: Dot3D glyphs for values beyond the whisker fences

  Data jitter strip
  -----------------
  - Raw data points plotted as Dot3D inside or beside the violin body,
    jittered along the depth (Z) axis to avoid overplotting ("beeswarm lite")
  - Jitter width auto-scales with the local KDE density at each point's Y value

  Multi-group layout
  ------------------
  - N violins arranged side by side along X with configurable spacing
  - Per-group labels below the X axis
  - Shared Y axis with ticks and numeric labels
  - Optional X-axis category labels with rotation

  Significance brackets
  ---------------------
  - Automatic pairwise significance testing (Mann-Whitney U or t-test)
  - Brackets drawn above violin pairs with p-value and star notation
  - Bracket stacking: multiple comparisons stacked vertically without overlap

  Visual groundwork
  -----------------
  - Floor grid plane at y=0 (semi-transparent)
  - Optional back-wall reference grid
  - Y-axis ticks with numeric labels
  - Violin group labels on X axis

  Animation suite
  ---------------
  - animate_grow(i)            : single violin surface expands from median upward and downward
  - animate_grow_all()         : all violins grow with a staggered lag
  - animate_reveal_boxplot(i)  : IQR box, whiskers, median, mean fade/grow in sequence
  - animate_drop_jitter(i)     : jitter dots rain down into position
  - animate_morph_bandwidth(i) : violin surface morphs as bandwidth changes (ValueTracker)
  - animate_compare_groups()   : significance brackets draw in one by one
  - animate_highlight_group(i) : flash-scale one entire violin
  - animate_split_reveal()     : for half-violin pairs, each half grows from the center line
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

from manim import (
    WHITE, BLACK, BLUE, BLUE_E, BLUE_B,
    YELLOW, RED, GREEN, ORANGE, PURPLE,
    GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    UP, DOWN, LEFT, RIGHT, OUT, IN,
    PI, TAU,
    VGroup, VMobject, Mobject,
    Arrow3D, Line3D, Sphere, Dot3D, Polygon,
    Text, MathTex,
    Surface,
    FadeIn, FadeOut, GrowFromCenter, GrowFromPoint, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    ValueTracker,
    interpolate_color,
    ManimColor,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Palette & shading constants
# ---------------------------------------------------------------------------

# Default series palette — up to 8 groups
VIOLIN_PALETTE: list[ManimColor] = [
    ManimColor("#1E88E5"),   # blue
    ManimColor("#E53935"),   # red
    ManimColor("#43A047"),   # green
    ManimColor("#FB8C00"),   # orange
    ManimColor("#8E24AA"),   # purple
    ManimColor("#00ACC1"),   # cyan
    ManimColor("#F4511E"),   # deep orange
    ManimColor("#6D4C41"),   # brown
]

MEDIAN_COLOR  = ManimColor("#FFFFFF")
MEAN_COLOR    = ManimColor("#FFD600")
WHISKER_COLOR = ManimColor("#90A4AE")
OUTLIER_COLOR = ManimColor("#FF5252")
BRACKET_COLOR = ManimColor("#B0BEC5")

# IQR zone tint: IQR body is lightened relative to the tail color
IQR_LIGHTEN   = 0.25
TAIL_DARKEN   = 0.30

FACE_DARKEN   = 0.40
FACE_LIGHTEN  = 0.18


def _darken(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lighten(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)


# ---------------------------------------------------------------------------
# KDE helper
# ---------------------------------------------------------------------------

def _compute_kde(
    data:      np.ndarray,
    bandwidth: float | str,
    n_points:  int = 256,
    padding:   float = 0.05,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Compute a kernel density estimate over a fine grid.

    Returns
    -------
    ys : (n_points,) grid of Y values in data space
    ds : (n_points,) corresponding KDE density values
    kde : the fitted scipy gaussian_kde object (for on-demand evaluation)
    """
    kde    = scipy_stats.gaussian_kde(data, bw_method=bandwidth)
    lo     = data.min()
    hi     = data.max()
    margin = (hi - lo) * padding
    ys     = np.linspace(lo - margin, hi + margin, n_points)
    ds     = kde(ys)
    return ys, ds, kde


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ViolinConfig:
    """All visual and statistical options for ViolinPlot3D.

    Violin geometry
    ---------------
    bandwidth : float | str
        KDE bandwidth passed to ``scipy.stats.gaussian_kde``.
        Use ``"scott"`` (default), ``"silverman"``, or an explicit float.
    surface_resolution_u : int
        Number of meridian segments (angular, around the revolution axis).
        Higher = smoother circular cross-sections.
    surface_resolution_v : int
        Number of parallel segments (vertical, along the KDE profile).
        Higher = smoother profile curve.
    violin_max_radius : float
        Maximum rendered radius of the widest violin cross-section (Manim units).
        All violins are normalised to this width.
    normalize_across_groups : bool
        If True, all violin widths are normalised to the global max KDE density
        across all groups (so widths are comparable between groups).
        If False, each violin uses its own max (all look equally wide).
    half_mode : str | None
        ``None``   — full rotation (default circular violin).
        ``"right"`` — right half only (u ∈ [0, π]).
        ``"left"``  — left half only (u ∈ [π, 2π]).
        Used to create mirrored split violins.
    zone_coloring : bool
        If True, apply three distinct colors: tail / IQR body / upper tail.
        If False, the entire surface uses one uniform base color.
    iqr_lighten : float
        How much lighter the IQR zone surface is vs the tail zones.
    tail_darken : float
        How much darker the tail zones are vs the base color.
    surface_opacity : float
        Fill opacity of the violin surface [0, 1].
    surface_stroke_width : float
        Mesh line stroke width on the surface.  0 = no mesh lines.
    surface_stroke_opacity : float
        Opacity of the mesh stroke lines.

    Inner box plot
    --------------
    show_box : bool
        Render the inner IQR box prism.
    box_width : float
        Width of the IQR box as a fraction of violin_max_radius.
    box_depth : float
        Depth (Z extent) of the IQR box prism.
    box_opacity : float
        Fill opacity of the IQR box faces.
    show_median : bool
        Render a sphere at the median value.
    median_radius : float
        Radius of the median sphere.
    show_mean : bool
        Render a diamond polygon at the mean value.
    mean_size : float
        Half-diagonal of the mean diamond.
    show_whiskers : bool
        Render Tukey fence whisker lines.
    whisker_stroke_width : float
        Stroke width of whisker lines.
    whisker_fence : float
        IQR multiplier for Tukey fences (default 1.5).
    show_outliers : bool
        Render outlier dots beyond the whisker fences.
    outlier_radius : float
        Radius of outlier dot glyphs.

    Jitter strip
    ------------
    show_jitter : bool
        Overlay raw data points jittered along Z inside the violin.
    jitter_max_z : float
        Maximum Z offset for jittered points.
    jitter_radius : float
        Radius of jitter dot glyphs.
    jitter_opacity : float
        Opacity of jitter dots.
    jitter_density_scale : bool
        If True, jitter width at each point scales with local KDE density,
        so denser regions spread wider.

    Significance brackets
    ---------------------
    show_significance : bool
        Compute pairwise tests and draw brackets.
    sig_test : str
        ``"mannwhitney"`` (default, non-parametric) or ``"ttest"`` (parametric).
    sig_alpha : float
        Significance threshold; pairs with p > alpha are labeled "n.s.".
    bracket_lift : float
        Base vertical clearance above the tallest violin before the first bracket.
    bracket_step : float
        Vertical spacing between stacked brackets.
    bracket_stroke_width : float
        Stroke width of bracket lines.

    Multi-group layout
    ------------------
    group_spacing : float
        Horizontal distance between violin centers.
    show_group_labels : bool
        Render group name labels below the X axis.
    group_label_font_size : int
        Font size of group labels.
    show_floor_grid : bool
        Render a semi-transparent floor grid at y=0.
    floor_grid_lines : int
        Number of lines in each direction on the floor grid.
    floor_opacity : float
        Opacity of floor grid lines.
    show_back_grid : bool
        Render a semi-transparent back wall grid.
    back_grid_opacity : float
        Opacity of back wall grid lines.
    n_y_ticks : int
        Number of Y-axis tick marks.
    axes_color : ManimColor
        Color of all axis lines, ticks, and labels.
    tick_length : float
        Tick mark length.
    axes_y_label : str
        Label for the vertical (value) axis.
    label_font_size : int
        Font size for axis tick labels and the Y axis label.

    Layout
    ------
    y_scale : float
        Total rendered height in Manim units (max data range → this height).
    """

    # ---- violin geometry ----
    bandwidth:               float | str = "scott"
    surface_resolution_u:    int         = 32
    surface_resolution_v:    int         = 128
    violin_max_radius:       float       = 0.55
    normalize_across_groups: bool        = True
    half_mode:               str | None  = None
    zone_coloring:           bool        = True
    iqr_lighten:             float       = IQR_LIGHTEN
    tail_darken:             float       = TAIL_DARKEN
    surface_opacity:         float       = 0.72
    surface_stroke_width:    float       = 0.20
    surface_stroke_opacity:  float       = 0.30

    # ---- inner box plot ----
    show_box:            bool  = True
    box_width:           float = 0.08
    box_depth:           float = 0.16
    box_opacity:         float = 0.90
    show_median:         bool  = True
    median_radius:       float = 0.07
    show_mean:           bool  = True
    mean_size:           float = 0.10
    show_whiskers:       bool  = True
    whisker_stroke_width: float = 1.5
    whisker_fence:       float = 1.5
    show_outliers:       bool  = True
    outlier_radius:      float = 0.04

    # ---- jitter strip ----
    show_jitter:          bool  = False
    jitter_max_z:         float = 0.28
    jitter_radius:        float = 0.03
    jitter_opacity:       float = 0.55
    jitter_density_scale: bool  = True

    # ---- significance brackets ----
    show_significance:    bool  = False
    sig_test:             str   = "mannwhitney"
    sig_alpha:            float = 0.05
    bracket_lift:         float = 0.30
    bracket_step:         float = 0.38
    bracket_stroke_width: float = 1.2

    # ---- multi-group layout ----
    group_spacing:         float      = 1.80
    show_group_labels:     bool       = True
    group_label_font_size: int        = 22
    show_floor_grid:       bool       = True
    floor_grid_lines:      int        = 8
    floor_opacity:         float      = 0.15
    show_back_grid:        bool       = True
    back_grid_opacity:     float      = 0.08
    n_y_ticks:             int        = 6
    axes_color:            ManimColor = GRAY_B
    tick_length:           float      = 0.12
    axes_y_label:          str        = "Value"
    label_font_size:       int        = 18

    # ---- layout ----
    y_scale: float = 5.0


# ---------------------------------------------------------------------------
# Data container for one group
# ---------------------------------------------------------------------------

@dataclass
class ViolinGroup:
    """One group / category in a ViolinPlot3D.

    Parameters
    ----------
    data : array-like
        1-D observations for this group.
    label : str
        Category label displayed below the X axis.
    color : ManimColor | None
        Override the automatic palette color.
    """
    data:  np.ndarray
    label: str        = ""
    color: ManimColor | None = None

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if len(self.data) < 2:
            raise ValueError(f"ViolinGroup '{self.label}' needs at least 2 observations.")

    # ---- descriptive statistics (computed once) ----
    @property
    def q1(self)     -> float: return float(np.percentile(self.data, 25))
    @property
    def q3(self)     -> float: return float(np.percentile(self.data, 75))
    @property
    def median(self) -> float: return float(np.median(self.data))
    @property
    def mean(self)   -> float: return float(np.mean(self.data))
    @property
    def iqr(self)    -> float: return self.q3 - self.q1

    def whisker_bounds(self, fence: float = 1.5) -> tuple[float, float]:
        lo = self.q1 - fence * self.iqr
        hi = self.q3 + fence * self.iqr
        inliers = self.data[(self.data >= lo) & (self.data <= hi)]
        w_lo = float(inliers.min()) if len(inliers) else lo
        w_hi = float(inliers.max()) if len(inliers) else hi
        return w_lo, w_hi

    def outliers(self, fence: float = 1.5) -> np.ndarray:
        lo, hi = self.whisker_bounds(fence)
        return self.data[(self.data < lo) | (self.data > hi)]


# ---------------------------------------------------------------------------
# Y-axis coordinate mapper
# ---------------------------------------------------------------------------

class _YMapper:
    """Maps data-space Y values to Manim world Y coordinates.

    All violins in the scene share the same mapper so their heights are
    directly comparable.
    """

    def __init__(
        self,
        all_groups:  list[ViolinGroup],
        y_scale:     float,
        padding:     float = 0.06,
    ):
        all_data  = np.concatenate([g.data for g in all_groups])
        lo, hi    = all_data.min(), all_data.max()
        span      = hi - lo if hi > lo else 1.0
        margin    = span * padding
        self.lo   = lo - margin
        self.hi   = hi + margin
        self.span = self.hi - self.lo
        self.y_scale = y_scale

    def __call__(self, v: float | np.ndarray) -> float | np.ndarray:
        return (np.asarray(v) - self.lo) / self.span * self.y_scale

    def inv(self, world_y: float) -> float:
        """Inverse: world Y → data value."""
        return world_y / self.y_scale * self.span + self.lo


# ---------------------------------------------------------------------------
# IQR box prism (3-face shaded, same pattern as histogram3d._BarPrism)
# ---------------------------------------------------------------------------

class _IQRBox(VGroup):
    """A thin 3D prism spanning Q1 → Q3 centered on the violin's X position.

    Front face is lighter, side face darker, top face lightest — replicating
    the per-face shading used throughout this module.
    """

    def __init__(
        self,
        x_center: float,
        y_bottom: float,    # world Y of Q1
        y_top:    float,    # world Y of Q3
        width:    float,
        depth:    float,
        color:    ManimColor,
        opacity:  float = 0.90,
    ):
        super().__init__()
        hw = width / 2
        hd = depth / 2

        front_color = color
        side_color  = _darken(color,  FACE_DARKEN)
        top_color   = _lighten(color, FACE_LIGHTEN)

        # 8 corners
        AFL = np.array([x_center - hw, y_bottom, -hd])
        AFR = np.array([x_center + hw, y_bottom, -hd])
        ABL = np.array([x_center - hw, y_bottom,  hd])
        ABR = np.array([x_center + hw, y_bottom,  hd])
        TFL = np.array([x_center - hw, y_top,    -hd])
        TFR = np.array([x_center + hw, y_top,    -hd])
        TBL = np.array([x_center - hw, y_top,     hd])
        TBR = np.array([x_center + hw, y_top,     hd])

        def _face(pts: list[np.ndarray], col: ManimColor) -> Polygon:
            p = Polygon(*pts, color=col)
            p.set_fill(color=col, opacity=opacity)
            p.set_stroke(color=_darken(col, 0.50), width=0.7, opacity=0.5)
            return p

        self.add(_face([AFL, AFR, TFR, TFL], front_color))  # front
        self.add(_face([AFR, ABR, TBR, TFR], side_color))   # right side
        self.add(_face([TFL, TFR, TBR, TBL], top_color))    # top


# ---------------------------------------------------------------------------
# Mean diamond marker
# ---------------------------------------------------------------------------

class _MeanDiamond(Polygon):
    """A flat diamond glyph at the mean value, oriented in the XY plane."""

    def __init__(
        self,
        x_center: float,
        y_pos:    float,
        z_pos:    float,
        size:     float,
        color:    ManimColor = MEAN_COLOR,
    ):
        pts = [
            np.array([x_center,        y_pos + size, z_pos]),
            np.array([x_center + size, y_pos,        z_pos]),
            np.array([x_center,        y_pos - size, z_pos]),
            np.array([x_center - size, y_pos,        z_pos]),
        ]
        super().__init__(*pts, color=color)
        self.set_fill(color=color, opacity=1.0)
        self.set_stroke(color=_darken(color, 0.45), width=1.0)


# ---------------------------------------------------------------------------
# Significance bracket
# ---------------------------------------------------------------------------

class _SigBracket(VGroup):
    """An L-shaped bracket drawn between two violin centers with a p-value label.

    Parameters
    ----------
    x_left, x_right : float
        World X positions of the two violins being compared.
    y_base : float
        World Y position of the bracket's horizontal bar.
    p_value : float
        The computed p-value; controls the star / "n.s." annotation.
    color : ManimColor
    stroke_width : float
    font_size : int
    """

    def __init__(
        self,
        x_left:       float,
        x_right:      float,
        y_base:       float,
        p_value:      float,
        color:        ManimColor = BRACKET_COLOR,
        stroke_width: float      = 1.2,
        font_size:    int        = 20,
        z_pos:        float      = 0.0,
    ):
        super().__init__()
        tick_h = 0.10   # height of the vertical tick at each end

        left_tick  = Line3D(
            start=np.array([x_left,  y_base - tick_h, z_pos]),
            end  =np.array([x_left,  y_base,          z_pos]),
            color=color, stroke_width=stroke_width,
        )
        horiz      = Line3D(
            start=np.array([x_left,  y_base, z_pos]),
            end  =np.array([x_right, y_base, z_pos]),
            color=color, stroke_width=stroke_width,
        )
        right_tick = Line3D(
            start=np.array([x_right, y_base,          z_pos]),
            end  =np.array([x_right, y_base - tick_h, z_pos]),
            color=color, stroke_width=stroke_width,
        )
        self.add(left_tick, horiz, right_tick)

        # Star annotation
        if p_value < 0.001:
            stars, lcolor = "***", ManimColor("#00E676")
        elif p_value < 0.01:
            stars, lcolor = "**",  ManimColor("#FFD600")
        elif p_value < 0.05:
            stars, lcolor = "*",   ManimColor("#FF9100")
        else:
            stars, lcolor = "n.s.", BRACKET_COLOR

        label = Text(stars, color=lcolor, font_size=font_size)
        label.move_to(np.array([(x_left + x_right) / 2, y_base + 0.16, z_pos]))
        self.add(label)


# ---------------------------------------------------------------------------
# Violin surface (surface of revolution)
# ---------------------------------------------------------------------------

class _ViolinSurface(VGroup):
    """A single violin rendered as a Surface of revolution.

    The KDE profile is rotated around the vertical (Y) axis to produce
    a true 3D solid of revolution. Three surface zones (lower tail, IQR
    body, upper tail) are rendered as separate surface objects so they
    can be colored independently and animated separately.

    Parameters
    ----------
    x_center : float
        World X position of this violin's center axis.
    ys_grid : (M,) ndarray
        Fine Y grid in world units at which the KDE is evaluated.
    radii : (M,) ndarray
        World-space radius at each Y position (already normalised).
    y_q1, y_q3 : float
        World Y positions of Q1 and Q3 — used to split into three zones.
    color : ManimColor
        Base color; zones are derived by lightening/darkening.
    cfg : ViolinConfig
    """

    def __init__(
        self,
        x_center: float,
        ys_grid:  np.ndarray,
        radii:    np.ndarray,
        y_q1:     float,
        y_q3:     float,
        color:    ManimColor,
        cfg:      ViolinConfig,
    ):
        super().__init__()
        self.x_center = x_center
        self.cfg      = cfg
        self._ys      = ys_grid
        self._radii   = radii
        self._color   = color

        # Angular range depends on half_mode
        if cfg.half_mode is None:
            u_lo, u_hi = 0.0, 1.0          # full revolution
        elif cfg.half_mode == "right":
            u_lo, u_hi = 0.0, 0.5          # front half (θ ∈ [0, π])
        else:                               # "left"
            u_lo, u_hi = 0.5, 1.0          # back half  (θ ∈ [π, 2π])

        n = len(ys_grid)

        def _interp_radius(y_world: float) -> float:
            """Linear interpolation into the radius array at a given Y."""
            idx = np.searchsorted(ys_grid, y_world)
            idx = int(np.clip(idx, 1, n - 1))
            t   = (y_world - ys_grid[idx - 1]) / max(ys_grid[idx] - ys_grid[idx - 1], 1e-12)
            return float(radii[idx - 1] * (1 - t) + radii[idx] * t)

        def _make_surface(
            v_lo:   float,
            v_hi:   float,
            surf_color: ManimColor,
        ) -> Surface:
            """Build one zone of the surface of revolution.

            u ∈ [u_lo, u_hi] → angle θ = u * TAU  (angular revolution)
            v ∈ [v_lo, v_hi] → Y position along the KDE profile
            """
            y_lo_w = ys_grid[0]
            y_hi_w = ys_grid[-1]

            def surf_fn(u: float, v: float) -> np.ndarray:
                theta  = u * TAU
                y_world = y_lo_w + v * (y_hi_w - y_lo_w)
                r       = _interp_radius(y_world)
                return np.array([
                    x_center + r * np.cos(theta),
                    y_world,
                    r * np.sin(theta),
                ])

            surf = Surface(
                surf_fn,
                u_range=[u_lo, u_hi],
                v_range=[v_lo, v_hi],
                resolution=(cfg.surface_resolution_u, cfg.surface_resolution_v),
            )
            surf.set_style(
                fill_color=surf_color,
                fill_opacity=cfg.surface_opacity,
                stroke_color=_darken(surf_color, 0.35),
                stroke_width=cfg.surface_stroke_width,
                stroke_opacity=cfg.surface_stroke_opacity,
            )
            return surf

        if cfg.zone_coloring:
            # Map Y world positions to v parameter values
            y_total_lo = ys_grid[0]
            y_total_hi = ys_grid[-1]
            y_span     = y_total_hi - y_total_lo

            v_q1 = (y_q1 - y_total_lo) / y_span
            v_q3 = (y_q3 - y_total_lo) / y_span

            # Clamp to valid range
            v_q1 = float(np.clip(v_q1, 0.02, 0.98))
            v_q3 = float(np.clip(v_q3, 0.02, 0.98))

            if v_q1 >= v_q3:
                # Degenerate IQR — use single surface
                self.surface = _make_surface(0.0, 1.0, color)
                self.add(self.surface)
            else:
                tail_color = _darken(color,  cfg.tail_darken)
                iqr_color  = _lighten(color, cfg.iqr_lighten)

                self.lower_tail = _make_surface(0.0,  v_q1, tail_color)
                self.iqr_body   = _make_surface(v_q1, v_q3, iqr_color)
                self.upper_tail = _make_surface(v_q3, 1.0,  tail_color)
                self.add(self.lower_tail, self.iqr_body, self.upper_tail)
        else:
            self.surface = _make_surface(0.0, 1.0, color)
            self.add(self.surface)

    def get_zone_surfaces(self) -> list:
        """Return individual zone surfaces in bottom-to-top order."""
        if hasattr(self, "lower_tail"):
            return [self.lower_tail, self.iqr_body, self.upper_tail]
        return [self.surface]


# ---------------------------------------------------------------------------
# Single violin assembly (surface + boxplot + jitter)
# ---------------------------------------------------------------------------

class _SingleViolin(VGroup):
    """All visual layers for one group's violin.

    Separates each layer into named sub-groups so they can be animated
    independently via the public API.
    """

    def __init__(
        self,
        group:    ViolinGroup,
        x_center: float,
        mapper:   _YMapper,
        global_max_density: float,
        cfg:      ViolinConfig,
        rng:      np.random.Generator,
    ):
        super().__init__()
        self.group    = group
        self.x_center = x_center
        self.mapper   = mapper
        self.cfg      = cfg

        # ---- KDE ----
        ys_data, ds, kde = _compute_kde(
            group.data, bandwidth=cfg.bandwidth, n_points=cfg.surface_resolution_v
        )
        ys_world = mapper(ys_data)
        # Normalise radius: either to this group's max or global max
        if cfg.normalize_across_groups:
            norm_denom = global_max_density
        else:
            norm_denom = ds.max()
        if norm_denom < 1e-12:
            norm_denom = 1.0
        radii = (ds / norm_denom) * cfg.violin_max_radius

        # Key statistics in world space
        y_q1     = mapper(group.q1)
        y_q3     = mapper(group.q3)
        y_median = mapper(group.median)
        y_mean   = mapper(group.mean)
        w_lo, w_hi = group.whisker_bounds(cfg.whisker_fence)
        y_wlo    = mapper(w_lo)
        y_whi    = mapper(w_hi)

        # ---- 1. Violin surface ----
        self.surface_group = _ViolinSurface(
            x_center=x_center,
            ys_grid=ys_world,
            radii=radii,
            y_q1=y_q1,
            y_q3=y_q3,
            color=group.color,
            cfg=cfg,
        )
        self.add(self.surface_group)

        # ---- 2. Inner box plot ----
        self.boxplot_group = VGroup()

        if cfg.show_whiskers:
            # Whisker line (lower)
            wl = Line3D(
                start=np.array([x_center, y_wlo, 0]),
                end  =np.array([x_center, y_q1,  0]),
                color=WHISKER_COLOR,
                stroke_width=cfg.whisker_stroke_width,
            )
            # Whisker line (upper)
            wu = Line3D(
                start=np.array([x_center, y_q3,  0]),
                end  =np.array([x_center, y_whi, 0]),
                color=WHISKER_COLOR,
                stroke_width=cfg.whisker_stroke_width,
            )
            # Whisker caps (small horizontal ticks)
            cap_hw = cfg.box_width * 0.8
            cap_lo = Line3D(
                start=np.array([x_center - cap_hw, y_wlo, 0]),
                end  =np.array([x_center + cap_hw, y_wlo, 0]),
                color=WHISKER_COLOR, stroke_width=cfg.whisker_stroke_width,
            )
            cap_hi = Line3D(
                start=np.array([x_center - cap_hw, y_whi, 0]),
                end  =np.array([x_center + cap_hw, y_whi, 0]),
                color=WHISKER_COLOR, stroke_width=cfg.whisker_stroke_width,
            )
            self.boxplot_group.add(wl, wu, cap_lo, cap_hi)

        if cfg.show_box:
            box_color = _lighten(group.color, 0.15)
            iqr_box   = _IQRBox(
                x_center=x_center,
                y_bottom=y_q1,
                y_top   =y_q3,
                width   =cfg.box_width * 2,
                depth   =cfg.box_depth,
                color   =box_color,
                opacity =cfg.box_opacity,
            )
            self.boxplot_group.add(iqr_box)

        if cfg.show_median:
            med_sphere = Sphere(radius=cfg.median_radius, resolution=(10, 10))
            med_sphere.set_color(MEDIAN_COLOR)
            med_sphere.set_opacity(1.0)
            med_sphere.move_to(np.array([x_center, y_median, 0]))
            self.boxplot_group.add(med_sphere)

        if cfg.show_mean:
            mean_diamond = _MeanDiamond(
                x_center=x_center,
                y_pos   =y_mean,
                z_pos   =0.0,
                size    =cfg.mean_size,
                color   =MEAN_COLOR,
            )
            self.boxplot_group.add(mean_diamond)

        if cfg.show_outliers:
            for val in group.outliers(cfg.whisker_fence):
                y_out = mapper(val)
                dot   = Dot3D(
                    point=np.array([x_center, y_out, 0]),
                    radius=cfg.outlier_radius,
                    color =OUTLIER_COLOR,
                )
                dot.set_opacity(0.85)
                self.boxplot_group.add(dot)

        self.add(self.boxplot_group)

        # ---- 3. Jitter strip ----
        self.jitter_group = VGroup()
        if cfg.show_jitter:
            # Pre-sort data for consistent display
            sorted_data = np.sort(group.data)
            for val in sorted_data:
                y_w   = mapper(val)
                # Density-scaled jitter width: evaluate KDE at this point
                local_density = float(kde(val)[0])
                max_density   = float(kde(ys_data[np.argmax(ds)])[0])
                if cfg.jitter_density_scale and max_density > 0:
                    jitter_scale = (local_density / max_density)
                else:
                    jitter_scale = 1.0
                # Random Z offset within allowed range
                z_off = rng.uniform(-jitter_scale * cfg.jitter_max_z,
                                     jitter_scale * cfg.jitter_max_z)
                dot = Dot3D(
                    point=np.array([x_center, y_w, z_off]),
                    radius=cfg.jitter_radius,
                    color=_lighten(group.color, 0.30),
                )
                dot.set_opacity(cfg.jitter_opacity)
                self.jitter_group.add(dot)
            self.add(self.jitter_group)

        # Store stats for external use
        self.y_q1     = y_q1
        self.y_q3     = y_q3
        self.y_median = y_median
        self.y_mean   = y_mean
        self.y_whi    = y_whi
        self.y_top    = float(ys_world[-1])   # topmost KDE extent
        self._kde     = kde
        self._ys_data = ys_data


# ---------------------------------------------------------------------------
# Floor grid
# ---------------------------------------------------------------------------

class _FloorGrid(VGroup):
    def __init__(
        self,
        x_range:  tuple[float, float],
        z_range:  tuple[float, float],
        n_lines:  int,
        color:    ManimColor,
        opacity:  float,
    ):
        super().__init__()
        x0, x1 = x_range
        z0, z1 = z_range
        for i in range(n_lines + 1):
            t  = i / n_lines
            xv = x0 + t * (x1 - x0)
            self.add(Line3D(start=np.array([xv, 0, z0]),
                            end  =np.array([xv, 0, z1]),
                            color=color, stroke_width=0.5).set_opacity(opacity))
            zv = z0 + t * (z1 - z0)
            self.add(Line3D(start=np.array([x0, 0, zv]),
                            end  =np.array([x1, 0, zv]),
                            color=color, stroke_width=0.5).set_opacity(opacity))


# ---------------------------------------------------------------------------
# Main ViolinPlot3D class
# ---------------------------------------------------------------------------

class ViolinPlot3D(VGroup):
    """A detailed 3D violin plot for Manim statistics animations.

    Supports single or multiple groups, full surface-of-revolution geometry,
    embedded box plots, jitter strips, significance brackets, and a rich
    animation API.

    Basic usage
    -----------
    >>> import numpy as np
    >>> from manim import *
    >>> from manim_stats.charts.violin_plot3d import ViolinPlot3D, ViolinConfig, ViolinGroup
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         rng = np.random.default_rng(0)
    ...         groups = [
    ...             ViolinGroup(rng.normal(0,   1,   200), label="Control"),
    ...             ViolinGroup(rng.normal(1.5, 0.8, 200), label="Treatment A"),
    ...             ViolinGroup(rng.normal(3.0, 1.2, 200), label="Treatment B"),
    ...         ]
    ...         cfg  = ViolinConfig(show_jitter=True, show_significance=True)
    ...         plot = ViolinPlot3D(groups, config=cfg)
    ...         self.set_camera_orientation(phi=65*DEGREES, theta=-50*DEGREES)
    ...         self.play(plot.animate_grow_all())
    ...         self.play(plot.animate_reveal_boxplots())
    ...         self.play(plot.animate_compare_groups())

    Parameters
    ----------
    groups : ViolinGroup | list[ViolinGroup]
        One or more data groups.
    config : ViolinConfig, optional
        Visual configuration.  Defaults to ``ViolinConfig()``.
    seed : int, optional
        Random seed for jitter dot placement.
    """

    def __init__(
        self,
        groups: ViolinGroup | list[ViolinGroup],
        config: ViolinConfig | None = None,
        seed:   int = 42,
    ):
        super().__init__()
        self.cfg = config or ViolinConfig()

        if isinstance(groups, ViolinGroup):
            self._groups = [groups]
        else:
            self._groups = list(groups)

        # Assign palette colors
        for i, g in enumerate(self._groups):
            if g.color is None:
                g.color = VIOLIN_PALETTE[i % len(VIOLIN_PALETTE)]

        self._rng    = np.random.default_rng(seed)
        self._mapper = _YMapper(self._groups, y_scale=self.cfg.y_scale)
        self._build()

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg    = self.cfg
        mapper = self._mapper
        n      = len(self._groups)

        # Total width spanned by all violins
        total_w = (n - 1) * cfg.group_spacing
        x_left  = -total_w / 2

        # Global max KDE density (for cross-group normalisation)
        global_max_density = 0.0
        if cfg.normalize_across_groups:
            for g in self._groups:
                _, ds, _ = _compute_kde(g.data, bandwidth=cfg.bandwidth)
                global_max_density = max(global_max_density, float(ds.max()))
        if global_max_density < 1e-12:
            global_max_density = 1.0

        # ---- 1. Floor grid ------------------------------------------
        if cfg.show_floor_grid:
            half_z = cfg.violin_max_radius * 1.6
            grid   = _FloorGrid(
                x_range=(x_left - cfg.violin_max_radius * 1.5,
                          -x_left + cfg.violin_max_radius * 1.5),
                z_range=(-half_z, half_z),
                n_lines=cfg.floor_grid_lines,
                color=cfg.axes_color,
                opacity=cfg.floor_opacity,
            )
            self.add(grid)
            self.floor_grid = grid

        # ---- 2. Back wall grid --------------------------------------
        if cfg.show_back_grid:
            back = _FloorGrid(
                x_range=(x_left - cfg.violin_max_radius * 1.5,
                          -x_left + cfg.violin_max_radius * 1.5),
                z_range=(0, cfg.y_scale * 1.05),
                n_lines=cfg.floor_grid_lines,
                color=cfg.axes_color,
                opacity=cfg.back_grid_opacity,
            )
            # Rotate to be a vertical wall — reuse _FloorGrid on XY plane
            # by just building it as XY lines (swap z/y in the grid)
            # We build it as a proper XY back wall instead:
            self._back_wall = VGroup()
            bw_x0 = x_left  - cfg.violin_max_radius * 1.5
            bw_x1 = -x_left + cfg.violin_max_radius * 1.5
            bw_y0 = 0.0
            bw_y1 = cfg.y_scale * 1.05
            n_bw  = cfg.floor_grid_lines

            for i in range(n_bw + 1):
                t   = i / n_bw
                xv  = bw_x0 + t * (bw_x1 - bw_x0)
                yv  = bw_y0 + t * (bw_y1 - bw_y0)
                self._back_wall.add(
                    Line3D(start=np.array([xv, bw_y0, 0]),
                           end  =np.array([xv, bw_y1, 0]),
                           color=cfg.axes_color, stroke_width=0.4
                           ).set_opacity(cfg.back_grid_opacity)
                )
                self._back_wall.add(
                    Line3D(start=np.array([bw_x0, yv, 0]),
                           end  =np.array([bw_x1, yv, 0]),
                           color=cfg.axes_color, stroke_width=0.4
                           ).set_opacity(cfg.back_grid_opacity)
                )
            self.add(self._back_wall)

        # ---- 3. Build each violin -----------------------------------
        self.violins:   list[_SingleViolin] = []
        self.x_centers: list[float]          = []

        for i, group in enumerate(self._groups):
            xc = x_left + i * cfg.group_spacing
            self.x_centers.append(xc)

            violin = _SingleViolin(
                group=group,
                x_center=xc,
                mapper=mapper,
                global_max_density=global_max_density,
                cfg=cfg,
                rng=self._rng,
            )
            self.violins.append(violin)
            self.add(violin)

        # ---- 4. Y axis ----------------------------------------------
        # Arrow
        ax_x = x_left - cfg.violin_max_radius - 0.40
        y_axis = Arrow3D(
            start=np.array([ax_x, -0.15,              0]),
            end  =np.array([ax_x,  cfg.y_scale * 1.12, 0]),
            color=cfg.axes_color, stroke_width=1.5,
        )
        self.add(y_axis)

        # Ticks + labels
        n_t        = cfg.n_y_ticks
        data_ticks = np.linspace(mapper.lo, mapper.hi, n_t)
        for dv in data_ticks:
            yw = float(mapper(dv))
            tick = Line3D(
                start=np.array([ax_x,                   yw, 0]),
                end  =np.array([ax_x - cfg.tick_length, yw, 0]),
                color=cfg.axes_color, stroke_width=0.8,
            )
            self.add(tick)
            # Format: integer if data is coarsely scaled, else 1dp
            fmt = f"{dv:.1f}"
            lbl = Text(fmt, color=cfg.axes_color, font_size=cfg.label_font_size)
            lbl.move_to(np.array([ax_x - cfg.tick_length - 0.25, yw, 0]))
            self.add(lbl)

        # Y axis label
        y_lbl = Text(cfg.axes_y_label, color=cfg.axes_color,
                     font_size=cfg.label_font_size + 4)
        y_lbl.move_to(np.array([ax_x - 0.65, cfg.y_scale / 2, 0]))
        self.add(y_lbl)

        # ---- 5. Group labels on X axis ------------------------------
        if cfg.show_group_labels:
            for xc, group in zip(self.x_centers, self._groups):
                lbl = Text(group.label or "",
                           color=cfg.axes_color,
                           font_size=cfg.group_label_font_size)
                lbl.move_to(np.array([xc, -0.42, 0]))
                self.add(lbl)

        # ---- 6. Significance brackets -------------------------------
        self._brackets: list[_SigBracket] = []
        self._brackets_group = VGroup()
        if cfg.show_significance and n > 1:
            self._build_significance_brackets()
        # Brackets added lazily in animate_compare_groups()

    def _build_significance_brackets(self) -> None:
        """Compute pairwise tests and create bracket objects, stacked vertically."""
        cfg     = self.cfg
        n       = len(self._groups)
        # Starting Y for the lowest bracket = topmost violin extent + lift
        y_max   = max(v.y_top for v in self.violins)
        y_base  = y_max + cfg.bracket_lift

        # All pairs in display order (adjacent pairs first, then spanning)
        pairs = []
        for gap in range(1, n):            # gap=1: adjacent, gap=2: skip-one, …
            for i in range(n - gap):
                pairs.append((i, i + gap))

        for stack_idx, (i, j) in enumerate(pairs):
            g1, g2 = self._groups[i], self._groups[j]
            x1, x2 = self.x_centers[i], self.x_centers[j]
            y_brk  = y_base + stack_idx * cfg.bracket_step

            if cfg.sig_test == "ttest":
                _, p = scipy_stats.ttest_ind(g1.data, g2.data, equal_var=False)
            else:
                _, p = scipy_stats.mannwhitneyu(
                    g1.data, g2.data, alternative="two-sided"
                )
            p = float(p)

            bracket = _SigBracket(
                x_left=x1, x_right=x2,
                y_base=y_brk,
                p_value=p,
                color=BRACKET_COLOR,
                stroke_width=cfg.bracket_stroke_width,
                font_size=cfg.group_label_font_size - 2,
            )
            self._brackets.append(bracket)
            self._brackets_group.add(bracket)

    # ------------------------------------------------------------------
    # Animation API
    # ------------------------------------------------------------------

    def animate_grow(
        self,
        index:    int,
        run_time: float = 2.0,
    ) -> LaggedStart:
        """Grow a single violin surface outward from the median line.

        The three zone surfaces expand in sequence: IQR body first,
        then lower tail and upper tail together.

        Parameters
        ----------
        index : int
            0-based index of the violin to animate.
        """
        v      = self.violins[index]
        zones  = v.surface_group.get_zone_surfaces()

        if len(zones) == 3:
            lower, body, upper = zones
            return LaggedStart(
                GrowFromPoint(body,  point=np.array([v.x_center, v.y_median, 0]),
                              run_time=run_time * 0.55),
                AnimationGroup(
                    GrowFromPoint(lower, point=np.array([v.x_center, v.y_q1, 0]),
                                  run_time=run_time * 0.45),
                    GrowFromPoint(upper, point=np.array([v.x_center, v.y_q3, 0]),
                                  run_time=run_time * 0.45),
                ),
                lag_ratio=0.35,
                run_time=run_time,
            )
        else:
            return LaggedStart(
                GrowFromPoint(zones[0],
                              point=np.array([v.x_center, v.y_median, 0]),
                              run_time=run_time),
                lag_ratio=0.0,
                run_time=run_time,
            )

    def animate_grow_all(
        self,
        lag_ratio: float = 0.18,
        run_time:  float = 3.5,
    ) -> LaggedStart:
        """Staggered grow-in of all violin surfaces left to right.

        Each violin uses ``animate_grow`` internally so the zone-sequenced
        expansion plays even in the staggered version::

            self.play(plot.animate_grow_all())
        """
        return LaggedStart(
            *[self.animate_grow(i, run_time=run_time * 0.7)
              for i in range(len(self.violins))],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_reveal_boxplot(
        self,
        index:    int,
        run_time: float = 1.5,
    ) -> LaggedStart:
        """Sequentially reveal whiskers → IQR box → median → mean for one violin.

        Elements appear in the order that communicates the statistics
        most clearly (range first, then spread, then centre measures).

        Parameters
        ----------
        index : int
            0-based index of the violin.
        """
        bp = self.violins[index].boxplot_group
        return LaggedStart(
            *[FadeIn(el, run_time=run_time * 0.5) for el in bp],
            lag_ratio=0.20,
            run_time=run_time,
        )

    def animate_reveal_boxplots(
        self,
        lag_ratio: float = 0.15,
        run_time:  float = 2.5,
    ) -> LaggedStart:
        """Reveal all box-plot layers across all violins with a staggered lag."""
        return LaggedStart(
            *[self.animate_reveal_boxplot(i, run_time=run_time * 0.6)
              for i in range(len(self.violins))],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_drop_jitter(
        self,
        index:    int,
        run_time: float = 2.0,
    ) -> LaggedStart:
        """Rain jitter dots downward into their final positions.

        Each dot falls from slightly above its target Y, creating a
        visual impression of data "settling" into the distribution.

        Parameters
        ----------
        index : int
            0-based index of the violin.
        """
        jg = self.violins[index].jitter_group
        if len(jg) == 0:
            return LaggedStart(FadeIn(VGroup()), lag_ratio=0, run_time=0.1)

        anims = []
        for dot in jg.submobjects:
            # Drop from above
            start_pos = dot.get_center() + UP * 0.8
            anims.append(
                Succession(
                    FadeIn(dot.copy().move_to(start_pos), run_time=0.01),
                    dot.animate(
                        run_time=run_time * 0.4,
                        rate_func=lambda t: t ** 2,   # accelerate downward
                    ).move_to(dot.get_center()),
                )
            )
        return LaggedStart(*anims, lag_ratio=0.008, run_time=run_time)

    def animate_drop_jitter_all(
        self,
        lag_ratio: float = 0.10,
        run_time:  float = 2.5,
    ) -> LaggedStart:
        """Drop jitter dots for all violins with a staggered inter-group lag."""
        return LaggedStart(
            *[self.animate_drop_jitter(i, run_time=run_time * 0.6)
              for i in range(len(self.violins))],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_compare_groups(
        self,
        lag_ratio: float = 0.30,
        run_time:  float = 2.0,
    ) -> LaggedStart:
        """Draw significance brackets one by one from adjacent to spanning pairs.

        Brackets are added to the scene on first call::

            self.play(plot.animate_compare_groups())
        """
        if self._brackets_group not in self.submobjects:
            self.add(self._brackets_group)
        return LaggedStart(
            *[Create(b, run_time=run_time * 0.5) for b in self._brackets],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_highlight_group(
        self,
        index:           int,
        scale_factor:    float      = 1.15,
        highlight_color: ManimColor | None = None,
        run_time:        float      = 0.6,
    ) -> Succession:
        """Scale and optionally recolor one violin, then restore it.

        All other violins are dimmed simultaneously to draw focus.

        Parameters
        ----------
        index : int
            0-based index of the violin to highlight.
        """
        v        = self.violins[index]
        others   = [self.violins[j] for j in range(len(self.violins)) if j != index]

        dim_in   = AnimationGroup(*[
            o.animate(run_time=run_time / 2).set_opacity(0.25) for o in others
        ])
        scale_in = v.animate(run_time=run_time / 2).scale(scale_factor)
        if highlight_color is not None:
            scale_in = AnimationGroup(
                scale_in,
                v.surface_group.animate(run_time=run_time / 2)
                 .set_color(highlight_color),
            )

        restore = AnimationGroup(
            *[o.animate(run_time=run_time / 2).set_opacity(1.0) for o in others],
            v.animate(run_time=run_time / 2).scale(1 / scale_factor),
        )

        return Succession(
            AnimationGroup(dim_in, scale_in),
            restore,
        )

    def animate_split_reveal(
        self,
        run_time: float = 2.5,
    ) -> AnimationGroup:
        """For half-violin configurations, grow each half from the centre line.

        Only meaningful when ``cfg.half_mode`` is ``"right"`` or ``"left"``.
        Falls back to ``animate_grow_all`` for full violins.
        """
        if self.cfg.half_mode is None:
            return self.animate_grow_all(run_time=run_time)

        anims = []
        for i, v in enumerate(self.violins):
            for surf in v.surface_group.get_zone_surfaces():
                anims.append(
                    GrowFromPoint(
                        surf,
                        point=np.array([v.x_center, v.y_median, 0]),
                        run_time=run_time,
                    )
                )
        return AnimationGroup(*anims)

    def animate_morph_bandwidth(
        self,
        index:         int,
        new_bandwidth: float,
        run_time:      float = 1.8,
    ) -> Transform:
        """Morph one violin to a different KDE bandwidth setting.

        Creates a new ``_SingleViolin`` with the updated bandwidth and
        transforms the existing one into it — useful for explaining the
        effect of smoothing on the density estimate::

            self.play(plot.animate_morph_bandwidth(0, new_bandwidth=0.3))

        Parameters
        ----------
        index : int
            0-based index of the violin to morph.
        new_bandwidth : float
            Target bandwidth value (must be a float, not "scott"/"silverman").
        """
        old_violin = self.violins[index]
        new_cfg    = ViolinConfig(**{
            **self.cfg.__dict__,
            "bandwidth": new_bandwidth,
        })
        new_violin = _SingleViolin(
            group=self._groups[index],
            x_center=old_violin.x_center,
            mapper=self._mapper,
            global_max_density=old_violin._kde(
                old_violin._ys_data[np.argmax(old_violin._kde(old_violin._ys_data))]
            ).max(),
            cfg=new_cfg,
            rng=np.random.default_rng(42),
        )
        return Transform(old_violin.surface_group, new_violin.surface_group,
                         run_time=run_time)

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def single(
        cls,
        data:   Sequence[float] | np.ndarray,
        label:  str   = "",
        config: ViolinConfig | None = None,
        seed:   int   = 42,
    ) -> "ViolinPlot3D":
        """Create a single-group violin plot."""
        g = ViolinGroup(data=np.asarray(data, dtype=float), label=label)
        return cls([g], config=config, seed=seed)

    @classmethod
    def from_normal_groups(
        cls,
        params:    list[tuple[float, float]],
        n:         int          = 200,
        labels:    list[str]    | None = None,
        config:    ViolinConfig | None = None,
        seed:      int          = 0,
    ) -> "ViolinPlot3D":
        """Create a multi-group violin from Gaussian parameters.

        Parameters
        ----------
        params : list of (mean, std) tuples
            One tuple per group.
        n : int
            Observations per group.
        labels : list[str], optional
            Group labels.  Auto-generated if None.

        Example
        -------
        >>> plot = ViolinPlot3D.from_normal_groups(
        ...     params=[(0, 1), (1.5, 0.7), (3, 1.3)],
        ...     labels=["Control", "Dose 1", "Dose 2"],
        ... )
        """
        rng    = np.random.default_rng(seed)
        groups = []
        for i, (mu, sigma) in enumerate(params):
            lbl  = (labels[i] if labels and i < len(labels) else f"Group {i + 1}")
            data = rng.normal(loc=mu, scale=sigma, size=n)
            groups.append(ViolinGroup(data=data, label=lbl))
        return cls(groups, config=config, seed=seed)

    @classmethod
    def from_skewed_groups(
        cls,
        n:      int          = 200,
        config: ViolinConfig | None = None,
        seed:   int          = 0,
    ) -> "ViolinPlot3D":
        """Three groups with varying skewness to showcase zone coloring.

        Group A: symmetric normal
        Group B: right-skewed (log-normal)
        Group C: bimodal (mixture)
        """
        rng  = np.random.default_rng(seed)
        a    = rng.normal(0, 1, n)
        b    = rng.lognormal(0, 0.6, n)
        c    = np.concatenate([rng.normal(-1.5, 0.5, n // 2),
                               rng.normal( 1.5, 0.5, n // 2)])
        groups = [
            ViolinGroup(a, label="Normal"),
            ViolinGroup(b, label="Log-Normal"),
            ViolinGroup(c, label="Bimodal"),
        ]
        return cls(groups, config=config, seed=seed)

    @classmethod
    def split_pair(
        cls,
        data_left:  Sequence[float] | np.ndarray,
        data_right: Sequence[float] | np.ndarray,
        label:      str          = "",
        config:     ViolinConfig | None = None,
        seed:       int          = 0,
    ) -> "ViolinPlot3D":
        """Create a mirrored split violin (left half vs right half) at x=0.

        Useful for before/after or male/female comparisons where both
        distributions share the same Y axis directly side by side.

        The left series uses ``half_mode="left"`` and the right series
        uses ``half_mode="right"``; they are positioned at the same X so
        they appear as two halves of one shape.
        """
        cfg_l = ViolinConfig(**(config.__dict__ if config else {}))
        cfg_l.half_mode      = "left"
        cfg_l.group_spacing  = 0.0   # both at x=0
        cfg_r = ViolinConfig(**(config.__dict__ if config else {}))
        cfg_r.half_mode      = "right"
        cfg_r.group_spacing  = 0.0

        gl = ViolinGroup(np.asarray(data_left,  dtype=float), label=label + " (L)")
        gr = ViolinGroup(np.asarray(data_right, dtype=float), label=label + " (R)")

        # Build as two separate single-violin plots and combine
        plot = cls.__new__(cls)
        VGroup.__init__(plot)
        plot.cfg     = config or ViolinConfig()
        plot._groups = [gl, gr]
        plot._rng    = np.random.default_rng(seed)
        # Assign colors
        gl.color = VIOLIN_PALETTE[0]
        gr.color = VIOLIN_PALETTE[1]
        plot._mapper  = _YMapper([gl, gr], y_scale=plot.cfg.y_scale)
        plot.violins  = []
        plot.x_centers = []
        plot._brackets = []
        plot._brackets_group = VGroup()

        for half, grp, cfg_h in [("left", gl, cfg_l), ("right", gr, cfg_r)]:
            _, ds, _ = _compute_kde(grp.data, bandwidth=cfg_h.bandwidth)
            gmd      = float(ds.max())
            v = _SingleViolin(
                group=grp,
                x_center=0.0,
                mapper=plot._mapper,
                global_max_density=gmd,
                cfg=cfg_h,
                rng=plot._rng,
            )
            plot.violins.append(v)
            plot.x_centers.append(0.0)
            plot.add(v)

        return plot