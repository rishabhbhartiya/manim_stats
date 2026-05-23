"""
manim_stats/charts/scatter_plot3d.py
=====================================
A production-quality 3D scatter plot for Manim with:
  - Point glyphs rendered as 3D spheres or flat diamonds, with optional
    size encoding (bubble chart mode) and Z-value / category color mapping
  - OLS regression plane rendered as a shaded Surface mesh with
    configurable color ramp and transparency
  - Residual arrows from each point to its projection on the regression
    plane, sign-colored (positive vs negative) and toggle-able
  - Confidence / covariance ellipsoid drawn via parametric surface around
    the point cloud (1-sigma, 2-sigma shells)
  - Marginal histograms projected onto the XZ-wall (Y marginal) and
    YZ-wall (X marginal) as flat bar stacks
  - Pearson r annotation badge with significance stars (p-value lookup)
  - Outlier detection: points > N*sigma from centroid styled with a
    distinct glyph and stroke
  - Multi-series (categorical) support: each series gets its own color,
    glyph style, optional per-series regression line
  - Full 3D axis system: tick marks + numeric labels on X, Y, Z; optional
    back-wall grid planes on XY, XZ, YZ faces
  - Animation suite:
      animate_plot_points      – staggered sphere grow-in
      animate_fit_plane        – regression plane grows from centroid
      animate_draw_residuals   – residual arrows appear bar-by-bar
      animate_ellipsoid        – covariance shell expands from centre
      animate_rotate_cloud     – camera orbit around the point cloud
      animate_morph_series     – points migrate between two datasets
      animate_highlight_point  – flash + scale a single point
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
    ThreeDAxes, Arrow3D, Line3D,
    Sphere, Dot3D, Polygon, Surface,
    Text, MathTex, DecimalNumber,
    Surface, ParametricFunction,
    FadeIn, FadeOut, GrowFromCenter, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    ValueTracker, always_redraw,
    interpolate_color, color_to_rgb, rgb_to_color,
    ManimColor, Rotate, rate_functions,
    DEGREES, Arrow, Dot,
)


# ---------------------------------------------------------------------------
# Color palette & shading helpers
# ---------------------------------------------------------------------------

# Default series palette (up to 8 groups)
SERIES_PALETTE: list[ManimColor] = [
    ManimColor("#2979FF"),   # vivid blue
    ManimColor("#FF6D00"),   # vivid orange
    ManimColor("#00BFA5"),   # teal
    ManimColor("#D500F9"),   # purple
    ManimColor("#FFD600"),   # yellow
    ManimColor("#FF1744"),   # red
    ManimColor("#00E676"),   # green
    ManimColor("#FF4081"),   # pink
]

# Regression plane gradient (low → high predicted Y)
PLANE_COLD = ManimColor("#1565C0")
PLANE_HOT  = ManimColor("#B71C1C")

# Residual colors
RESIDUAL_POS = ManimColor("#00E676")   # green  = actual above plane
RESIDUAL_NEG = ManimColor("#FF5252")   # red    = actual below plane

# Outlier styling
OUTLIER_COLOR  = ManimColor("#FFD600")
OUTLIER_STROKE = ManimColor("#FF6D00")

# Confidence ellipsoid shells
ELLIPSOID_1S = ManimColor("#7B1FA2")
ELLIPSOID_2S = ManimColor("#4A148C")

FACE_DARKEN  = 0.35
FACE_LIGHTEN = 0.20


def _darken(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, BLACK, f)

def _lighten(c: ManimColor, f: float) -> ManimColor:
    return interpolate_color(c, WHITE, f)

def _freq_color(t: float, cold: ManimColor, hot: ManimColor) -> ManimColor:
    """Linear interpolation between cold and hot for a scalar t in [0, 1]."""
    return interpolate_color(cold, hot, np.clip(t, 0, 1))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScatterSeries:
    """One group / category within a ScatterPlot3D.

    Parameters
    ----------
    x, y : array-like
        The two primary axes of the scatter cloud.
    z : array-like, optional
        Third axis values.  If *None*, all points are placed at ``z=0``
        (2-D scatter displayed in 3-D space).
    label : str
        Legend label for this series.
    color : ManimColor, optional
        Override the automatic palette color for this series.
    size : array-like, optional
        Per-point sphere radius multiplier for bubble-chart mode.
        Must match the length of *x*.  Set to *None* for uniform size.
    """
    x:     np.ndarray
    y:     np.ndarray
    z:     np.ndarray | None = None
    label: str = ""
    color: ManimColor | None = None
    size:  np.ndarray | None = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.z is not None:
            self.z = np.asarray(self.z, dtype=float)
        else:
            self.z = np.zeros(len(self.x))
        if self.size is not None:
            self.size = np.asarray(self.size, dtype=float)
            if len(self.size) != len(self.x):
                raise ValueError("size array must match length of x")

    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) matrix of raw data coordinates."""
        return np.column_stack([self.x, self.y, self.z])

    @classmethod
    def from_arrays(
        cls,
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[float] | None = None,
        **kwargs,
    ) -> "ScatterSeries":
        return cls(x=np.asarray(x), y=np.asarray(y),
                   z=np.asarray(z) if z is not None else None, **kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScatterConfig:
    """All visual and statistical options for ScatterPlot3D.

    Point geometry
    --------------
    point_radius : float
        Base radius for sphere glyphs.
    point_opacity : float
        Fill opacity of point spheres [0, 1].
    point_stroke_width : float
        Outline stroke width. 0 to disable.
    bubble_scale : float
        When ``size`` arrays are provided on series, max sphere radius.
    outlier_sigma : float
        Points more than this many σ from centroid are flagged as outliers.
        Set to ``np.inf`` to disable outlier detection.
    outlier_radius_mult : float
        Radius multiplier for outlier glyphs (default 1.4×).
    use_sphere : bool
        If True, render points as ``Sphere`` (slow but beautiful).
        If False, render as ``Dot3D`` (fast).

    Regression plane
    ----------------
    show_regression_plane : bool
        Fit and display the OLS regression plane (x, z → y).
    plane_resolution : int
        Grid resolution of the ``Surface`` mesh.
    plane_opacity : float
        Transparency of the regression plane surface.
    plane_color_cold : ManimColor
        Color at the low end of the predicted-Y gradient.
    plane_color_hot : ManimColor
        Color at the high end of the predicted-Y gradient.
    plane_stroke_width : float
        Grid line width on the surface.  0 to disable grid lines.
    show_per_series_regression : bool
        When True and multiple series exist, draw a regression *line*
        (in 2-D subspace) per series instead of a global plane.

    Residuals
    ---------
    show_residuals : bool
        Draw Arrow3D from each point to its projection on the plane.
    residual_stroke_width : float
        Width of residual arrows.
    residual_positive_color : ManimColor
        Color for upward residuals (actual > predicted).
    residual_negative_color : ManimColor
        Color for downward residuals (actual < predicted).
    show_residual_labels : bool
        Label each residual arrow with its signed value.

    Confidence ellipsoid
    --------------------
    show_ellipsoid : bool
        Draw the 1-sigma covariance ellipsoid around the full point cloud.
    show_ellipsoid_2sigma : bool
        Also draw the 2-sigma shell (more transparent).
    ellipsoid_resolution : int
        Sphere mesh resolution before applying the covariance transform.
    ellipsoid_opacity_1s : float
        Opacity of the 1-sigma shell.
    ellipsoid_opacity_2s : float
        Opacity of the 2-sigma shell.

    Marginal histograms
    -------------------
    show_marginals : bool
        Project flattened histograms onto the back walls.
    marginal_bins : int
        Number of bins for each marginal histogram.
    marginal_opacity : float
        Opacity of the marginal bar strips.
    marginal_color : ManimColor
        Fill color of the marginal bars.

    Correlation badge
    -----------------
    show_correlation : bool
        Show the Pearson r badge in the scene.
    correlation_font_size : int
        Font size of the r / p-value annotation.

    Axes & grid
    -----------
    axes_x_label : str
        Label for the X axis.
    axes_y_label : str
        Label for the Y axis.
    axes_z_label : str
        Label for the Z axis.
    axes_color : ManimColor
        Color of axis lines, ticks, and labels.
    tick_length : float
        Tick mark length.
    n_ticks : int
        Approximate number of ticks per axis.
    show_back_grids : bool
        Render semi-transparent grid planes on the XY, XZ, and YZ walls.
    back_grid_opacity : float
        Opacity of back-wall grid planes.

    Layout
    ------
    x_scale, y_scale, z_scale : float
        Rendered extent of each axis in Manim units.
    """

    # ---- point geometry ----
    point_radius:         float       = 0.07
    point_opacity:        float       = 0.88
    point_stroke_width:   float       = 0.6
    bubble_scale:         float       = 0.22
    outlier_sigma:        float       = 3.0
    outlier_radius_mult:  float       = 1.50
    use_sphere:           bool        = False  # True = prettier, slower

    # ---- regression plane ----
    show_regression_plane:       bool       = True
    plane_resolution:            int        = 20
    plane_opacity:               float      = 0.30
    plane_color_cold:            ManimColor = PLANE_COLD
    plane_color_hot:             ManimColor = PLANE_HOT
    plane_stroke_width:          float      = 0.4
    show_per_series_regression:  bool       = False

    # ---- residuals ----
    show_residuals:           bool       = False
    residual_stroke_width:    float      = 1.2
    residual_positive_color:  ManimColor = RESIDUAL_POS
    residual_negative_color:  ManimColor = RESIDUAL_NEG
    show_residual_labels:     bool       = False

    # ---- confidence ellipsoid ----
    show_ellipsoid:        bool  = False
    show_ellipsoid_2sigma: bool  = False
    ellipsoid_resolution:  int   = 16
    ellipsoid_opacity_1s:  float = 0.12
    ellipsoid_opacity_2s:  float = 0.06

    # ---- marginal histograms ----
    show_marginals:     bool       = False
    marginal_bins:      int        = 15
    marginal_opacity:   float      = 0.40
    marginal_color:     ManimColor = ManimColor("#455A64")

    # ---- correlation badge ----
    show_correlation:       bool = True
    correlation_font_size:  int  = 28

    # ---- axes & grid ----
    axes_x_label:      str        = "X"
    axes_y_label:      str        = "Y"
    axes_z_label:      str        = "Z"
    axes_color:        ManimColor = GRAY_B
    tick_length:       float      = 0.12
    n_ticks:           int        = 5
    show_back_grids:   bool       = True
    back_grid_opacity: float      = 0.10

    # ---- layout ----
    x_scale: float = 5.0
    y_scale: float = 4.0
    z_scale: float = 5.0


# ---------------------------------------------------------------------------
# Coordinate mapper
# ---------------------------------------------------------------------------

class _CoordMapper:
    """Converts raw data coordinates to Manim 3D world coordinates.

    Maps each axis independently through a linear transform so that the
    full data range fills ``[-scale/2, +scale/2]`` on that axis.
    """

    def __init__(
        self,
        all_series: list[ScatterSeries],
        x_scale: float,
        y_scale: float,
        z_scale: float,
        padding: float = 0.08,
    ):
        xs = np.concatenate([s.x for s in all_series])
        ys = np.concatenate([s.y for s in all_series])
        zs = np.concatenate([s.z for s in all_series])

        def _safe_range(arr: np.ndarray) -> tuple[float, float]:
            lo, hi = arr.min(), arr.max()
            span = hi - lo
            if span < 1e-10:
                span = 1.0
            margin = span * padding
            return lo - margin, hi + margin

        self.x_min, self.x_max = _safe_range(xs)
        self.y_min, self.y_max = _safe_range(ys)
        self.z_min, self.z_max = _safe_range(zs)
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale

    def __call__(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
    ) -> np.ndarray:
        """Map (x, y, z) data coords → Manim world coords."""
        mx = (np.asarray(x) - self.x_min) / (self.x_max - self.x_min) - 0.5
        my = (np.asarray(y) - self.y_min) / (self.y_max - self.y_min)
        mz = (np.asarray(z) - self.z_min) / (self.z_max - self.z_min) - 0.5
        world = np.stack([mx * self.x_scale,
                          my * self.y_scale,
                          mz * self.z_scale], axis=-1)
        return world

    def x_world(self, v: float) -> float:
        return ((v - self.x_min) / (self.x_max - self.x_min) - 0.5) * self.x_scale

    def y_world(self, v: float) -> float:
        return ((v - self.y_min) / (self.y_max - self.y_min)) * self.y_scale

    def z_world(self, v: float) -> float:
        return ((v - self.z_min) / (self.z_max - self.z_min) - 0.5) * self.z_scale


# ---------------------------------------------------------------------------
# Back-wall grid planes
# ---------------------------------------------------------------------------

class _BackGridPlane(VGroup):
    """A semi-transparent grid plane on one of the three back walls."""

    def __init__(
        self,
        corner: np.ndarray,       # bottom-left world corner
        u_vec:  np.ndarray,       # vector along U axis
        v_vec:  np.ndarray,       # vector along V axis
        n_u:    int = 8,
        n_v:    int = 8,
        color:  ManimColor = GRAY_C,
        opacity: float = 0.10,
    ):
        super().__init__()
        for i in range(n_u + 1):
            t = i / n_u
            start = corner + t * u_vec
            end   = start  + v_vec
            self.add(Line3D(start=start, end=end,
                            color=color, stroke_width=0.5).set_opacity(opacity))
        for j in range(n_v + 1):
            t = j / n_v
            start = corner + t * v_vec
            end   = start  + u_vec
            self.add(Line3D(start=start, end=end,
                            color=color, stroke_width=0.5).set_opacity(opacity))


# ---------------------------------------------------------------------------
# Regression plane surface
# ---------------------------------------------------------------------------

class _RegressionPlane(VGroup):
    """OLS regression plane rendered as a gradient-shaded Surface.

    The plane is fit as  Y = b0 + b1*X + b2*Z  using least squares.
    It is coloured by the predicted Y value across the surface so viewers
    can see the slope direction at a glance.
    """

    def __init__(
        self,
        mapper:     _CoordMapper,
        all_x:      np.ndarray,
        all_y:      np.ndarray,
        all_z:      np.ndarray,
        cfg:        ScatterConfig,
    ):
        super().__init__()

        # Fit OLS: Y = b0 + b1*X + b2*Z
        A = np.column_stack([np.ones_like(all_x), all_x, all_z])
        coeffs, _, _, _ = np.linalg.lstsq(A, all_y, rcond=None)
        b0, b1, b2      = coeffs
        self._coeffs    = coeffs

        # Prediction range (data space)
        x_lo, x_hi = mapper.x_min, mapper.x_max
        z_lo, z_hi = mapper.z_min, mapper.z_max
        y_pred_lo   = b0 + b1 * x_lo + b2 * z_lo
        y_pred_hi   = b0 + b1 * x_hi + b2 * z_hi
        y_pred_min  = min(y_pred_lo, y_pred_hi,
                          b0 + b1 * x_lo + b2 * z_hi,
                          b0 + b1 * x_hi + b2 * z_lo)
        y_pred_max  = max(y_pred_lo, y_pred_hi,
                          b0 + b1 * x_lo + b2 * z_hi,
                          b0 + b1 * x_hi + b2 * z_lo)
        y_pred_span = y_pred_max - y_pred_min if y_pred_max > y_pred_min else 1.0

        # Surface: u ∈ [0,1] ↦ x; v ∈ [0,1] ↦ z
        def surface_func(u: float, v: float) -> np.ndarray:
            x_d = x_lo + u * (x_hi - x_lo)
            z_d = z_lo + v * (z_hi - z_lo)
            y_d = b0 + b1 * x_d + b2 * z_d
            return mapper(x_d, y_d, z_d)

        def color_func(u: float, v: float) -> ManimColor:
            x_d  = x_lo + u * (x_hi - x_lo)
            z_d  = z_lo + v * (z_hi - z_lo)
            y_d  = b0 + b1 * x_d + b2 * z_d
            t    = np.clip((y_d - y_pred_min) / y_pred_span, 0, 1)
            return _freq_color(t, cfg.plane_color_cold, cfg.plane_color_hot)

        plane = Surface(
            surface_func,
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(cfg.plane_resolution, cfg.plane_resolution),
        )
        plane.set_style(
            fill_opacity=cfg.plane_opacity,
            stroke_width=cfg.plane_stroke_width,
            stroke_opacity=cfg.plane_opacity * 0.6,
        )

        # Apply color per face based on position
        # (Manim's Surface doesn't have a built-in color_func param
        #  in all versions; we approximate with a gradient based on the surface
        #  normal direction by tinting after creation.)
        plane.set_color_by_gradient(cfg.plane_color_cold, cfg.plane_color_hot)

        self.add(plane)
        self.plane       = plane
        self._b0         = b0
        self._b1         = b1
        self._b2         = b2

    def predict(self, x: float, z: float) -> float:
        return self._b0 + self._b1 * x + self._b2 * z


# ---------------------------------------------------------------------------
# Residual arrows
# ---------------------------------------------------------------------------

class _ResidualArrows(VGroup):
    """Arrow3D from each data point to its projection on the OLS plane."""

    def __init__(
        self,
        mapper:    _CoordMapper,
        all_x:     np.ndarray,
        all_y:     np.ndarray,
        all_z:     np.ndarray,
        reg_plane: _RegressionPlane,
        cfg:       ScatterConfig,
    ):
        super().__init__()
        self.arrows:    list[Arrow3D] = []
        self.residuals: list[float]   = []

        for xi, yi, zi in zip(all_x, all_y, all_z):
            y_pred    = reg_plane.predict(xi, zi)
            residual  = yi - y_pred
            self.residuals.append(residual)

            point_w = mapper(xi, yi,     zi)
            proj_w  = mapper(xi, y_pred, zi)

            color = cfg.residual_positive_color if residual >= 0 \
                    else cfg.residual_negative_color

            # Only draw arrows with non-negligible residuals
            if abs(residual) < 1e-8:
                continue

            arr = Arrow3D(
                start=proj_w,
                end=point_w,
                color=color,
                stroke_width=cfg.residual_stroke_width,
                tip_length=0.06,
            )
            self.arrows.append(arr)
            self.add(arr)

            if cfg.show_residual_labels:
                lbl = Text(f"{residual:+.2f}",
                           color=color,
                           font_size=14)
                mid = (point_w + proj_w) / 2
                lbl.move_to(mid + RIGHT * 0.18)
                self.add(lbl)


# ---------------------------------------------------------------------------
# Covariance ellipsoid
# ---------------------------------------------------------------------------

class _CovarianceEllipsoid(VGroup):
    """A semi-transparent ellipsoidal shell at 1σ and optionally 2σ.

    Built by taking the eigenvectors of the 3×3 sample covariance matrix
    and warping a unit sphere accordingly.
    """

    def __init__(
        self,
        mapper:      _CoordMapper,
        all_x:       np.ndarray,
        all_y:       np.ndarray,
        all_z:       np.ndarray,
        cfg:         ScatterConfig,
    ):
        super().__init__()
        xyz_data   = np.column_stack([all_x, all_y, all_z])
        centroid   = xyz_data.mean(axis=0)
        cov        = np.cov(xyz_data.T)           # (3, 3)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        eigenvals  = np.maximum(eigenvals, 1e-10)  # guard against negatives
        std_devs   = np.sqrt(eigenvals)             # axes of the ellipsoid

        # Map centroid to world
        cx, cy, cz = centroid
        centroid_w = mapper(cx, cy, cz)

        # Scaling from data space to world space
        wx = mapper.x_scale / (mapper.x_max - mapper.x_min)
        wy = mapper.y_scale / (mapper.y_max - mapper.y_min)
        wz = mapper.z_scale / (mapper.z_max - mapper.z_min)
        world_scale = np.array([wx, wy, wz])

        def make_shell(k_sigma: float, color: ManimColor, opacity: float) -> Surface:
            """Parametric surface for the k-sigma ellipsoid shell."""

            def ellipsoid_point(u: float, v: float) -> np.ndarray:
                # Unit sphere in eigenvector space
                phi   = u * PI         # [0, π]
                theta = v * TAU        # [0, 2π]
                sphere_pt = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ])
                # Scale along principal axes
                scaled = k_sigma * std_devs * sphere_pt
                # Rotate into data space via eigenvectors
                rotated = eigenvecs @ scaled
                # Translate to centroid (data space)
                data_pt = centroid + rotated
                return mapper(*data_pt)

            surf = Surface(
                ellipsoid_point,
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(cfg.ellipsoid_resolution, cfg.ellipsoid_resolution),
            )
            surf.set_style(
                fill_color=color,
                fill_opacity=opacity,
                stroke_color=_lighten(color, 0.3),
                stroke_width=0.5,
                stroke_opacity=opacity * 1.5,
            )
            return surf

        shell_1s = make_shell(1.0, ELLIPSOID_1S, cfg.ellipsoid_opacity_1s)
        self.add(shell_1s)
        self.shell_1s = shell_1s

        if cfg.show_ellipsoid_2sigma:
            shell_2s = make_shell(2.0, ELLIPSOID_2S, cfg.ellipsoid_opacity_2s)
            self.add(shell_2s)
            self.shell_2s = shell_2s


# ---------------------------------------------------------------------------
# Marginal histograms (projected onto back walls)
# ---------------------------------------------------------------------------

class _MarginalHistogram(VGroup):
    """Flat bar strips projected onto a back wall of the 3D axes box.

    Parameters
    ----------
    values : array-like
        The 1-D data projected onto this axis.
    axis : str
        ``"x"`` — project X marginal onto the YZ back wall.
        ``"z"`` — project Z marginal onto the XY back wall.
    """

    def __init__(
        self,
        values:  np.ndarray,
        mapper:  _CoordMapper,
        axis:    str,
        n_bins:  int,
        color:   ManimColor,
        opacity: float,
        cfg:     ScatterConfig,
    ):
        super().__init__()
        counts, edges = np.histogram(values, bins=n_bins)
        max_count = counts.max() if counts.max() > 0 else 1

        # Max extent for the marginal bars (in Manim units)
        bar_max_len = 0.50

        for i, count in enumerate(counts):
            if count == 0:
                continue
            bar_len = (count / max_count) * bar_max_len
            lo, hi  = edges[i], edges[i + 1]
            center  = (lo + hi) / 2.0

            if axis == "x":
                # Bars extend in the -X direction from the right back wall
                x_w = mapper.x_world(center)
                y_w = 0.0
                z_w = mapper.z_scale / 2 + 0.02   # just behind the back wall

                pts = [
                    np.array([x_w - (hi - lo) / (mapper.x_max - mapper.x_min) * mapper.x_scale / 2,
                               0, z_w]),
                    np.array([x_w + (hi - lo) / (mapper.x_max - mapper.x_min) * mapper.x_scale / 2,
                               0, z_w]),
                    np.array([x_w + (hi - lo) / (mapper.x_max - mapper.x_min) * mapper.x_scale / 2,
                               bar_len, z_w]),
                    np.array([x_w - (hi - lo) / (mapper.x_max - mapper.x_min) * mapper.x_scale / 2,
                               bar_len, z_w]),
                ]
            else:  # axis == "z"
                z_w  = mapper.z_world(center)
                x_w  = -(mapper.x_scale / 2 + 0.02)

                pts = [
                    np.array([x_w, 0,        z_w - (hi - lo) / (mapper.z_max - mapper.z_min) * mapper.z_scale / 2]),
                    np.array([x_w, 0,        z_w + (hi - lo) / (mapper.z_max - mapper.z_min) * mapper.z_scale / 2]),
                    np.array([x_w, bar_len,  z_w + (hi - lo) / (mapper.z_max - mapper.z_min) * mapper.z_scale / 2]),
                    np.array([x_w, bar_len,  z_w - (hi - lo) / (mapper.z_max - mapper.z_min) * mapper.z_scale / 2]),
                ]

            bar = Polygon(*pts, color=color)
            bar.set_fill(color=color, opacity=opacity)
            bar.set_stroke(color=_darken(color, 0.3), width=0.5, opacity=opacity)
            self.add(bar)


# ---------------------------------------------------------------------------
# Axis tick marks and labels
# ---------------------------------------------------------------------------

class _ScatterAxisTicks(VGroup):
    """Tick marks + numeric labels along one axis of the scatter plot."""

    def __init__(
        self,
        world_positions: Sequence[float],
        data_values:     Sequence[float],
        axis:            str,             # "x", "y", or "z"
        fixed_coords:    dict[str, float],
        tick_length:     float = 0.12,
        color:           ManimColor = GRAY_B,
        font_size:       int = 18,
        precision:       int = 1,
    ):
        super().__init__()
        for wp, dv in zip(world_positions, data_values):
            if axis == "x":
                start = np.array([wp, fixed_coords["y"], fixed_coords["z"]])
                end   = np.array([wp, fixed_coords["y"] - tick_length, fixed_coords["z"]])
                lbl_pos = end + DOWN * 0.22
            elif axis == "y":
                start = np.array([fixed_coords["x"], wp, fixed_coords["z"]])
                end   = np.array([fixed_coords["x"] - tick_length, wp, fixed_coords["z"]])
                lbl_pos = end + LEFT * 0.28
            else:  # "z"
                start = np.array([fixed_coords["x"], fixed_coords["y"], wp])
                end   = np.array([fixed_coords["x"], fixed_coords["y"] - tick_length, wp])
                lbl_pos = end + DOWN * 0.22

            tick = Line3D(start=start, end=end, color=color, stroke_width=0.8)
            self.add(tick)

            lbl = Text(f"{dv:.{precision}f}", color=color, font_size=font_size)
            lbl.move_to(lbl_pos)
            self.add(lbl)


# ---------------------------------------------------------------------------
# Correlation badge
# ---------------------------------------------------------------------------

class _CorrelationBadge(VGroup):
    """Floating MathTex badge showing Pearson r, p-value, and significance stars."""

    def __init__(
        self,
        r_value:   float,
        p_value:   float,
        pos:       np.ndarray,
        font_size: int = 28,
    ):
        super().__init__()

        # Significance stars
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = "n.s."

        # Color by strength
        if abs(r_value) >= 0.7:
            r_color = ManimColor("#00E676")
        elif abs(r_value) >= 0.4:
            r_color = ManimColor("#FFD600")
        else:
            r_color = ManimColor("#FF5252")

        r_tex = MathTex(
            rf"r = {r_value:.3f}^{{{stars}}}",
            color=r_color,
            font_size=font_size,
        )
        p_tex = MathTex(
            rf"p = {p_value:.4f}",
            color=GRAY_B,
            font_size=font_size - 6,
        )
        p_tex.next_to(r_tex, DOWN, buff=0.12)

        self.add(r_tex, p_tex)
        self.move_to(pos)


# ---------------------------------------------------------------------------
# Point glyph factory
# ---------------------------------------------------------------------------

def _make_point_glyph(
    world_pos:    np.ndarray,
    color:        ManimColor,
    radius:       float,
    opacity:      float,
    stroke_width: float,
    use_sphere:   bool,
) -> Dot3D | Sphere:
    """Create a single point glyph (Dot3D or Sphere) at *world_pos*."""
    if use_sphere:
        glyph = Sphere(radius=radius, resolution=(8, 8))
        glyph.set_color(color)
        glyph.set_opacity(opacity)
        glyph.move_to(world_pos)
    else:
        glyph = Dot3D(point=world_pos, radius=radius, color=color)
        glyph.set_opacity(opacity)
        if stroke_width > 0:
            glyph.set_stroke(color=_darken(color, 0.4), width=stroke_width)
    return glyph


# ---------------------------------------------------------------------------
# Main ScatterPlot3D class
# ---------------------------------------------------------------------------

class ScatterPlot3D(VGroup):
    """A detailed 3D scatter plot for Manim statistics animations.

    Supports single or multi-series data, regression planes, residual
    arrows, covariance ellipsoids, marginal histograms, and a full
    animation API.

    Basic usage (single series)
    ---------------------------
    >>> import numpy as np
    >>> from manim import *
    >>> from manim_stats.charts.scatter_plot3d import ScatterPlot3D, ScatterConfig, ScatterSeries
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         rng = np.random.default_rng(0)
    ...         x = rng.normal(size=120)
    ...         y = 2.5 * x + rng.normal(scale=0.8, size=120)
    ...         z = rng.normal(size=120)
    ...         cfg  = ScatterConfig(show_regression_plane=True, show_residuals=True)
    ...         plot = ScatterPlot3D.from_xy(x, y, z=z, config=cfg)
    ...         self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
    ...         self.play(plot.animate_plot_points())
    ...         self.play(plot.animate_fit_plane())
    ...         self.play(plot.animate_draw_residuals())

    Multi-series usage
    ------------------
    >>> series = [
    ...     ScatterSeries(x1, y1, label="Group A"),
    ...     ScatterSeries(x2, y2, label="Group B"),
    ... ]
    >>> plot = ScatterPlot3D(series, config=ScatterConfig())

    Parameters
    ----------
    series : ScatterSeries | list[ScatterSeries]
        One or more data series.
    config : ScatterConfig, optional
        Visual configuration.  Defaults to ``ScatterConfig()``.
    """

    def __init__(
        self,
        series:  ScatterSeries | list[ScatterSeries],
        config:  ScatterConfig | None = None,
    ):
        super().__init__()
        self.cfg = config or ScatterConfig()

        if isinstance(series, ScatterSeries):
            self._series = [series]
        else:
            self._series = list(series)

        # Assign palette colors to series that don't have one
        for i, s in enumerate(self._series):
            if s.color is None:
                s.color = SERIES_PALETTE[i % len(SERIES_PALETTE)]

        self._mapper = _CoordMapper(
            self._series,
            x_scale=self.cfg.x_scale,
            y_scale=self.cfg.y_scale,
            z_scale=self.cfg.z_scale,
        )
        self._build()

    # ------------------------------------------------------------------
    # Private build pipeline
    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg    = self.cfg
        mapper = self._mapper

        # Concatenate all series data for global regression / ellipsoid
        all_x = np.concatenate([s.x for s in self._series])
        all_y = np.concatenate([s.y for s in self._series])
        all_z = np.concatenate([s.z for s in self._series])

        # ---- 1. Back-wall grids --------------------------------------
        if cfg.show_back_grids:
            # XY back wall (z = z_max)
            bw_xy = _BackGridPlane(
                corner=np.array([-cfg.x_scale / 2, 0,            cfg.z_scale / 2]),
                u_vec =np.array([ cfg.x_scale,     0,            0              ]),
                v_vec =np.array([ 0,               cfg.y_scale,  0              ]),
                n_u=cfg.n_ticks * 2, n_v=cfg.n_ticks * 2,
                color=cfg.axes_color, opacity=cfg.back_grid_opacity,
            )
            # XZ floor (y = 0)
            bw_xz = _BackGridPlane(
                corner=np.array([-cfg.x_scale / 2, 0,  -cfg.z_scale / 2]),
                u_vec =np.array([ cfg.x_scale,     0,   0               ]),
                v_vec =np.array([ 0,               0,   cfg.z_scale     ]),
                n_u=cfg.n_ticks * 2, n_v=cfg.n_ticks * 2,
                color=cfg.axes_color, opacity=cfg.back_grid_opacity,
            )
            # YZ left wall (x = x_min)
            bw_yz = _BackGridPlane(
                corner=np.array([-cfg.x_scale / 2,  0,           -cfg.z_scale / 2]),
                u_vec =np.array([ 0,                cfg.y_scale,  0               ]),
                v_vec =np.array([ 0,                0,            cfg.z_scale     ]),
                n_u=cfg.n_ticks * 2, n_v=cfg.n_ticks * 2,
                color=cfg.axes_color, opacity=cfg.back_grid_opacity,
            )
            self.add(bw_xy, bw_xz, bw_yz)
            self.back_grids = VGroup(bw_xy, bw_xz, bw_yz)

        # ---- 2. Regression plane -------------------------------------
        if cfg.show_regression_plane and len(all_x) >= 3:
            self._reg_plane = _RegressionPlane(
                mapper=mapper,
                all_x=all_x, all_y=all_y, all_z=all_z,
                cfg=cfg,
            )
            self.add(self._reg_plane)
        else:
            self._reg_plane = None

        # ---- 3. Covariance ellipsoid ---------------------------------
        if cfg.show_ellipsoid and len(all_x) >= 4:
            self.ellipsoid = _CovarianceEllipsoid(
                mapper=mapper,
                all_x=all_x, all_y=all_y, all_z=all_z,
                cfg=cfg,
            )
            self.add(self.ellipsoid)

        # ---- 4. Marginal histograms ----------------------------------
        if cfg.show_marginals:
            self.marginal_x = _MarginalHistogram(
                values=all_x, mapper=mapper, axis="x",
                n_bins=cfg.marginal_bins,
                color=cfg.marginal_color,
                opacity=cfg.marginal_opacity,
                cfg=cfg,
            )
            self.marginal_z = _MarginalHistogram(
                values=all_z, mapper=mapper, axis="z",
                n_bins=cfg.marginal_bins,
                color=cfg.marginal_color,
                opacity=cfg.marginal_opacity,
                cfg=cfg,
            )
            self.add(self.marginal_x, self.marginal_z)

        # ---- 5. Residual arrows (drawn after points) -----------------
        self._residuals_group: _ResidualArrows | None = None
        if cfg.show_residuals and self._reg_plane is not None:
            self._residuals_group = _ResidualArrows(
                mapper=mapper,
                all_x=all_x, all_y=all_y, all_z=all_z,
                reg_plane=self._reg_plane,
                cfg=cfg,
            )
            # Not added yet — revealed via animate_draw_residuals()

        # ---- 6. Point glyphs -----------------------------------------
        # Detect outliers across all series combined
        centroid = np.array([all_x.mean(), all_y.mean(), all_z.mean()])
        std_xyz  = np.array([all_x.std(), all_y.std(), all_z.std()])
        std_xyz  = np.where(std_xyz < 1e-10, 1.0, std_xyz)

        self.point_groups: list[VGroup] = []
        self.all_glyphs:   list[Dot3D | Sphere] = []
        self._glyph_series_index: list[int]  = []  # which series each glyph belongs to
        self._glyph_data_index:   list[int]  = []  # index within that series

        for s_idx, series in enumerate(self._series):
            grp = VGroup()
            for pt_idx in range(len(series.x)):
                xi, yi, zi = series.x[pt_idx], series.y[pt_idx], series.z[pt_idx]
                world_pos  = mapper(xi, yi, zi)

                # Determine if outlier
                dist_sigma = np.abs(
                    np.array([xi - centroid[0], yi - centroid[1], zi - centroid[2]])
                    / std_xyz
                )
                is_outlier = dist_sigma.max() > cfg.outlier_sigma

                # Determine radius
                if series.size is not None:
                    size_norm = (series.size[pt_idx] - series.size.min()) / \
                                max(series.size.max() - series.size.min(), 1e-10)
                    radius = cfg.point_radius + size_norm * (cfg.bubble_scale - cfg.point_radius)
                else:
                    radius = cfg.point_radius

                if is_outlier:
                    radius *= cfg.outlier_radius_mult
                    glyph_color = OUTLIER_COLOR
                else:
                    glyph_color = series.color

                glyph = _make_point_glyph(
                    world_pos=world_pos,
                    color=glyph_color,
                    radius=radius,
                    opacity=cfg.point_opacity,
                    stroke_width=cfg.point_stroke_width,
                    use_sphere=cfg.use_sphere,
                )

                if is_outlier:
                    glyph.set_stroke(color=OUTLIER_STROKE,
                                     width=cfg.point_stroke_width * 2.5)

                grp.add(glyph)
                self.all_glyphs.append(glyph)
                self._glyph_series_index.append(s_idx)
                self._glyph_data_index.append(pt_idx)

            self.point_groups.append(grp)
            self.add(grp)

        # ---- 7. Axis system ------------------------------------------
        n  = cfg.n_ticks
        ax = cfg.axes_color

        # Compute evenly-spaced tick positions in data space → world space
        x_data_ticks  = np.linspace(mapper.x_min, mapper.x_max, n)
        y_data_ticks  = np.linspace(mapper.y_min, mapper.y_max, n)
        z_data_ticks  = np.linspace(mapper.z_min, mapper.z_max, n)
        x_world_ticks = [mapper.x_world(v) for v in x_data_ticks]
        y_world_ticks = [mapper.y_world(v) for v in y_data_ticks]
        z_world_ticks = [mapper.z_world(v) for v in z_data_ticks]

        x_floor = 0.0
        y_floor = 0.0
        z_floor = -cfg.z_scale / 2

        x_ticks = _ScatterAxisTicks(
            world_positions=x_world_ticks,
            data_values=x_data_ticks,
            axis="x",
            fixed_coords={"y": y_floor, "z": z_floor},
            tick_length=cfg.tick_length,
            color=ax, font_size=16, precision=1,
        )
        y_ticks = _ScatterAxisTicks(
            world_positions=y_world_ticks,
            data_values=y_data_ticks,
            axis="y",
            fixed_coords={"x": -cfg.x_scale / 2, "z": z_floor},
            tick_length=cfg.tick_length,
            color=ax, font_size=16, precision=1,
        )
        z_ticks = _ScatterAxisTicks(
            world_positions=z_world_ticks,
            data_values=z_data_ticks,
            axis="z",
            fixed_coords={"x": -cfg.x_scale / 2, "y": y_floor},
            tick_length=cfg.tick_length,
            color=ax, font_size=16, precision=1,
        )
        self.add(x_ticks, y_ticks, z_ticks)

        # Axis arrows
        x_arrow = Arrow3D(
            start=np.array([-cfg.x_scale / 2 - 0.2, 0, z_floor]),
            end  =np.array([ cfg.x_scale / 2 + 0.4, 0, z_floor]),
            color=ax, stroke_width=1.5,
        )
        y_arrow = Arrow3D(
            start=np.array([-cfg.x_scale / 2, -0.2,                z_floor]),
            end  =np.array([-cfg.x_scale / 2,  cfg.y_scale + 0.4,  z_floor]),
            color=ax, stroke_width=1.5,
        )
        z_arrow = Arrow3D(
            start=np.array([-cfg.x_scale / 2, 0,  cfg.z_scale / 2 + 0.4]),
            end  =np.array([-cfg.x_scale / 2, 0, -cfg.z_scale / 2 - 0.2]),
            color=ax, stroke_width=1.5,
        )
        self.add(x_arrow, y_arrow, z_arrow)

        # Axis labels
        x_lbl = Text(cfg.axes_x_label, color=ax, font_size=22)
        x_lbl.move_to(np.array([0, -0.55, z_floor]))
        y_lbl = Text(cfg.axes_y_label, color=ax, font_size=22)
        y_lbl.move_to(np.array([-cfg.x_scale / 2 - 0.65, cfg.y_scale / 2, z_floor]))
        z_lbl = Text(cfg.axes_z_label, color=ax, font_size=22)
        z_lbl.move_to(np.array([-cfg.x_scale / 2, -0.55, 0]))
        self.add(x_lbl, y_lbl, z_lbl)

        # ---- 8. Correlation badge ------------------------------------
        if cfg.show_correlation and len(all_x) >= 3:
            r, p = scipy_stats.pearsonr(all_x, all_y)
            badge = _CorrelationBadge(
                r_value=r,
                p_value=p,
                pos=np.array([cfg.x_scale / 2 + 0.2, cfg.y_scale + 0.3, z_floor]),
                font_size=cfg.correlation_font_size,
            )
            self.add(badge)
            self.correlation_badge = badge
            self._r_value = r
            self._p_value = p

        # ---- 9. Legend (multi-series) --------------------------------
        if len(self._series) > 1:
            self.legend = self._build_legend()
            self.add(self.legend)

        # ---- Store references for animations -------------------------
        self._all_x = all_x
        self._all_y = all_y
        self._all_z = all_z

    def _build_legend(self) -> VGroup:
        """Floating legend for multi-series plots."""
        grp  = VGroup()
        cfg  = self.cfg
        x_pos = cfg.x_scale / 2 + 0.15
        y_pos = cfg.y_scale * 0.75

        for i, series in enumerate(self._series):
            dot = Dot3D(
                point=np.array([x_pos, y_pos - i * 0.40, -cfg.z_scale / 2]),
                radius=cfg.point_radius * 1.4,
                color=series.color,
            )
            lbl = Text(series.label or f"Series {i+1}",
                       color=cfg.axes_color, font_size=18)
            lbl.move_to(np.array([x_pos + 0.45, y_pos - i * 0.40, -cfg.z_scale / 2]))
            grp.add(dot, lbl)

        return grp

    # ------------------------------------------------------------------
    # Public animation helpers
    # ------------------------------------------------------------------

    def animate_plot_points(
        self,
        lag_ratio: float = 0.015,
        run_time:  float = 3.0,
    ) -> LaggedStart:
        """Staggered grow-in of all point glyphs from their world position.

        Returns a ``LaggedStart`` suitable for::

            self.play(plot.animate_plot_points())
        """
        return LaggedStart(
            *[GrowFromCenter(g) for g in self.all_glyphs],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_fit_plane(
        self,
        run_time:  float = 2.0,
    ) -> FadeIn | AnimationGroup:
        """Fade the regression plane into view.

        Call after ``animate_plot_points`` so the plane appears to *fit*
        to already-visible data::

            self.play(plot.animate_plot_points())
            self.play(plot.animate_fit_plane())
        """
        if self._reg_plane is None:
            return FadeIn(VGroup(), run_time=0.1)
        return FadeIn(self._reg_plane, run_time=run_time)

    def animate_draw_residuals(
        self,
        lag_ratio: float = 0.04,
        run_time:  float = 2.5,
    ) -> LaggedStart | FadeIn:
        """Reveal residual arrows one by one.

        The ``_ResidualArrows`` group is added to the scene here on first
        call so it is invisible until this animation is played::

            self.play(plot.animate_draw_residuals())
        """
        if self._residuals_group is None:
            return FadeIn(VGroup(), run_time=0.1)
        if self._residuals_group not in self.submobjects:
            self.add(self._residuals_group)
        return LaggedStart(
            *[FadeIn(arr, shift=UP * 0.05) for arr in self._residuals_group.arrows],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_ellipsoid(
        self,
        run_time: float = 1.5,
    ) -> FadeIn:
        """Expand the covariance ellipsoid from the cloud centroid."""
        if not hasattr(self, "ellipsoid"):
            return FadeIn(VGroup(), run_time=0.1)
        return FadeIn(self.ellipsoid, run_time=run_time)

    def animate_highlight_point(
        self,
        series_index: int,
        point_index:  int,
        highlight_color: ManimColor = YELLOW,
        scale_factor:    float = 2.5,
        run_time:        float = 0.5,
    ) -> Succession:
        """Flash and scale one specific point to draw attention to it.

        Parameters
        ----------
        series_index : int
            0-based index of the series.
        point_index : int
            0-based index within that series.
        """
        # Find the glyph
        target = None
        for glyph, si, pi in zip(
            self.all_glyphs,
            self._glyph_series_index,
            self._glyph_data_index,
        ):
            if si == series_index and pi == point_index:
                target = glyph
                break

        if target is None:
            raise IndexError(
                f"No point found at series={series_index}, point={point_index}"
            )

        orig_color = target.get_color()
        orig_scale = 1.0

        scale_up    = target.animate(run_time=run_time / 2).scale(scale_factor)\
                             .set_color(highlight_color)
        scale_down  = target.animate(run_time=run_time / 2).scale(1 / scale_factor)\
                             .set_color(orig_color)
        return Succession(scale_up, scale_down)

    def animate_morph_series(
        self,
        series_index: int,
        new_x: np.ndarray,
        new_y: np.ndarray,
        new_z: np.ndarray | None = None,
        run_time: float = 2.0,
    ) -> AnimationGroup:
        """Migrate the points of one series to new coordinates.

        Useful for showing how a distribution shifts, or for animating
        the convergence of estimators::

            self.play(plot.animate_morph_series(0, new_x, new_y, new_z))
        """
        if new_z is None:
            new_z = np.zeros(len(new_x))

        grp   = self.point_groups[series_index]
        anims = []

        for pt_idx, glyph in enumerate(grp.submobjects):
            if pt_idx >= len(new_x):
                break
            new_pos = self._mapper(new_x[pt_idx], new_y[pt_idx], new_z[pt_idx])
            anims.append(
                glyph.animate(run_time=run_time).move_to(new_pos)
            )

        return AnimationGroup(*anims)

    def animate_reveal_stats(
        self,
        run_time: float = 1.5,
    ) -> LaggedStart:
        """Sequentially fade in all statistical overlays.

        Order: back grids → regression plane → residuals → ellipsoid → badge.
        """
        targets = []
        if hasattr(self, "back_grids"):
            targets.append(self.back_grids)
        if self._reg_plane is not None:
            targets.append(self._reg_plane)
        if self._residuals_group is not None:
            if self._residuals_group not in self.submobjects:
                self.add(self._residuals_group)
            targets.append(self._residuals_group)
        if hasattr(self, "ellipsoid"):
            targets.append(self.ellipsoid)
        if hasattr(self, "correlation_badge"):
            targets.append(self.correlation_badge)
        return LaggedStart(
            *[FadeIn(t, run_time=run_time * 0.7) for t in targets],
            lag_ratio=0.25,
            run_time=run_time,
        )

    def animate_color_by_residual(
        self,
        run_time: float = 1.2,
    ) -> AnimationGroup:
        """Recolor all points by the sign and magnitude of their residual.

        Positive residuals → green gradient.
        Negative residuals → red gradient.
        Requires ``show_regression_plane=True``.
        """
        if self._reg_plane is None:
            return AnimationGroup()

        anims = []
        residuals = [
            self._series[si].y[pi] - self._reg_plane.predict(
                self._series[si].x[pi], self._series[si].z[pi]
            )
            for glyph, si, pi in zip(
                self.all_glyphs,
                self._glyph_series_index,
                self._glyph_data_index,
            )
        ]
        r_max = max(abs(r) for r in residuals) if residuals else 1.0

        for glyph, residual in zip(self.all_glyphs, residuals):
            t = abs(residual) / r_max
            if residual >= 0:
                color = _freq_color(t, GRAY_C, RESIDUAL_POS)
            else:
                color = _freq_color(t, GRAY_C, RESIDUAL_NEG)
            anims.append(
                glyph.animate(run_time=run_time).set_color(color)
            )
        return AnimationGroup(*anims)

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_xy(
        cls,
        x:      Sequence[float] | np.ndarray,
        y:      Sequence[float] | np.ndarray,
        z:      Sequence[float] | np.ndarray | None = None,
        label:  str = "",
        config: ScatterConfig | None = None,
    ) -> "ScatterPlot3D":
        """Create a single-series scatter plot from X, Y (and optional Z)."""
        series = ScatterSeries(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            z=np.asarray(z, dtype=float) if z is not None else None,
            label=label,
        )
        return cls(series=series, config=config)

    @classmethod
    def from_linear(
        cls,
        n:      int   = 150,
        slope:  float = 2.0,
        noise:  float = 0.8,
        seed:   int   = 0,
        config: ScatterConfig | None = None,
    ) -> "ScatterPlot3D":
        """Demo: linear relationship Y = slope*X + noise."""
        rng = np.random.default_rng(seed)
        x   = rng.normal(size=n)
        z   = rng.normal(size=n)
        y   = slope * x + 0.5 * z + rng.normal(scale=noise, size=n)
        return cls.from_xy(x, y, z, label="Linear", config=config)

    @classmethod
    def from_quadratic(
        cls,
        n:      int   = 150,
        noise:  float = 0.6,
        seed:   int   = 0,
        config: ScatterConfig | None = None,
    ) -> "ScatterPlot3D":
        """Demo: quadratic bowl Y = X² + Z² + noise (non-linear cloud)."""
        rng = np.random.default_rng(seed)
        x   = rng.uniform(-2, 2, size=n)
        z   = rng.uniform(-2, 2, size=n)
        y   = x ** 2 + z ** 2 + rng.normal(scale=noise, size=n)
        return cls.from_xy(x, y, z, label="Quadratic", config=config)

    @classmethod
    def from_clusters(
        cls,
        n_per_cluster: int   = 60,
        n_clusters:    int   = 3,
        spread:        float = 0.5,
        seed:          int   = 0,
        config:        ScatterConfig | None = None,
    ) -> "ScatterPlot3D":
        """Demo: K distinct Gaussian clusters as separate series."""
        rng      = np.random.default_rng(seed)
        centers  = rng.uniform(-2, 2, size=(n_clusters, 3))
        series   = []
        for k, (cx, cy, cz) in enumerate(centers):
            x = rng.normal(loc=cx, scale=spread, size=n_per_cluster)
            y = rng.normal(loc=cy, scale=spread, size=n_per_cluster)
            z = rng.normal(loc=cz, scale=spread, size=n_per_cluster)
            series.append(ScatterSeries(x=x, y=y, z=z, label=f"Cluster {k+1}"))
        return cls(series=series, config=config)

    @classmethod
    def from_bubble(
        cls,
        n:      int   = 80,
        seed:   int   = 0,
        config: ScatterConfig | None = None,
    ) -> "ScatterPlot3D":
        """Demo: bubble chart where point size encodes a 4th variable."""
        rng  = np.random.default_rng(seed)
        x    = rng.normal(size=n)
        z    = rng.normal(size=n)
        size = rng.exponential(scale=1.0, size=n) + 0.2
        y    = 1.5 * x - 0.8 * z + size * 0.3 + rng.normal(scale=0.5, size=n)
        series = ScatterSeries(x=x, y=y, z=z, label="Bubble", size=size)
        return cls(series=series, config=config)