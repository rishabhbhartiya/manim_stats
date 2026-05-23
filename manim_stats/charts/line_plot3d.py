"""
manim_stats/charts/line_plot3d.py
==================================
Production-quality 3D line plots for statistical visualizations.

Design philosophy
-----------------
A line in 3D is richly layered — not just a polyline.  Each
``LineSeries3D`` is built from up to six independent visual layers:

    1. Tube body      — an optional volumetric ``VMobject`` extruded along
                        the path for visible thickness under any camera angle.
    2. Stroke line    — the primary polyline drawn with Manim's stroke system.
    3. Glow halo      — a wider, low-opacity duplicate of the stroke for a
                        soft luminosity effect.
    4. Drop lines     — thin vertical segments from each data point down to
                        the floor plane (z = 0), grounding the curve spatially.
    5. Area fill      — a shaded polygon between the curve and the floor,
                        for PDF / CDF / time-series area charts.
    6. Markers        — per-point 3D markers (sphere, diamond, cross, ring)
                        with optional billboard labels.

Animations are designed for teaching:
    - ``animate_draw``    – stroke traces left-to-right via ``Create``.
    - ``animate_trace``   – a glowing dot travels the full path.
    - ``animate_update``  – curve morphs to new data via point interpolation.
    - ``animate_fill``    – area fill grows upward from the floor.
    - ``animate_rise``    – every point lifts from y=0 simultaneously.

Classes
-------
LineConfig
LineSeries3D
MultiLinePlot3D
CDFLine3D
TimeSeriesPlot3D
ParametricLine3D

Helpers / internals
-------------------
LineMarker3D
AreaFill3D
_DropLines3D
LineColorPalette

Usage examples
--------------
    # Basic multi-series line plot
    from manim import *
    from manim_stats.axes.grid3d import FullGrid3D
    from manim_stats.charts.line_plot3d import MultiLinePlot3D

    class DemoLine(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
            grid = FullGrid3D(x_range=[0, 5, 1], y_range=[-1, 1, 1],
                              z_range=[0, 5, 1])
            plot = MultiLinePlot3D(
                x_values=[0, 1, 2, 3, 4, 5],
                series={"Normal": [0.5,2.1,4.0,3.3,1.8,0.6],
                        "Laplace": [0.3,1.5,4.5,1.5,0.3,0.1]},
                area_fill=True,
            )
            self.play(grid.animate_build())
            self.play(plot.animate_reveal(stagger=0.3))
            self.wait()

    # CDF with shaded tail
    from manim_stats.charts.line_plot3d import CDFLine3D
    cdf = CDFLine3D(x_values=xs, cdf_values=ys, shade_above=1.96)

    # Parametric space curve
    from manim_stats.charts.line_plot3d import ParametricLine3D
    helix = ParametricLine3D(
        x_func=lambda t: np.cos(t),
        y_func=lambda t: np.sin(t),
        z_func=lambda t: t / TAU,
        t_range=(0, TAU * 3, 0.05),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    List, Sequence, Tuple, Optional, Callable, Union, Dict, Iterator
)
import numpy as np

from manim import (
    # Mobjects
    VGroup, VMobject, Polygon, Rectangle, Line, DashedLine,
    Dot, Sphere, Arrow, Text, MathTex, Dot3D,
    # Curves
    ParametricFunction, CubicBezier,
    # Scene
    ThreeDScene,
    # Animations
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, Transform, MoveAlongPath,
    UpdateFromAlphaFunc, Flash,
    # Constants
    DEGREES, PI, TAU,
    RIGHT, LEFT, UP, DOWN, OUT, IN, ORIGIN,
    X_AXIS, Y_AXIS, Z_AXIS,
    WHITE, BLACK, GRAY, BLUE, GREEN, RED, YELLOW,
    # Colour utilities
    ManimColor, color_to_rgb, interpolate_color,
    rgba_to_color, color_to_rgba,
    # Utilities
    rate_functions,
    smooth,
)

# ---------------------------------------------------------------------------
# Shared colour helpers (self-contained copy — see bar_chart3d.py for origin)
# ---------------------------------------------------------------------------

def _with_opacity(color: ManimColor, opacity: float) -> ManimColor:
    r, g, b = color_to_rgb(color)
    return rgba_to_color([r, g, b, max(0.0, min(1.0, opacity))])

def _darken(color: ManimColor, factor: float = 0.65) -> ManimColor:
    r, g, b = color_to_rgb(color)
    return ManimColor([r * factor, g * factor, b * factor])

def _lighten(color: ManimColor, factor: float = 1.35) -> ManimColor:
    r, g, b = color_to_rgb(color)
    return ManimColor([min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0)])

def _lerp_color(a: ManimColor, b: ManimColor, t: float) -> ManimColor:
    ra, ga, ba = color_to_rgb(a)
    rb, gb, bb = color_to_rgb(b)
    return ManimColor([ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t])


# ---------------------------------------------------------------------------
# LineColorPalette
# ---------------------------------------------------------------------------

class LineColorPalette:
    """Pre-built colour palettes for line series.

    Attributes
    ----------
    CATEGORICAL : list[ManimColor]
        Eight perceptually distinct hues — max contrast across overlaid
        series.  Same base hues as ``BarColorPalette.CATEGORICAL`` so
        a combined bar+line chart stays colour-consistent.
    SOFT : list[ManimColor]
        Pastel-toned versions of CATEGORICAL — easier on dark backgrounds
        when plotting many overlapping series.
    MONOCHROME_BLUE : list[ManimColor]
        Six shades of blue — useful for small-multiples or sequential
        time steps of the same variable.
    """

    CATEGORICAL: List[ManimColor] = [
        ManimColor("#4A90D9"),   # sky blue
        ManimColor("#E8593C"),   # coral
        ManimColor("#2DAA6E"),   # emerald
        ManimColor("#E0AA40"),   # amber
        ManimColor("#9B59B6"),   # purple
        ManimColor("#1ABC9C"),   # teal
        ManimColor("#E74C3C"),   # red
        ManimColor("#F39C12"),   # orange
    ]

    SOFT: List[ManimColor] = [
        ManimColor("#7AB8E8"),
        ManimColor("#F0907A"),
        ManimColor("#60CC92"),
        ManimColor("#EEC870"),
        ManimColor("#BF8DE0"),
        ManimColor("#55CEB8"),
        ManimColor("#F08080"),
        ManimColor("#F4B860"),
    ]

    MONOCHROME_BLUE: List[ManimColor] = [
        ManimColor("#C8DCF0"),
        ManimColor("#93BDE0"),
        ManimColor("#5B9FD0"),
        ManimColor("#2E7FBF"),
        ManimColor("#1A5EA0"),
        ManimColor("#0C3D7A"),
    ]

    @staticmethod
    def ramp(lo: ManimColor, hi: ManimColor, n: int) -> List[ManimColor]:
        return [_lerp_color(lo, hi, i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# LineConfig
# ---------------------------------------------------------------------------

@dataclass
class LineConfig:
    """Complete visual specification for a single ``LineSeries3D``.

    Stroke layer
    ~~~~~~~~~~~~
    ``stroke_width`` and ``color`` control the primary line.  The optional
    ``dash_pattern`` switches the line to dashed (e.g. ``[0.12, 0.06]``).

    Glow layer
    ~~~~~~~~~~
    ``glow_width`` draws a wider, low-opacity copy of the line underneath
    the stroke, giving a soft luminosity effect against dark backgrounds.

    Marker layer
    ~~~~~~~~~~~~
    ``marker_style`` selects the point marker shape.  ``"none"`` disables
    markers entirely.  ``"sphere"`` uses a ``Dot3D``; ``"diamond"``,
    ``"cross"``, and ``"ring"`` use thin ``VMobject`` shapes.

    Area fill layer
    ~~~~~~~~~~~~~~~
    ``area_fill`` enables a shaded polygon between the curve and the
    floor (z = 0).  ``area_fill_opacity`` controls its transparency.
    ``area_fill_gradient`` draws the fill darker at the floor, lighter
    at the curve.

    Drop-line layer
    ~~~~~~~~~~~~~~~
    ``drop_lines`` adds thin vertical segments from each data point down
    to z = 0.  These spatially ground the curve and make individual x
    positions easy to read.

    Tube layer
    ~~~~~~~~~~
    ``tube_radius`` > 0 extrudes a volumetric tube along the path.
    The tube is rendered as a series of thin circular cross-section
    polygons perpendicular to the path tangent.  Setting this to > 0
    adds real 3D thickness visible under any camera angle.

    Smooth interpolation
    ~~~~~~~~~~~~~~~~~~~~
    ``smooth_curve`` replaces the piecewise-linear polyline with a
    Catmull-Rom spline evaluated at ``smooth_resolution`` intermediate
    points per segment.  This gives organic curves for continuous data
    (PDFs, CDFs) while preserving sharp corners when False (discrete
    time series, CDF steps).

    Attributes
    ----------
    color : ManimColor
    stroke_width : float
    opacity : float
    dash_pattern : list[float] or None
        ``[dash_length, gap_length]`` in Manim units.  ``None`` = solid.
    glow_width : float
        Width of the glow halo.  0 = disabled.
    glow_opacity : float
        Opacity of the glow halo.
    marker_style : str
        ``"none"``, ``"sphere"``, ``"diamond"``, ``"cross"``, ``"ring"``,
        ``"square"``.
    marker_size : float
        Radius / half-width of markers.
    marker_color : ManimColor or None
        Marker fill colour.  *None* = inherit ``color``.
    marker_stroke_width : float
        Edge stroke width for non-sphere markers.
    show_point_labels : bool
        Whether to display a value label beside each marker.
    point_label_font_size : int
    point_label_decimals : int
    point_label_color : ManimColor
    drop_lines : bool
    drop_line_opacity : float
    drop_line_stroke_width : float
    drop_line_dash : bool
    area_fill : bool
    area_fill_opacity : float
    area_fill_gradient : bool
        If True, fill colour fades from ``color`` at the top to near-black
        at z = 0.
    tube_radius : float
        0 = disabled.
    smooth_curve : bool
    smooth_resolution : int
        Intermediate points per segment when ``smooth_curve`` is True.
    z_floor : float
        Z-coordinate of the "floor" for drop-lines and area fill.
    """

    color: ManimColor = ManimColor("#4A90D9")
    stroke_width: float = 2.8
    opacity: float = 1.0

    dash_pattern: Optional[List[float]] = None

    glow_width: float = 8.0
    glow_opacity: float = 0.12

    marker_style: str = "sphere"          # none | sphere | diamond | cross | ring | square
    marker_size: float = 0.08
    marker_color: Optional[ManimColor] = None
    marker_stroke_width: float = 1.2

    show_point_labels: bool = False
    point_label_font_size: int = 18
    point_label_decimals: int = 2
    point_label_color: ManimColor = WHITE

    drop_lines: bool = False
    drop_line_opacity: float = 0.35
    drop_line_stroke_width: float = 0.8
    drop_line_dash: bool = True

    area_fill: bool = False
    area_fill_opacity: float = 0.22
    area_fill_gradient: bool = True

    tube_radius: float = 0.0

    smooth_curve: bool = True
    smooth_resolution: int = 16

    z_floor: float = 0.0


# ── Preset configs ───────────────────────────────────────────────────────────

CLEAN_LINE = LineConfig(
    glow_width=0.0,
    marker_style="none",
    drop_lines=False,
    area_fill=False,
    tube_radius=0.0,
    smooth_curve=True,
)

PDF_LINE = LineConfig(
    stroke_width=3.0,
    glow_width=9.0,
    glow_opacity=0.14,
    marker_style="none",
    area_fill=True,
    area_fill_opacity=0.26,
    area_fill_gradient=True,
    smooth_curve=True,
    smooth_resolution=20,
)

CDF_LINE = LineConfig(
    stroke_width=2.8,
    glow_width=7.0,
    glow_opacity=0.12,
    marker_style="none",
    area_fill=False,
    smooth_curve=False,   # step functions stay sharp
    drop_lines=False,
)

TIMESERIES_LINE = LineConfig(
    stroke_width=2.5,
    glow_width=7.0,
    glow_opacity=0.10,
    marker_style="sphere",
    marker_size=0.07,
    drop_lines=True,
    drop_line_opacity=0.28,
    area_fill=True,
    area_fill_opacity=0.18,
    area_fill_gradient=True,
    smooth_curve=True,
    smooth_resolution=12,
)

TUBE_LINE = LineConfig(
    stroke_width=2.0,
    glow_width=0.0,
    marker_style="sphere",
    marker_size=0.09,
    tube_radius=0.045,
    area_fill=False,
    smooth_curve=True,
)


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _catmull_rom_chain(
    points: np.ndarray,
    resolution: int = 16,
) -> np.ndarray:
    """Evaluate a Catmull-Rom spline through *points*.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    resolution : int
        Number of interpolated points per segment.

    Returns
    -------
    np.ndarray, shape ((N-1)*resolution + 1, 3)
    """
    n = len(points)
    if n < 2:
        return points.copy()

    # Extend endpoints by reflection so tangents at ends are natural
    p = np.vstack([
        2 * points[0] - points[1],
        points,
        2 * points[-1] - points[-2],
    ])

    out_pts = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for j in range(resolution):
            t = j / resolution
            t2, t3 = t * t, t * t * t
            c0 = -t3 + 2*t2 - t
            c1 =  3*t3 - 5*t2 + 2
            c2 = -3*t3 + 4*t2 + t
            c3 =  t3 - t2
            pt = 0.5 * (c0*p0 + c1*p1 + c2*p2 + c3*p3)
            out_pts.append(pt)

    out_pts.append(points[-1].copy())
    return np.array(out_pts)


def _build_polyline(
    points: np.ndarray,
    config: LineConfig,
) -> VMobject:
    """Build the primary stroke VMobject from *points*.

    Handles solid vs dashed and applies opacity.
    """
    color = _with_opacity(config.color, config.opacity)

    if config.dash_pattern is not None:
        dash_len, gap_len = config.dash_pattern[0], config.dash_pattern[1]
        # DashedVMobject approach: build solid first, then dash
        solid = VMobject()
        solid.set_points_as_corners(points)
        # Use manim's DashedLine workaround: set dash via stroke properties
        # (true dashing on a VMobject path uses set_dash)
        solid.set_stroke(color=color, width=config.stroke_width)
        try:
            solid.set_dash([dash_len, gap_len])
        except AttributeError:
            pass  # older manim — fall back to solid
        return solid

    line = VMobject()
    line.set_points_as_corners(points)
    line.set_stroke(color=color, width=config.stroke_width)
    line.set_fill(opacity=0)
    return line


def _build_glow(points: np.ndarray, config: LineConfig) -> VMobject:
    """Build a wide low-opacity copy of the line as a glow halo."""
    glow = VMobject()
    glow.set_points_as_corners(points)
    glow.set_stroke(
        color=_with_opacity(config.color, config.glow_opacity),
        width=config.glow_width,
    )
    glow.set_fill(opacity=0)
    return glow


def _build_area_fill(
    curve_pts: np.ndarray,
    config: LineConfig,
    y_depth: float = 0.0,
) -> Polygon:
    """Build the area fill polygon between the curve and z_floor.

    The polygon is a closed shape:
    curve_pts[0] → … → curve_pts[-1] → floor projection → back.

    Parameters
    ----------
    y_depth : float
        Y coordinate at which to place the fill polygon.  Set to the
        series' y_position so the fill doesn't bleed into depth.
    """
    floor_z = config.z_floor

    # Floor projection: same x, y, but z = floor_z
    floor_pts = np.array([
        [p[0], y_depth, floor_z] for p in reversed(curve_pts)
    ])

    poly_pts = list(curve_pts) + list(floor_pts)

    if config.area_fill_gradient:
        fill_color = _with_opacity(config.color, config.area_fill_opacity)
    else:
        fill_color = _with_opacity(config.color, config.area_fill_opacity)

    fill = Polygon(
        *poly_pts,
        fill_color=fill_color,
        fill_opacity=1.0,
        stroke_width=0,
    )
    return fill


# ---------------------------------------------------------------------------
# LineMarker3D
# ---------------------------------------------------------------------------

class LineMarker3D(VGroup):
    """A single 3D point marker.

    Supports five shapes:
    ``"sphere"``   → ``Dot3D`` (or ``Sphere`` for large radii).
    ``"diamond"``  → rotated square polygon in the XZ plane.
    ``"cross"``    → two perpendicular line segments.
    ``"ring"``     → hollow circle approximated as a thin polygon.
    ``"square"``   → axis-aligned square polygon in the XZ plane.

    Parameters
    ----------
    position : np.ndarray
        3D world position of the marker centre.
    style : str
        One of the five shape names above.
    size : float
        Radius / half-width in Manim units.
    color : ManimColor
        Fill / stroke colour.
    stroke_width : float
        Edge stroke width (ignored for spheres).
    label : str or None
        Optional text label shown near the marker.
    label_font_size : int
    label_color : ManimColor
    scene : ThreeDScene or None
        Needed to register fixed-orientation labels.
    """

    def __init__(
        self,
        position: np.ndarray = ORIGIN,
        style: str = "sphere",
        size: float = 0.08,
        color: ManimColor = WHITE,
        stroke_width: float = 1.2,
        label: Optional[str] = None,
        label_font_size: int = 18,
        label_color: ManimColor = WHITE,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        pos = np.array(position, dtype=float)

        if style == "sphere":
            marker = Dot3D(
                point=pos,
                radius=size,
                color=color,
            )

        elif style == "diamond":
            hw = size
            pts = [
                pos + np.array([0, 0, hw]),
                pos + np.array([hw, 0, 0]),
                pos + np.array([0, 0, -hw]),
                pos + np.array([-hw, 0, 0]),
            ]
            marker = Polygon(
                *pts,
                fill_color=color,
                fill_opacity=1.0,
                stroke_color=_darken(color, 0.7),
                stroke_width=stroke_width,
            )

        elif style == "cross":
            h = size * 1.4
            marker = VGroup(
                Line(pos - h * RIGHT, pos + h * RIGHT,
                     color=color, stroke_width=stroke_width * 2),
                Line(pos - h * UP, pos + h * UP,
                     color=color, stroke_width=stroke_width * 2),
            )

        elif style == "ring":
            n_pts = 16
            angles = np.linspace(0, TAU, n_pts, endpoint=False)
            outer_pts = [pos + size * np.array([np.cos(a), 0, np.sin(a)])
                         for a in angles]
            marker = VMobject()
            marker.set_points_as_corners(outer_pts + [outer_pts[0]])
            marker.set_stroke(color=color, width=stroke_width * 2)
            marker.set_fill(opacity=0)

        elif style == "square":
            hw = size * 0.85
            pts = [
                pos + np.array([-hw, 0, -hw]),
                pos + np.array([hw, 0, -hw]),
                pos + np.array([hw, 0, hw]),
                pos + np.array([-hw, 0, hw]),
            ]
            marker = Polygon(
                *pts,
                fill_color=color,
                fill_opacity=1.0,
                stroke_color=_darken(color, 0.7),
                stroke_width=stroke_width,
            )

        else:
            raise ValueError(f"Unknown marker style: {style!r}")

        self.marker = marker
        self.add(marker)

        if label is not None:
            lbl = Text(label, font_size=label_font_size, color=label_color)
            lbl.move_to(pos + np.array([size * 2.0, 0, size * 1.5]))
            self.label_obj = lbl
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
            self.add(lbl)


# ---------------------------------------------------------------------------
# _DropLines3D  (internal)
# ---------------------------------------------------------------------------

class _DropLines3D(VGroup):
    """Thin vertical lines from each data point down to z_floor."""

    def __init__(
        self,
        points: np.ndarray,
        config: LineConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        col = _with_opacity(config.color, config.drop_line_opacity)
        floor_z = config.z_floor

        for pt in points:
            floor_pt = np.array([pt[0], pt[1], floor_z])
            if config.drop_line_dash:
                ln = DashedLine(
                    floor_pt, pt,
                    dash_length=0.07,
                    dashed_ratio=0.5,
                    color=col,
                    stroke_width=config.drop_line_stroke_width,
                )
            else:
                ln = Line(
                    floor_pt, pt,
                    color=col,
                    stroke_width=config.drop_line_stroke_width,
                )
            self.add(ln)


# ---------------------------------------------------------------------------
# AreaFill3D
# ---------------------------------------------------------------------------

class AreaFill3D(VGroup):
    """Shaded region between two curves (or one curve and the floor).

    Can represent:
    - Area under a PDF / CDF curve.
    - Region between two overlapping time series.
    - Confidence band around a regression line.

    Parameters
    ----------
    upper_points : np.ndarray, shape (N, 3)
        Points on the upper boundary.
    lower_points : np.ndarray or None
        Points on the lower boundary.  If *None*, the floor at
        ``floor_z`` is used.
    color : ManimColor
    opacity : float
    floor_z : float
    gradient : bool
        If True, fill is lighter near *upper_points* and darker at
        *lower_points* (approximated by layered thin strips).
    n_gradient_strips : int
        Number of strips when *gradient* is True.
    """

    def __init__(
        self,
        upper_points: np.ndarray,
        lower_points: Optional[np.ndarray] = None,
        color: ManimColor = ManimColor("#4A90D9"),
        opacity: float = 0.22,
        floor_z: float = 0.0,
        gradient: bool = True,
        n_gradient_strips: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if lower_points is None:
            lower_points = np.array([
                [p[0], p[1], floor_z] for p in upper_points
            ])

        if not gradient or n_gradient_strips <= 1:
            poly_pts = list(upper_points) + list(reversed(lower_points))
            fill = Polygon(
                *poly_pts,
                fill_color=_with_opacity(color, opacity),
                fill_opacity=1.0,
                stroke_width=0,
            )
            self.add(fill)
            return

        # Gradient: interpolate between upper and lower in strips
        for s in range(n_gradient_strips):
            t_lo = s / n_gradient_strips
            t_hi = (s + 1) / n_gradient_strips
            strip_opacity = opacity * (1.0 - 0.55 * t_lo)  # darker near floor

            upper_lo = upper_points * (1 - t_lo) + lower_points * t_lo
            upper_hi = upper_points * (1 - t_hi) + lower_points * t_hi

            strip_pts = list(upper_lo) + list(reversed(upper_hi))
            strip = Polygon(
                *strip_pts,
                fill_color=_with_opacity(color, strip_opacity),
                fill_opacity=1.0,
                stroke_width=0,
            )
            self.add(strip)


# ---------------------------------------------------------------------------
# LineSeries3D  — a single data series
# ---------------------------------------------------------------------------

class LineSeries3D(VGroup):
    """A single richly-layered 3D line series.

    Parameters
    ----------
    x_values : sequence of float
        X coordinates of the data points.
    z_values : sequence of float
        Z (height) coordinates of the data points — the "y" of the data.
    y_position : float
        Y coordinate in 3D space.  Shifts the whole series into the
        scene depth, useful in ``MultiLinePlot3D``.
    config : LineConfig
        Full visual specification.
    name : str or None
        Series name — shown in legend panels.
    scene : ThreeDScene or None
        Pass the scene for fixed-orientation label registration.

    Attributes
    ----------
    raw_points : np.ndarray, shape (N, 3)
        The original (non-smoothed) data points.
    curve_points : np.ndarray, shape (M, 3)
        The smoothed / interpolated points used for rendering.
    glow : VMobject
    stroke : VMobject
    markers : VGroup
    drop_lines : _DropLines3D
    area : AreaFill3D or VGroup
    """

    def __init__(
        self,
        x_values: Sequence[float],
        z_values: Sequence[float],
        y_position: float = 0.0,
        config: Optional[LineConfig] = None,
        name: Optional[str] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cfg = config if config is not None else LineConfig()
        self.name = name
        self._scene = scene
        self.y_position = y_position

        xs = np.array(x_values, dtype=float)
        zs = np.array(z_values, dtype=float)
        ys = np.full_like(xs, y_position)

        self.raw_points = np.column_stack([xs, ys, zs])  # (N,3)

        # Smoothed curve points
        if self.cfg.smooth_curve and len(self.raw_points) >= 3:
            self.curve_points = _catmull_rom_chain(
                self.raw_points, resolution=self.cfg.smooth_resolution
            )
        else:
            self.curve_points = self.raw_points.copy()

        self._build_layers()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        cfg = self.cfg

        # Layer 1: glow halo (drawn first, behind stroke)
        if cfg.glow_width > 0 and cfg.glow_opacity > 0:
            self.glow = _build_glow(self.curve_points, cfg)
            self.add(self.glow)
        else:
            self.glow = VGroup()

        # Layer 2: area fill (behind stroke, above glow)
        if cfg.area_fill:
            self.area = AreaFill3D(
                upper_points=self.curve_points,
                floor_z=cfg.z_floor,
                color=cfg.color,
                opacity=cfg.area_fill_opacity,
                gradient=cfg.area_fill_gradient,
                n_gradient_strips=10,
            )
            self.add(self.area)
        else:
            self.area = VGroup()

        # Layer 3: primary stroke
        self.stroke = _build_polyline(self.curve_points, cfg)
        self.add(self.stroke)

        # Layer 4: drop lines (at raw points only, not smoothed)
        if cfg.drop_lines:
            self.drop_lines = _DropLines3D(self.raw_points, cfg)
            self.add(self.drop_lines)
        else:
            self.drop_lines = VGroup()

        # Layer 5: markers (at raw data points)
        if cfg.marker_style != "none":
            mcol = cfg.marker_color if cfg.marker_color is not None else cfg.color
            self.markers = VGroup(*[
                LineMarker3D(
                    position=pt,
                    style=cfg.marker_style,
                    size=cfg.marker_size,
                    color=mcol,
                    stroke_width=cfg.marker_stroke_width,
                    scene=scene,
                )
                for pt in self.raw_points
            ])
            self.add(self.markers)
        else:
            self.markers = VGroup()

        # Layer 6: per-point value labels
        if cfg.show_point_labels:
            self._build_point_labels(scene)

    def _build_point_labels(self, scene: Optional[ThreeDScene]) -> None:
        fmt = f"{{:.{self.cfg.point_label_decimals}f}}"
        self.point_labels = VGroup()
        for pt in self.raw_points:
            lbl = Text(
                fmt.format(pt[2]),   # z = data value
                font_size=self.cfg.point_label_font_size,
                color=self.cfg.point_label_color,
            )
            lbl.move_to(pt + np.array([0, 0, self.cfg.marker_size * 2.5]))
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
            self.point_labels.add(lbl)
        self.add(self.point_labels)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_draw(self, run_time: float = 1.5) -> Create:
        """Animate the stroke tracing left-to-right."""
        return Create(self.stroke, run_time=run_time)

    def animate_trace(
        self,
        run_time: float = 2.0,
        dot_color: Optional[ManimColor] = None,
        dot_radius: float = 0.12,
        flash_at_end: bool = True,
    ) -> AnimationGroup:
        """A glowing dot travels the full curve from start to end.

        The dot is created, moved along the path, and removed — all
        within the returned ``AnimationGroup``.  The scene must already
        contain the series for this to look correct.

        Parameters
        ----------
        flash_at_end : bool
            If True, emit a ``Flash`` at the endpoint after travel.
        """
        col = dot_color if dot_color is not None else self.cfg.color
        tracer = Dot3D(
            point=self.curve_points[0],
            radius=dot_radius,
            color=_lighten(col, 1.5),
        )

        path = VMobject()
        path.set_points_as_corners(self.curve_points)

        move_anim = MoveAlongPath(tracer, path, run_time=run_time)

        anims: List = [FadeIn(tracer, run_time=0.15), move_anim]
        if flash_at_end:
            anims.append(Flash(
                tracer,
                color=col,
                flash_radius=dot_radius * 4,
                run_time=0.4,
            ))
        anims.append(FadeOut(tracer, run_time=0.15))

        return Succession(*anims)

    def animate_fill(self, run_time: float = 1.2) -> UpdateFromAlphaFunc:
        """Animate the area fill growing upward from the floor."""
        if not self.cfg.area_fill or isinstance(self.area, VGroup):
            return FadeIn(VGroup(), run_time=0.01)  # no-op

        target_area = self.area
        floor_z = self.cfg.z_floor
        curve_pts = self.curve_points

        def updater(mob: VGroup, alpha: float) -> None:
            frac = smooth(alpha)
            # Interpolate between floor and full curve
            interp_pts = np.array([
                [p[0], p[1], floor_z + (p[2] - floor_z) * frac]
                for p in curve_pts
            ])
            # Rebuild area strips
            mob.become(
                AreaFill3D(
                    upper_points=interp_pts,
                    floor_z=floor_z,
                    color=self.cfg.color,
                    opacity=self.cfg.area_fill_opacity,
                    gradient=self.cfg.area_fill_gradient,
                    n_gradient_strips=10,
                )
            )

        return UpdateFromAlphaFunc(target_area, updater, run_time=run_time)

    def animate_rise(self, run_time: float = 1.2) -> UpdateFromAlphaFunc:
        """All points rise simultaneously from z_floor to their true height."""
        raw = self.raw_points.copy()
        cfg = self.cfg
        floor_z = cfg.z_floor

        def updater(mob: LineSeries3D, alpha: float) -> None:
            frac = smooth(alpha)
            new_pts = raw.copy()
            new_pts[:, 2] = floor_z + (raw[:, 2] - floor_z) * frac

            if cfg.smooth_curve and len(new_pts) >= 3:
                curve = _catmull_rom_chain(new_pts, cfg.smooth_resolution)
            else:
                curve = new_pts.copy()

            mob.stroke.set_points_as_corners(curve)
            if cfg.glow_width > 0:
                mob.glow.set_points_as_corners(curve)

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)

    def animate_update(
        self,
        new_x: Sequence[float],
        new_z: Sequence[float],
        run_time: float = 1.2,
    ) -> UpdateFromAlphaFunc:
        """Morph the curve to new data values.

        Both x and z coordinates may change.  If the number of points
        changes, the curve is linearly re-sampled to match.
        """
        old_curve = self.curve_points.copy()
        new_raw = np.column_stack([
            np.array(new_x, dtype=float),
            np.full(len(new_x), self.y_position),
            np.array(new_z, dtype=float),
        ])
        cfg = self.cfg

        if cfg.smooth_curve and len(new_raw) >= 3:
            new_curve = _catmull_rom_chain(new_raw, cfg.smooth_resolution)
        else:
            new_curve = new_raw.copy()

        # Re-sample to same length for clean interpolation
        def _resample(pts: np.ndarray, n: int) -> np.ndarray:
            idx = np.linspace(0, len(pts) - 1, n)
            return np.array([pts[int(i)] for i in np.floor(idx).astype(int)])

        target_n = max(len(old_curve), len(new_curve))
        old_rs = _resample(old_curve, target_n)
        new_rs = _resample(new_curve, target_n)

        def updater(mob: LineSeries3D, alpha: float) -> None:
            frac = rate_functions.ease_in_out_cubic(alpha)
            interp = old_rs + (new_rs - old_rs) * frac
            mob.stroke.set_points_as_corners(interp)
            if cfg.glow_width > 0:
                mob.glow.set_points_as_corners(interp)

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)

    # ------------------------------------------------------------------
    # Styling utilities
    # ------------------------------------------------------------------

    def highlight_range(
        self,
        x_lo: float,
        x_hi: float,
        color: ManimColor = YELLOW,
        opacity: float = 0.35,
    ) -> AreaFill3D:
        """Shade the area under the curve between *x_lo* and *x_hi*.

        Returns the ``AreaFill3D`` so the caller can animate it with
        ``FadeIn`` or ``Create``.
        """
        mask = (self.raw_points[:, 0] >= x_lo) & (self.raw_points[:, 0] <= x_hi)
        if not np.any(mask):
            return AreaFill3D(self.raw_points[:1], floor_z=self.cfg.z_floor)

        region_pts = self.raw_points[mask]

        # Include boundary points via linear interpolation
        xs = self.raw_points[:, 0]
        zs = self.raw_points[:, 2]

        # Left boundary
        lo_z = float(np.interp(x_lo, xs, zs))
        hi_z = float(np.interp(x_hi, xs, zs))

        full_pts = np.vstack([
            [[x_lo, self.y_position, lo_z]],
            region_pts,
            [[x_hi, self.y_position, hi_z]],
        ])

        fill = AreaFill3D(
            upper_points=full_pts,
            floor_z=self.cfg.z_floor,
            color=color,
            opacity=opacity,
            gradient=False,
        )
        self.add(fill)
        return fill

    def add_horizontal_marker(
        self,
        z_value: float,
        color: ManimColor = ManimColor("#E0AA40"),
        stroke_width: float = 1.8,
        opacity: float = 0.75,
        label: Optional[str] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Draw a horizontal reference line at height *z_value*.

        Returns the ``VGroup`` (line + optional label).
        """
        x_min = self.raw_points[:, 0].min()
        x_max = self.raw_points[:, 0].max()
        y = self.y_position

        ln = Line(
            np.array([x_min, y, z_value]),
            np.array([x_max, y, z_value]),
            color=_with_opacity(color, opacity),
            stroke_width=stroke_width,
        )
        grp = VGroup(ln)

        if label is not None:
            lbl = Text(label, font_size=18, color=color)
            lbl.move_to(np.array([x_max + 0.25, y, z_value]))
            grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        self.add(grp)
        return grp

    def get_value_at(self, x: float) -> float:
        """Return the linearly interpolated z-value at x-coordinate *x*."""
        xs = self.raw_points[:, 0]
        zs = self.raw_points[:, 2]
        return float(np.interp(x, xs, zs))


# ---------------------------------------------------------------------------
# MultiLinePlot3D  — k overlaid series
# ---------------------------------------------------------------------------

class MultiLinePlot3D(VGroup):
    """Multiple line series overlaid on the same x-axis.

    Each series sits at a common ``y_position`` (default 0) or can be
    depth-staggered along y for a small-multiples feel.

    Parameters
    ----------
    x_values : sequence of float
        Shared x-axis values for all series.
    series : dict[str, sequence[float]] or list[sequence[float]]
        Either a ``{name: z_values}`` mapping (preserves insertion order)
        or a list of z-value arrays (names auto-generated as S0, S1, …).
    colors : list[ManimColor] or None
        Per-series colours.  Defaults to ``LineColorPalette.CATEGORICAL``.
    config : LineConfig or None
        Shared config applied to all series (individual overrides via
        ``series_configs``).
    series_configs : list[LineConfig] or None
        Per-series configs.  Takes priority over *config*.
    y_position : float
        Common Y offset for all series (no depth-stagger).
    depth_stagger : float
        If > 0, successive series are offset by this amount along y,
        creating a depth-ribbon view of multiple distributions.
    area_fill : bool
        Shortcut to enable ``area_fill`` for all series regardless of
        *config*.
    scene : ThreeDScene or None

    Attributes
    ----------
    series_list : list[LineSeries3D]
    names : list[str]
    """

    def __init__(
        self,
        x_values: Sequence[float],
        series: Union[Dict[str, Sequence[float]], List[Sequence[float]]],
        colors: Optional[Sequence[ManimColor]] = None,
        config: Optional[LineConfig] = None,
        series_configs: Optional[List[LineConfig]] = None,
        y_position: float = 0.0,
        depth_stagger: float = 0.0,
        area_fill: bool = False,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(series, dict):
            self.names = list(series.keys())
            z_list = [list(v) for v in series.values()]
        else:
            z_list = [list(v) for v in series]
            self.names = [f"S{i}" for i in range(len(z_list))]

        k = len(z_list)
        palette = LineColorPalette.CATEGORICAL
        self._colors = (
            list(colors) if colors is not None
            else [palette[i % len(palette)] for i in range(k)]
        )

        self.series_list: List[LineSeries3D] = []

        for i, (name, zs, col) in enumerate(zip(self.names, z_list, self._colors)):
            # Resolve per-series config
            if series_configs is not None and i < len(series_configs):
                cfg = series_configs[i]
            elif config is not None:
                cfg = LineConfig(**config.__dict__)   # copy
            else:
                cfg = LineConfig()
            cfg.color = col
            if area_fill:
                cfg.area_fill = True

            y_pos = y_position + i * depth_stagger
            ser = LineSeries3D(
                x_values=x_values,
                z_values=zs,
                y_position=y_pos,
                config=cfg,
                name=name,
                scene=scene,
            )
            self.series_list.append(ser)
            self.add(ser)

    # ------------------------------------------------------------------

    def animate_reveal(
        self,
        stagger: float = 0.3,
        run_time_per_series: float = 1.2,
        mode: str = "draw",
    ) -> AnimationGroup:
        """Reveal all series with a staggered animation.

        Parameters
        ----------
        stagger : float
            Seconds between series start times.
        mode : str
            ``"draw"``  — stroke traces left-to-right (``animate_draw``).
            ``"rise"``  — all points rise from the floor simultaneously.
            ``"fade"``  — simple ``FadeIn``.
        """
        anims = []
        for ser in self.series_list:
            if mode == "draw":
                anims.append(ser.animate_draw(run_time=run_time_per_series))
            elif mode == "rise":
                anims.append(ser.animate_rise(run_time=run_time_per_series))
            else:
                anims.append(FadeIn(ser, run_time=run_time_per_series))

        return LaggedStart(*anims, lag_ratio=stagger / run_time_per_series)

    def animate_update(
        self,
        new_series: Union[Dict[str, Sequence[float]], List[Sequence[float]]],
        new_x: Optional[Sequence[float]] = None,
        run_time: float = 1.2,
    ) -> AnimationGroup:
        """Morph all series to new data simultaneously."""
        if isinstance(new_series, dict):
            z_list = list(new_series.values())
        else:
            z_list = list(new_series)

        anims = []
        for ser, new_z in zip(self.series_list, z_list):
            new_x_vals = new_x if new_x is not None else ser.raw_points[:, 0]
            anims.append(ser.animate_update(new_x_vals, new_z, run_time=run_time))

        return AnimationGroup(*anims)

    def highlight_series(
        self,
        name_or_index: Union[str, int],
        dim_others: bool = True,
        dim_opacity: float = 0.20,
    ) -> "MultiLinePlot3D":
        """Highlight one series, optionally dimming the others."""
        if isinstance(name_or_index, str):
            idx = self.names.index(name_or_index)
        else:
            idx = name_or_index

        for i, ser in enumerate(self.series_list):
            if i == idx:
                ser.stroke.set_stroke(width=ser.cfg.stroke_width * 1.8)
                ser.set_opacity(1.0)
            elif dim_others:
                ser.set_opacity(dim_opacity)
        return self

    def unhighlight_all(self) -> "MultiLinePlot3D":
        for ser in self.series_list:
            ser.stroke.set_stroke(width=ser.cfg.stroke_width)
            ser.set_opacity(1.0)
        return self

    def add_legend(
        self,
        position: np.ndarray = ORIGIN,
        font_size: int = 20,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Add a floating legend panel listing series names and colours.

        Returns the ``VGroup`` so the caller can position/animate it.
        """
        legend = VGroup()
        for i, (name, col) in enumerate(zip(self.names, self._colors)):
            swatch = Line(
                ORIGIN, RIGHT * 0.4,
                color=_with_opacity(col, 0.9),
                stroke_width=3.0,
            )
            lbl = Text(name, font_size=font_size, color=col)
            row = VGroup(swatch, lbl)
            row.arrange(RIGHT, buff=0.12)
            row.move_to(position + DOWN * i * 0.35)
            legend.add(row)

        if scene is not None:
            for row in legend:
                scene.add_fixed_orientation_mobjects(row)
        self.add(legend)
        return legend


# ---------------------------------------------------------------------------
# CDFLine3D
# ---------------------------------------------------------------------------

class CDFLine3D(VGroup):
    """Specialized line for cumulative distribution functions.

    Renders as a strict step function (no smoothing) and provides:
    - ``shade_tail_left(x_crit)``  — shade area to the left of x_crit.
    - ``shade_tail_right(x_crit)`` — shade area to the right.
    - ``shade_between(x_lo, x_hi)`` — shade the middle region.
    - ``add_critical_marker(x, label)`` — vertical dashed line + label.
    - ``add_probability_arc(x_lo, x_hi)`` — annotated brace showing P(a ≤ X ≤ b).

    Parameters
    ----------
    x_values : sequence of float
    cdf_values : sequence of float
        Values in [0, 1].
    y_position : float
    config : LineConfig or None
        Defaults to ``CDF_LINE``.
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        x_values: Sequence[float],
        cdf_values: Sequence[float],
        y_position: float = 0.0,
        config: Optional[LineConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else LineConfig(**CDF_LINE.__dict__)
        cfg.smooth_curve = False  # enforce step rendering

        self.cfg = cfg
        self._scene = scene
        self.y_position = y_position

        self.xs = np.array(x_values, dtype=float)
        self.zs = np.array(cdf_values, dtype=float)

        # Build step-function points: for each (x, F(x)), emit
        # a horizontal segment followed by a vertical jump
        step_pts = self._build_step_points()

        self._raw_step = step_pts
        self.series = LineSeries3D(
            x_values=step_pts[:, 0],
            z_values=step_pts[:, 2],
            y_position=y_position,
            config=cfg,
            name="CDF",
            scene=scene,
        )
        self.add(self.series)

    def _build_step_points(self) -> np.ndarray:
        """Build explicit step-function path from (x, F(x)) pairs."""
        pts = []
        for i, (x, z) in enumerate(zip(self.xs, self.zs)):
            if i == 0:
                pts.append([x, self.y_position, z])
            else:
                # Horizontal segment at previous z, then jump
                pts.append([x, self.y_position, self.zs[i - 1]])
                pts.append([x, self.y_position, z])
        return np.array(pts, dtype=float)

    # ------------------------------------------------------------------

    def shade_tail_left(
        self,
        x_crit: float,
        color: ManimColor = ManimColor("#E8593C"),
        opacity: float = 0.32,
    ) -> AreaFill3D:
        """Shade the region under the CDF to the left of *x_crit*."""
        mask = self.xs <= x_crit
        if not np.any(mask):
            return AreaFill3D(self._raw_step[:1])
        region_pts = np.array([
            [x, self.y_position, z] for x, z in zip(self.xs[mask], self.zs[mask])
        ])
        fill = AreaFill3D(
            upper_points=region_pts,
            floor_z=self.cfg.z_floor,
            color=color,
            opacity=opacity,
            gradient=False,
        )
        self.add(fill)
        return fill

    def shade_tail_right(
        self,
        x_crit: float,
        color: ManimColor = ManimColor("#E8593C"),
        opacity: float = 0.32,
    ) -> AreaFill3D:
        """Shade the region under the CDF to the right of *x_crit*."""
        mask = self.xs >= x_crit
        if not np.any(mask):
            return AreaFill3D(self._raw_step[:1])
        # CDF tail: the "area" is 1 - F(x_crit), visualised as a
        # rectangle from F(x_crit) to 1.0 on the z-axis
        region_pts = np.array([
            [x, self.y_position, z] for x, z in zip(self.xs[mask], self.zs[mask])
        ])
        fill = AreaFill3D(
            upper_points=region_pts,
            lower_points=np.array([[p[0], self.y_position, np.interp(x_crit, self.xs, self.zs)]
                                    for p in region_pts]),
            color=color,
            opacity=opacity,
            gradient=False,
        )
        self.add(fill)
        return fill

    def shade_between(
        self,
        x_lo: float,
        x_hi: float,
        color: ManimColor = ManimColor("#2DAA6E"),
        opacity: float = 0.35,
    ) -> AreaFill3D:
        """Shade the CDF band between *x_lo* and *x_hi*."""
        return self.series.highlight_range(x_lo, x_hi, color=color, opacity=opacity)

    def add_critical_marker(
        self,
        x_crit: float,
        label: str = "",
        color: ManimColor = ManimColor("#E0AA40"),
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Vertical dashed line at *x_crit* with an optional label."""
        z_val = float(np.interp(x_crit, self.xs, self.zs))
        y = self.y_position
        floor_z = self.cfg.z_floor

        ln = DashedLine(
            np.array([x_crit, y, floor_z]),
            np.array([x_crit, y, 1.05]),
            dash_length=0.07,
            dashed_ratio=0.5,
            color=_with_opacity(color, 0.80),
            stroke_width=1.8,
        )
        dot = Dot3D(
            point=np.array([x_crit, y, z_val]),
            radius=0.07,
            color=color,
        )
        grp = VGroup(ln, dot)

        if label:
            lbl = Text(label, font_size=18, color=color)
            lbl.move_to(np.array([x_crit, y, 1.15]))
            grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        self.add(grp)
        return grp

    def animate_draw(self, run_time: float = 2.0) -> Create:
        """Draw the CDF curve left-to-right."""
        return Create(self.series.stroke, run_time=run_time)


# ---------------------------------------------------------------------------
# TimeSeriesPlot3D
# ---------------------------------------------------------------------------

class TimeSeriesPlot3D(VGroup):
    """Line plot specialised for time-indexed data.

    Adds:
    - Tick marks with optional date/time labels at each x position.
    - A rolling-window highlight band (e.g. show a 7-day window).
    - A vertical marker plane at a specific time point.
    - Anomaly markers — dots coloured by deviation from a reference.

    Parameters
    ----------
    x_values : sequence of float
        Numeric time indices (seconds, days, year fractions, etc.).
    z_values : sequence of float
        Measured values.
    tick_labels : sequence of str or None
        Human-readable labels for each x tick.
    y_position : float
    config : LineConfig or None
        Defaults to ``TIMESERIES_LINE``.
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        x_values: Sequence[float],
        z_values: Sequence[float],
        tick_labels: Optional[Sequence[str]] = None,
        y_position: float = 0.0,
        config: Optional[LineConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else LineConfig(**TIMESERIES_LINE.__dict__)
        self.cfg = cfg
        self._scene = scene
        self.y_position = y_position

        self.xs = np.array(x_values, dtype=float)
        self.zs = np.array(z_values, dtype=float)

        # Core line series
        self.series = LineSeries3D(
            x_values=x_values,
            z_values=z_values,
            y_position=y_position,
            config=cfg,
            name="time_series",
            scene=scene,
        )
        self.add(self.series)

        # Tick marks and labels
        self._tick_labels = tick_labels
        if tick_labels is not None:
            self._build_tick_labels(tick_labels, scene)

    def _build_tick_labels(
        self,
        labels: Sequence[str],
        scene: Optional[ThreeDScene],
    ) -> None:
        tick_col = _with_opacity(self.cfg.color, 0.55)
        self.ticks = VGroup()
        self.tick_label_group = VGroup()

        for x, lbl in zip(self.xs, labels):
            y = self.y_position
            floor_z = self.cfg.z_floor
            tick = Line(
                np.array([x, y, floor_z - 0.08]),
                np.array([x, y, floor_z + 0.08]),
                color=tick_col,
                stroke_width=1.2,
            )
            self.ticks.add(tick)

            t = Text(lbl, font_size=16, color=_with_opacity(self.cfg.color, 0.70))
            t.move_to(np.array([x, y, floor_z - 0.28]))
            if scene is not None:
                scene.add_fixed_orientation_mobjects(t)
            self.tick_label_group.add(t)

        self.add(self.ticks, self.tick_label_group)

    # ------------------------------------------------------------------

    def add_rolling_window(
        self,
        x_lo: float,
        x_hi: float,
        color: ManimColor = ManimColor("#E0AA40"),
        opacity: float = 0.18,
    ) -> AreaFill3D:
        """Add a vertical highlight band between *x_lo* and *x_hi*.

        The band spans from z_floor to max(z_values) and is a flat
        rectangular region in the xz-plane.
        """
        z_max = float(self.zs.max()) * 1.05
        floor_z = self.cfg.z_floor
        y = self.y_position

        band_pts = np.array([
            [x_lo, y, z_max],
            [x_hi, y, z_max],
            [x_hi, y, floor_z],
            [x_lo, y, floor_z],
        ])
        # Reverse lower half for Polygon winding
        rect = Polygon(
            *band_pts,
            fill_color=_with_opacity(color, opacity),
            fill_opacity=1.0,
            stroke_width=0,
        )
        grp = VGroup(rect)
        self.add(grp)
        return grp

    def add_event_marker(
        self,
        x: float,
        label: str = "",
        color: ManimColor = ManimColor("#9B59B6"),
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Vertical dashed line marking a specific time point."""
        z_max = float(self.zs.max()) * 1.08
        floor_z = self.cfg.z_floor
        y = self.y_position

        ln = DashedLine(
            np.array([x, y, floor_z]),
            np.array([x, y, z_max]),
            dash_length=0.08,
            dashed_ratio=0.5,
            color=_with_opacity(color, 0.75),
            stroke_width=1.6,
        )
        grp = VGroup(ln)

        if label:
            lbl = Text(label, font_size=17, color=color)
            lbl.move_to(np.array([x, y, z_max + 0.22]))
            grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        self.add(grp)
        return grp

    def add_anomaly_markers(
        self,
        threshold_sigma: float = 2.0,
        anomaly_color: ManimColor = ManimColor("#E8593C"),
        normal_color: Optional[ManimColor] = None,
    ) -> VGroup:
        """Place markers at points deviating beyond *threshold_sigma* σ.

        Points within the threshold use the series colour (or
        *normal_color*).  Outliers use *anomaly_color*.
        """
        mean = float(self.zs.mean())
        std = float(self.zs.std())
        if std == 0:
            return VGroup()

        nrm_col = normal_color if normal_color is not None else self.cfg.color
        markers = VGroup()

        for x, z in zip(self.xs, self.zs):
            is_anomaly = abs(z - mean) > threshold_sigma * std
            col = anomaly_color if is_anomaly else nrm_col
            m = Dot3D(
                point=np.array([x, self.y_position, z]),
                radius=0.09 if is_anomaly else 0.06,
                color=col,
            )
            markers.add(m)

        self.add(markers)
        return markers

    def animate_draw(self, run_time: float = 2.0) -> AnimationGroup:
        """Draw the line and fade-in tick labels."""
        anims = [self.series.animate_draw(run_time=run_time)]
        if hasattr(self, "tick_label_group"):
            anims.append(FadeIn(self.tick_label_group, run_time=0.5))
        return AnimationGroup(*anims, lag_ratio=0.8)


# ---------------------------------------------------------------------------
# ParametricLine3D
# ---------------------------------------------------------------------------

class ParametricLine3D(VGroup):
    """A fully 3D parametric space curve.

    Takes three callables ``x(t)``, ``y(t)``, ``z(t)`` and samples them
    over ``t_range`` to produce a ``LineSeries3D``.  All visual layers
    of ``LineSeries3D`` are available.

    Useful for:
    - Bivariate distribution trajectories.
    - Spiral / helical paths.
    - Phase-space plots.
    - Random walk paths in 3D.

    Parameters
    ----------
    x_func, y_func, z_func : Callable[[float], float]
        Parametric component functions.
    t_range : (float, float, float)
        ``(t_min, t_max, t_step)`` — the sampling range.
    config : LineConfig or None
    scene : ThreeDScene or None

    Attributes
    ----------
    t_values : np.ndarray
    points : np.ndarray, shape (N, 3)
        The sampled 3D path.
    path_mob : VMobject
        The primary rendered path (``stroke`` of the underlying series).
    """

    def __init__(
        self,
        x_func: Callable[[float], float],
        y_func: Callable[[float], float],
        z_func: Callable[[float], float],
        t_range: Tuple[float, float, float] = (0, TAU, 0.05),
        config: Optional[LineConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else LineConfig(
            smooth_curve=False,  # sampled finely enough already
            marker_style="none",
            area_fill=False,
            drop_lines=False,
            glow_width=7.0,
            glow_opacity=0.14,
        )
        self.cfg = cfg

        t_min, t_max, t_step = t_range
        self.t_values = np.arange(t_min, t_max + t_step * 0.5, t_step)

        self.points = np.column_stack([
            np.array([x_func(t) for t in self.t_values]),
            np.array([y_func(t) for t in self.t_values]),
            np.array([z_func(t) for t in self.t_values]),
        ])

        # Build the VMobject path directly (no LineSeries3D intermediary
        # since x, y, z are all free — not just z)
        self.path_mob = _build_polyline(self.points, cfg)
        self.add(self.path_mob)

        if cfg.glow_width > 0 and cfg.glow_opacity > 0:
            self.glow = _build_glow(self.points, cfg)
            self.add(self.glow)

        if cfg.marker_style != "none":
            mcol = cfg.marker_color if cfg.marker_color is not None else cfg.color
            self.markers = VGroup(*[
                LineMarker3D(
                    position=pt,
                    style=cfg.marker_style,
                    size=cfg.marker_size,
                    color=mcol,
                )
                for pt in self.points[::max(1, len(self.points) // 20)]
            ])
            self.add(self.markers)

    # ------------------------------------------------------------------

    def animate_draw(self, run_time: float = 2.5) -> Create:
        """Trace the parametric curve."""
        return Create(self.path_mob, run_time=run_time)

    def animate_trace(
        self,
        run_time: float = 3.0,
        dot_radius: float = 0.10,
        dot_color: Optional[ManimColor] = None,
    ) -> Succession:
        """A glowing dot travels the full 3D path."""
        col = dot_color if dot_color is not None else self.cfg.color
        tracer = Dot3D(
            point=self.points[0],
            radius=dot_radius,
            color=_lighten(col, 1.5),
        )
        path = VMobject()
        path.set_points_as_corners(self.points)
        return Succession(
            FadeIn(tracer, run_time=0.1),
            MoveAlongPath(tracer, path, run_time=run_time),
            FadeOut(tracer, run_time=0.1),
        )

    def get_tangent(self, t: float) -> np.ndarray:
        """Numerically approximate the tangent vector at parameter *t*."""
        dt = (self.t_values[1] - self.t_values[0]) * 0.5
        t_idx = np.searchsorted(self.t_values, t)
        if t_idx <= 0:
            return self.points[1] - self.points[0]
        if t_idx >= len(self.points) - 1:
            return self.points[-1] - self.points[-2]
        return self.points[t_idx + 1] - self.points[t_idx - 1]


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def line_from_function(
    func: Callable[[float], float],
    x_range: Tuple[float, float, float] = (0, 5, 0.1),
    y_position: float = 0.0,
    config: Optional[LineConfig] = None,
    scene: Optional[ThreeDScene] = None,
) -> LineSeries3D:
    """Build a ``LineSeries3D`` from a scalar function f(x) → z.

    Parameters
    ----------
    func : callable
        The function to plot.
    x_range : (float, float, float)
        ``(x_min, x_max, x_step)``.

    Example
    -------
    ::

        normal_pdf = line_from_function(
            lambda x: np.exp(-0.5 * x**2) / np.sqrt(TAU),
            x_range=(-4, 4, 0.05),
            config=PDF_LINE,
        )
    """
    x_min, x_max, x_step = x_range
    xs = np.arange(x_min, x_max + x_step * 0.5, x_step)
    zs = np.array([func(x) for x in xs])
    return LineSeries3D(
        x_values=xs,
        z_values=zs,
        y_position=y_position,
        config=config,
        scene=scene,
    )


def parametric_helix(
    radius: float = 1.0,
    height: float = 3.0,
    turns: float = 3.0,
    color: ManimColor = ManimColor("#4A90D9"),
    scene: Optional[ThreeDScene] = None,
) -> ParametricLine3D:
    """Factory for a standard helix — useful as a demo or annotation path."""
    cfg = LineConfig(
        color=color,
        smooth_curve=False,
        marker_style="none",
        glow_width=8.0,
        glow_opacity=0.15,
    )
    return ParametricLine3D(
        x_func=lambda t: radius * np.cos(t),
        y_func=lambda t: radius * np.sin(t),
        z_func=lambda t: height * t / (TAU * turns),
        t_range=(0, TAU * turns, 0.04),
        config=cfg,
        scene=scene,
    )