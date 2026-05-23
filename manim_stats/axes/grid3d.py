"""
manim_stats/axes/grid3d.py
==========================
Production-quality 3D grid system for statistical visualizations.

Classes
-------
GridConfig
    Dataclass holding all styling parameters for a grid plane.

GridPlane3D
    A single richly-styled grid plane (XY, XZ, or YZ) with:
    - Major/minor line subdivision
    - Depth-fade along grid edges
    - Zero-line emphasis
    - Subtle fill behind the plane
    - Animated sequential build-in

GridBoundingBox3D
    A full wireframe bounding box around a 3D data region,
    with optional corner tick marks and face-diagonal guides.

FloatingGrid3D
    A detached grid backdrop not anchored to any axis system.
    Useful as a chart background or floor plane.

FullGrid3D
    Composite of all three axis-aligned GridPlane3D objects,
    with a unified colour theme and synchronized animations.

BillboardLabel3D
    A text label that always rotates to face the active camera,
    used for axis tick labels in 3D.

GridSnapHelper
    Utility (non-rendered) that maps data values to snapped
    grid positions, ensuring plotted objects land exactly on
    grid intersections.

Usage example
-------------
    from manim import *
    from manim_stats.axes.grid3d import FullGrid3D, GridConfig

    class Demo(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65*DEGREES, theta=-45*DEGREES)

            grid = FullGrid3D(
                x_range=[-4, 4, 1],
                y_range=[-4, 4, 1],
                z_range=[0, 6, 1],
                minor_subdivisions=4,
            )
            self.play(grid.animate_build(lag=0.03))
            self.wait()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple, Optional, Callable
import numpy as np

from manim import (
    VGroup, VMobject, Line, DashedLine, Rectangle, Polygon,
    Text, MathTex, DecimalNumber,
    ThreeDScene, always_rotate,
    DEGREES, PI, TAU,
    RIGHT, LEFT, UP, DOWN, OUT, IN,
    WHITE, BLACK, GRAY, BLUE, GREEN, RED, YELLOW,
    ORIGIN, X_AXIS, Y_AXIS, Z_AXIS,
    interpolate_color, color_to_rgba, rgba_to_color,
    ManimColor, color_to_rgb,
    AnimationGroup, LaggedStart, Create, FadeIn, DrawBorderThenFill,
    Scene,
    config,
)

# ---------------------------------------------------------------------------
# Internal colour helpers
# ---------------------------------------------------------------------------

def _with_opacity(color: ManimColor, opacity: float) -> ManimColor:
    """Return *color* with its opacity replaced by *opacity* (0–1)."""
    r, g, b = color_to_rgb(color)
    return rgba_to_color([r, g, b, opacity])


def _fade_color(
    color: ManimColor,
    base_opacity: float,
    fade_factor: float,
) -> ManimColor:
    """Multiply a colour's opacity by *fade_factor* (0–1)."""
    return _with_opacity(color, base_opacity * fade_factor)


# ---------------------------------------------------------------------------
# GridConfig
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """All visual parameters for a single grid plane.

    Attributes
    ----------
    major_color : ManimColor
        Colour of major grid lines (multiples of the step).
    minor_color : ManimColor
        Colour of minor (subdivision) grid lines.
    zero_color : ManimColor
        Colour of the zero-crossing lines on this plane.
    fill_color : ManimColor
        Background fill of the plane rectangle.
    major_stroke_width : float
        Stroke width for major grid lines.
    minor_stroke_width : float
        Stroke width for minor (subdivision) grid lines.
    zero_stroke_width : float
        Stroke width for the zero-crossing lines.
    major_opacity : float
        Opacity for major grid lines (before depth-fade is applied).
    minor_opacity : float
        Opacity for minor grid lines.
    fill_opacity : float
        Opacity of the background fill rectangle.
    edge_fade : bool
        If True, grid lines fade to transparent near the plane edges.
    edge_fade_fraction : float
        Fraction of the plane width/height over which the edge fade
        is applied (0 = no fade zone, 0.2 = fade over 20% near edges).
    dashed_minor : bool
        If True, minor grid lines are drawn as dashed lines.
    dash_length : float
        Length of each dash segment for dashed minor lines.
    dash_ratio : float
        Ratio of dash to gap for dashed minor lines (0–1).
    """

    major_color: ManimColor = BLUE
    minor_color: ManimColor = BLUE
    zero_color: ManimColor = WHITE
    fill_color: ManimColor = BLUE

    major_stroke_width: float = 1.0
    minor_stroke_width: float = 0.4
    zero_stroke_width: float = 2.0

    major_opacity: float = 0.45
    minor_opacity: float = 0.18
    zero_opacity: float = 0.70
    fill_opacity: float = 0.04

    edge_fade: bool = True
    edge_fade_fraction: float = 0.12

    dashed_minor: bool = False
    dash_length: float = 0.05
    dash_ratio: float = 0.5


# Default theme presets

DARK_GRID = GridConfig(
    major_color=ManimColor("#4A90D9"),
    minor_color=ManimColor("#3A70A9"),
    zero_color=ManimColor("#88BBFF"),
    fill_color=ManimColor("#1A2A4A"),
    major_opacity=0.50,
    minor_opacity=0.20,
    zero_opacity=0.80,
    fill_opacity=0.06,
)

LIGHT_GRID = GridConfig(
    major_color=ManimColor("#607080"),
    minor_color=ManimColor("#9AABB8"),
    zero_color=ManimColor("#2244AA"),
    fill_color=ManimColor("#D0DCE8"),
    major_opacity=0.40,
    minor_opacity=0.18,
    zero_opacity=0.65,
    fill_opacity=0.08,
)

STATS_GRID = GridConfig(
    major_color=ManimColor("#5B7FA6"),
    minor_color=ManimColor("#3D5A78"),
    zero_color=ManimColor("#E0AA40"),
    fill_color=ManimColor("#0E1D2E"),
    major_stroke_width=1.2,
    minor_stroke_width=0.5,
    zero_stroke_width=2.4,
    major_opacity=0.55,
    minor_opacity=0.22,
    zero_opacity=0.85,
    fill_opacity=0.07,
    edge_fade=True,
    edge_fade_fraction=0.15,
    dashed_minor=True,
    dash_length=0.06,
    dash_ratio=0.45,
)


# ---------------------------------------------------------------------------
# GridPlane3D
# ---------------------------------------------------------------------------

class GridPlane3D(VGroup):
    """A richly styled single grid plane for 3D scenes.

    Parameters
    ----------
    axis_u : np.ndarray
        Direction of the first (horizontal) axis on the plane.
        Typically RIGHT, UP, or OUT.
    axis_v : np.ndarray
        Direction of the second (vertical) axis on the plane.
    u_range : (float, float, float)
        (min, max, step) for the u-axis grid lines.
    v_range : (float, float, float)
        (min, max, step) for the v-axis grid lines.
    minor_subdivisions : int
        Number of minor divisions per major step (0 = no minors).
    origin : np.ndarray
        3D point where u=0, v=0 maps.
    config : GridConfig
        Visual styling config.  If *None*, uses ``STATS_GRID``.
    label_u_values : bool
        Whether to add tick-value labels along the u-axis edge.
    label_v_values : bool
        Whether to add tick-value labels along the v-axis edge.
    label_font_size : int
        Font size for tick labels.
    num_decimal_places : int
        Decimal places shown in tick labels.

    Notes
    -----
    All grid lines are built as individual ``Line`` / ``DashedLine``
    objects so that they can be animated independently (e.g. drawn
    one at a time with ``LaggedStart``).
    """

    def __init__(
        self,
        axis_u: np.ndarray = RIGHT,
        axis_v: np.ndarray = UP,
        u_range: Tuple[float, float, float] = (-4, 4, 1),
        v_range: Tuple[float, float, float] = (-4, 4, 1),
        minor_subdivisions: int = 4,
        origin: np.ndarray = ORIGIN,
        config: Optional[GridConfig] = None,
        label_u_values: bool = False,
        label_v_values: bool = False,
        label_font_size: int = 18,
        num_decimal_places: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cfg = config if config is not None else STATS_GRID
        self.axis_u = np.array(axis_u, dtype=float)
        self.axis_v = np.array(axis_v, dtype=float)
        self.u_range = u_range
        self.v_range = v_range
        self.minor_subdivisions = minor_subdivisions
        self.origin = np.array(origin, dtype=float)

        # Sub-groups (public so callers can animate selectively)
        self.fill_plane = VGroup()
        self.zero_lines = VGroup()
        self.major_lines = VGroup()
        self.minor_lines = VGroup()
        self.tick_labels = VGroup()

        self._build_fill()
        self._build_grid_lines()
        if label_u_values or label_v_values:
            self._build_tick_labels(
                label_u=label_u_values,
                label_v=label_v_values,
                font_size=label_font_size,
                decimals=num_decimal_places,
            )

        self.add(
            self.fill_plane,
            self.minor_lines,
            self.major_lines,
            self.zero_lines,
            self.tick_labels,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _uv_to_3d(self, u: float, v: float) -> np.ndarray:
        return self.origin + u * self.axis_u + v * self.axis_v

    def _edge_fade_opacity(
        self,
        t: float,
        t_min: float,
        t_max: float,
        base_opacity: float,
    ) -> float:
        """Compute fade-adjusted opacity for a grid line at parameter *t*.

        Lines near t_min or t_max fade toward zero.  Lines in the
        interior keep *base_opacity*.
        """
        if not self.cfg.edge_fade:
            return base_opacity
        frac = self.cfg.edge_fade_fraction
        t_range = t_max - t_min
        if t_range == 0:
            return base_opacity
        rel = (t - t_min) / t_range  # 0 … 1
        # Smooth step that is 0 at edges, 1 in centre
        lo_fade = np.clip(rel / frac, 0, 1)
        hi_fade = np.clip((1 - rel) / frac, 0, 1)
        fade = min(lo_fade, hi_fade)
        # Smooth: 3t²-2t³
        smooth = fade * fade * (3 - 2 * fade)
        return base_opacity * smooth

    def _make_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: ManimColor,
        opacity: float,
        stroke_width: float,
        dashed: bool = False,
    ) -> VMobject:
        if dashed:
            ln = DashedLine(
                start, end,
                dash_length=self.cfg.dash_length,
                dashed_ratio=self.cfg.dash_ratio,
                color=_with_opacity(color, opacity),
                stroke_width=stroke_width,
            )
        else:
            ln = Line(
                start, end,
                color=_with_opacity(color, opacity),
                stroke_width=stroke_width,
            )
        return ln

    def _build_fill(self) -> None:
        u_min, u_max, _ = self.u_range
        v_min, v_max, _ = self.v_range
        corners = [
            self._uv_to_3d(u_min, v_min),
            self._uv_to_3d(u_max, v_min),
            self._uv_to_3d(u_max, v_max),
            self._uv_to_3d(u_min, v_max),
        ]
        rect = Polygon(
            *corners,
            fill_color=_with_opacity(self.cfg.fill_color, self.cfg.fill_opacity),
            fill_opacity=1.0,  # opacity already baked into colour
            stroke_width=0,
        )
        self.fill_plane.add(rect)

    def _build_grid_lines(self) -> None:
        u_min, u_max, u_step = self.u_range
        v_min, v_max, v_step = self.v_range
        cfg = self.cfg

        # --- lines parallel to axis_v (varying u) ---
        self._add_lines_along_axis(
            fixed_axis="u",
            fixed_min=u_min,
            fixed_max=u_max,
            fixed_step=u_step,
            other_min=v_min,
            other_max=v_max,
        )

        # --- lines parallel to axis_u (varying v) ---
        self._add_lines_along_axis(
            fixed_axis="v",
            fixed_min=v_min,
            fixed_max=v_max,
            fixed_step=v_step,
            other_min=u_min,
            other_max=u_max,
        )

    def _add_lines_along_axis(
        self,
        fixed_axis: str,   # "u" or "v"
        fixed_min: float,
        fixed_max: float,
        fixed_step: float,
        other_min: float,
        other_max: float,
    ) -> None:
        """Add all major, minor, and zero lines sweeping across one axis."""
        cfg = self.cfg
        n_major = round((fixed_max - fixed_min) / fixed_step)

        for i in range(n_major + 1):
            t_major = fixed_min + i * fixed_step
            is_zero = abs(t_major) < 1e-9

            fade_op = self._edge_fade_opacity(
                t_major, fixed_min, fixed_max,
                cfg.zero_opacity if is_zero else cfg.major_opacity,
            )

            if fixed_axis == "u":
                start = self._uv_to_3d(t_major, other_min)
                end = self._uv_to_3d(t_major, other_max)
            else:
                start = self._uv_to_3d(other_min, t_major)
                end = self._uv_to_3d(other_max, t_major)

            if is_zero:
                ln = self._make_line(
                    start, end,
                    cfg.zero_color, fade_op,
                    cfg.zero_stroke_width, dashed=False,
                )
                self.zero_lines.add(ln)
            else:
                ln = self._make_line(
                    start, end,
                    cfg.major_color, fade_op,
                    cfg.major_stroke_width, dashed=False,
                )
                self.major_lines.add(ln)

            # Minor lines between this major tick and the next
            if self.minor_subdivisions > 0 and i < n_major:
                minor_step = fixed_step / self.minor_subdivisions
                for j in range(1, self.minor_subdivisions):
                    t_minor = t_major + j * minor_step
                    fade_minor = self._edge_fade_opacity(
                        t_minor, fixed_min, fixed_max,
                        cfg.minor_opacity,
                    )
                    if fixed_axis == "u":
                        ms = self._uv_to_3d(t_minor, other_min)
                        me = self._uv_to_3d(t_minor, other_max)
                    else:
                        ms = self._uv_to_3d(other_min, t_minor)
                        me = self._uv_to_3d(other_max, t_minor)

                    mln = self._make_line(
                        ms, me,
                        cfg.minor_color, fade_minor,
                        cfg.minor_stroke_width, dashed=cfg.dashed_minor,
                    )
                    self.minor_lines.add(mln)

    def _build_tick_labels(
        self,
        label_u: bool,
        label_v: bool,
        font_size: int,
        decimals: int,
    ) -> None:
        """Add numeric tick labels along the outer edge of the plane."""
        u_min, u_max, u_step = self.u_range
        v_min, v_max, v_step = self.v_range
        fmt = f"{{:.{decimals}f}}"

        label_offset_scale = 0.25  # how far outside the plane edge

        if label_u:
            n = round((u_max - u_min) / u_step)
            for i in range(n + 1):
                val = u_min + i * u_step
                pos = (
                    self._uv_to_3d(val, v_min)
                    - label_offset_scale * self.axis_v
                )
                lbl = Text(
                    fmt.format(val),
                    font_size=font_size,
                    color=_with_opacity(self.cfg.major_color, 0.70),
                )
                lbl.move_to(pos)
                self.tick_labels.add(lbl)

        if label_v:
            n = round((v_max - v_min) / v_step)
            for i in range(n + 1):
                val = v_min + i * v_step
                pos = (
                    self._uv_to_3d(u_min, val)
                    - label_offset_scale * self.axis_u
                )
                lbl = Text(
                    fmt.format(val),
                    font_size=font_size,
                    color=_with_opacity(self.cfg.major_color, 0.70),
                )
                lbl.move_to(pos)
                self.tick_labels.add(lbl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def animate_build(
        self,
        lag: float = 0.02,
        run_time_per_line: float = 0.4,
    ) -> AnimationGroup:
        """Return an animation that draws the plane line-by-line.

        The build order is: fill → minor lines → major lines → zero lines.
        Tick labels fade in at the end.

        Parameters
        ----------
        lag : float
            Seconds between the start of consecutive line animations.
        run_time_per_line : float
            Duration of each individual line's ``Create`` animation.
        """
        anims = []

        anims.append(FadeIn(self.fill_plane, run_time=0.3))

        line_groups = [self.minor_lines, self.major_lines, self.zero_lines]
        for group in line_groups:
            for ln in group:
                anims.append(Create(ln, run_time=run_time_per_line))

        if len(self.tick_labels) > 0:
            anims.append(FadeIn(self.tick_labels, run_time=0.5))

        return LaggedStart(*anims, lag_ratio=lag)

    def set_opacity(self, opacity: float) -> "GridPlane3D":
        """Scale all line opacities by *opacity* (0–1). Returns self."""
        for group in (self.major_lines, self.minor_lines, self.zero_lines):
            for ln in group:
                ln.set_stroke(opacity=opacity)
        self.fill_plane.set_fill(opacity=self.cfg.fill_opacity * opacity)
        return self

    def highlight_line(
        self,
        axis: str,
        value: float,
        color: ManimColor = YELLOW,
        stroke_width: float = 3.0,
        opacity: float = 1.0,
    ) -> Optional[VMobject]:
        """Find the major grid line nearest to *value* on *axis* and
        temporarily highlight it.  Returns the line (caller can animate).

        Parameters
        ----------
        axis : str
            ``"u"`` or ``"v"`` – which axis family to search.
        value : float
            The data value whose nearest grid line to highlight.
        """
        u_min, u_max, u_step = self.u_range
        v_min, v_max, v_step = self.v_range

        if axis == "u":
            snap = round((value - u_min) / u_step) * u_step + u_min
            start = self._uv_to_3d(snap, v_min)
            end = self._uv_to_3d(snap, v_max)
        else:
            snap = round((value - v_min) / v_step) * v_step + v_min
            start = self._uv_to_3d(u_min, snap)
            end = self._uv_to_3d(u_max, snap)

        ln = Line(
            start, end,
            color=_with_opacity(color, opacity),
            stroke_width=stroke_width,
        )
        self.add(ln)
        return ln

    def get_snap_position(self, u_val: float, v_val: float) -> np.ndarray:
        """Return the 3D position of the grid point nearest to (u_val, v_val)."""
        u_min, _, u_step = self.u_range
        v_min, _, v_step = self.v_range
        u_snapped = round((u_val - u_min) / u_step) * u_step + u_min
        v_snapped = round((v_val - v_min) / v_step) * v_step + v_min
        return self._uv_to_3d(u_snapped, v_snapped)


# ---------------------------------------------------------------------------
# GridBoundingBox3D
# ---------------------------------------------------------------------------

class GridBoundingBox3D(VGroup):
    """A wireframe bounding box around a 3D data region.

    Draws the 12 edges of a rectangular prism, with optional corner
    tick marks and face-diagonal guides (useful for showing axes).

    Parameters
    ----------
    x_range, y_range, z_range : (float, float)
        (min, max) extents along each axis.
    color : ManimColor
        Edge colour.
    stroke_width : float
        Edge thickness.
    opacity : float
        Edge opacity.
    corner_ticks : bool
        Whether to draw small tick marks at each of the 8 corners.
    tick_length : float
        Length of each corner tick.
    face_diagonals : bool
        If True, draw light diagonal guides on each face.
    diagonal_opacity : float
        Opacity of the face diagonals.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        z_range: Tuple[float, float] = (0, 4),
        color: ManimColor = ManimColor("#5B7FA6"),
        stroke_width: float = 1.2,
        opacity: float = 0.60,
        corner_ticks: bool = True,
        tick_length: float = 0.12,
        face_diagonals: bool = False,
        diagonal_opacity: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        x0, x1 = x_range
        y0, y1 = y_range
        z0, z1 = z_range

        # 8 corners of the bounding box
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1],
            [x1, y1, z1], [x0, y1, z1],
        ])

        # 12 edges  (pairs of corner indices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
        ]

        edge_color = _with_opacity(color, opacity)
        self.edges = VGroup()
        for a, b in edges:
            ln = Line(
                corners[a], corners[b],
                color=edge_color,
                stroke_width=stroke_width,
            )
            self.edges.add(ln)
        self.add(self.edges)

        # Corner tick marks
        if corner_ticks:
            self.ticks = VGroup()
            tick_dirs = [
                np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
            ]
            tick_color = _with_opacity(color, min(opacity * 1.3, 1.0))
            for c in corners:
                for d in tick_dirs:
                    t = Line(
                        c - d * tick_length / 2,
                        c + d * tick_length / 2,
                        color=tick_color,
                        stroke_width=stroke_width * 1.5,
                    )
                    self.ticks.add(t)
            self.add(self.ticks)

        # Face diagonal guides
        if face_diagonals:
            self.diagonals = VGroup()
            diag_color = _with_opacity(color, diagonal_opacity)
            face_pairs = [
                (0, 6), (1, 7),  # bottom-to-top diagonals across faces
                (2, 4), (3, 5),
            ]
            for a, b in face_pairs:
                d = DashedLine(
                    corners[a], corners[b],
                    dash_length=0.08,
                    dashed_ratio=0.5,
                    color=diag_color,
                    stroke_width=0.5,
                )
                self.diagonals.add(d)
            self.add(self.diagonals)

    def animate_build(self, run_time: float = 1.5) -> AnimationGroup:
        """Draw all edges, then ticks, then diagonals."""
        anims = [LaggedStart(
            *[Create(e) for e in self.edges],
            lag_ratio=0.06,
            run_time=run_time,
        )]
        if hasattr(self, "ticks"):
            anims.append(FadeIn(self.ticks, run_time=0.4))
        if hasattr(self, "diagonals"):
            anims.append(LaggedStart(
                *[Create(d) for d in self.diagonals],
                lag_ratio=0.08,
                run_time=0.6,
            ))
        return AnimationGroup(*anims, lag_ratio=0.15)


# ---------------------------------------------------------------------------
# FloatingGrid3D
# ---------------------------------------------------------------------------

class FloatingGrid3D(GridPlane3D):
    """A grid plane that lives anywhere in 3D space, not necessarily
    through the origin.

    This is a thin convenience wrapper over :class:`GridPlane3D` that
    accepts a ``center`` keyword argument and pre-translates the origin.

    Parameters
    ----------
    center : np.ndarray
        3D centre of the grid.
    width : float
        Full width of the grid (axis_u direction).
    height : float
        Full height of the grid (axis_v direction).
    step : float
        Major grid step in both directions.
    minor_subdivisions : int
        Minor subdivision count per major step.
    axis_u, axis_v : np.ndarray
        Plane basis vectors.
    config : GridConfig
        Visual config. Defaults to ``STATS_GRID``.
    """

    def __init__(
        self,
        center: np.ndarray = ORIGIN,
        width: float = 8.0,
        height: float = 8.0,
        step: float = 1.0,
        minor_subdivisions: int = 4,
        axis_u: np.ndarray = RIGHT,
        axis_v: np.ndarray = UP,
        config: Optional[GridConfig] = None,
        **kwargs,
    ):
        hw = width / 2
        hh = height / 2
        super().__init__(
            axis_u=axis_u,
            axis_v=axis_v,
            u_range=(-hw, hw, step),
            v_range=(-hh, hh, step),
            minor_subdivisions=minor_subdivisions,
            origin=np.array(center, dtype=float),
            config=config,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# FullGrid3D
# ---------------------------------------------------------------------------

class FullGrid3D(VGroup):
    """Three axis-aligned grid planes (XY, XZ, YZ) forming a full 3D grid.

    Parameters
    ----------
    x_range, y_range, z_range : (float, float, float)
        (min, max, step) for each axis.
    minor_subdivisions : int
        Minor subdivisions per major step (applies to all planes).
    config : GridConfig
        Shared visual config.  If *None*, uses ``STATS_GRID``.
    show_xy : bool
        Whether to include the XY (floor) plane.
    show_xz : bool
        Whether to include the XZ (back wall) plane.
    show_yz : bool
        Whether to include the YZ (side wall) plane.
    label_axes : bool
        Whether to add tick labels on each plane.

    Attributes
    ----------
    xy_plane : GridPlane3D
    xz_plane : GridPlane3D
    yz_plane : GridPlane3D
    bounding_box : GridBoundingBox3D
    """

    def __init__(
        self,
        x_range: Tuple[float, float, float] = (-4, 4, 1),
        y_range: Tuple[float, float, float] = (-4, 4, 1),
        z_range: Tuple[float, float, float] = (0, 5, 1),
        minor_subdivisions: int = 4,
        config: Optional[GridConfig] = None,
        show_xy: bool = True,
        show_xz: bool = True,
        show_yz: bool = True,
        show_bounding_box: bool = False,
        label_axes: bool = False,
        label_font_size: int = 18,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else STATS_GRID
        x0, x1, xs = x_range
        y0, y1, ys = y_range
        z0, z1, zs = z_range

        # XY plane (floor) — normal is Z
        if show_xy:
            xy_cfg = GridConfig(**cfg.__dict__)  # copy
            self.xy_plane = GridPlane3D(
                axis_u=RIGHT,
                axis_v=OUT,        # Manim's OUT = +y direction (in 3D scene)
                u_range=x_range,
                v_range=y_range,
                minor_subdivisions=minor_subdivisions,
                origin=np.array([0, 0, z0]),
                config=xy_cfg,
                label_u_values=label_axes,
                label_v_values=label_axes,
                label_font_size=label_font_size,
            )
            self.add(self.xy_plane)

        # XZ plane (back wall) — normal is Y
        if show_xz:
            xz_cfg = GridConfig(**cfg.__dict__)
            # Make the xz plane slightly more transparent to not overwhelm
            xz_cfg.major_opacity = cfg.major_opacity * 0.75
            xz_cfg.minor_opacity = cfg.minor_opacity * 0.75
            xz_cfg.fill_opacity = cfg.fill_opacity * 0.5
            self.xz_plane = GridPlane3D(
                axis_u=RIGHT,
                axis_v=UP,
                u_range=x_range,
                v_range=z_range,
                minor_subdivisions=minor_subdivisions,
                origin=np.array([0, y1, 0]),
                config=xz_cfg,
                label_u_values=label_axes,
                label_v_values=label_axes,
                label_font_size=label_font_size,
            )
            self.add(self.xz_plane)

        # YZ plane (side wall) — normal is X
        if show_yz:
            yz_cfg = GridConfig(**cfg.__dict__)
            yz_cfg.major_opacity = cfg.major_opacity * 0.65
            yz_cfg.minor_opacity = cfg.minor_opacity * 0.65
            yz_cfg.fill_opacity = cfg.fill_opacity * 0.40
            self.yz_plane = GridPlane3D(
                axis_u=OUT,        # Manim's OUT = +y
                axis_v=UP,
                u_range=y_range,
                v_range=z_range,
                minor_subdivisions=minor_subdivisions,
                origin=np.array([x0, 0, 0]),
                config=yz_cfg,
                label_u_values=label_axes,
                label_v_values=label_axes,
                label_font_size=label_font_size,
            )
            self.add(self.yz_plane)

        # Optional bounding box
        if show_bounding_box:
            self.bounding_box = GridBoundingBox3D(
                x_range=(x0, x1),
                y_range=(y0, y1),
                z_range=(z0, z1),
                color=cfg.major_color,
                opacity=0.50,
            )
            self.add(self.bounding_box)

    # ------------------------------------------------------------------

    def animate_build(
        self,
        order: Sequence[str] = ("xy", "xz", "yz", "bounding_box"),
        lag_between_planes: float = 0.2,
        line_lag: float = 0.015,
    ) -> AnimationGroup:
        """Animate each plane building in sequence.

        Parameters
        ----------
        order : sequence of str
            Which planes to animate and in what order.
            Valid values: ``"xy"``, ``"xz"``, ``"yz"``, ``"bounding_box"``.
        lag_between_planes : float
            Seconds between the start of each plane's animation.
        line_lag : float
            Seconds between individual lines within a plane.
        """
        anims = []
        for name in order:
            plane = getattr(self, f"{name}_plane" if name != "bounding_box" else "bounding_box", None)
            if plane is None:
                continue
            if hasattr(plane, "animate_build"):
                anims.append(plane.animate_build(lag=line_lag))

        return LaggedStart(*anims, lag_ratio=lag_between_planes)

    def set_floor_opacity(self, opacity: float) -> "FullGrid3D":
        """Convenience: set XY plane opacity."""
        if hasattr(self, "xy_plane"):
            self.xy_plane.set_opacity(opacity)
        return self

    def highlight_axes(
        self,
        color: ManimColor = YELLOW,
        stroke_width: float = 2.5,
    ) -> "FullGrid3D":
        """Draw thick zero-crossing lines on all planes to emphasise axes."""
        for attr in ("xy_plane", "xz_plane", "yz_plane"):
            plane: Optional[GridPlane3D] = getattr(self, attr, None)
            if plane is None:
                continue
            # Re-style existing zero lines
            for ln in plane.zero_lines:
                ln.set_stroke(color=_with_opacity(color, 0.85), width=stroke_width)
        return self


# ---------------------------------------------------------------------------
# BillboardLabel3D
# ---------------------------------------------------------------------------

class BillboardLabel3D(VGroup):
    """A 3D text label that is always rotated to face the camera.

    Manim's ``ThreeDScene`` can auto-rotate labels via
    ``self.add_fixed_orientation_mobjects()``, but this class packages
    the label + offset together so it can be positioned relative to a
    grid tick.

    Parameters
    ----------
    text : str
        Label text.  Rendered via :class:`~manim.Text`.
    position : np.ndarray
        3D world position of the label's centre.
    font_size : int
        Font size.
    color : ManimColor
        Label colour.
    scene : ThreeDScene
        The scene in which this label lives.  Needed to call
        ``add_fixed_orientation_mobjects``.
    """

    def __init__(
        self,
        text: str,
        position: np.ndarray = ORIGIN,
        font_size: int = 20,
        color: ManimColor = WHITE,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label = Text(text, font_size=font_size, color=color)
        self.label.move_to(np.array(position, dtype=float))
        self.add(self.label)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(self.label)


# ---------------------------------------------------------------------------
# GridSnapHelper
# ---------------------------------------------------------------------------

class GridSnapHelper:
    """Utility class (no Manim rendering) that maps data values to
    snapped grid positions in 3D space.

    Useful when placing chart objects (bars, spheres, labels) exactly
    on grid intersections rather than at raw data coordinates.

    Parameters
    ----------
    x_range, y_range, z_range : (float, float, float)
        (min, max, step) for each axis — must match your ``FullGrid3D``.
    origin : np.ndarray
        3D origin of the grid coordinate system.
    """

    def __init__(
        self,
        x_range: Tuple[float, float, float] = (-4, 4, 1),
        y_range: Tuple[float, float, float] = (-4, 4, 1),
        z_range: Tuple[float, float, float] = (0, 5, 1),
        origin: np.ndarray = ORIGIN,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.origin = np.array(origin, dtype=float)

    def _snap(self, val: float, rng: Tuple[float, float, float]) -> float:
        mn, mx, step = rng
        snapped = round((val - mn) / step) * step + mn
        return float(np.clip(snapped, mn, mx))

    def snap(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> np.ndarray:
        """Return the snapped 3D position for data coordinates (x, y, z)."""
        sx = self._snap(x, self.x_range)
        sy = self._snap(y, self.y_range)
        sz = self._snap(z, self.z_range)
        return self.origin + np.array([sx, sy, sz])

    def data_to_grid(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Batch-snap an (N, 3) array of data points.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Rows of (x, y, z) data coordinates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Snapped grid positions.
        """
        out = np.zeros_like(points)
        for i, (x, y, z) in enumerate(points):
            out[i] = self.snap(x, y, z)
        return out

    def axes_ticks(self, axis: str = "x") -> np.ndarray:
        """Return a 1-D array of major tick values along *axis*."""
        rng = {"x": self.x_range, "y": self.y_range, "z": self.z_range}[axis]
        mn, mx, step = rng
        return np.arange(mn, mx + step * 0.5, step)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_stats_grid(
    x_range: Tuple[float, float, float] = (-4, 4, 1),
    y_range: Tuple[float, float, float] = (-4, 4, 1),
    z_range: Tuple[float, float, float] = (0, 6, 1),
    theme: str = "stats",
    minor: int = 4,
    label_axes: bool = False,
    show_bounding_box: bool = False,
) -> FullGrid3D:
    """Factory that wires ``FullGrid3D`` + ``GridSnapHelper`` together.

    Parameters
    ----------
    theme : str
        ``"stats"`` (default), ``"dark"``, or ``"light"``.

    Returns
    -------
    FullGrid3D
        Ready to add to a scene.

    Example
    -------
    ::

        grid = make_stats_grid(x_range=(-3, 3, 1), z_range=(0, 8, 1))
        snap = GridSnapHelper(x_range=(-3, 3, 1), z_range=(0, 8, 1))
        self.play(grid.animate_build())
        bar_pos = snap.snap(x=1.0, y=0.0, z=3.5)
    """
    themes = {"stats": STATS_GRID, "dark": DARK_GRID, "light": LIGHT_GRID}
    cfg = themes.get(theme, STATS_GRID)
    return FullGrid3D(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        minor_subdivisions=minor,
        config=cfg,
        show_bounding_box=show_bounding_box,
        label_axes=label_axes,
    )