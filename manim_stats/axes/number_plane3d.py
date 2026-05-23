"""
manim_stats/axes/number_plane3d.py
===================================
Full 3-D number plane system for the Manim Statistics Extension.

Unlike ``GridPlane3D`` (which only draws backdrop grid lines for axes),
every class here is a **self-contained 2-D coordinate plane embedded in
3-D space**.  A plane can be:

    • Snapped to an axis face  (``FaceNumberPlane3D``)
    • Floating at any data slice  (``FloatingPlane3D``)
    • Freely oriented  (``NumberPlane3D`` with custom normal + basis)

Each plane exposes its own coordinate system so that 2-D statistical
objects (curves, regions, scatter, vectors) can be drawn directly on
any face of the 3-D scene without the caller doing any trigonometry.

Architecture
------------
NumberPlane3D
    ├── PlaneGeometry          — background fill rectangle + edge border
    ├── PlaneGrid              — major / minor line meshes in plane-local coords
    ├── PlaneAxes              — two in-plane axis spines with tips
    ├── PlaneTickSystem        — ticks + labels on both plane axes
    ├── PlaneOriginMark        — cross / dot at the plane origin
    └── PlaneDrawingLayer      — VGroup that receives curves / regions etc.

Coordinate conventions
-----------------------
Every plane has:
    ``origin_3d``   — 3-D scene point corresponding to (0,0) on the plane
    ``basis_u``     — 3-D unit vector for the plane's "x" direction
    ``basis_v``     — 3-D unit vector for the plane's "y" direction
    ``normal``      — basis_u × basis_v (outward-facing normal)

Data ↔ scene mapping:
    plane_c2p(u, v) → origin_3d + u_scene * basis_u + v_scene * basis_v
    plane_p2c(pt)   → (u_data, v_data)  via dot-product projection

Shadow projections
------------------
``ShadowProjection3D`` takes any VGroup of 3-D Dot3Ds / Line3Ds and
collapses them orthogonally onto a target ``NumberPlane3D``, producing
a matching set of projected dots / lines with drop-lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Literal,
    Optional, Sequence, Tuple, Union,
)

import numpy as np
from numpy.typing import ArrayLike

from manim import (
    VGroup, VMobject, Mobject, Group,
    # Geometry
    Line3D, Arrow3D, Dot3D, Sphere,
    Line, DashedLine, DashedVMobject,
    Polygon, Rectangle, RoundedRectangle,
    Text, MathTex,
    # Curves
    ParametricFunction,
    # Animations
    Animation, AnimationGroup, Succession,
    Create, FadeIn, FadeOut, Write,
    DrawBorderThenFill, Transform,
    UpdateFromAlphaFunc,
    # Constants
    ORIGIN, UP, DOWN, LEFT, RIGHT, IN, OUT,
    DEGREES, PI, TAU, WHITE, BLACK,
    # Utils
    interpolate_color, normalize,
    rotation_matrix, smooth, there_and_back,
    ValueTracker, always_redraw,
)

from ..core.base import (
    StatsObject3D,
    StatsTheme, StatsColorPalette,
    MaterialConfig, MaterialApplicator,
    LabelConfig, LabelAnchor,
    AnimationConfig, BuildStyle,
    HighlightStyle, HighlightSystem,
    ThemeMode,
)
from ..core.math_utils import (
    auto_range, generate_ticks, nice_number,
    smooth_curve, FloatArray,
)
from .axes3d import (
    AxisScaleMode, GridStyle, AxisConfig,
    GridConfig, AxesConfig, ScaleTransform,
    StatsAxes3D,
)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  ENUMERATIONS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class PlaneFace(Enum):
    """Which face of the bounding box a ``FaceNumberPlane3D`` occupies."""
    XY_FRONT  = auto()   # z = z_max  (front-facing, normal = +Z)
    XY_BACK   = auto()   # z = z_min  (back wall,    normal = −Z)
    XZ_TOP    = auto()   # y = y_max  (ceiling,      normal = +Y)
    XZ_BOTTOM = auto()   # y = y_min  (floor,        normal = −Y)
    YZ_RIGHT  = auto()   # x = x_max  (right wall,   normal = +X)
    YZ_LEFT   = auto()   # x = x_min  (left wall,    normal = −X)


class PlaneMaterial(Enum):
    SOLID       = auto()   # opaque fill
    FROSTED     = auto()   # translucent, high opacity
    GLASS       = auto()   # very translucent
    WIREFRAME   = auto()   # no fill, only border


@dataclass
class PlaneGridConfig:
    """Grid configuration for a single number plane."""
    show_major_grid:    bool      = True
    show_minor_grid:    bool      = True
    major_spacing:      float     = 1.0     # data units
    minor_per_major:    int       = 4
    major_color:        Optional[str] = None
    minor_color:        Optional[str] = None
    major_opacity:      float     = 0.55
    minor_opacity:      float     = 0.18
    major_width:        float     = 1.0
    minor_width:        float     = 0.5
    style:              GridStyle = GridStyle.SOLID
    fade_edges:         bool      = True
    fade_amount:        float     = 0.5     # 0 = no fade, 1 = fully transparent edge


@dataclass
class PlaneAxisConfig:
    """Configuration for the in-plane axis spines."""
    show_u_axis:        bool      = True    # "horizontal" axis of the plane
    show_v_axis:        bool      = True    # "vertical" axis
    color:              Optional[str] = None
    width:              float     = 2.2
    include_tip:        bool      = True
    tip_length:         float     = 0.18
    show_ticks:         bool      = True
    tick_length:        float     = 0.14
    show_labels:        bool      = True
    label_font_size:    float     = 20
    label_offset:       float     = 0.30
    u_label:            str       = ""
    v_label:            str       = ""
    u_label_is_math:    bool      = True
    v_label_is_math:    bool      = True


@dataclass
class PlaneConfig:
    """
    Complete configuration for a ``NumberPlane3D``.

    u_range / v_range describe the data-space extent of the two
    in-plane axes (u = "horizontal", v = "vertical" of the plane).
    """
    u_range:        Tuple[float, float] = (-5.0, 5.0)
    v_range:        Tuple[float, float] = (-5.0, 5.0)
    u_length:       float               = 6.0      # scene units
    v_length:       float               = 6.0

    grid:           PlaneGridConfig     = field(default_factory=PlaneGridConfig)
    axes:           PlaneAxisConfig     = field(default_factory=PlaneAxisConfig)
    material:       PlaneMaterial       = PlaneMaterial.FROSTED

    fill_color:     Optional[str]       = None    # None → theme.surface
    fill_opacity:   float               = 0.12
    border_color:   Optional[str]       = None    # None → theme.border
    border_width:   float               = 1.4
    border_opacity: float               = 0.6

    show_origin_mark: bool              = True
    origin_radius:    float             = 0.055

    depth_offset:     float             = 0.005   # push in front of background


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PLANE GEOMETRY  (background fill + border)
# ─────────────────────────────────────────────────────────────────────────────

class PlaneGeometry(VGroup):
    """
    A filled quadrilateral representing the plane's background surface.
    Constructed from the four corners in 3-D scene space.

    Using a ``Polygon`` (rather than a ``Rectangle``) means this works
    for any orientation — not just axis-aligned planes.
    """

    def __init__(
        self,
        corners:  List[np.ndarray],     # 4 corners in 3-D, CCW order
        cfg:      PlaneConfig,
        theme:    StatsColorPalette,
    ) -> None:
        super().__init__()

        fill   = cfg.fill_color   or theme.surface
        border = cfg.border_color or theme.border

        # Background fill polygon
        if cfg.material != PlaneMaterial.WIREFRAME:
            poly = Polygon(*corners)
            poly.set_fill(fill, opacity=cfg.fill_opacity)
            poly.set_stroke(width=0)
            self.add(poly)

        # Border edges
        n = len(corners)
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            edge = Line3D(a, b, color=border,
                          thickness=cfg.border_width * 0.005)
            edge.set_opacity(cfg.border_opacity)
            self.add(edge)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PLANE GRID
# ─────────────────────────────────────────────────────────────────────────────

class PlaneGrid(VGroup):
    """
    Major and minor grid lines drawn in the plane's local (u, v) coordinate
    system, then mapped to 3-D world space via the plane's basis vectors.
    """

    def __init__(
        self,
        cfg:          PlaneConfig,
        grid_cfg:     PlaneGridConfig,
        theme:        StatsColorPalette,
        plane_c2p_fn: Callable[[float, float], np.ndarray],
    ) -> None:
        super().__init__()
        self._c2p   = plane_c2p_fn
        self._theme = theme

        major_color = grid_cfg.major_color or theme.border
        minor_color = grid_cfg.minor_color or theme.border

        u_lo, u_hi = cfg.u_range
        v_lo, v_hi = cfg.v_range
        spacing    = grid_cfg.major_spacing
        mspacing   = spacing / grid_cfg.minor_per_major

        # Major grid lines parallel to V axis (vertical on plane)
        u_majors = np.arange(
            math.ceil(u_lo / spacing) * spacing,
            u_hi + spacing * 1e-6, spacing)
        for u in u_majors:
            line = self._make_line(
                (u, v_lo), (u, v_hi),
                major_color, grid_cfg.major_width,
                grid_cfg.major_opacity, grid_cfg.style,
            )
            if line: self.add(line)

        # Major grid lines parallel to U axis (horizontal on plane)
        v_majors = np.arange(
            math.ceil(v_lo / spacing) * spacing,
            v_hi + spacing * 1e-6, spacing)
        for v in v_majors:
            line = self._make_line(
                (u_lo, v), (u_hi, v),
                major_color, grid_cfg.major_width,
                grid_cfg.major_opacity, grid_cfg.style,
            )
            if line: self.add(line)

        if grid_cfg.show_minor_grid:
            # Minor lines parallel to V
            u_minors = np.arange(
                math.ceil(u_lo / mspacing) * mspacing,
                u_hi + mspacing * 1e-6, mspacing)
            u_minors = u_minors[~np.isin(
                np.round(u_minors, 10), np.round(u_majors, 10))]
            for u in u_minors:
                line = self._make_line(
                    (u, v_lo), (u, v_hi),
                    minor_color, grid_cfg.minor_width,
                    grid_cfg.minor_opacity, grid_cfg.style,
                )
                if line: self.add(line)

            # Minor lines parallel to U
            v_minors = np.arange(
                math.ceil(v_lo / mspacing) * mspacing,
                v_hi + mspacing * 1e-6, mspacing)
            v_minors = v_minors[~np.isin(
                np.round(v_minors, 10), np.round(v_majors, 10))]
            for v in v_minors:
                line = self._make_line(
                    (u_lo, v), (u_hi, v),
                    minor_color, grid_cfg.minor_width,
                    grid_cfg.minor_opacity, grid_cfg.style,
                )
                if line: self.add(line)

    # ── helpers ───────────────────────────────────────────────────────────

    def _make_line(
        self,
        uv1:     Tuple[float, float],
        uv2:     Tuple[float, float],
        color:   str,
        width:   float,
        opacity: float,
        style:   GridStyle,
    ) -> Optional[VMobject]:
        try:
            p1 = self._c2p(*uv1)
            p2 = self._c2p(*uv2)
            if np.allclose(p1, p2):
                return None
            if style == GridStyle.SOLID:
                line = Line3D(p1, p2, color=color,
                              thickness=width * 0.005)
            elif style == GridStyle.DASHED:
                line = DashedLine(p1, p2, color=color,
                                  stroke_width=width * 1.3,
                                  dash_length=0.12)
            else:  # DOTTED
                line = DashedLine(p1, p2, color=color,
                                  stroke_width=width,
                                  dash_length=0.04,
                                  dashed_ratio=0.3)
            line.set_opacity(opacity)
            return line
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PLANE AXES
# ─────────────────────────────────────────────────────────────────────────────

class PlaneAxes(VGroup):
    """
    Two in-plane axis spines (U and V) with tips and title labels.
    Both spines pass through the plane origin.
    """

    def __init__(
        self,
        cfg:          PlaneConfig,
        theme:        StatsColorPalette,
        plane_c2p_fn: Callable[[float, float], np.ndarray],
    ) -> None:
        super().__init__()
        self._c2p  = plane_c2p_fn
        ax_cfg     = cfg.axes
        color      = ax_cfg.color or theme.neutral

        u_lo, u_hi = cfg.u_range
        v_lo, v_hi = cfg.v_range

        # U axis (horizontal of the plane)
        if ax_cfg.show_u_axis:
            p_start = plane_c2p_fn(u_lo, 0.0)
            p_end   = plane_c2p_fn(u_hi, 0.0)
            self._add_spine(p_start, p_end, color, ax_cfg)
            # U title label
            if ax_cfg.u_label:
                lbl_cls = MathTex if ax_cfg.u_label_is_math else Text
                lbl = lbl_cls(ax_cfg.u_label,
                               font_size=ax_cfg.label_font_size,
                               color=color)
                lbl.move_to(p_end + RIGHT * 0.35 + DOWN * 0.1)
                self.add(lbl)

        # V axis (vertical of the plane)
        if ax_cfg.show_v_axis:
            p_start = plane_c2p_fn(0.0, v_lo)
            p_end   = plane_c2p_fn(0.0, v_hi)
            self._add_spine(p_start, p_end, color, ax_cfg)
            # V title label
            if ax_cfg.v_label:
                lbl_cls = MathTex if ax_cfg.v_label_is_math else Text
                lbl = lbl_cls(ax_cfg.v_label,
                               font_size=ax_cfg.label_font_size,
                               color=color)
                lbl.move_to(p_end + UP * 0.35)
                self.add(lbl)

    def _add_spine(
        self,
        start:  np.ndarray,
        end:    np.ndarray,
        color:  str,
        ax_cfg: PlaneAxisConfig,
    ) -> None:
        if ax_cfg.include_tip:
            arrow = Arrow3D(start=start, end=end,
                            color=color,
                            thickness=ax_cfg.width * 0.007,
                            tip_length=ax_cfg.tip_length)
            self.add(arrow)
        else:
            line = Line3D(start=start, end=end,
                          color=color,
                          thickness=ax_cfg.width * 0.007)
            self.add(line)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLANE TICK SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class PlaneTickSystem(VGroup):
    """
    Tick marks and numeric labels on both in-plane axes.
    Ticks are short Line3D segments perpendicular to each axis
    (within the plane surface).
    """

    def __init__(
        self,
        cfg:          PlaneConfig,
        theme:        StatsColorPalette,
        plane_c2p_fn: Callable[[float, float], np.ndarray],
        basis_u:      np.ndarray,    # 3-D unit vector along plane U
        basis_v:      np.ndarray,    # 3-D unit vector along plane V
    ) -> None:
        super().__init__()
        ax_cfg = cfg.axes
        if not ax_cfg.show_ticks:
            return

        color    = ax_cfg.color or theme.neutral
        lbl_color = theme.text_secondary
        spacing   = cfg.grid.major_spacing
        tlen      = ax_cfg.tick_length
        loffset   = ax_cfg.label_offset

        u_lo, u_hi = cfg.u_range
        v_lo, v_hi = cfg.v_range

        # Ticks on U axis (perpendicular = basis_v)
        u_vals = np.arange(
            math.ceil(u_lo / spacing) * spacing,
            u_hi + spacing * 1e-6, spacing)
        for u in u_vals:
            if abs(u) < 1e-9:
                continue
            centre = plane_c2p_fn(u, 0.0)
            t1     = centre - basis_v * tlen * 0.5
            t2     = centre + basis_v * tlen * 0.5
            tick   = Line3D(t1, t2, color=color, thickness=0.005)
            tick.set_opacity(0.8)
            self.add(tick)
            if ax_cfg.show_labels:
                lbl = self._fmt_label(u, ax_cfg.label_font_size, lbl_color)
                lbl.move_to(centre - basis_v * loffset)
                self.add(lbl)

        # Ticks on V axis (perpendicular = basis_u)
        v_vals = np.arange(
            math.ceil(v_lo / spacing) * spacing,
            v_hi + spacing * 1e-6, spacing)
        for v in v_vals:
            if abs(v) < 1e-9:
                continue
            centre = plane_c2p_fn(0.0, v)
            t1     = centre - basis_u * tlen * 0.5
            t2     = centre + basis_u * tlen * 0.5
            tick   = Line3D(t1, t2, color=color, thickness=0.005)
            tick.set_opacity(0.8)
            self.add(tick)
            if ax_cfg.show_labels:
                lbl = self._fmt_label(v, ax_cfg.label_font_size, lbl_color)
                lbl.move_to(centre - basis_u * loffset)
                self.add(lbl)

    @staticmethod
    def _fmt_label(val: float, size: float, color: str) -> VMobject:
        text = f"{val:.0f}" if val == int(val) else f"{val:.2g}"
        return Text(text, font_size=size, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLANE ORIGIN MARK
# ─────────────────────────────────────────────────────────────────────────────

class PlaneOriginMark(VGroup):
    """
    Crosshair + dot at the plane's (0, 0) origin.
    The crosshair arms lie within the plane surface.
    """

    def __init__(
        self,
        origin_3d:  np.ndarray,
        basis_u:    np.ndarray,
        basis_v:    np.ndarray,
        cfg:        PlaneConfig,
        theme:      StatsColorPalette,
    ) -> None:
        super().__init__()
        if not cfg.show_origin_mark:
            return

        color = theme.accent
        r     = cfg.origin_radius
        ext   = r * 2.5

        # Central dot
        dot = Dot3D(origin_3d, radius=r, color=color)
        dot.set_opacity(0.9)
        self.add(dot)

        # Crosshair arms along basis_u and basis_v
        for direction in [basis_u, -basis_u, basis_v, -basis_v]:
            arm = Line3D(origin_3d,
                         origin_3d + direction * ext,
                         color=color, thickness=0.007)
            arm.set_opacity(0.55)
            self.add(arm)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLANE DRAWING LAYER
# ─────────────────────────────────────────────────────────────────────────────

class PlaneDrawingLayer(VGroup):
    """
    A VGroup that sits on top of a ``NumberPlane3D`` and receives
    all user-drawn content: curves, regions, scatter points, vectors.

    Drawing methods accept data-space (u, v) coordinates and
    internally call ``plane_c2p`` to convert to 3-D scene points.
    """

    def __init__(
        self,
        plane_c2p_fn: Callable[[float, float], np.ndarray],
        theme:        StatsColorPalette,
        basis_u:      np.ndarray,
        basis_v:      np.ndarray,
        normal:       np.ndarray,
    ) -> None:
        super().__init__()
        self._c2p    = plane_c2p_fn
        self._theme  = theme
        self._bu     = basis_u
        self._bv     = basis_v
        self._normal = normal
        self._objects: Dict[str, VMobject] = {}

    # ── curve ─────────────────────────────────────────────────────────────

    def draw_curve(
        self,
        key:       str,
        u_vals:    ArrayLike,
        v_vals:    ArrayLike,
        color:     Optional[str]  = None,
        width:     float          = 2.5,
        opacity:   float          = 1.0,
        smooth_n:  int            = 0,       # 0 = no smoothing
    ) -> VGroup:
        """
        Draw a 2-D curve (u[i], v[i]) on the plane.
        Optionally smooth with cubic interpolation first.
        """
        us = np.asarray(u_vals, float)
        vs = np.asarray(v_vals, float)

        if smooth_n > len(us):
            from ..core.math_utils import smooth_curve as _sc
            us, vs = _sc(us, vs, n_out=smooth_n)

        c = color or self._theme.primary
        grp = VGroup()

        for i in range(len(us) - 1):
            p1 = self._c2p(us[i], vs[i])
            p2 = self._c2p(us[i + 1], vs[i + 1])
            if np.allclose(p1, p2):
                continue
            seg = Line3D(p1, p2, color=c, thickness=width * 0.006)
            seg.set_opacity(opacity)
            grp.add(seg)

        self._objects[key] = grp
        self.add(grp)
        return grp

    # ── filled region ─────────────────────────────────────────────────────

    def draw_region(
        self,
        key:         str,
        u_vals:      ArrayLike,
        v_lo_vals:   ArrayLike,
        v_hi_vals:   ArrayLike,
        color:       Optional[str]  = None,
        fill_opacity: float         = 0.35,
        stroke_width: float         = 0.0,
    ) -> VGroup:
        """
        Fill the region between ``v_lo_vals`` and ``v_hi_vals`` along ``u_vals``.
        Suitable for shaded PDF areas, confidence bands, etc.
        The polygon is assembled as [upper_path, reversed_lower_path].
        """
        us   = np.asarray(u_vals,    float)
        v_lo = np.asarray(v_lo_vals, float)
        v_hi = np.asarray(v_hi_vals, float)
        c    = color or self._theme.positive

        # Build polygon: upper edge L→R, lower edge R→L
        upper = [self._c2p(u, vh) for u, vh in zip(us, v_hi)]
        lower = [self._c2p(u, vl) for u, vl in reversed(
            list(zip(us, v_lo)))]
        all_pts = upper + lower

        if len(all_pts) < 3:
            return VGroup()

        poly = Polygon(*all_pts)
        poly.set_fill(c, opacity=fill_opacity)
        poly.set_stroke(c, width=stroke_width, opacity=fill_opacity * 1.4)

        grp = VGroup(poly)
        self._objects[key] = grp
        self.add(grp)
        return grp

    # ── scatter ────────────────────────────────────────────────────────────

    def draw_scatter(
        self,
        key:          str,
        u_vals:       ArrayLike,
        v_vals:       ArrayLike,
        dot_radius:   float              = 0.06,
        colors:       Optional[List[str]] = None,
        opacity:      float              = 0.85,
        jitter_normal: float             = 0.003,   # tiny push off plane to avoid z-fighting
    ) -> VGroup:
        """
        Place Dot3D points on the plane at each (u, v) data point.
        """
        us = np.asarray(u_vals, float)
        vs = np.asarray(v_vals, float)
        n  = len(us)
        grp = VGroup()

        for i, (u, v) in enumerate(zip(us, vs)):
            c    = colors[i % len(colors)] if colors else self._theme.primary
            pos  = self._c2p(u, v) + self._normal * jitter_normal
            dot  = Dot3D(pos, radius=dot_radius, color=c)
            dot.set_opacity(opacity)
            grp.add(dot)

        self._objects[key] = grp
        self.add(grp)
        return grp

    # ── vector ────────────────────────────────────────────────────────────

    def draw_vector(
        self,
        key:      str,
        origin:   Tuple[float, float],
        tip:      Tuple[float, float],
        color:    Optional[str]  = None,
        width:    float          = 2.0,
        tip_size: float          = 0.18,
    ) -> Arrow3D:
        """
        Draw an arrow from *origin* to *tip* (both in plane data coords).
        """
        c   = color or self._theme.accent
        p1  = self._c2p(*origin)
        p2  = self._c2p(*tip)
        arr = Arrow3D(p1, p2, color=c,
                      thickness=width * 0.006,
                      tip_length=tip_size)
        self._objects[key] = arr
        self.add(arr)
        return arr

    # ── text annotation ───────────────────────────────────────────────────

    def draw_label(
        self,
        key:       str,
        u:         float,
        v:         float,
        text:      str,
        is_math:   bool           = False,
        font_size: float          = 22,
        color:     Optional[str]  = None,
        bg:        bool           = True,
    ) -> VGroup:
        """
        Place a floating text label at plane data-coords (u, v).
        """
        c   = color or self._theme.text_primary
        pos = self._c2p(u, v)

        mob_cls = MathTex if is_math else Text
        mob     = mob_cls(text, font_size=font_size, color=c)
        mob.move_to(pos)

        if bg:
            from ..core.base import StatsTheme
            t   = StatsTheme.current
            bg_r = RoundedRectangle(
                corner_radius=0.10,
                width=mob.width + 0.25,
                height=mob.height + 0.18,
            )
            bg_r.set_fill(t.surface, opacity=0.80)
            bg_r.set_stroke(t.border, width=1.0)
            bg_r.move_to(mob)
            grp = VGroup(bg_r, mob)
        else:
            grp = VGroup(mob)

        self._objects[key] = grp
        self.add(grp)
        return grp

    # ── removal ───────────────────────────────────────────────────────────

    def remove_object(self, key: str) -> None:
        if key in self._objects:
            self.remove(self._objects.pop(key))

    def clear_all(self) -> None:
        for key in list(self._objects):
            self.remove_object(key)

    def get_object(self, key: str) -> Optional[VMobject]:
        return self._objects.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  NUMBER PLANE 3D  (master class)
# ─────────────────────────────────────────────────────────────────────────────

class NumberPlane3D(StatsObject3D):
    """
    A fully-featured 2-D number plane embedded in 3-D space.

    The plane is defined by:
        ``origin_3d``  — 3-D scene position of data (0,0)
        ``basis_u``    — 3-D unit vector for the plane's U ("x") direction
        ``basis_v``    — 3-D unit vector for the plane's V ("y") direction

    The ``normal`` is computed as ``basis_u × basis_v``.

    Coordinate mapping
    ------------------
    ::

        scene_pt = plane.plane_c2p(u, v)
        u, v     = plane.plane_p2c(scene_pt)

    Drawing API
    -----------
    All drawing methods are on ``plane.drawing`` (``PlaneDrawingLayer``):
    ::

        plane.drawing.draw_curve("pdf", u_arr, v_arr, color=BLUE)
        plane.drawing.draw_region("shade", u_arr, v0, v_hi, color=GREEN)
        plane.drawing.draw_scatter("pts", u_arr, v_arr)
        plane.drawing.draw_vector("mu", (0,0), (1.5,0))
        plane.drawing.draw_label("note", 2, 0.3, r"\\mu + \\sigma")
    """

    def __init__(
        self,
        origin_3d:   Optional[np.ndarray]  = None,
        basis_u:     Optional[np.ndarray]  = None,
        basis_v:     Optional[np.ndarray]  = None,
        plane_config: Optional[PlaneConfig] = None,
        **kwargs,
    ) -> None:
        # Basis vectors — default: plane lies in XY scene plane
        self._origin_3d = np.asarray(origin_3d) if origin_3d is not None \
                          else np.array([0., 0., 0.])
        self._basis_u   = _normalize(np.asarray(basis_u) if basis_u is not None
                                     else np.array([1., 0., 0.]))
        self._basis_v   = _normalize(np.asarray(basis_v) if basis_v is not None
                                     else np.array([0., 1., 0.]))
        self._normal    = _normalize(np.cross(self._basis_u, self._basis_v))
        self._plane_cfg = plane_config or PlaneConfig()

        # Sub-components (populated in _build_geometry)
        self.geometry:  PlaneGeometry    = None   # type: ignore[assignment]
        self.grid:      PlaneGrid        = None   # type: ignore[assignment]
        self.axes:      PlaneAxes        = None   # type: ignore[assignment]
        self.ticks:     PlaneTickSystem  = None   # type: ignore[assignment]
        self.origin_mk: PlaneOriginMark  = None   # type: ignore[assignment]
        self.drawing:   PlaneDrawingLayer = None  # type: ignore[assignment]

        super().__init__(**kwargs)

    # ── coordinate transform ──────────────────────────────────────────────

    def plane_c2p(self, u: float, v: float) -> np.ndarray:
        """
        Plane data coords (u, v) → 3-D scene point.
        """
        cfg  = self._plane_cfg
        u_lo, u_hi = cfg.u_range
        v_lo, v_hi = cfg.v_range

        # Normalise to [-0.5, 0.5], then scale to scene length
        tu = (u - u_lo) / (u_hi - u_lo) - 0.5 if u_hi != u_lo else 0.
        tv = (v - v_lo) / (v_hi - v_lo) - 0.5 if v_hi != v_lo else 0.

        su = tu * cfg.u_length
        sv = tv * cfg.v_length

        return self._origin_3d + su * self._basis_u + sv * self._basis_v

    def plane_p2c(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        3-D scene point → plane data coords (u, v) via projection.
        """
        cfg  = self._plane_cfg
        u_lo, u_hi = cfg.u_range
        v_lo, v_hi = cfg.v_range

        delta = np.asarray(point_3d) - self._origin_3d
        su    = float(np.dot(delta, self._basis_u))
        sv    = float(np.dot(delta, self._basis_v))

        tu = su / cfg.u_length + 0.5
        tv = sv / cfg.v_length + 0.5

        u = u_lo + tu * (u_hi - u_lo)
        v = v_lo + tv * (v_hi - v_lo)
        return u, v

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """
        Orthogonally project a 3-D point onto this plane's surface.
        Returns the 3-D scene position of the projected point.
        """
        delta = np.asarray(point_3d) - self._origin_3d
        dist  = float(np.dot(delta, self._normal))
        return np.asarray(point_3d) - dist * self._normal

    # ── geometry build ────────────────────────────────────────────────────

    def _build_geometry(self) -> None:
        cfg   = self._plane_cfg
        theme = self._palette
        c2p   = self.plane_c2p

        u_lo, u_hi = cfg.u_range
        v_lo, v_hi = cfg.v_range

        # 4 corners of the plane in 3-D
        corners = [
            c2p(u_lo, v_lo), c2p(u_hi, v_lo),
            c2p(u_hi, v_hi), c2p(u_lo, v_hi),
        ]

        # Depth-push along normal to avoid z-fighting
        offset = self._normal * cfg.depth_offset

        # Background geometry
        self.geometry = PlaneGeometry(
            [pt + offset for pt in corners], cfg, theme)
        self.add(self.geometry)

        # Grid
        if cfg.grid.show_major_grid:
            self.grid = PlaneGrid(cfg, cfg.grid, theme, c2p)
            self.add(self.grid)

        # Axes
        if cfg.axes.show_u_axis or cfg.axes.show_v_axis:
            self.axes = PlaneAxes(cfg, theme, c2p)
            self.add(self.axes)

        # Ticks
        if cfg.axes.show_ticks:
            self.ticks = PlaneTickSystem(
                cfg, theme, c2p, self._basis_u, self._basis_v)
            self.add(self.ticks)

        # Origin mark
        self.origin_mk = PlaneOriginMark(
            self._origin_3d, self._basis_u, self._basis_v, cfg, theme)
        self.add(self.origin_mk)

        # Drawing layer (empty at build time)
        self.drawing = PlaneDrawingLayer(
            c2p, theme, self._basis_u, self._basis_v, self._normal)
        self.add(self.drawing)

    # ── animation protocol ────────────────────────────────────────────────

    def animate_build(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        """
        The plane materialises by sliding in from behind its normal vector,
        then the grid draws itself, then axes and labels appear.
        """
        cfg = cfg or self._anim_cfg
        t   = cfg.run_time

        geo_in   = FadeIn(self.geometry, run_time=t * 0.35, rate_func=smooth)
        grid_in  = FadeIn(self.grid,     run_time=t * 0.40,
                          shift=self._normal * (-0.3),
                          rate_func=smooth)
        axes_in  = AnimationGroup(
            FadeIn(self.axes,      run_time=t * 0.30, rate_func=smooth),
            FadeIn(self.ticks,     run_time=t * 0.30, rate_func=smooth),
            FadeIn(self.origin_mk, run_time=t * 0.20, rate_func=smooth),
            lag_ratio=0.2,
        )
        return Succession(geo_in, grid_in, axes_in, lag_ratio=0)

    def animate_update(
        self, new_data: Any, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return FadeIn(self, run_time=cfg.run_time * 0.5)

    def animate_highlight(
        self,
        style: HighlightStyle = HighlightStyle.GLOW,
        cfg:   Optional[AnimationConfig] = None,
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return HighlightSystem.glow(self.geometry, cfg)

    def animate_exit(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return FadeOut(self, shift=self._normal * 0.4,
                       run_time=cfg.run_time * 0.6)

    # ── accessors ─────────────────────────────────────────────────────────

    @property
    def normal(self) -> np.ndarray:
        return self._normal.copy()

    @property
    def basis(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._basis_u.copy(), self._basis_v.copy()

    @property
    def origin_scene(self) -> np.ndarray:
        return self._origin_3d.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FACE NUMBER PLANE
# ─────────────────────────────────────────────────────────────────────────────

class FaceNumberPlane3D(NumberPlane3D):
    """
    A ``NumberPlane3D`` that snaps exactly to one face of a ``StatsAxes3D``
    bounding box.

    The plane's (u, v) data range is automatically inherited from the
    two axes that form that face, so ``plane_c2p`` is aligned with
    the parent ``StatsAxes3D.c2p`` for the same data values.

    Usage
    -----
    ::

        axes = StatsAxes3D.for_3d_surface(...)
        floor = FaceNumberPlane3D(axes, face=PlaneFace.XZ_BOTTOM)
        scene.add(axes, floor)
    """

    # Maps face → (u_axis_key, v_axis_key, normal_direction, fixed_axis_key)
    _FACE_MAP: Dict[PlaneFace, Dict] = {
        PlaneFace.XY_FRONT:  dict(u="x", v="y", normal=OUT,  fixed="z", sign=+1),
        PlaneFace.XY_BACK:   dict(u="x", v="y", normal=IN,   fixed="z", sign=-1),
        PlaneFace.XZ_TOP:    dict(u="x", v="z", normal=UP,   fixed="y", sign=+1),
        PlaneFace.XZ_BOTTOM: dict(u="x", v="z", normal=DOWN, fixed="y", sign=-1),
        PlaneFace.YZ_RIGHT:  dict(u="y", v="z", normal=RIGHT,fixed="x", sign=+1),
        PlaneFace.YZ_LEFT:   dict(u="y", v="z", normal=LEFT, fixed="x", sign=-1),
    }

    # Maps normal direction string → (basis_u, basis_v)
    _BASIS_MAP: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "OUT":   (np.array([1,0,0]), np.array([0,1,0])),
        "IN":    (np.array([-1,0,0]),np.array([0,1,0])),
        "UP":    (np.array([1,0,0]), np.array([0,0,1])),
        "DOWN":  (np.array([1,0,0]), np.array([0,0,-1])),
        "RIGHT": (np.array([0,1,0]), np.array([0,0,1])),
        "LEFT":  (np.array([0,-1,0]),np.array([0,0,1])),
    }

    def __init__(
        self,
        parent_axes:   StatsAxes3D,
        face:          PlaneFace    = PlaneFace.XY_FRONT,
        plane_config:  Optional[PlaneConfig] = None,
        **kwargs,
    ) -> None:
        fm      = self._FACE_MAP[face]
        axes_cfg = parent_axes.config
        pcfg    = parent_axes.c2p   # use parent's c2p to compute origin

        # Resolve u, v ranges from parent axes
        u_ax = getattr(axes_cfg, fm["u"])
        v_ax = getattr(axes_cfg, fm["v"])
        f_ax = getattr(axes_cfg, fm["fixed"])
        u_range = u_ax.range[:2]
        v_range = v_ax.range[:2]
        f_val   = f_ax.range[1] if fm["sign"] > 0 else f_ax.range[0]

        # Compute origin_3d: the scene point at (u_mid, v_mid, f_val)
        u_mid = (u_range[0] + u_range[1]) / 2
        v_mid = (v_range[0] + v_range[1]) / 2

        coord_map = {"x": 0.0, "y": 0.0, "z": 0.0}
        coord_map[fm["u"]]     = u_mid
        coord_map[fm["v"]]     = v_mid
        coord_map[fm["fixed"]] = f_val
        origin_3d = pcfg(coord_map["x"], coord_map["y"], coord_map["z"])

        # Basis vectors
        normal_key = {OUT: "OUT", IN: "IN", UP: "UP",
                       DOWN: "DOWN", RIGHT: "RIGHT", LEFT: "LEFT"}[fm["normal"]]
        basis_u, basis_v = self._BASIS_MAP[normal_key]

        # Build plane config with inherited ranges
        cfg = plane_config or PlaneConfig()
        cfg.u_range = u_range
        cfg.v_range = v_range
        cfg.u_length = parent_axes._x_length if fm["u"] == "x" else \
                       parent_axes._y_length if fm["u"] == "y" else \
                       parent_axes._z_length
        cfg.v_length = parent_axes._x_length if fm["v"] == "x" else \
                       parent_axes._y_length if fm["v"] == "y" else \
                       parent_axes._z_length

        self._face      = face
        self._parent_c2p = pcfg

        super().__init__(
            origin_3d=origin_3d,
            basis_u=basis_u,
            basis_v=basis_v,
            plane_config=cfg,
            **kwargs,
        )

    # Override plane_c2p to use parent axes coordinate system exactly
    def plane_c2p(self, u: float, v: float) -> np.ndarray:
        fm   = self._FACE_MAP[self._face]
        f_ax = getattr(self._plane_cfg, "fixed_val", 0.0)
        coords: Dict[str, float] = {}
        coords[fm["u"]]     = u
        coords[fm["v"]]     = v
        coords[fm["fixed"]] = f_ax
        return self._parent_c2p(coords.get("x", 0.0),
                                 coords.get("y", 0.0),
                                 coords.get("z", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  FLOATING PLANE 3D
# ─────────────────────────────────────────────────────────────────────────────

class FloatingPlane3D(NumberPlane3D):
    """
    A ``NumberPlane3D`` that lives at a **variable data-value slice**
    along one axis of a ``StatsAxes3D``.

    The slice position is controlled by a ``ValueTracker``, enabling
    smooth ``animate_slice()`` animations through the data volume.

    Typical use: slicing a bivariate normal surface at z = const,
    showing a cross-sectional distribution as a glowing plane.

    Usage
    -----
    ::

        axes  = StatsAxes3D.for_3d_surface(...)
        plane = FloatingPlane3D(axes, slice_axis=AxisID.Z,
                                slice_value=0.0)
        scene.add(plane)
        scene.play(plane.animate_slice(target_value=2.0, run_time=3))
    """

    def __init__(
        self,
        parent_axes:  StatsAxes3D,
        slice_axis:   "AxisID"      = None,
        slice_value:  float         = 0.0,
        plane_config: Optional[PlaneConfig] = None,
        **kwargs,
    ) -> None:
        from .axes3d import AxisID as _AID
        slice_axis = slice_axis or _AID.Z

        self._parent_axes  = parent_axes
        self._slice_axis   = slice_axis
        self._slice_tracker = ValueTracker(slice_value)

        # Determine orientation based on which axis is sliced
        axes_cfg = parent_axes.config
        if slice_axis == _AID.Z:
            u_range = axes_cfg.x.range[:2]
            v_range = axes_cfg.y.range[:2]
            basis_u = np.array([1., 0., 0.])
            basis_v = np.array([0., 1., 0.])
            def _origin():
                sv = self._slice_tracker.get_value()
                return parent_axes.c2p(
                    (u_range[0]+u_range[1])/2,
                    (v_range[0]+v_range[1])/2,
                    sv)
        elif slice_axis == _AID.X:
            u_range = axes_cfg.y.range[:2]
            v_range = axes_cfg.z.range[:2]
            basis_u = np.array([0., 1., 0.])
            basis_v = np.array([0., 0., 1.])
            def _origin():
                sv = self._slice_tracker.get_value()
                return parent_axes.c2p(
                    sv,
                    (u_range[0]+u_range[1])/2,
                    (v_range[0]+v_range[1])/2)
        else:  # Y
            u_range = axes_cfg.x.range[:2]
            v_range = axes_cfg.z.range[:2]
            basis_u = np.array([1., 0., 0.])
            basis_v = np.array([0., 0., 1.])
            def _origin():
                sv = self._slice_tracker.get_value()
                return parent_axes.c2p(
                    (u_range[0]+u_range[1])/2,
                    sv,
                    (v_range[0]+v_range[1])/2)

        self._origin_fn = _origin

        cfg = plane_config or PlaneConfig(
            u_range=u_range,
            v_range=v_range,
            fill_opacity=0.08,
            material=PlaneMaterial.GLASS,
        )

        super().__init__(
            origin_3d=_origin(),
            basis_u=basis_u,
            basis_v=basis_v,
            plane_config=cfg,
            **kwargs,
        )

        # Attach updater so the plane follows the tracker
        self.add_updater(lambda m, dt: m._update_slice_position())

    def _update_slice_position(self) -> None:
        """Move the plane to the current tracker value."""
        new_origin = self._origin_fn()
        delta = new_origin - self._origin_3d
        self.shift(delta)
        self._origin_3d = new_origin

    def animate_slice(
        self,
        target_value: float,
        run_time:     float = 2.5,
        rate_func:    Callable = smooth,
    ) -> Animation:
        """
        Animate the plane sliding to *target_value* along its slice axis.
        """
        return self._slice_tracker.animate(
    run_time=run_time,
    rate_func=rate_func,
).set_value(target_value)

    @property
    def slice_value(self) -> float:
        return self._slice_tracker.get_value()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SHADOW PROJECTION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowConfig:
    """Configuration for shadow projection."""
    dot_radius:       float         = 0.045
    dot_opacity:      float         = 0.45
    dot_color:        Optional[str] = None    # None → dim version of source color
    show_drop_lines:  bool          = True
    drop_line_color:  Optional[str] = None
    drop_line_opacity: float        = 0.25
    drop_line_width:  float         = 0.8
    drop_line_style:  GridStyle     = GridStyle.DASHED
    shadow_offset:    float         = 0.005   # push shadow onto plane surface


class ShadowProjection3D(VGroup):
    """
    Projects a group of 3-D Dot3D points (or any mobjects with a
    ``get_center()`` method) orthogonally onto a target ``NumberPlane3D``.

    For each source point it creates:
        • A Dot3D on the plane surface (the shadow)
        • An optional dashed drop-line from the source to the shadow

    The projected dots and lines are stored in two sub-VGroups:
        ``self.shadows``    — projected dots on the plane
        ``self.drop_lines`` — dashed connector lines

    Usage
    -----
    ::

        scatter_3d = VGroup(*[Dot3D(pt) for pt in pts])
        floor = FaceNumberPlane3D(axes, PlaneFace.XZ_BOTTOM)
        proj  = ShadowProjection3D(scatter_3d, floor,
                                   ShadowConfig(dot_color=BLUE_A))
        scene.add(proj)
        scene.play(FadeIn(proj))
    """

    def __init__(
        self,
        source_mobs:  VGroup,
        target_plane: NumberPlane3D,
        cfg:          Optional[ShadowConfig] = None,
        theme:        Optional[StatsColorPalette] = None,
    ) -> None:
        super().__init__()
        cfg   = cfg   or ShadowConfig()
        theme = theme or StatsTheme.current

        self.shadows    = VGroup()
        self.drop_lines = VGroup()

        for mob in source_mobs.submobjects:
            src_pt   = mob.get_center()
            proj_pt  = target_plane.project_point(src_pt)
            proj_pt  = proj_pt + target_plane.normal * cfg.shadow_offset

            # Shadow colour: dim the source colour if none specified
            src_color = mob.get_fill_color() or theme.primary
            s_color   = cfg.dot_color or _dim_color(src_color, 0.55)

            dot = Dot3D(proj_pt, radius=cfg.dot_radius, color=s_color)
            dot.set_opacity(cfg.dot_opacity)
            self.shadows.add(dot)

            if cfg.show_drop_lines:
                dl_color = cfg.drop_line_color or theme.neutral
                if cfg.drop_line_style == GridStyle.DASHED:
                    dl = DashedLine(src_pt, proj_pt,
                                    color=dl_color,
                                    stroke_width=cfg.drop_line_width * 1.2,
                                    dash_length=0.08)
                else:
                    dl = Line3D(src_pt, proj_pt, color=dl_color,
                                thickness=cfg.drop_line_width * 0.005)
                dl.set_opacity(cfg.drop_line_opacity)
                self.drop_lines.add(dl)

        self.add(self.drop_lines, self.shadows)

    def animate_project(
        self,
        source_mobs: VGroup,
        target_plane: NumberPlane3D,
        run_time:     float = 1.5,
        lag_ratio:    float = 0.03,
    ) -> Animation:
        """
        Animate each source point collapsing onto its projected shadow.
        Each source dot moves along its drop-line to its shadow position.
        """
        anims = []
        for src, shadow, dl in zip(
            source_mobs.submobjects,
            self.shadows.submobjects,
            self.drop_lines.submobjects,
        ):
            anims.append(AnimationGroup(
                Create(dl, run_time=run_time * 0.6),
                FadeIn(shadow, run_time=run_time * 0.4),
            ))
        return AnimationGroup(*anims, lag_ratio=lag_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  NUMBER PLANE 3D SYSTEM  (manages multiple planes)
# ─────────────────────────────────────────────────────────────────────────────

class NumberPlane3DSystem(VGroup):
    """
    Manages a coordinated set of ``NumberPlane3D`` objects — typically
    the three face-planes of a ``StatsAxes3D`` bounding box.

    Provides:
        • Simultaneous build / exit animations for all planes
        • Coordinated shadow projection from 3-D data objects
        • Easy access to individual planes by face key

    Usage
    -----
    ::

        axes   = StatsAxes3D.for_3d_surface(...)
        system = NumberPlane3DSystem.for_axes(axes,
                     faces=[PlaneFace.XZ_BOTTOM,
                            PlaneFace.YZ_LEFT,
                            PlaneFace.XY_BACK])
        scene.add(axes, system)
        scene.play(system.animate_build_all())
    """

    def __init__(self) -> None:
        super().__init__()
        self._planes: Dict[str, NumberPlane3D] = {}

    def add_plane(self, key: str, plane: NumberPlane3D) -> None:
        """Register and add a plane under *key*."""
        self._planes[key] = plane
        self.add(plane)

    def get_plane(self, key: str) -> Optional[NumberPlane3D]:
        return self._planes.get(key)

    def animate_build_all(
        self,
        cfg: Optional[AnimationConfig] = None,
        lag_ratio: float = 0.15,
    ) -> Animation:
        """Build all planes with a staggered reveal."""
        cfg = cfg or AnimationConfig()
        anims = [p.animate_build(cfg) for p in self._planes.values()]
        return AnimationGroup(*anims, lag_ratio=lag_ratio)

    def animate_exit_all(
        self,
        cfg: Optional[AnimationConfig] = None,
        lag_ratio: float = 0.1,
    ) -> Animation:
        cfg = cfg or AnimationConfig()
        anims = [p.animate_exit(cfg) for p in self._planes.values()]
        return AnimationGroup(*anims, lag_ratio=lag_ratio)

    def project_to_all(
        self,
        source_mobs:  VGroup,
        shadow_cfg:   Optional[ShadowConfig] = None,
    ) -> Dict[str, ShadowProjection3D]:
        """
        Project *source_mobs* onto every registered plane.
        Returns dict[key, ShadowProjection3D].
        """
        projections = {}
        for key, plane in self._planes.items():
            proj = ShadowProjection3D(source_mobs, plane, shadow_cfg)
            self.add(proj)
            projections[key] = proj
        return projections

    def animate_project_to_all(
        self,
        source_mobs:  VGroup,
        shadow_cfg:   Optional[ShadowConfig] = None,
        run_time:     float = 2.0,
    ) -> Animation:
        """Animate projections onto all planes simultaneously."""
        projections = self.project_to_all(source_mobs, shadow_cfg)
        anims = [
            proj.animate_project(source_mobs, plane, run_time=run_time)
            for (key, plane), proj in zip(self._planes.items(),
                                           projections.values())
        ]
        return AnimationGroup(*anims)

    # ── factory ───────────────────────────────────────────────────────────

    @classmethod
    def for_axes(
        cls,
        parent_axes:  StatsAxes3D,
        faces:        List[PlaneFace] = None,
        plane_config: Optional[PlaneConfig] = None,
    ) -> "NumberPlane3DSystem":
        """
        Build a system of ``FaceNumberPlane3D`` objects for the given *faces*.
        Defaults to bottom floor + back wall + left wall.
        """
        faces = faces or [
            PlaneFace.XZ_BOTTOM,
            PlaneFace.XY_BACK,
            PlaneFace.YZ_LEFT,
        ]
        system = cls()
        for face in faces:
            plane = FaceNumberPlane3D(parent_axes, face, plane_config)
            system.add_plane(face.name, plane)
        return system


# ─────────────────────────────────────────────────────────────────────────────
# 12.  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    """Safe unit-vector normalisation."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _dim_color(hex_color: str, factor: float = 0.6) -> str:
    """
    Darken *hex_color* by *factor* (0 = black, 1 = original).
    Used for shadow dot colours.
    """
    from colour import Color
    try:
        c = Color(hex_color)
        c.luminance = max(0.0, c.luminance * factor)
        return c.hex_l
    except Exception:
        return hex_color


def make_plane_for_distribution(
    u_range:  Tuple[float, float] = (-4.0, 4.0),
    v_range:  Tuple[float, float] = (0.0, 0.45),
    origin:   Optional[np.ndarray] = None,
    facing:   Literal["front", "right", "top"] = "front",
    theme:    Optional[StatsColorPalette] = None,
) -> NumberPlane3D:
    """
    Convenience factory: return a ``NumberPlane3D`` pre-configured for
    displaying a 1-D probability distribution curve on a given face.
    """
    t = theme or StatsTheme.current
    cfg = PlaneConfig(
        u_range=u_range,
        v_range=v_range,
        fill_opacity=0.10,
        material=PlaneMaterial.FROSTED,
        fill_color=t.surface,
        border_color=t.border,
        grid=PlaneGridConfig(
            major_spacing=1.0,
            major_opacity=0.35,
            minor_opacity=0.12,
            style=GridStyle.DASHED,
        ),
        axes=PlaneAxisConfig(
            u_label="x", v_label="f(x)",
            show_ticks=True,
        ),
    )
    basis_map = {
        "front": (np.array([1,0,0]), np.array([0,1,0])),
        "right": (np.array([0,1,0]), np.array([0,0,1])),
        "top":   (np.array([1,0,0]), np.array([0,0,1])),
    }
    bu, bv = basis_map.get(facing, (np.array([1,0,0]), np.array([0,1,0])))
    return NumberPlane3D(
        origin_3d=origin if origin is not None else np.zeros(3),
        basis_u=bu,
        basis_v=bv,
        plane_config=cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 13.  MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enumerations
    "PlaneFace",
    "PlaneMaterial",
    # Config dataclasses
    "PlaneGridConfig",
    "PlaneAxisConfig",
    "PlaneConfig",
    "ShadowConfig",
    # Sub-components
    "PlaneGeometry",
    "PlaneGrid",
    "PlaneAxes",
    "PlaneTickSystem",
    "PlaneOriginMark",
    "PlaneDrawingLayer",
    # Main classes
    "NumberPlane3D",
    "FaceNumberPlane3D",
    "FloatingPlane3D",
    "ShadowProjection3D",
    "NumberPlane3DSystem",
    # Utilities
    "make_plane_for_distribution",
]