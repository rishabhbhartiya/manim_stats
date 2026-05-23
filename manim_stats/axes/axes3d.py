"""
manim_stats/axes/axes3d.py
==========================
Master 3-D axes system for the Manim Statistics Extension.

Every chart, distribution, regression, and inference asset uses
``StatsAxes3D`` as its coordinate backbone.

Architecture
------------
StatsAxes3D
    ├── AxisSpine3D          — individual X / Y / Z spine with tip
    ├── TickSystem3D         — major + minor ticks on one axis
    ├── TickLabelSystem3D    — numeric / custom labels for one axis
    ├── GridPlane3D          — one face of the grid box (XY / YZ / XZ)
    ├── GridSystem3D         — all active grid planes together
    ├── OriginDecoration3D   — dot, crosshair, zero-plane highlight
    ├── BoundingBox3D        — optional wireframe cube around axes
    ├── ReferenceLineSystem  — h-lines, v-lines, planes, brackets
    ├── AnnotationSystem3D   — point annotations, region highlights
    └── AxisLabel3D          — per-axis MathTex / Text title

Coordinate systems
------------------
Data space   — the (x, y, z) numbers the user works with
Scene space  — Manim 3-D world units centred at ORIGIN
Axes map data → scene via ``c2p(x, y, z)`` (coordinate to point)
and scene → data via ``p2c(px, py, pz)`` (point to coordinate).

Scale modes (per axis independently)
--------------------------------------
    LINEAR   — standard linear mapping
    LOG      — log₁₀ mapping  (x_data > 0 required)
    SYMLOG   — signed-log  (safe for data crossing zero)
    LOGIT    — logit mapping  (x_data ∈ (0, 1) required)
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
from numpy.typing import ArrayLike, NDArray

from manim import (
    # Mobjects
    VGroup, VMobject, Group, Mobject,
    Line3D, Arrow3D, Dot3D, Sphere, Cylinder,
    Text, MathTex, Tex,
    Line, DashedLine, DashedVMobject,
    Dot, Rectangle, RoundedRectangle,
    SurroundingRectangle, Polygon,
    # Animations
    Animation, AnimationGroup, Succession,
    Create, FadeIn, FadeOut, Write,
    DrawBorderThenFill, GrowArrow,
    Transform, MoveAlongPath,
    UpdateFromAlphaFunc,
    # Constants
    ORIGIN, UP, DOWN, LEFT, RIGHT, IN, OUT,
    X_AXIS, Y_AXIS, Z_AXIS,
    WHITE, BLACK, GRAY, GRAY_A, GRAY_B, GRAY_C,
    DEGREES, PI, TAU,
    # Utilities
    interpolate_color, normalize,
    rotation_matrix, angle_of_vector,
    smooth, rate_functions, there_and_back,
    DecimalNumber, Integer,
    # 3-D
    ThreeDAxes, NumberPlane,
    always_redraw, ValueTracker,
)

from ..core.base import (
    StatsObject3D, StatsProp3D,
    StatsTheme, StatsColorPalette,
    MaterialConfig, MaterialApplicator,
    LabelConfig, LabelAnchor, LabelAttachment,
    AnimationConfig, BuildStyle, DataUpdateMode,
    HighlightStyle, HighlightSystem,
    BoundData, ThemeMode,
)
from ..core.math_utils import (
    auto_range, generate_ticks, nice_number, format_stat_value,
    FloatArray,
)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  ENUMERATIONS & CONFIG DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

class AxisScaleMode(Enum):
    LINEAR = auto()
    LOG    = auto()   # log₁₀
    SYMLOG = auto()   # signed log
    LOGIT  = auto()   # logit (0–1 data)


class TickStyle(Enum):
    OUTWARD = auto()   # tick extends away from plot area
    INWARD  = auto()   # tick extends into plot area
    CROSS   = auto()   # tick extends both ways


class GridStyle(Enum):
    SOLID  = auto()
    DASHED = auto()
    DOTTED = auto()


class AxisID(Enum):
    X = "x"
    Y = "y"
    Z = "z"


@dataclass
class AxisConfig:
    """
    Complete configuration for one axis (X, Y, or Z).
    """
    # Range: (min, max, step).  step=None → auto-computed
    range:          Tuple[float, float, Optional[float]] = (-5.0, 5.0, None)
    scale_mode:     AxisScaleMode   = AxisScaleMode.LINEAR
    symlog_thresh:  float           = 1.0       # linear threshold for SYMLOG

    # Spine
    visible:        bool            = True
    spine_color:    Optional[str]   = None      # None → theme.neutral
    spine_width:    float           = 2.5
    include_tip:    bool            = True
    tip_length:     float           = 0.2
    tip_width:      float           = 0.12
    length:         float           = 6.0       # scene units

    # Ticks
    show_major_ticks:  bool         = True
    show_minor_ticks:  bool         = True
    major_tick_length: float        = 0.18
    minor_tick_length: float        = 0.09
    tick_style:        TickStyle    = TickStyle.OUTWARD
    tick_color:        Optional[str] = None     # None → spine_color
    minor_per_major:   int          = 4
    custom_ticks:      Optional[List[float]] = None  # override auto ticks

    # Labels
    show_tick_labels:  bool         = True
    tick_font_size:    float        = 20
    tick_color_label:  Optional[str] = None
    tick_label_offset: float        = 0.35
    tick_decimal_places: int        = 1
    use_sci_notation:  bool         = False     # force sci notation
    sci_threshold:     float        = 1e-3      # auto sci if |val| < this
    custom_tick_labels: Optional[Dict[float, str]] = None  # val → str overrides

    # Axis title
    label:          str             = ""
    label_is_math:  bool            = True
    label_font_size: float          = 28
    label_color:    Optional[str]   = None
    label_offset:   float           = 0.6       # beyond last tick label

    # Extras
    include_zero_line:  bool        = False
    zero_line_color:    Optional[str] = None
    zero_line_width:    float       = 1.2
    zero_line_opacity:  float       = 0.5


@dataclass
class GridConfig:
    """Configuration for the 3-D grid system."""
    show_xy_plane:  bool       = True
    show_xz_plane:  bool       = False   # floor grid
    show_yz_plane:  bool       = False

    style:          GridStyle  = GridStyle.SOLID
    major_color:    Optional[str] = None   # None → theme.border
    minor_color:    Optional[str] = None
    major_opacity:  float      = 0.55
    minor_opacity:  float      = 0.20
    major_width:    float      = 1.0
    minor_width:    float      = 0.5

    # Fade out toward edges
    fade_edges:     bool       = True
    fade_strength:  float      = 0.6    # 0 = no fade, 1 = fully transparent at edge

    # Depth
    z_offset:       float      = -0.01  # push slightly behind data


@dataclass
class AxesConfig:
    """
    Top-level configuration for a complete ``StatsAxes3D`` instance.
    Holds sub-configs for each axis and the grid.
    """
    x: AxisConfig = field(default_factory=AxisConfig)
    y: AxisConfig = field(default_factory=lambda: AxisConfig(
        range=(0.0, 1.0, None), length=5.0, label="y"))
    z: AxisConfig = field(default_factory=lambda: AxisConfig(
        range=(-5.0, 5.0, None), length=6.0, label="z",
        visible=False,
        show_tick_labels=False,
        show_major_ticks=False,
    ))
    grid:            GridConfig   = field(default_factory=GridConfig)

    # Global scene-space dimensions
    x_length:        float        = 7.0
    y_length:        float        = 5.0
    z_length:        float        = 5.0

    # Origin
    show_origin_dot:     bool     = True
    origin_dot_radius:   float    = 0.055
    origin_dot_color:    Optional[str] = None  # None → theme.accent

    # Bounding box
    show_bounding_box:   bool     = False
    bounding_box_color:  Optional[str] = None
    bounding_box_opacity: float   = 0.12
    bounding_box_width:  float    = 0.8

    # Global appearance
    background_rect:     bool     = False   # filled background rectangle
    background_opacity:  float    = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SCALE TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

class ScaleTransform:
    """
    Forward (data → normalised [0,1]) and inverse transforms for each scale mode.
    Used internally by ``StatsAxes3D.c2p()`` and ``p2c()``.
    """

    @staticmethod
    def forward(
        value:  float,
        lo:     float,
        hi:     float,
        mode:   AxisScaleMode,
        thresh: float = 1.0,
    ) -> float:
        """Map *value* ∈ [lo, hi] → [0, 1]."""
        if mode == AxisScaleMode.LINEAR:
            return (value - lo) / (hi - lo) if hi != lo else 0.0

        elif mode == AxisScaleMode.LOG:
            if lo <= 0:
                raise ValueError("LOG scale requires lo > 0.")
            lv  = math.log10(max(value, lo * 1e-9))
            llo = math.log10(lo)
            lhi = math.log10(hi)
            return (lv - llo) / (lhi - llo) if lhi != llo else 0.0

        elif mode == AxisScaleMode.SYMLOG:
            def _symlog(v: float) -> float:
                if abs(v) <= thresh:
                    return v / thresh * math.log10(1 + 1)
                sign = 1 if v > 0 else -1
                return sign * (math.log10(abs(v)) - math.log10(thresh) +
                               math.log10(1 + 1))
            sv  = _symlog(value)
            slo = _symlog(lo)
            shi = _symlog(hi)
            return (sv - slo) / (shi - slo) if shi != slo else 0.0

        elif mode == AxisScaleMode.LOGIT:
            v   = max(min(value, 1 - 1e-9), 1e-9)
            lv  = math.log(v / (1 - v))
            llo = math.log(lo / (1 - lo))
            lhi = math.log(hi / (1 - hi))
            return (lv - llo) / (lhi - llo) if lhi != llo else 0.0

        return 0.0

    @staticmethod
    def inverse(
        t:      float,
        lo:     float,
        hi:     float,
        mode:   AxisScaleMode,
        thresh: float = 1.0,
    ) -> float:
        """Map *t* ∈ [0, 1] → data value in [lo, hi]."""
        if mode == AxisScaleMode.LINEAR:
            return lo + t * (hi - lo)

        elif mode == AxisScaleMode.LOG:
            llo = math.log10(lo)
            lhi = math.log10(hi)
            return 10 ** (llo + t * (lhi - llo))

        elif mode == AxisScaleMode.SYMLOG:
            def _symlog(v: float) -> float:
                if abs(v) <= thresh:
                    return v / thresh * math.log10(2)
                sign = 1 if v > 0 else -1
                return sign * (math.log10(abs(v)) - math.log10(thresh) +
                               math.log10(2))
            slo = _symlog(lo); shi = _symlog(hi)
            sv  = slo + t * (shi - slo)
            c   = math.log10(2)
            if abs(sv) <= c:
                return sv / c * thresh
            sign = 1 if sv > 0 else -1
            return sign * (10 ** (abs(sv) - c + math.log10(thresh)))

        elif mode == AxisScaleMode.LOGIT:
            llo = math.log(lo / (1 - lo))
            lhi = math.log(hi / (1 - hi))
            lv  = llo + t * (lhi - llo)
            return 1 / (1 + math.exp(-lv))

        return lo


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AXIS SPINE
# ─────────────────────────────────────────────────────────────────────────────

class AxisSpine3D(VGroup):
    """
    One 3-D axis spine: a Line3D body + optional Arrow3D tip.

    The spine runs from ``start`` to ``end`` in scene space.
    A negative extension (``neg_extend``) can push the spine slightly
    past the origin so it covers tick marks that fall at the boundary.
    """

    def __init__(
        self,
        start:      np.ndarray,
        end:        np.ndarray,
        cfg:        AxisConfig,
        theme:      StatsColorPalette,
        neg_extend: float = 0.1,
    ) -> None:
        super().__init__()
        self._cfg   = cfg
        self._theme = theme

        color = cfg.spine_color or theme.neutral
        direction = end - start
        length    = np.linalg.norm(direction)

        # Extend slightly past origin
        unit = direction / length if length > 0 else RIGHT
        body_start = start - unit * neg_extend

        if cfg.include_tip:
            # Use Arrow3D: shaft + tip
            arrow = Arrow3D(
                start=body_start,
                end=end,
                color=color,
                thickness=cfg.spine_width * 0.008,   # Arrow3D uses radius
                tip_length=cfg.tip_length,
                tip_base_radius=cfg.tip_width * 0.6,
            )
            self.add(arrow)
            self._body = arrow
        else:
            line = Line3D(
                start=body_start, end=end,
                color=color,
                thickness=cfg.spine_width * 0.008,
            )
            self.add(line)
            self._body = line

        self.set_color(color)

    def set_spine_color(self, color: str) -> None:
        self._body.set_color(color)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TICK SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class TickSystem3D(VGroup):
    """
    Major + minor tick marks for one axis.

    Ticks are short Line3D segments placed perpendicular to the spine
    in the plane of the nearest grid face.
    """

    def __init__(
        self,
        axis_id:    AxisID,
        cfg:        AxisConfig,
        axes_cfg:   AxesConfig,
        theme:      StatsColorPalette,
        c2p_fn:     Callable[[float, float, float], np.ndarray],
    ) -> None:
        super().__init__()
        self._axis_id  = axis_id
        self._cfg      = cfg
        self._theme    = theme
        self._c2p      = c2p_fn

        lo, hi, step = self._resolve_range(cfg)
        tick_info    = generate_ticks(lo, hi, step,
                                       cfg.minor_per_major)
        color = cfg.tick_color or cfg.spine_color or theme.neutral

        if cfg.show_major_ticks:
            for tv in tick_info["major"]:
                tick = self._make_tick(tv, cfg.major_tick_length,
                                       color, cfg.spine_width * 0.7)
                if tick is not None:
                    self.add(tick)

        if cfg.show_minor_ticks:
            for tv in tick_info["minor"]:
                tick = self._make_tick(tv, cfg.minor_tick_length,
                                       color, cfg.spine_width * 0.45)
                if tick is not None:
                    self.add(tick)

    # ── helpers ───────────────────────────────────────────────────────────

    def _resolve_range(
        self, cfg: AxisConfig
    ) -> Tuple[float, float, float]:
        lo, hi, step = cfg.range
        if step is None:
            lo, hi, step = auto_range([lo, hi], padding=0.0, n_ticks=5)
        return lo, hi, step

    def _make_tick(
        self,
        data_val:   float,
        length:     float,
        color:      str,
        width:      float,
    ) -> Optional[Line3D]:
        """Build one tick Line3D at *data_val* along this axis."""
        axis   = self._axis_id
        c2p    = self._c2p
        style  = self._cfg.tick_style

        # Scene position of the tick centre
        if axis == AxisID.X:
            pos = c2p(data_val, 0.0, 0.0)
            perp = UP      # ticks point up (in XY plane)
        elif axis == AxisID.Y:
            pos = c2p(0.0, data_val, 0.0)
            perp = RIGHT
        else:
            pos = c2p(0.0, 0.0, data_val)
            perp = UP

        if style == TickStyle.OUTWARD:
            start_pt = pos
            end_pt   = pos - perp * length   # "outside" the axes box
        elif style == TickStyle.INWARD:
            start_pt = pos
            end_pt   = pos + perp * length
        else:  # CROSS
            start_pt = pos - perp * length * 0.5
            end_pt   = pos + perp * length * 0.5

        try:
            tick = Line3D(start=start_pt, end=end_pt,
                          color=color, thickness=width * 0.006)
            return tick
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TICK LABEL SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class TickLabelSystem3D(VGroup):
    """
    Numeric (or custom) labels for the tick marks on one axis.

    Labels are always-redraw VMobjects that optionally face the camera.
    """

    def __init__(
        self,
        axis_id:    AxisID,
        cfg:        AxisConfig,
        theme:      StatsColorPalette,
        c2p_fn:     Callable[[float, float, float], np.ndarray],
    ) -> None:
        super().__init__()
        self._axis_id = axis_id
        self._cfg     = cfg
        self._theme   = theme
        self._c2p     = c2p_fn
        self._mobs:   List[VMobject] = []

        lo, hi, step = self._resolve_range(cfg)
        tick_info    = generate_ticks(lo, hi, step, cfg.minor_per_major)
        color        = cfg.tick_color_label or theme.text_secondary

        for tv, lbl_str in zip(tick_info["major"], tick_info["labels"]):
            # Custom label override
            if cfg.custom_tick_labels and tv in cfg.custom_tick_labels:
                lbl_str = cfg.custom_tick_labels[tv]
            elif cfg.use_sci_notation or abs(tv) > 0 and abs(tv) < cfg.sci_threshold:
                lbl_str = f"{tv:.2e}"

            mob = self._make_label(tv, lbl_str, color)
            if mob is not None:
                self.add(mob)
                self._mobs.append(mob)

    # ── helpers ───────────────────────────────────────────────────────────

    def _resolve_range(
        self, cfg: AxisConfig
    ) -> Tuple[float, float, float]:
        lo, hi, step = cfg.range
        if step is None:
            lo, hi, step = auto_range([lo, hi], padding=0.0, n_ticks=5)
        return lo, hi, step

    def _make_label(
        self,
        data_val:   float,
        text:       str,
        color:      str,
    ) -> Optional[VMobject]:
        axis   = self._axis_id
        c2p    = self._c2p
        offset = self._cfg.tick_label_offset

        if axis == AxisID.X:
            pos = c2p(data_val, 0.0, 0.0) + DOWN * offset
        elif axis == AxisID.Y:
            pos = c2p(0.0, data_val, 0.0) + LEFT * offset
        else:
            pos = c2p(0.0, 0.0, data_val) + LEFT * offset + DOWN * 0.1

        try:
            mob = Text(text,
                       font_size=self._cfg.tick_font_size,
                       color=color)
            mob.move_to(pos)
            return mob
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 5.  AXIS LABEL (TITLE)
# ─────────────────────────────────────────────────────────────────────────────

class AxisLabel3D(VGroup):
    """
    Axis title label with MathTex / Text support and camera-facing rotation.
    Positioned beyond the last tick label, offset along the perpendicular.
    """

    def __init__(
        self,
        axis_id:    AxisID,
        cfg:        AxisConfig,
        theme:      StatsColorPalette,
        scene_end:  np.ndarray,          # scene-space end of the axis spine
    ) -> None:
        super().__init__()
        if not cfg.label:
            return

        color = cfg.label_color or theme.text_primary
        if cfg.label_is_math:
            mob = MathTex(cfg.label,
                          font_size=cfg.label_font_size,
                          color=color)
        else:
            mob = Text(cfg.label,
                       font_size=cfg.label_font_size,
                       color=color)

        # Position beyond spine tip
        if axis_id == AxisID.X:
            mob.next_to(scene_end, RIGHT * cfg.label_offset + DOWN * 0.25)
        elif axis_id == AxisID.Y:
            mob.next_to(scene_end, UP * cfg.label_offset)
        else:
            mob.next_to(scene_end, OUT * cfg.label_offset + RIGHT * 0.3)

        self.add(mob)
        self._label_mob = mob

    def get_label_mob(self) -> Optional[VMobject]:
        return getattr(self, "_label_mob", None)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GRID PLANE
# ─────────────────────────────────────────────────────────────────────────────

class GridPlane3D(VGroup):
    """
    One face of the 3-D grid box.

    Draws major and minor grid lines on one plane (XY, YZ, or XZ),
    with optional edge-fade opacity gradient.

    The plane face is parameterised by two axes (``axis_a``, ``axis_b``)
    and a fixed value on the third axis (``fixed_val``, default 0).
    """

    def __init__(
        self,
        axis_a:     AxisID,          # "horizontal" axis of this plane
        axis_b:     AxisID,          # "vertical" axis of this plane
        cfg_a:      AxisConfig,      # config for axis_a
        cfg_b:      AxisConfig,      # config for axis_b
        grid_cfg:   GridConfig,
        theme:      StatsColorPalette,
        c2p_fn:     Callable[[float, float, float], np.ndarray],
        fixed_val:  float = 0.0,
    ) -> None:
        super().__init__()
        self._theme   = theme
        self._grid    = grid_cfg
        self._c2p     = c2p_fn

        major_color = grid_cfg.major_color or theme.border
        minor_color = grid_cfg.minor_color or theme.border

        lo_a, hi_a, step_a = self._resolve(cfg_a)
        lo_b, hi_b, step_b = self._resolve(cfg_b)

        ticks_a = generate_ticks(lo_a, hi_a, step_a, cfg_a.minor_per_major)
        ticks_b = generate_ticks(lo_b, hi_b, step_b, cfg_b.minor_per_major)

        # Major gridlines along axis_a (lines parallel to axis_b)
        for val in ticks_a["major"]:
            line = self._make_line(axis_a, axis_b, val,
                                   lo_b, hi_b, fixed_val,
                                   major_color, grid_cfg.major_width,
                                   grid_cfg.major_opacity, grid_cfg.style)
            if line: self.add(line)

        # Major gridlines along axis_b (lines parallel to axis_a)
        for val in ticks_b["major"]:
            line = self._make_line(axis_b, axis_a, val,
                                   lo_a, hi_a, fixed_val,
                                   major_color, grid_cfg.major_width,
                                   grid_cfg.major_opacity, grid_cfg.style)
            if line: self.add(line)

        # Minor gridlines along axis_a
        for val in ticks_a["minor"]:
            line = self._make_line(axis_a, axis_b, val,
                                   lo_b, hi_b, fixed_val,
                                   minor_color, grid_cfg.minor_width,
                                   grid_cfg.minor_opacity, grid_cfg.style)
            if line: self.add(line)

        # Minor gridlines along axis_b
        for val in ticks_b["minor"]:
            line = self._make_line(axis_b, axis_a, val,
                                   lo_a, hi_a, fixed_val,
                                   minor_color, grid_cfg.minor_width,
                                   grid_cfg.minor_opacity, grid_cfg.style)
            if line: self.add(line)

        # Push plane behind data
        self.shift(self._plane_normal(axis_a, axis_b) * grid_cfg.z_offset)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve(cfg: AxisConfig) -> Tuple[float, float, float]:
        lo, hi, step = cfg.range
        if step is None:
            lo, hi, step = auto_range([lo, hi], padding=0.0, n_ticks=5)
        return lo, hi, step

    def _coord(
        self,
        fixed_axis: AxisID,
        fixed_val:  float,
        free_axis:  AxisID,
        free_val:   float,
        plane_fixed_val: float,
    ) -> np.ndarray:
        """
        Build a (x, y, z) data coordinate where one axis is fixed at
        *fixed_val*, one runs freely at *free_val*, and the plane's
        constant axis is at *plane_fixed_val*.
        """
        coords: Dict[AxisID, float] = {
            AxisID.X: plane_fixed_val,
            AxisID.Y: plane_fixed_val,
            AxisID.Z: plane_fixed_val,
        }
        coords[fixed_axis] = fixed_val
        coords[free_axis]  = free_val
        return self._c2p(coords[AxisID.X], coords[AxisID.Y], coords[AxisID.Z])

    def _make_line(
        self,
        fixed_axis:  AxisID,
        free_axis:   AxisID,
        fixed_val:   float,
        free_lo:     float,
        free_hi:     float,
        plane_val:   float,
        color:       str,
        width:       float,
        opacity:     float,
        style:       GridStyle,
    ) -> Optional[VMobject]:
        """Draw one grid line at *fixed_val* on *fixed_axis*, spanning *free_axis*."""
        try:
            # Determine which axis is the "plane's normal"
            # so we can correctly assign coordinate slots
            used  = {fixed_axis, free_axis}
            third = next(a for a in AxisID if a not in used)

            def _pt(fv: float) -> np.ndarray:
                d: Dict[AxisID, float] = {third: plane_val,
                                           fixed_axis: fixed_val,
                                           free_axis: fv}
                return self._c2p(d[AxisID.X], d[AxisID.Y], d[AxisID.Z])

            p1 = _pt(free_lo)
            p2 = _pt(free_hi)

            if style == GridStyle.SOLID:
                line = Line3D(start=p1, end=p2,
                              color=color,
                              thickness=width * 0.005)
                line.set_opacity(opacity)
            elif style == GridStyle.DASHED:
                line = DashedLine(start=p1, end=p2,
                                  color=color,
                                  stroke_width=width * 1.2,
                                  dash_length=0.12)
                line.set_opacity(opacity)
            else:  # DOTTED
                line = DashedLine(start=p1, end=p2,
                                  color=color,
                                  stroke_width=width,
                                  dash_length=0.04,
                                  dashed_ratio=0.3)
                line.set_opacity(opacity)

            if self._grid.fade_edges:
                # Reduce opacity at end segments (edge fade approximation)
                line.set_opacity(opacity * (1.0 - self._grid.fade_strength * 0.3))

            return line
        except Exception:
            return None

    @staticmethod
    def _plane_normal(axis_a: AxisID, axis_b: AxisID) -> np.ndarray:
        """Return a unit vector pointing 'into' the scene for this plane."""
        used   = {axis_a, axis_b}
        normal_axis = next(a for a in AxisID if a not in used)
        return {AxisID.X: RIGHT, AxisID.Y: UP, AxisID.Z: OUT}[normal_axis]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRID SYSTEM (composite of all planes)
# ─────────────────────────────────────────────────────────────────────────────

class GridSystem3D(VGroup):
    """
    Aggregates all active ``GridPlane3D`` objects.
    """

    def __init__(
        self,
        axes_cfg:   AxesConfig,
        theme:      StatsColorPalette,
        c2p_fn:     Callable[[float, float, float], np.ndarray],
    ) -> None:
        super().__init__()
        gc = axes_cfg.grid
        xc, yc, zc = axes_cfg.x, axes_cfg.y, axes_cfg.z

        if gc.show_xy_plane:
            plane = GridPlane3D(AxisID.X, AxisID.Y, xc, yc, gc,
                                theme, c2p_fn, fixed_val=0.0)
            self.add(plane)

        if gc.show_xz_plane:
            lo_y = axes_cfg.y.range[0]
            plane = GridPlane3D(AxisID.X, AxisID.Z, xc, zc, gc,
                                theme, c2p_fn, fixed_val=lo_y)
            self.add(plane)

        if gc.show_yz_plane:
            lo_x = axes_cfg.x.range[0]
            plane = GridPlane3D(AxisID.Y, AxisID.Z, yc, zc, gc,
                                theme, c2p_fn, fixed_val=lo_x)
            self.add(plane)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ORIGIN DECORATION
# ─────────────────────────────────────────────────────────────────────────────

class OriginDecoration3D(VGroup):
    """
    Visual elements at the axes origin:
        • Dot3D sphere
        • Optional crosshair lines (tiny extensions along each axis)
        • Optional zero-plane highlight ring
    """

    def __init__(
        self,
        cfg:        AxesConfig,
        theme:      StatsColorPalette,
        c2p_fn:     Callable[[float, float, float], np.ndarray],
    ) -> None:
        super().__init__()
        if not cfg.show_origin_dot:
            return

        origin_pt = c2p_fn(0.0, 0.0, 0.0)
        color     = cfg.origin_dot_color or theme.accent

        # Origin sphere
        dot = Dot3D(point=origin_pt,
                    radius=cfg.origin_dot_radius,
                    color=color)
        dot.set_opacity(0.9)
        self.add(dot)

        # Crosshair micro-lines
        ext = 0.15
        for direction, ax_id in [
            (RIGHT, AxisID.X),
            (UP,    AxisID.Y),
            (OUT,   AxisID.Z),
        ]:
            h1 = Line3D(origin_pt - direction * ext,
                        origin_pt + direction * ext,
                        color=color,
                        thickness=0.004)
            h1.set_opacity(0.5)
            self.add(h1)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  BOUNDING BOX
# ─────────────────────────────────────────────────────────────────────────────

class BoundingBox3D(VGroup):
    """
    Wireframe cube drawn at the boundary of the 3-D axes.
    Useful for communicating the full 3-D data volume.
    """

    def __init__(
        self,
        axes_cfg:   AxesConfig,
        theme:      StatsColorPalette,
        c2p_fn:     Callable[[float, float, float], np.ndarray],
    ) -> None:
        super().__init__()
        xlo, xhi = axes_cfg.x.range[0], axes_cfg.x.range[1]
        ylo, yhi = axes_cfg.y.range[0], axes_cfg.y.range[1]
        zlo, zhi = axes_cfg.z.range[0], axes_cfg.z.range[1]

        color   = axes_cfg.bounding_box_color or theme.border
        opacity = axes_cfg.bounding_box_opacity
        width   = axes_cfg.bounding_box_width

        corners = {
            "lll": c2p_fn(xlo, ylo, zlo), "lhl": c2p_fn(xlo, yhi, zlo),
            "hll": c2p_fn(xhi, ylo, zlo), "hhl": c2p_fn(xhi, yhi, zlo),
            "llh": c2p_fn(xlo, ylo, zhi), "lhh": c2p_fn(xlo, yhi, zhi),
            "hlh": c2p_fn(xhi, ylo, zhi), "hhh": c2p_fn(xhi, yhi, zhi),
        }

        edges = [
            # Bottom face
            ("lll","hll"), ("hll","hhl"), ("hhl","lhl"), ("lhl","lll"),
            # Top face
            ("llh","hlh"), ("hlh","hhh"), ("hhh","lhh"), ("lhh","llh"),
            # Verticals
            ("lll","llh"), ("hll","hlh"), ("hhl","hhh"), ("lhl","lhh"),
        ]

        for a, b in edges:
            edge = Line3D(corners[a], corners[b],
                          color=color, thickness=width * 0.005)
            edge.set_opacity(opacity)
            self.add(edge)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  REFERENCE LINE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReferenceLineConfig:
    value:      float
    axis:       AxisID
    color:      Optional[str]   = None
    width:      float           = 1.8
    opacity:    float           = 0.85
    style:      GridStyle       = GridStyle.DASHED
    label:      str             = ""
    label_side: Literal["left", "right", "top", "bottom"] = "right"
    label_font_size: float      = 22


class ReferenceLineSystem(VGroup):
    """
    Adds horizontal, vertical, or z-level reference lines to the axes.

    Typical uses:
        - Mean / median lines on a histogram
        - Alpha threshold on a p-value axis
        - Decision boundary on a classification plot
    """

    def __init__(
        self,
        theme: StatsColorPalette,
        c2p_fn: Callable[[float, float, float], np.ndarray],
        axes_cfg: AxesConfig,
    ) -> None:
        super().__init__()
        self._theme   = theme
        self._c2p     = c2p_fn
        self._axes    = axes_cfg
        self._lines:  Dict[str, VGroup] = {}

    def add_line(self, key: str, cfg: ReferenceLineConfig) -> VGroup:
        """
        Draw one reference line and optionally label it.
        Returns the VGroup of [line, label] for external animation.
        """
        color = cfg.color or self._theme.accent
        xlo, xhi = self._axes.x.range[:2]
        ylo, yhi = self._axes.y.range[:2]
        zlo, zhi = self._axes.z.range[:2]

        if cfg.axis == AxisID.Y:
            p1 = self._c2p(xlo, cfg.value, 0.0)
            p2 = self._c2p(xhi, cfg.value, 0.0)
        elif cfg.axis == AxisID.X:
            p1 = self._c2p(cfg.value, ylo, 0.0)
            p2 = self._c2p(cfg.value, yhi, 0.0)
        else:
            p1 = self._c2p(0.0, ylo, cfg.value)
            p2 = self._c2p(0.0, yhi, cfg.value)

        if cfg.style == GridStyle.SOLID:
            line = Line3D(p1, p2, color=color,
                          thickness=cfg.width * 0.007)
        else:
            line = DashedLine(p1, p2, color=color,
                              stroke_width=cfg.width * 1.4,
                              dash_length=0.15)
        line.set_opacity(cfg.opacity)

        grp = VGroup(line)

        if cfg.label:
            lbl_pos = p2 + {
                "right":  RIGHT * 0.25,
                "left":   LEFT  * 0.25,
                "top":    UP    * 0.25,
                "bottom": DOWN  * 0.25,
            }.get(cfg.label_side, RIGHT * 0.25)
            lbl = Text(cfg.label,
                       font_size=cfg.label_font_size,
                       color=color)
            lbl.move_to(lbl_pos)
            grp.add(lbl)

        self._lines[key] = grp
        self.add(grp)
        return grp

    def remove_line(self, key: str) -> None:
        if key in self._lines:
            self.remove(self._lines.pop(key))

    def update_line_value(self, key: str, new_value: float) -> None:
        """Remove and redraw a reference line at a new data value."""
        if key in self._lines:
            cfg_old = self._lines[key]
            self.remove_line(key)

    def add_span(
        self,
        key:         str,
        lo:          float,
        hi:          float,
        axis:        AxisID,
        color:       Optional[str]  = None,
        opacity:     float          = 0.18,
        label:       str            = "",
        label_font_size: float      = 20,
    ) -> VGroup:
        """
        Shade a region between *lo* and *hi* along *axis*.
        Returns the shaded VGroup.
        """
        c   = color or self._theme.accent
        xlo, xhi = self._axes.x.range[:2]
        ylo, yhi = self._axes.y.range[:2]

        if axis == AxisID.X:
            corners = [
                self._c2p(lo, ylo, 0), self._c2p(hi, ylo, 0),
                self._c2p(hi, yhi, 0), self._c2p(lo, yhi, 0),
            ]
        else:  # Y axis span
            corners = [
                self._c2p(xlo, lo, 0), self._c2p(xhi, lo, 0),
                self._c2p(xhi, hi, 0), self._c2p(xlo, hi, 0),
            ]

        poly = Polygon(*corners)
        poly.set_fill(c, opacity=opacity)
        poly.set_stroke(c, width=1.0, opacity=opacity * 1.5)

        grp = VGroup(poly)

        if label:
            mid_x = (lo + hi) / 2
            mid_y = (ylo + yhi) / 2
            if axis == AxisID.X:
                lbl_pt = self._c2p(mid_x, yhi, 0) + UP * 0.2
            else:
                lbl_pt = self._c2p(xhi, mid_y, 0) + RIGHT * 0.2
            lbl = Text(label, font_size=label_font_size, color=c)
            lbl.move_to(lbl_pt)
            grp.add(lbl)

        self._lines[key] = grp
        self.add(grp)
        return grp

    def add_bracket(
        self,
        key:        str,
        lo:         float,
        hi:         float,
        axis:       AxisID,
        color:      Optional[str]  = None,
        offset:     float          = 0.4,
        label:      str            = "",
        label_font_size: float     = 22,
    ) -> VGroup:
        """
        Draw a bracket indicating a span [lo, hi] along *axis*,
        offset from the axis by *offset* scene units.
        Useful for annotating IQR, CI, critical regions.
        """
        c     = color or self._theme.accent
        yval  = self._axes.y.range[0] - offset    # below x-axis

        if axis == AxisID.X:
            p_lo  = self._c2p(lo, yval, 0)
            p_hi  = self._c2p(hi, yval, 0)
            mid   = self._c2p((lo + hi) / 2, yval, 0)
            # Bracket shape: p_lo─────p_hi  with tiny serifs
            bar   = Line3D(p_lo, p_hi, color=c, thickness=0.01)
            s_lo  = Line3D(p_lo, p_lo + UP * 0.15, color=c, thickness=0.008)
            s_hi  = Line3D(p_hi, p_hi + UP * 0.15, color=c, thickness=0.008)
            grp   = VGroup(bar, s_lo, s_hi)
        else:
            # Vertical bracket
            p_lo  = self._c2p(self._axes.x.range[0] - offset, lo, 0)
            p_hi  = self._c2p(self._axes.x.range[0] - offset, hi, 0)
            mid   = self._c2p(self._axes.x.range[0] - offset, (lo + hi)/2, 0)
            bar   = Line3D(p_lo, p_hi, color=c, thickness=0.01)
            s_lo  = Line3D(p_lo, p_lo + RIGHT * 0.15, color=c, thickness=0.008)
            s_hi  = Line3D(p_hi, p_hi + RIGHT * 0.15, color=c, thickness=0.008)
            grp   = VGroup(bar, s_lo, s_hi)

        if label:
            lbl = Text(label, font_size=label_font_size, color=c)
            lbl.move_to(mid + DOWN * 0.35 if axis == AxisID.X
                        else mid + LEFT * 0.4)
            grp.add(lbl)

        self._lines[key] = grp
        self.add(grp)
        return grp


# ─────────────────────────────────────────────────────────────────────────────
# 11.  ANNOTATION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PointAnnotationConfig:
    x:          float
    y:          float
    z:          float          = 0.0
    label:      str            = ""
    label_is_math: bool        = False
    dot_color:  Optional[str]  = None
    dot_radius: float          = 0.08
    line_color: Optional[str]  = None
    line_style: GridStyle      = GridStyle.DASHED
    line_width: float          = 1.5
    line_opacity: float        = 0.7
    label_offset: np.ndarray   = field(default_factory=lambda: np.array([0.3, 0.3, 0.0]))
    label_font_size: float     = 22
    show_drop_lines: bool      = True   # drop lines to axes
    drop_line_opacity: float   = 0.35


class AnnotationSystem3D(VGroup):
    """
    Point annotations, drop lines, and region labels for 3-D axes.

    Each annotation consists of:
        • A Dot3D at the data point
        • A text/math label with a leader line
        • Optional drop lines to the axis planes (like a 3-D coordinate cross)
    """

    def __init__(
        self,
        theme:    StatsColorPalette,
        c2p_fn:   Callable[[float, float, float], np.ndarray],
        axes_cfg: AxesConfig,
    ) -> None:
        super().__init__()
        self._theme      = theme
        self._c2p        = c2p_fn
        self._axes       = axes_cfg
        self._annotations: Dict[str, VGroup] = {}

    def add_point(self, key: str, cfg: PointAnnotationConfig) -> VGroup:
        """Annotate a data point at (x, y, z)."""
        t     = self._theme
        color = cfg.dot_color or t.accent

        scene_pt = self._c2p(cfg.x, cfg.y, cfg.z)
        dot      = Dot3D(scene_pt, radius=cfg.dot_radius, color=color)
        grp      = VGroup(dot)

        # Leader line + label
        if cfg.label:
            label_pt = scene_pt + cfg.label_offset
            if cfg.line_style == GridStyle.DASHED:
                leader = DashedLine(scene_pt, label_pt,
                                    color=cfg.line_color or color,
                                    stroke_width=cfg.line_width,
                                    dash_length=0.08)
            else:
                leader = Line3D(scene_pt, label_pt,
                                color=cfg.line_color or color,
                                thickness=cfg.line_width * 0.004)
            leader.set_opacity(cfg.line_opacity)

            mob_cls = MathTex if cfg.label_is_math else Text
            lbl     = mob_cls(cfg.label,
                              font_size=cfg.label_font_size,
                              color=color)
            lbl.move_to(label_pt + RIGHT * lbl.width / 2 + UP * 0.05)
            grp.add(leader, lbl)

        # Drop lines to the three axis planes
        if cfg.show_drop_lines:
            origin = self._c2p(0, 0, 0)
            xlo    = self._axes.x.range[0]
            ylo    = self._axes.y.range[0]

            for end_pt, opacity in [
                (self._c2p(cfg.x, ylo, cfg.z),  cfg.drop_line_opacity),    # to XZ floor
                (self._c2p(xlo, cfg.y, cfg.z),  cfg.drop_line_opacity),    # to YZ wall
                (self._c2p(cfg.x, cfg.y, 0.0),  cfg.drop_line_opacity),    # to XY back
            ]:
                dl = DashedLine(scene_pt, end_pt,
                                color=t.neutral,
                                stroke_width=0.9,
                                dash_length=0.07)
                dl.set_opacity(opacity)
                grp.add(dl)

        self._annotations[key] = grp
        self.add(grp)
        return grp

    def remove_annotation(self, key: str) -> None:
        if key in self._annotations:
            self.remove(self._annotations.pop(key))

    def add_region_label(
        self,
        key:    str,
        x:      float,
        y:      float,
        label:  str,
        color:  Optional[str] = None,
        font_size: float      = 24,
        bg:     bool          = True,
    ) -> VMobject:
        """
        Floating label at (x, y) with optional pill background.
        Useful for labelling distribution regions (e.g. "Rejection Region").
        """
        t   = self._theme
        c   = color or t.text_primary
        pos = self._c2p(x, y, 0.0)

        mob = Text(label, font_size=font_size, color=c)
        mob.move_to(pos)

        if bg:
            bg_rect = RoundedRectangle(
                corner_radius=0.12,
                width=mob.width + 0.3,
                height=mob.height + 0.2,
            )
            bg_rect.set_fill(t.surface, opacity=0.8)
            bg_rect.set_stroke(c, width=1.2, opacity=0.6)
            bg_rect.move_to(mob)
            grp = VGroup(bg_rect, mob)
        else:
            grp = VGroup(mob)

        self._annotations[key] = grp
        self.add(grp)
        return grp


# ─────────────────────────────────────────────────────────────────────────────
# 12.  ZERO LINES
# ─────────────────────────────────────────────────────────────────────────────

class ZeroLines3D(VGroup):
    """
    Optional zero-crossing lines (x=0, y=0 spine extensions)
    drawn with reduced opacity when the data range crosses zero.
    Used for signed data axes like z-scores, residuals, log-ratios.
    """

    def __init__(
        self,
        axes_cfg:  AxesConfig,
        theme:     StatsColorPalette,
        c2p_fn:    Callable[[float, float, float], np.ndarray],
    ) -> None:
        super().__init__()
        for axis_id, axis_cfg in [
            (AxisID.X, axes_cfg.x),
            (AxisID.Y, axes_cfg.y),
        ]:
            if not axis_cfg.include_zero_line:
                continue
            lo, hi = axis_cfg.range[:2]
            if lo >= 0 or hi <= 0:
                continue   # zero not in range — nothing to do
            color = axis_cfg.zero_line_color or theme.neutral
            if axis_id == AxisID.X:
                p1 = c2p_fn(0.0, axes_cfg.y.range[0], 0.0)
                p2 = c2p_fn(0.0, axes_cfg.y.range[1], 0.0)
            else:
                p1 = c2p_fn(axes_cfg.x.range[0], 0.0, 0.0)
                p2 = c2p_fn(axes_cfg.x.range[1], 0.0, 0.0)
            line = Line3D(p1, p2, color=color,
                          thickness=axis_cfg.zero_line_width * 0.005)
            line.set_opacity(axis_cfg.zero_line_opacity)
            self.add(line)


# ─────────────────────────────────────────────────────────────────────────────
# 13.  MASTER CLASS — StatsAxes3D
# ─────────────────────────────────────────────────────────────────────────────

class StatsAxes3D(StatsObject3D):
    """
    Master 3-D axes object for the Manim Statistics Extension.

    Every chart, histogram, distribution, scatter, and inference asset
    owns a ``StatsAxes3D`` for its coordinate system.

    Quick start
    -----------
    ::

        axes = StatsAxes3D(
            x_range=(-3, 3, 1),
            y_range=(0, 0.5, 0.1),
            x_label="x",
            y_label="f(x)",
        )
        scene.add(axes)
        scene.play(axes.animate_build())

    Or use a preset factory::

        axes = StatsAxes3D.for_distribution(
            x_range=(-4, 4, 1),
            y_range=(0, 0.45, 0.1),
        )

    Coordinate transform
    --------------------
    ``axes.c2p(x, y, z)``  → scene np.ndarray
    ``axes.p2c(px, py, pz)`` → data (x, y, z) tuple

    Adding reference lines
    ----------------------
    ::

        axes.add_h_line("mean", value=0.0, label=r"\\mu", color=BLUE)
        axes.add_v_line("alpha", value=1.96, label=r"z_{\\alpha/2}")
        axes.add_span("ci", lo=-1.96, hi=1.96, axis=AxisID.X)

    Annotating points
    -----------------
    ::

        axes.annotate_point("mode", x=0, y=0.4,
                            label="Mode", show_drop_lines=True)
    """

    def __init__(
        self,
        # Convenience shortcuts (override AxesConfig if supplied)
        x_range:    Optional[Tuple[float, float, float]] = None,
        y_range:    Optional[Tuple[float, float, float]] = None,
        z_range:    Optional[Tuple[float, float, float]] = None,
        x_label:    Optional[str] = None,
        y_label:    Optional[str] = None,
        z_label:    Optional[str] = None,
        x_length:   float         = 7.0,
        y_length:   float         = 5.0,
        z_length:   float         = 5.0,
        # Full config override
        axes_config: Optional[AxesConfig] = None,
        # base class pass-through
        **kwargs,
    ) -> None:

        # ── resolve config ────────────────────────────────────────────────
        cfg = axes_config or AxesConfig()

        if x_range is not None: cfg.x.range = x_range
        if y_range is not None: cfg.y.range = y_range
        if z_range is not None: cfg.z.range = z_range
        if x_label is not None: cfg.x.label = x_label
        if y_label is not None: cfg.y.label = y_label
        if z_label is not None: cfg.z.label = z_label
        cfg.x_length = x_length
        cfg.y_length = y_length
        cfg.z_length = z_length

        self._axes_cfg = cfg

        # Scene-space axis lengths mapped from config
        self._x_length = x_length
        self._y_length = y_length
        self._z_length = z_length

        # Sub-system handles (populated in _build_geometry)
        self.spines:      VGroup               = VGroup()
        self.tick_system: VGroup               = VGroup()
        self.label_system: VGroup              = VGroup()
        self.grid:         GridSystem3D        = None    # type: ignore[assignment]
        self.origin_deco:  OriginDecoration3D  = None    # type: ignore[assignment]
        self.bbox:         BoundingBox3D       = None    # type: ignore[assignment]
        self.ref_lines:    ReferenceLineSystem = None    # type: ignore[assignment]
        self.annotations:  AnnotationSystem3D  = None    # type: ignore[assignment]
        self.zero_lines:   ZeroLines3D         = None    # type: ignore[assignment]

        super().__init__(**kwargs)

    # ── geometry build ────────────────────────────────────────────────────

    def _build_geometry(self) -> None:
        cfg   = self._axes_cfg
        theme = self._palette

        # ── spines ────────────────────────────────────────────────────────
        ox = self.c2p(cfg.x.range[0], 0, 0)
        ex = self.c2p(cfg.x.range[1], 0, 0)
        oy = self.c2p(0, cfg.y.range[0], 0)
        ey = self.c2p(0, cfg.y.range[1], 0)
        oz = self.c2p(0, 0, cfg.z.range[0])
        ez = self.c2p(0, 0, cfg.z.range[1])

        if cfg.x.visible:
            sx = AxisSpine3D(ox, ex, cfg.x, theme)
            self.spines.add(sx)
        if cfg.y.visible:
            sy = AxisSpine3D(oy, ey, cfg.y, theme)
            self.spines.add(sy)
        if cfg.z.visible:
            sz = AxisSpine3D(oz, ez, cfg.z, theme)
            self.spines.add(sz)
        self.add(self.spines)

        # ── ticks ─────────────────────────────────────────────────────────
        for axis_id, axis_cfg in [
            (AxisID.X, cfg.x),
            (AxisID.Y, cfg.y),
            (AxisID.Z, cfg.z),
        ]:
            if axis_cfg.visible:
                ts = TickSystem3D(axis_id, axis_cfg, cfg, theme, self.c2p)
                self.tick_system.add(ts)
        self.add(self.tick_system)

        # ── tick labels ───────────────────────────────────────────────────
        for axis_id, axis_cfg in [
            (AxisID.X, cfg.x),
            (AxisID.Y, cfg.y),
            (AxisID.Z, cfg.z),
        ]:
            if axis_cfg.visible and axis_cfg.show_tick_labels:
                tls = TickLabelSystem3D(axis_id, axis_cfg, theme, self.c2p)
                self.label_system.add(tls)
        self.add(self.label_system)

        # ── axis title labels ─────────────────────────────────────────────
        axis_titles = VGroup()
        for axis_id, axis_cfg, spine_end in [
            (AxisID.X, cfg.x, ex),
            (AxisID.Y, cfg.y, ey),
            (AxisID.Z, cfg.z, ez),
        ]:
            if axis_cfg.visible and axis_cfg.label:
                al = AxisLabel3D(axis_id, axis_cfg, theme, spine_end)
                axis_titles.add(al)
        self.add(axis_titles)

        # ── grid ──────────────────────────────────────────────────────────
        self.grid = GridSystem3D(cfg, theme, self.c2p)
        self.add(self.grid)

        # ── origin decoration ─────────────────────────────────────────────
        self.origin_deco = OriginDecoration3D(cfg, theme, self.c2p)
        self.add(self.origin_deco)

        # ── bounding box ──────────────────────────────────────────────────
        if cfg.show_bounding_box:
            self.bbox = BoundingBox3D(cfg, theme, self.c2p)
            self.add(self.bbox)

        # ── reference lines & annotations (empty at build time) ───────────
        self.ref_lines   = ReferenceLineSystem(theme, self.c2p, cfg)
        self.annotations = AnnotationSystem3D(theme, self.c2p, cfg)
        self.zero_lines  = ZeroLines3D(cfg, theme, self.c2p)
        self.add(self.ref_lines)
        self.add(self.annotations)
        self.add(self.zero_lines)

    # ── coordinate transforms ─────────────────────────────────────────────

    def c2p(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        """
        Data coordinates → scene 3-D point (np.ndarray).
        Handles linear, log, symlog, logit per-axis independently.
        """
        cfg = self._axes_cfg

        tx = ScaleTransform.forward(
            x, cfg.x.range[0], cfg.x.range[1],
            cfg.x.scale_mode, cfg.x.symlog_thresh,
        )
        ty = ScaleTransform.forward(
            y, cfg.y.range[0], cfg.y.range[1],
            cfg.y.scale_mode, cfg.y.symlog_thresh,
        )
        tz = ScaleTransform.forward(
            z, cfg.z.range[0], cfg.z.range[1],
            cfg.z.scale_mode, cfg.z.symlog_thresh,
        ) if cfg.z.visible else 0.0

        # Map [0,1] → scene length, centred at ORIGIN
        px = (tx - 0.5) * self._x_length
        py = (ty - 0.5) * self._y_length
        pz = (tz - 0.5) * self._z_length if cfg.z.visible else 0.0

        return np.array([px, py, pz])

    def p2c(
        self, px: float, py: float, pz: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Scene 3-D point → data coordinates.
        """
        cfg = self._axes_cfg

        tx = px / self._x_length + 0.5
        ty = py / self._y_length + 0.5
        tz = pz / self._z_length + 0.5 if cfg.z.visible else 0.5

        x = ScaleTransform.inverse(tx, cfg.x.range[0], cfg.x.range[1],
                                    cfg.x.scale_mode, cfg.x.symlog_thresh)
        y = ScaleTransform.inverse(ty, cfg.y.range[0], cfg.y.range[1],
                                    cfg.y.scale_mode, cfg.y.symlog_thresh)
        z = ScaleTransform.inverse(tz, cfg.z.range[0], cfg.z.range[1],
                                    cfg.z.scale_mode, cfg.z.symlog_thresh)
        return x, y, z

    def data_to_scene_length(self, data_delta: float, axis: AxisID) -> float:
        """
        Convert a data-space distance to a scene-space length.
        Useful for sizing bars, error-bar heights, etc.
        For non-linear scales this is an approximation at the midpoint.
        """
        cfg = self._axes_cfg
        ax_cfg = {AxisID.X: cfg.x, AxisID.Y: cfg.y, AxisID.Z: cfg.z}[axis]
        lo, hi = ax_cfg.range[:2]
        mid = (lo + hi) / 2
        p1 = self.c2p(*{AxisID.X: (mid, 0, 0),
                         AxisID.Y: (0, mid, 0),
                         AxisID.Z: (0, 0, mid)}[axis])
        p2 = self.c2p(*{AxisID.X: (mid + data_delta, 0, 0),
                         AxisID.Y: (0, mid + data_delta, 0),
                         AxisID.Z: (0, 0, mid + data_delta)}[axis])
        return float(np.linalg.norm(p2 - p1))

    # ── reference line convenience API ────────────────────────────────────

    def add_h_line(
        self,
        key:     str,
        value:   float,
        color:   Optional[str] = None,
        width:   float         = 1.8,
        opacity: float         = 0.85,
        style:   GridStyle     = GridStyle.DASHED,
        label:   str           = "",
    ) -> VGroup:
        """Add a horizontal reference line at y = *value*."""
        cfg = ReferenceLineConfig(
            value=value, axis=AxisID.Y,
            color=color, width=width, opacity=opacity,
            style=style, label=label,
        )
        return self.ref_lines.add_line(key, cfg)

    def add_v_line(
        self,
        key:     str,
        value:   float,
        color:   Optional[str] = None,
        width:   float         = 1.8,
        opacity: float         = 0.85,
        style:   GridStyle     = GridStyle.DASHED,
        label:   str           = "",
    ) -> VGroup:
        """Add a vertical reference line at x = *value*."""
        cfg = ReferenceLineConfig(
            value=value, axis=AxisID.X,
            color=color, width=width, opacity=opacity,
            style=style, label=label,
        )
        return self.ref_lines.add_line(key, cfg)

    def add_span(
        self,
        key:     str,
        lo:      float,
        hi:      float,
        axis:    AxisID     = AxisID.X,
        color:   Optional[str] = None,
        opacity: float      = 0.18,
        label:   str        = "",
    ) -> VGroup:
        """Shade a region between *lo* and *hi* along *axis*."""
        return self.ref_lines.add_span(key, lo, hi, axis,
                                       color, opacity, label)

    def add_bracket(
        self,
        key:    str,
        lo:     float,
        hi:     float,
        axis:   AxisID     = AxisID.X,
        color:  Optional[str] = None,
        offset: float      = 0.4,
        label:  str        = "",
    ) -> VGroup:
        """Draw a bracket indicating span [lo, hi] along *axis*."""
        return self.ref_lines.add_bracket(key, lo, hi, axis,
                                          color, offset, label)

    def annotate_point(
        self,
        key:            str,
        x:              float,
        y:              float,
        z:              float  = 0.0,
        label:          str    = "",
        show_drop_lines: bool  = True,
        label_is_math:  bool   = False,
        color:          Optional[str] = None,
        label_offset:   Optional[np.ndarray] = None,
    ) -> VGroup:
        """Annotate a data point with a dot, leader line, and label."""
        offset = label_offset if label_offset is not None \
                 else np.array([0.3, 0.3, 0.0])
        cfg = PointAnnotationConfig(
            x=x, y=y, z=z,
            label=label,
            label_is_math=label_is_math,
            dot_color=color,
            line_color=color,
            show_drop_lines=show_drop_lines,
            label_offset=offset,
        )
        return self.annotations.add_point(key, cfg)

    # ── animations ────────────────────────────────────────────────────────

    def animate_build(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        """
        Reveal the axes in a structured sequence:
            1. Grid planes fade in softly
            2. Spines grow outward from origin
            3. Ticks and labels write themselves
            4. Origin dot pulses once
        """
        cfg = cfg or self._anim_cfg
        t   = cfg.run_time

        grid_in = FadeIn(self.grid, run_time=t * 0.4,
                         rate_func=smooth)
        spines_in = AnimationGroup(
            *[Create(s, run_time=t * 0.5, rate_func=smooth)
              for s in self.spines.submobjects],
            lag_ratio=0.15,
        )
        ticks_in = FadeIn(self.tick_system,
                          run_time=t * 0.3, rate_func=smooth)
        labels_in = AnimationGroup(
            *[Write(lbl, run_time=t * 0.4)
              for lbl in self.label_system.submobjects],
            lag_ratio=0.05,
        )
        origin_in = FadeIn(self.origin_deco,
                           run_time=t * 0.2, rate_func=smooth)
        return Succession(
            grid_in,
            spines_in,
            AnimationGroup(ticks_in, labels_in, origin_in, lag_ratio=0.1),
            lag_ratio=0,
        )

    def animate_update(
        self,
        new_data:   Any,
        cfg:        Optional[AnimationConfig] = None,
    ) -> Animation:
        """
        Alias for ``animate_update_range`` when *new_data* is a
        dict of {axis_key: (lo, hi, step)}.
        """
        if isinstance(new_data, dict):
            return self.animate_update_range(cfg=cfg, **new_data)
        return FadeIn(self, run_time=0.01)

    def animate_update_range(
        self,
        x_range: Optional[Tuple[float, float, float]] = None,
        y_range: Optional[Tuple[float, float, float]] = None,
        z_range: Optional[Tuple[float, float, float]] = None,
        cfg:     Optional[AnimationConfig] = None,
    ) -> Animation:
        """
        Smoothly rescale axes to new data ranges.
        Rebuilds geometry and returns a Transform animation.
        """
        cfg = cfg or self._anim_cfg
        old = self.copy()

        if x_range: self._axes_cfg.x.range = x_range
        if y_range: self._axes_cfg.y.range = y_range
        if z_range: self._axes_cfg.z.range = z_range

        # Remove old sub-objects and rebuild
        self.submobjects.clear()
        self._build_geometry()

        return Transform(old, self,
                         run_time=cfg.run_time,
                         rate_func=cfg.rate_func)

    def animate_highlight(
        self,
        style: HighlightStyle = HighlightStyle.PULSE,
        cfg:   Optional[AnimationConfig] = None,
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return HighlightSystem.pulse(self.spines, cfg)

    def animate_exit(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return FadeOut(self, run_time=cfg.run_time * 0.6,
                       rate_func=smooth)

    # ── scale mode setters ────────────────────────────────────────────────

    def set_x_scale(self, mode: AxisScaleMode, thresh: float = 1.0) -> "StatsAxes3D":
        self._axes_cfg.x.scale_mode    = mode
        self._axes_cfg.x.symlog_thresh = thresh
        return self

    def set_y_scale(self, mode: AxisScaleMode, thresh: float = 1.0) -> "StatsAxes3D":
        self._axes_cfg.y.scale_mode    = mode
        self._axes_cfg.y.symlog_thresh = thresh
        return self

    def set_z_scale(self, mode: AxisScaleMode, thresh: float = 1.0) -> "StatsAxes3D":
        self._axes_cfg.z.scale_mode    = mode
        self._axes_cfg.z.symlog_thresh = thresh
        return self

    # ── preset factory methods ────────────────────────────────────────────

    @classmethod
    def for_distribution(
        cls,
        x_range: Tuple[float, float, float] = (-4.0, 4.0, 1.0),
        y_range: Tuple[float, float, float] = (0.0, 0.45, 0.1),
        x_label: str = "x",
        y_label: str = "f(x)",
        **kwargs,
    ) -> "StatsAxes3D":
        """
        Preset for PDF / PMF distribution plots.
        Y-axis starts at 0, no z-axis shown, XY grid enabled.
        """
        cfg = AxesConfig(
            x=AxisConfig(range=x_range, label=x_label,
                         include_zero_line=True),
            y=AxisConfig(range=y_range, label=y_label,
                         length=5.0),
            z=AxisConfig(visible=False),
            grid=GridConfig(show_xy_plane=True,
                            show_xz_plane=False,
                            show_yz_plane=False,
                            style=GridStyle.SOLID,
                            fade_edges=True),
            show_origin_dot=True,
        )
        return cls(axes_config=cfg, **kwargs)

    @classmethod
    def for_scatter(
        cls,
        x_range: Tuple[float, float, float] = (-3.0, 3.0, 1.0),
        y_range: Tuple[float, float, float] = (-3.0, 3.0, 1.0),
        x_label: str = "X",
        y_label: str = "Y",
        **kwargs,
    ) -> "StatsAxes3D":
        """Preset for scatter plots and regression. Both axes cross zero."""
        cfg = AxesConfig(
            x=AxisConfig(range=x_range, label=x_label,
                         include_zero_line=True),
            y=AxisConfig(range=y_range, label=y_label,
                         include_zero_line=True),
            z=AxisConfig(visible=False),
            grid=GridConfig(show_xy_plane=True,
                            style=GridStyle.SOLID,
                            major_opacity=0.45,
                            minor_opacity=0.15),
            show_bounding_box=False,
        )
        return cls(axes_config=cfg, **kwargs)

    @classmethod
    def for_histogram(
        cls,
        x_range: Tuple[float, float, float] = (-4.0, 4.0, 1.0),
        y_range: Tuple[float, float, float] = (0.0, 1.0, 0.2),
        x_label: str = "Value",
        y_label: str = "Frequency",
        **kwargs,
    ) -> "StatsAxes3D":
        """Preset for histograms: y starts at 0, no minor grid clutter."""
        cfg = AxesConfig(
            x=AxisConfig(range=x_range, label=x_label),
            y=AxisConfig(range=y_range, label=y_label,
                         show_minor_ticks=False),
            z=AxisConfig(visible=False),
            grid=GridConfig(show_xy_plane=True,
                            style=GridStyle.DASHED,
                            major_opacity=0.35,
                            minor_opacity=0.0),
        )
        return cls(axes_config=cfg, **kwargs)

    @classmethod
    def for_3d_surface(
        cls,
        x_range: Tuple[float, float, float] = (-3.0, 3.0, 1.0),
        y_range: Tuple[float, float, float] = (-3.0, 3.0, 1.0),
        z_range: Tuple[float, float, float] = (0.0,  0.5, 0.1),
        x_label: str = "x",
        y_label: str = "y",
        z_label: str = "f(x,y)",
        **kwargs,
    ) -> "StatsAxes3D":
        """
        Preset for bivariate distribution surfaces and 3-D scatter.
        All three axes visible with XY floor grid.
        """
        cfg = AxesConfig(
            x=AxisConfig(range=x_range, label=x_label),
            y=AxisConfig(range=y_range, label=y_label),
            z=AxisConfig(range=z_range, label=z_label,
                         visible=True,
                         show_tick_labels=True,
                         show_major_ticks=True),
            grid=GridConfig(show_xy_plane=False,
                            show_xz_plane=True,    # floor
                            show_yz_plane=False,
                            style=GridStyle.SOLID,
                            major_opacity=0.30),
            show_bounding_box=True,
            bounding_box_opacity=0.10,
        )
        return cls(axes_config=cfg, z_length=4.0, **kwargs)

    @classmethod
    def for_correlation(
        cls,
        n_vars:  int    = 4,
        var_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "StatsAxes3D":
        """
        Preset for correlation matrix heatmaps.
        X and Y axes use categorical integer ticks with variable names.
        """
        names = var_names or [f"X_{i+1}" for i in range(n_vars)]
        custom = {float(i): names[i] for i in range(n_vars)}
        cfg = AxesConfig(
            x=AxisConfig(range=(-0.5, n_vars - 0.5, 1.0),
                         label="",
                         custom_tick_labels=custom,
                         show_minor_ticks=False),
            y=AxisConfig(range=(-0.5, n_vars - 0.5, 1.0),
                         label="",
                         custom_tick_labels=custom,
                         show_minor_ticks=False),
            z=AxisConfig(visible=False),
            grid=GridConfig(show_xy_plane=False),
            show_origin_dot=False,
        )
        return cls(axes_config=cfg, **kwargs)

    @classmethod
    def for_hypothesis(
        cls,
        dist_name: Literal["normal", "t", "chi2", "f"] = "normal",
        df:        Optional[float]                       = None,
        **kwargs,
    ) -> "StatsAxes3D":
        """
        Preset for hypothesis testing visualisations.
        X = test statistic, Y = probability density.
        Grid shows critical region spans.
        """
        x_ranges = {
            "normal": (-4.0, 4.0, 1.0),
            "t":      (-4.5, 4.5, 1.0),
            "chi2":   (0.0, 20.0, 2.0),
            "f":      (0.0, 6.0, 1.0),
        }
        xr = x_ranges.get(dist_name, (-4.0, 4.0, 1.0))
        cfg = AxesConfig(
            x=AxisConfig(range=xr,
                         label={"normal": "z", "t": "t",
                                "chi2": r"\chi^2", "f": "F"}[dist_name],
                         include_zero_line=dist_name in ("normal", "t")),
            y=AxisConfig(range=(0.0, 0.45, 0.1),
                         label="f(x)"),
            z=AxisConfig(visible=False),
            grid=GridConfig(show_xy_plane=True,
                            style=GridStyle.DASHED,
                            major_opacity=0.30),
        )
        return cls(axes_config=cfg, **kwargs)

    @classmethod
    def for_time_series(
        cls,
        x_range: Tuple[float, float, float] = (0.0, 100.0, 10.0),
        y_range: Tuple[float, float, float] = (-3.0, 3.0, 1.0),
        x_label: str = "t",
        y_label: str = "X_t",
        **kwargs,
    ) -> "StatsAxes3D":
        """Preset for time-series / running statistics plots."""
        cfg = AxesConfig(
            x=AxisConfig(range=x_range, label=x_label,
                         tick_decimal_places=0),
            y=AxisConfig(range=y_range, label=y_label,
                         include_zero_line=True),
            z=AxisConfig(visible=False),
            grid=GridConfig(show_xy_plane=True,
                            style=GridStyle.DASHED,
                            major_opacity=0.25),
        )
        return cls(axes_config=cfg, **kwargs)

    # ── property accessors ────────────────────────────────────────────────

    @property
    def x_range(self) -> Tuple[float, float, float]:
        return tuple(self._axes_cfg.x.range)   # type: ignore[return-value]

    @property
    def y_range(self) -> Tuple[float, float, float]:
        return tuple(self._axes_cfg.y.range)   # type: ignore[return-value]

    @property
    def z_range(self) -> Tuple[float, float, float]:
        return tuple(self._axes_cfg.z.range)   # type: ignore[return-value]

    @property
    def config(self) -> AxesConfig:
        return self._axes_cfg

    def __repr__(self) -> str:
        xr = self._axes_cfg.x.range
        yr = self._axes_cfg.y.range
        return (f"StatsAxes3D("
                f"x=[{xr[0]}, {xr[1]}], "
                f"y=[{yr[0]}, {yr[1]}])")


# ─────────────────────────────────────────────────────────────────────────────
# 14.  MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enumerations
    "AxisScaleMode",
    "TickStyle",
    "GridStyle",
    "AxisID",
    # Config dataclasses
    "AxisConfig",
    "GridConfig",
    "AxesConfig",
    "ReferenceLineConfig",
    "PointAnnotationConfig",
    # Scale transform
    "ScaleTransform",
    # Sub-components
    "AxisSpine3D",
    "TickSystem3D",
    "TickLabelSystem3D",
    "AxisLabel3D",
    "GridPlane3D",
    "GridSystem3D",
    "OriginDecoration3D",
    "BoundingBox3D",
    "ReferenceLineSystem",
    "AnnotationSystem3D",
    "ZeroLines3D",
    # Master class
    "StatsAxes3D",
]