"""
manim_stats/charts/bar_chart3d.py
==================================
Production-quality 3D bar charts for statistical visualizations.

Design philosophy
-----------------
Every bar is a true 3-faced prism (front, right side, top) with
independent shading per face, an edge-stroke silhouette, and an
optional gloss highlight strip.  This gives bars visible volume in
3D even under orthographic projection.

The chart system is layered:

    Bar3D               – single bar prism, all geometry
    BarChart3D          – single-series chart of N bars
    GroupedBarChart3D   – k series × N categories, bars side-by-side
                          along the depth (y) axis
    StackedBarChart3D   – k series × N categories, bars stacked in z

All charts integrate cleanly with ``FullGrid3D`` / ``GridSnapHelper``
from ``manim_stats.axes.grid3d``.

Classes
-------
BarConfig
BarChart3D
GroupedBarChart3D
StackedBarChart3D

Helpers / internals
-------------------
Bar3D
BarColorPalette
_ValueLabel3D

Usage example
-------------
    from manim import *
    from manim_stats.axes.grid3d import FullGrid3D
    from manim_stats.charts.bar_chart3d import BarChart3D, BarConfig

    class DemoBar(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65*DEGREES, theta=-50*DEGREES)

            grid = FullGrid3D(x_range=[-0.5, 5.5, 1], y_range=[-1, 1, 1],
                              z_range=[0, 6, 1])
            chart = BarChart3D(
                values=[2.1, 4.8, 3.3, 5.5, 1.9],
                labels=["Mon", "Tue", "Wed", "Thu", "Fri"],
                bar_width=0.60,
                bar_depth=0.40,
                z_scale=1.0,
            )
            self.play(grid.animate_build())
            self.play(chart.animate_grow(lag=0.08))
            self.wait()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    List, Sequence, Tuple, Optional, Callable, Union, Dict
)
import numpy as np

from manim import (
    # Mobjects
    VGroup, VMobject, Polygon, Rectangle, Line, DashedLine,
    Ellipse, Text, MathTex, Dot,
    # Scene
    ThreeDScene,
    # Animations
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, GrowFromEdge, Transform,
    UpdateFromAlphaFunc,
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
)

# Re-use colour helpers from grid3d (copy inline to keep the module
# self-contained when grid3d is not yet importable in a fresh checkout)
def _with_opacity(color: ManimColor, opacity: float) -> ManimColor:
    r, g, b = color_to_rgb(color)
    return rgba_to_color([r, g, b, max(0.0, min(1.0, opacity))])


def _darken(color: ManimColor, factor: float = 0.6) -> ManimColor:
    """Return *color* darkened by *factor* (0 = black, 1 = original)."""
    r, g, b = color_to_rgb(color)
    return ManimColor([r * factor, g * factor, b * factor])


def _lighten(color: ManimColor, factor: float = 1.4) -> ManimColor:
    """Return *color* lightened by *factor* (clamped to 1)."""
    r, g, b = color_to_rgb(color)
    return ManimColor([min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0)])


def _lerp_color(a: ManimColor, b: ManimColor, t: float) -> ManimColor:
    ra, ga, ba = color_to_rgb(a)
    rb, gb, bb = color_to_rgb(b)
    return ManimColor([ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t])


# ---------------------------------------------------------------------------
# BarColorPalette
# ---------------------------------------------------------------------------

class BarColorPalette:
    """Pre-built colour palettes for bar charts.

    Each palette is a list of base colours for individual bars or series.
    Face shading (front/side/top) is derived automatically.

    Attributes
    ----------
    CATEGORICAL : list of ManimColor
        Eight distinct hues for categorical data — max contrast.
    SEQUENTIAL_BLUE : list of ManimColor
        Six shades from light-blue → deep-blue for ordered data.
    SEQUENTIAL_TEAL : list of ManimColor
        Six shades from light-teal → deep-teal.
    DIVERGING : list of ManimColor
        Seven steps from deep-red → neutral-gray → deep-blue,
        centred at index 3. Useful for signed bar charts.
    MONOCHROME : list of ManimColor
        Single colour repeated — chart colour is set globally.
    """

    CATEGORICAL: List[ManimColor] = [
        ManimColor("#4A90D9"),  # sky blue
        ManimColor("#E8593C"),  # coral
        ManimColor("#2DAA6E"),  # emerald
        ManimColor("#E0AA40"),  # amber
        ManimColor("#9B59B6"),  # purple
        ManimColor("#1ABC9C"),  # teal
        ManimColor("#E74C3C"),  # red
        ManimColor("#F39C12"),  # orange
    ]

    SEQUENTIAL_BLUE: List[ManimColor] = [
        ManimColor("#C8DCF0"),
        ManimColor("#93BDE0"),
        ManimColor("#5B9FD0"),
        ManimColor("#2E7FBF"),
        ManimColor("#1A5EA0"),
        ManimColor("#0C3D7A"),
    ]

    SEQUENTIAL_TEAL: List[ManimColor] = [
        ManimColor("#B2E8D8"),
        ManimColor("#70CDB5"),
        ManimColor("#2DB092"),
        ManimColor("#188F74"),
        ManimColor("#0D6B58"),
        ManimColor("#064840"),
    ]

    DIVERGING: List[ManimColor] = [
        ManimColor("#A32D2D"),  # deep red  (most negative)
        ManimColor("#D05555"),
        ManimColor("#E89090"),
        ManimColor("#888880"),  # neutral
        ManimColor("#7AABD0"),
        ManimColor("#3A7AB8"),
        ManimColor("#0C4478"),  # deep blue (most positive)
    ]

    MONOCHROME: List[ManimColor] = [ManimColor("#4A90D9")] * 8

    @staticmethod
    def sequential_ramp(
        lo: ManimColor,
        hi: ManimColor,
        n: int,
    ) -> List[ManimColor]:
        """Generate *n* colours linearly interpolated from *lo* to *hi*."""
        return [_lerp_color(lo, hi, i / max(n - 1, 1)) for i in range(n)]

    @staticmethod
    def value_mapped(
        values: Sequence[float],
        lo_color: ManimColor = ManimColor("#A32D2D"),
        hi_color: ManimColor = ManimColor("#0C4478"),
        neutral_color: Optional[ManimColor] = None,
    ) -> List[ManimColor]:
        """Map each value to a colour proportional to its magnitude.

        If *neutral_color* is given, it is used at value=0 for a
        three-way diverging ramp (lo_color … neutral … hi_color).
        """
        arr = np.array(values, dtype=float)
        lo, hi = arr.min(), arr.max()
        span = hi - lo if hi != lo else 1.0
        out = []
        for v in arr:
            t = (v - lo) / span
            if neutral_color is None:
                out.append(_lerp_color(lo_color, hi_color, t))
            else:
                if t < 0.5:
                    out.append(_lerp_color(lo_color, neutral_color, t * 2))
                else:
                    out.append(_lerp_color(neutral_color, hi_color, (t - 0.5) * 2))
        return out


# ---------------------------------------------------------------------------
# BarConfig
# ---------------------------------------------------------------------------

@dataclass
class BarConfig:
    """Visual parameters for Bar3D geometry and shading.

    Shading model
    ~~~~~~~~~~~~~
    Each bar has three visible faces:
    - **Front** (facing the viewer along -y): main body colour.
    - **Right side** (facing +x): darkened by ``side_shade_factor``.
    - **Top** (facing +z): lightened by ``top_shade_factor``.

    Additional layers:
    - Edge strokes on every visible polygon edge.
    - A gloss highlight: a thin semi-transparent strip along the top
      of the front face (``gloss_opacity``).
    - A floor shadow: a soft ellipse on z=0 (``shadow_opacity``).
    - A value label floating above the bar (``show_value_label``).

    Attributes
    ----------
    base_color : ManimColor
        Front-face base colour.  Overridden per-bar when a palette is used.
    side_shade_factor : float
        Multiplier applied to darken the right-side face (0–1).
    top_shade_factor : float
        Multiplier applied to lighten the top face (≥1 lightens).
    edge_color : ManimColor
        Colour of edge stroke lines.
    edge_stroke_width : float
        Stroke width of edge lines.
    edge_opacity : float
        Opacity of edge lines.
    gloss_opacity : float
        Opacity of the gloss highlight strip.  Set 0 to disable.
    gloss_height_fraction : float
        Height of the gloss strip as a fraction of bar height (0–1).
    shadow_opacity : float
        Opacity of the floor shadow ellipse.  Set 0 to disable.
    shadow_x_scale : float
        X-radius of shadow ellipse relative to bar width.
    shadow_y_scale : float
        Y-radius of shadow ellipse relative to bar depth.
    show_value_label : bool
        Whether to show a numeric label above each bar.
    value_label_font_size : int
        Font size for value labels.
    value_label_color : ManimColor
        Colour for value labels.
    value_label_decimals : int
        Decimal places in value labels.
    value_label_offset : float
        Distance above bar top to place the label.
    show_category_label : bool
        Whether to show category labels beneath each bar.
    category_label_font_size : int
        Font size for category labels.
    category_label_color : ManimColor
        Colour of category labels.
    category_label_offset : float
        Distance below bar base to place the label.
    grow_rate_func : Callable
        Manim rate function for the grow-from-floor animation.
    """

    base_color: ManimColor = ManimColor("#4A90D9")
    side_shade_factor: float = 0.62
    top_shade_factor: float = 1.35
    edge_color: ManimColor = ManimColor("#FFFFFF")
    edge_stroke_width: float = 0.6
    edge_opacity: float = 0.25
    gloss_opacity: float = 0.18
    gloss_height_fraction: float = 0.30
    shadow_opacity: float = 0.20
    shadow_x_scale: float = 1.10
    shadow_y_scale: float = 0.55
    show_value_label: bool = True
    value_label_font_size: int = 22
    value_label_color: ManimColor = WHITE
    value_label_decimals: int = 1
    value_label_offset: float = 0.18
    show_category_label: bool = True
    category_label_font_size: int = 20
    category_label_color: ManimColor = ManimColor("#AABBCC")
    category_label_offset: float = 0.25
    grow_rate_func: Callable = rate_functions.ease_out_cubic


# Default preset configs

MINIMAL_BAR = BarConfig(
    gloss_opacity=0.0,
    shadow_opacity=0.0,
    edge_opacity=0.15,
    show_value_label=False,
    show_category_label=False,
)

POLISHED_BAR = BarConfig(
    side_shade_factor=0.58,
    top_shade_factor=1.45,
    edge_stroke_width=0.8,
    edge_opacity=0.30,
    gloss_opacity=0.22,
    shadow_opacity=0.25,
    show_value_label=True,
    value_label_font_size=24,
)

NEON_BAR = BarConfig(
    side_shade_factor=0.40,
    top_shade_factor=1.60,
    edge_color=WHITE,
    edge_stroke_width=1.2,
    edge_opacity=0.55,
    gloss_opacity=0.35,
    shadow_opacity=0.30,
    show_value_label=True,
)


# ---------------------------------------------------------------------------
# _ValueLabel3D  (internal)
# ---------------------------------------------------------------------------

class _ValueLabel3D(VGroup):
    """Floating numeric label positioned above a bar.

    Renders as a ``Text`` mobject.  The scene should call
    ``add_fixed_orientation_mobjects(label.text)`` so it faces the camera.
    """

    def __init__(
        self,
        value: float,
        position: np.ndarray,
        decimals: int = 1,
        font_size: int = 22,
        color: ManimColor = WHITE,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        fmt = f"{value:.{decimals}f}"
        self.text = Text(fmt, font_size=font_size, color=color)
        self.text.move_to(position)
        self.add(self.text)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(self.text)


# ---------------------------------------------------------------------------
# Bar3D  (single bar prism)
# ---------------------------------------------------------------------------

class Bar3D(VGroup):
    """A single 3D bar prism with rich face shading.

    The bar is a right-angled prism with:
    - Three visible faces: front (−y), right (+x), top (+z).
    - Three hidden faces: back (+y), left (−x), bottom (−z)
      are omitted — they are never visible in standard 3/4 view.
    - Edge strokes along all visible silhouette edges.
    - A gloss highlight strip near the top of the front face.
    - A floor shadow ellipse beneath the bar.

    Parameters
    ----------
    width : float
        Bar dimension along the x axis.
    depth : float
        Bar dimension along the y axis (into the scene).
    height : float
        Bar dimension along the z axis (the data value).
    position : np.ndarray
        3D position of the bar's bottom-centre (x, y, z=0).
    color : ManimColor
        Base colour for the front face.  Side/top are auto-shaded.
    config : BarConfig
        Full visual config.
    scene : ThreeDScene or None
        If provided, value/category labels are registered as
        fixed-orientation objects.

    Attributes
    ----------
    face_front : Polygon
    face_right : Polygon
    face_top   : Polygon
    edges      : VGroup
    gloss      : Polygon
    shadow     : Ellipse
    value_label : _ValueLabel3D or None
    category_label : _ValueLabel3D or None
    """

    def __init__(
        self,
        width: float = 0.60,
        depth: float = 0.40,
        height: float = 1.0,
        position: np.ndarray = ORIGIN,
        color: Optional[ManimColor] = None,
        config: Optional[BarConfig] = None,
        value: Optional[float] = None,
        label: Optional[str] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cfg = config if config is not None else BarConfig()
        self.bar_width = width
        self.bar_depth = depth
        self.bar_height = height
        self.bar_position = np.array(position, dtype=float)
        self._value = value if value is not None else height
        self._label_text = label

        base_color = color if color is not None else self.cfg.base_color
        self._base_color = base_color

        # Derived face colours
        self._color_front = base_color
        self._color_right = _darken(base_color, self.cfg.side_shade_factor)
        self._color_top = _lighten(base_color, self.cfg.top_shade_factor)

        # Build all sub-objects
        self._build_faces()
        self._build_edges()
        self._build_gloss()
        self._build_shadow()
        if self.cfg.show_value_label:
            self._build_value_label(scene)
        if self.cfg.show_category_label and label is not None:
            self._build_category_label(scene)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _corners(self, h: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Compute the 8 corners of the prism at *height* h."""
        if h is None:
            h = self.bar_height
        x, y, z = self.bar_position
        hw = self.bar_width / 2
        hd = self.bar_depth / 2
        return {
            # bottom
            "BFL": np.array([x - hw, y - hd, z]),       # bottom-front-left
            "BFR": np.array([x + hw, y - hd, z]),       # bottom-front-right
            "BBR": np.array([x + hw, y + hd, z]),       # bottom-back-right
            "BBL": np.array([x - hw, y + hd, z]),       # bottom-back-left
            # top
            "TFL": np.array([x - hw, y - hd, z + h]),
            "TFR": np.array([x + hw, y - hd, z + h]),
            "TBR": np.array([x + hw, y + hd, z + h]),
            "TBL": np.array([x - hw, y + hd, z + h]),
        }

    def _build_faces(self) -> None:
        c = self._corners()

        # Front face  (y = position.y - depth/2)
        self.face_front = Polygon(
            c["BFL"], c["BFR"], c["TFR"], c["TFL"],
            fill_color=self._color_front,
            fill_opacity=1.0,
            stroke_width=0,
        )

        # Right face  (x = position.x + width/2)
        self.face_right = Polygon(
            c["BFR"], c["BBR"], c["TBR"], c["TFR"],
            fill_color=self._color_right,
            fill_opacity=1.0,
            stroke_width=0,
        )

        # Top face  (z = position.z + height)
        self.face_top = Polygon(
            c["TFL"], c["TFR"], c["TBR"], c["TBL"],
            fill_color=self._color_top,
            fill_opacity=1.0,
            stroke_width=0,
        )

        self.faces = VGroup(self.face_front, self.face_right, self.face_top)
        self.add(self.faces)

    def _build_edges(self) -> None:
        """Silhouette edges — only the visible outline in 3/4 view."""
        c = self._corners()
        ecol = _with_opacity(self.cfg.edge_color, self.cfg.edge_opacity)
        sw = self.cfg.edge_stroke_width

        edge_pairs = [
            # Front face outline
            (c["BFL"], c["BFR"]),
            (c["BFR"], c["TFR"]),
            (c["TFR"], c["TFL"]),
            (c["TFL"], c["BFL"]),
            # Right face extra edges
            (c["BBR"], c["TBR"]),
            (c["TBR"], c["TFR"]),  # shared with front top
            (c["BFR"], c["BBR"]),
            # Top face extra edges
            (c["TFL"], c["TBL"]),
            (c["TBL"], c["TBR"]),
            # Bottom back
            (c["BBR"], c["BBL"]),
            (c["BBL"], c["BFL"]),
        ]
        self.edges = VGroup(*[
            Line(a, b, color=ecol, stroke_width=sw)
            for a, b in edge_pairs
        ])
        self.add(self.edges)

    def _build_gloss(self) -> None:
        """Semi-transparent highlight strip near top of front face."""
        if self.cfg.gloss_opacity <= 0:
            self.gloss = VGroup()
            self.add(self.gloss)
            return

        c = self._corners()
        gloss_h = self.bar_height * self.cfg.gloss_height_fraction
        z_bot = self.bar_position[2] + self.bar_height - gloss_h
        z_top = self.bar_position[2] + self.bar_height
        x, y, _ = self.bar_position
        hw = self.bar_width / 2
        hd = self.bar_depth / 2

        gloss_pts = [
            np.array([x - hw, y - hd, z_bot]),
            np.array([x + hw, y - hd, z_bot]),
            np.array([x + hw, y - hd, z_top]),
            np.array([x - hw, y - hd, z_top]),
        ]
        self.gloss = Polygon(
            *gloss_pts,
            fill_color=_with_opacity(WHITE, self.cfg.gloss_opacity),
            fill_opacity=1.0,
            stroke_width=0,
        )
        self.add(self.gloss)

    def _build_shadow(self) -> None:
        """Soft floor shadow ellipse at z = bar_position.z."""
        if self.cfg.shadow_opacity <= 0:
            self.shadow = VGroup()
            self.add(self.shadow)
            return

        x, y, z = self.bar_position
        sx = self.bar_width * self.cfg.shadow_x_scale
        sy = self.bar_depth * self.cfg.shadow_y_scale
        self.shadow = Ellipse(
            width=sx,
            height=sy,
            fill_color=_with_opacity(BLACK, self.cfg.shadow_opacity),
            fill_opacity=1.0,
            stroke_width=0,
        )
        # Ellipse lives in its own plane; move to bar base
        self.shadow.move_to(np.array([x, y, z]))
        self.add(self.shadow)

    def _build_value_label(self, scene: Optional[ThreeDScene]) -> None:
        x, y, z = self.bar_position
        label_pos = np.array([x, y - self.bar_depth / 2,
                               z + self.bar_height + self.cfg.value_label_offset])
        self.value_label = _ValueLabel3D(
            self._value,
            label_pos,
            decimals=self.cfg.value_label_decimals,
            font_size=self.cfg.value_label_font_size,
            color=self.cfg.value_label_color,
            scene=scene,
        )
        self.add(self.value_label)

    def _build_category_label(self, scene: Optional[ThreeDScene]) -> None:
        x, y, z = self.bar_position
        label_pos = np.array([x, y - self.bar_depth / 2,
                               z - self.cfg.category_label_offset])
        lbl = Text(
            self._label_text,
            font_size=self.cfg.category_label_font_size,
            color=self.cfg.category_label_color,
        )
        lbl.move_to(label_pos)
        self.category_label = VGroup(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)
        self.add(self.category_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def animate_grow(
        self,
        run_time: float = 0.8,
    ) -> UpdateFromAlphaFunc:
        """Return an animation that grows the bar from height=0 to full height.

        The entire bar (faces, edges, gloss, shadow, labels) scales up
        from the floor using a custom updater so the geometry reshapes
        correctly (not just scales).
        """
        target_height = self.bar_height
        cfg = self.cfg
        base_color = self._base_color
        position = self.bar_position.copy()
        width = self.bar_width
        depth = self.bar_depth
        value = self._value
        label_text = self._label_text

        def updater(mob: Bar3D, alpha: float) -> None:
            h = cfg.grow_rate_func(alpha) * target_height
            if h < 1e-4:
                mob.set_opacity(0)
                return
            mob.set_opacity(1)
            # Rebuild only the height-dependent geometry
            # We manipulate the existing polygons in-place by
            # recomputing their points.
            x, y, z = position
            hw, hd = width / 2, depth / 2

            # Front face
            pts_front = [
                [x - hw, y - hd, z],
                [x + hw, y - hd, z],
                [x + hw, y - hd, z + h],
                [x - hw, y - hd, z + h],
            ]
            mob.face_front.set_points_as_corners(
                [np.array(p) for p in pts_front] + [np.array(pts_front[0])]
            )

            # Right face
            pts_right = [
                [x + hw, y - hd, z],
                [x + hw, y + hd, z],
                [x + hw, y + hd, z + h],
                [x + hw, y - hd, z + h],
            ]
            mob.face_right.set_points_as_corners(
                [np.array(p) for p in pts_right] + [np.array(pts_right[0])]
            )

            # Top face
            pts_top = [
                [x - hw, y - hd, z + h],
                [x + hw, y - hd, z + h],
                [x + hw, y + hd, z + h],
                [x - hw, y + hd, z + h],
            ]
            mob.face_top.set_points_as_corners(
                [np.array(p) for p in pts_top] + [np.array(pts_top[0])]
            )

            # Gloss strip
            if cfg.gloss_opacity > 0:
                gloss_h = h * cfg.gloss_height_fraction
                z_bot = z + h - gloss_h
                z_top_g = z + h
                pts_gloss = [
                    [x - hw, y - hd, z_bot],
                    [x + hw, y - hd, z_bot],
                    [x + hw, y - hd, z_top_g],
                    [x - hw, y - hd, z_top_g],
                ]
                mob.gloss.set_points_as_corners(
                    [np.array(p) for p in pts_gloss] + [np.array(pts_gloss[0])]
                )

            # Value label — float above current top
            if cfg.show_value_label and hasattr(mob, "value_label"):
                new_pos = np.array([x, y - hd, z + h + cfg.value_label_offset])
                mob.value_label.move_to(new_pos)
                label_alpha = min(alpha * 3, 1.0)  # fade in last third
                mob.value_label.set_opacity(label_alpha)

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)

    def set_height(self, new_height: float) -> "Bar3D":
        """Instantly resize the bar to *new_height*. Returns self."""
        self.bar_height = new_height
        self._value = new_height
        # Rebuild geometry in place
        self.remove(self.faces, self.edges, self.gloss)
        if hasattr(self, "shadow"):
            self.remove(self.shadow)
        if hasattr(self, "value_label"):
            self.remove(self.value_label)
        if hasattr(self, "category_label"):
            self.remove(self.category_label)
        self._color_front = self._base_color
        self._color_right = _darken(self._base_color, self.cfg.side_shade_factor)
        self._color_top = _lighten(self._base_color, self.cfg.top_shade_factor)
        self._build_faces()
        self._build_edges()
        self._build_gloss()
        self._build_shadow()
        return self

    def get_top_center(self) -> np.ndarray:
        """Return the 3D centre of the top face."""
        x, y, z = self.bar_position
        return np.array([x, y, z + self.bar_height])

    def set_bar_color(self, color: ManimColor) -> "Bar3D":
        """Change bar colour without rebuilding geometry."""
        self._base_color = color
        self.face_front.set_fill(_darken(color, 1.0))
        self.face_right.set_fill(_darken(color, self.cfg.side_shade_factor))
        self.face_top.set_fill(_lighten(color, self.cfg.top_shade_factor))
        return self

    def highlight(
        self,
        color: ManimColor = YELLOW,
        edge_width: float = 2.0,
        edge_opacity: float = 0.9,
    ) -> "Bar3D":
        """Emphasise this bar with thick bright edges. Returns self."""
        for edge in self.edges:
            edge.set_stroke(
                color=_with_opacity(color, edge_opacity),
                width=edge_width,
            )
        return self

    def unhighlight(self) -> "Bar3D":
        """Restore default edge styling."""
        ecol = _with_opacity(self.cfg.edge_color, self.cfg.edge_opacity)
        for edge in self.edges:
            edge.set_stroke(color=ecol, width=self.cfg.edge_stroke_width)
        return self


# ---------------------------------------------------------------------------
# BarChart3D  — single series
# ---------------------------------------------------------------------------

class BarChart3D(VGroup):
    """A single-series 3D bar chart.

    Bars are laid out at integer x-positions (0, 1, 2, …) along the
    x-axis.  Pass a ``GridSnapHelper`` via ``snap_helper`` to align
    them to an existing ``FullGrid3D``.

    Parameters
    ----------
    values : sequence of float
        Data values — one bar per value.
    labels : sequence of str or None
        Category labels shown beneath each bar.
    x_start : float
        X coordinate of the first bar's centre.
    bar_spacing : float
        Centre-to-centre distance between consecutive bars.
    bar_width : float
        Width of each bar (x direction).
    bar_depth : float
        Depth of each bar (y direction, into the scene).
    z_scale : float
        Multiplier applied to raw values to get bar heights.
        E.g. ``z_scale=0.5`` maps a value of 4 to a bar of height 2.
    y_position : float
        Y offset for all bars.  Useful when compositing with a grid.
    colors : list of ManimColor or None
        Per-bar colours.  If *None*, cycles through
        ``BarColorPalette.CATEGORICAL``.
    config : BarConfig
        Shared visual config for all bars.
    scene : ThreeDScene or None
        Pass the scene to register labels for fixed-orientation rendering.

    Attributes
    ----------
    bars : list of Bar3D
        Direct access to each bar object.
    bar_group : VGroup
        All bars as a group (for batch transforms).
    """

    def __init__(
        self,
        values: Sequence[float],
        labels: Optional[Sequence[str]] = None,
        x_start: float = 0.0,
        bar_spacing: float = 1.0,
        bar_width: float = 0.60,
        bar_depth: float = 0.40,
        z_scale: float = 1.0,
        y_position: float = 0.0,
        colors: Optional[Sequence[ManimColor]] = None,
        config: Optional[BarConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.values = list(values)
        self.labels = list(labels) if labels else [None] * len(values)
        self.x_start = x_start
        self.bar_spacing = bar_spacing
        self.bar_width = bar_width
        self.bar_depth = bar_depth
        self.z_scale = z_scale
        self.y_position = y_position
        self.cfg = config if config is not None else POLISHED_BAR
        self._scene = scene

        # Resolve colours
        palette = BarColorPalette.CATEGORICAL
        if colors is None:
            self._colors = [palette[i % len(palette)] for i in range(len(values))]
        else:
            self._colors = list(colors)

        self.bars: List[Bar3D] = []
        self.bar_group = VGroup()

        self._build_bars()
        self.add(self.bar_group)

    # ------------------------------------------------------------------

    def _build_bars(self) -> None:
        for i, (val, lbl, col) in enumerate(
            zip(self.values, self.labels, self._colors)
        ):
            x = self.x_start + i * self.bar_spacing
            pos = np.array([x, self.y_position, 0.0])
            bar = Bar3D(
                width=self.bar_width,
                depth=self.bar_depth,
                height=max(val * self.z_scale, 1e-3),
                position=pos,
                color=col,
                config=self.cfg,
                value=val,
                label=lbl,
                scene=self._scene,
            )
            self.bars.append(bar)
            self.bar_group.add(bar)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_grow(
        self,
        lag: float = 0.06,
        run_time_per_bar: float = 0.8,
        sequential: bool = True,
    ) -> AnimationGroup:
        """Grow all bars from the floor.

        Parameters
        ----------
        lag : float
            Seconds between bar start times.
        run_time_per_bar : float
            Duration of each bar's grow animation.
        sequential : bool
            If True (default), bars grow left-to-right.
            If False, all bars start simultaneously.
        """
        anims = [bar.animate_grow(run_time=run_time_per_bar) for bar in self.bars]
        if sequential:
            return LaggedStart(*anims, lag_ratio=lag)
        return AnimationGroup(*anims)

    def animate_update(
        self,
        new_values: Sequence[float],
        run_time: float = 1.0,
    ) -> AnimationGroup:
        """Animate bars reshaping to *new_values*.

        Uses ``UpdateFromAlphaFunc`` per bar so heights interpolate
        smoothly without rebuilding geometry.
        """
        if len(new_values) != len(self.bars):
            raise ValueError(
                f"new_values length {len(new_values)} != "
                f"chart bar count {len(self.bars)}"
            )

        anims = []
        for bar, new_val in zip(self.bars, new_values):
            old_h = bar.bar_height
            new_h = max(new_val * self.z_scale, 1e-3)
            cfg = self.cfg
            x, y, z = bar.bar_position
            hw, hd = bar.bar_width / 2, bar.bar_depth / 2

            def make_updater(b, oh, nh, bx, by, bz, bhw, bhd, bcfg):
                def updater(mob, alpha):
                    h = oh + (nh - oh) * rate_functions.ease_in_out_cubic(alpha)
                    # Front face
                    mob.face_front.set_points_as_corners([
                        np.array([bx - bhw, by - bhd, bz]),
                        np.array([bx + bhw, by - bhd, bz]),
                        np.array([bx + bhw, by - bhd, bz + h]),
                        np.array([bx - bhw, by - bhd, bz + h]),
                        np.array([bx - bhw, by - bhd, bz]),
                    ])
                    # Right face
                    mob.face_right.set_points_as_corners([
                        np.array([bx + bhw, by - bhd, bz]),
                        np.array([bx + bhw, by + bhd, bz]),
                        np.array([bx + bhw, by + bhd, bz + h]),
                        np.array([bx + bhw, by - bhd, bz + h]),
                        np.array([bx + bhw, by - bhd, bz]),
                    ])
                    # Top face
                    mob.face_top.set_points_as_corners([
                        np.array([bx - bhw, by - bhd, bz + h]),
                        np.array([bx + bhw, by - bhd, bz + h]),
                        np.array([bx + bhw, by + bhd, bz + h]),
                        np.array([bx - bhw, by + bhd, bz + h]),
                        np.array([bx - bhw, by - bhd, bz + h]),
                    ])
                    # Gloss
                    if bcfg.gloss_opacity > 0:
                        gh = h * bcfg.gloss_height_fraction
                        zb = bz + h - gh
                        mob.gloss.set_points_as_corners([
                            np.array([bx - bhw, by - bhd, zb]),
                            np.array([bx + bhw, by - bhd, zb]),
                            np.array([bx + bhw, by - bhd, bz + h]),
                            np.array([bx - bhw, by - bhd, bz + h]),
                            np.array([bx - bhw, by - bhd, zb]),
                        ])
                    # Value label
                    if bcfg.show_value_label and hasattr(mob, "value_label"):
                        mob.value_label.move_to(
                            np.array([bx, by - bhd, bz + h + bcfg.value_label_offset])
                        )
                return updater

            anims.append(UpdateFromAlphaFunc(
                bar,
                make_updater(bar, old_h, new_h, x, y, z, hw, hd, cfg),
                run_time=run_time,
            ))
            bar.bar_height = new_h
            bar._value = new_val

        return AnimationGroup(*anims)

    # ------------------------------------------------------------------
    # Selection / highlighting
    # ------------------------------------------------------------------

    def highlight_bar(
        self,
        index: int,
        color: ManimColor = YELLOW,
        dim_others: bool = True,
        dim_opacity: float = 0.30,
    ) -> "BarChart3D":
        """Highlight bar at *index*, optionally dimming all others.

        Returns self so this can be chained.
        """
        for i, bar in enumerate(self.bars):
            if i == index:
                bar.highlight(color=color)
            elif dim_others:
                bar.set_opacity(dim_opacity)
        return self

    def unhighlight_all(self) -> "BarChart3D":
        """Remove all highlights and restore full opacity."""
        for bar in self.bars:
            bar.unhighlight()
            bar.set_opacity(1.0)
        return self

    def highlight_max(self, color: ManimColor = ManimColor("#2DAA6E")) -> "BarChart3D":
        """Highlight the tallest bar in *color*."""
        idx = int(np.argmax(self.values))
        return self.highlight_bar(idx, color=color)

    def highlight_min(self, color: ManimColor = ManimColor("#E8593C")) -> "BarChart3D":
        """Highlight the shortest bar in *color*."""
        idx = int(np.argmin(self.values))
        return self.highlight_bar(idx, color=color)

    # ------------------------------------------------------------------
    # Sort / reorder
    # ------------------------------------------------------------------

    def sorted_values(self, ascending: bool = True) -> List[float]:
        """Return values sorted for use with ``animate_update``."""
        s = sorted(self.values)
        return s if ascending else s[::-1]

    # ------------------------------------------------------------------
    # Stats overlays
    # ------------------------------------------------------------------

    def add_mean_line(
        self,
        color: ManimColor = ManimColor("#E0AA40"),
        stroke_width: float = 2.0,
        opacity: float = 0.85,
        label: bool = True,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Add a horizontal plane / line at the mean value.

        Returns the ``VGroup`` containing the mean line (+ optional
        label) so the caller can animate it.
        """
        mean_val = float(np.mean(self.values)) * self.z_scale
        x0 = self.x_start - self.bar_width
        x1 = self.x_start + (len(self.bars) - 1) * self.bar_spacing + self.bar_width
        y = self.y_position
        z = mean_val

        line = Line(
            np.array([x0, y, z]),
            np.array([x1, y, z]),
            color=_with_opacity(color, opacity),
            stroke_width=stroke_width,
        )
        grp = VGroup(line)

        if label:
            mean_text = Text(
                f"μ = {np.mean(self.values):.2f}",
                font_size=20,
                color=color,
            )
            mean_text.move_to(np.array([x1 + 0.3, y, z]))
            grp.add(mean_text)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(mean_text)

        self.add(grp)
        return grp

    def add_value_labels(self, scene: Optional[ThreeDScene] = None) -> "BarChart3D":
        """Force-add value labels to all bars (even if config said no)."""
        for bar in self.bars:
            if not hasattr(bar, "value_label"):
                bar._build_value_label(scene)
        return self

    # ------------------------------------------------------------------
    # Color utilities
    # ------------------------------------------------------------------

    def apply_value_coloring(
        self,
        lo_color: ManimColor = ManimColor("#A32D2D"),
        hi_color: ManimColor = ManimColor("#0C4478"),
    ) -> "BarChart3D":
        """Recolour bars based on their relative value."""
        colors = BarColorPalette.value_mapped(
            self.values, lo_color=lo_color, hi_color=hi_color
        )
        for bar, col in zip(self.bars, colors):
            bar.set_bar_color(col)
        return self

    def apply_sequential_coloring(
        self,
        palette: Optional[List[ManimColor]] = None,
    ) -> "BarChart3D":
        """Apply a sequential palette (light → dark) across bars."""
        if palette is None:
            palette = BarColorPalette.sequential_ramp(
                ManimColor("#93BDE0"), ManimColor("#0C3D7A"), len(self.bars)
            )
        for bar, col in zip(self.bars, palette):
            bar.set_bar_color(col)
        return self


# ---------------------------------------------------------------------------
# GroupedBarChart3D  — multiple series, side-by-side
# ---------------------------------------------------------------------------

class GroupedBarChart3D(VGroup):
    """Side-by-side grouped bar chart with series along the depth axis.

    For k series × N categories, bars in the same category are arranged
    next to each other along the y-axis (depth), one group per x position.

    Parameters
    ----------
    values : 2-D sequence, shape (k, N)
        ``values[s][c]`` is the height for series *s*, category *c*.
    category_labels : sequence of str or None
        N category labels along the x-axis.
    series_labels : sequence of str or None
        k series names — used in the legend.
    series_colors : sequence of ManimColor or None
        One colour per series.  Defaults to ``BarColorPalette.CATEGORICAL``.
    x_start : float
        X position of the first group.
    group_spacing : float
        Centre-to-centre distance between category groups.
    bar_width : float
        Width of a single bar along x.
    bar_depth : float
        Depth of a single bar along y.
    series_gap : float
        Gap between bars within the same group.
    z_scale : float
        Value → height multiplier.
    config : BarConfig
        Shared visual config.
    scene : ThreeDScene or None

    Attributes
    ----------
    bars : list of list of Bar3D
        ``bars[s][c]`` gives the Bar3D for series *s*, category *c*.
    series_groups : list of VGroup
        One VGroup per series.
    """

    def __init__(
        self,
        values: Sequence[Sequence[float]],
        category_labels: Optional[Sequence[str]] = None,
        series_labels: Optional[Sequence[str]] = None,
        series_colors: Optional[Sequence[ManimColor]] = None,
        x_start: float = 0.0,
        group_spacing: float = 1.2,
        bar_width: float = 0.30,
        bar_depth: float = 0.35,
        series_gap: float = 0.05,
        z_scale: float = 1.0,
        config: Optional[BarConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._values = [list(row) for row in values]
        k = len(self._values)       # number of series
        n = len(self._values[0])    # number of categories

        self._cat_labels = list(category_labels) if category_labels else [None] * n
        self._ser_labels = list(series_labels) if series_labels else [f"S{i}" for i in range(k)]

        palette = BarColorPalette.CATEGORICAL
        self._ser_colors = (
            list(series_colors) if series_colors
            else [palette[s % len(palette)] for s in range(k)]
        )

        self.cfg = config if config is not None else POLISHED_BAR
        cfg_no_cat = BarConfig(**self.cfg.__dict__)
        cfg_no_cat.show_category_label = False  # we'll add group labels manually

        self.bars: List[List[Bar3D]] = [[] for _ in range(k)]
        self.series_groups: List[VGroup] = [VGroup() for _ in range(k)]

        # Total depth per group
        total_group_depth = k * bar_depth + (k - 1) * series_gap
        y_offsets = [
            -total_group_depth / 2 + s * (bar_depth + series_gap) + bar_depth / 2
            for s in range(k)
        ]

        for s in range(k):
            for c in range(n):
                x = x_start + c * group_spacing
                y = y_offsets[s]
                bar = Bar3D(
                    width=bar_width,
                    depth=bar_depth,
                    height=max(self._values[s][c] * z_scale, 1e-3),
                    position=np.array([x, y, 0.0]),
                    color=self._ser_colors[s],
                    config=cfg_no_cat,
                    value=self._values[s][c],
                    label=None,
                    scene=scene,
                )
                self.bars[s].append(bar)
                self.series_groups[s].add(bar)

        for sg in self.series_groups:
            self.add(sg)

        # Category labels below groups
        if any(lbl is not None for lbl in self._cat_labels):
            self._add_category_labels(
                x_start, group_spacing, n, scene
            )

    def _add_category_labels(
        self,
        x_start: float,
        spacing: float,
        n: int,
        scene: Optional[ThreeDScene],
    ) -> None:
        for c, lbl in enumerate(self._cat_labels):
            if lbl is None:
                continue
            x = x_start + c * spacing
            t = Text(
                lbl,
                font_size=self.cfg.category_label_font_size,
                color=self.cfg.category_label_color,
            )
            t.move_to(np.array([x, 0, -self.cfg.category_label_offset]))
            if scene is not None:
                scene.add_fixed_orientation_mobjects(t)
            self.add(t)

    # ------------------------------------------------------------------

    def animate_grow(
        self,
        lag: float = 0.05,
        run_time_per_bar: float = 0.75,
        by_series: bool = False,
    ) -> AnimationGroup:
        """Grow all bars.

        Parameters
        ----------
        by_series : bool
            If True, grow each series in turn (series 0, then 1, …).
            If False (default), grow column-by-column (all series for
            category 0, then category 1, …).
        """
        k = len(self.bars)
        n = len(self.bars[0])

        if by_series:
            anims = [
                LaggedStart(
                    *[self.bars[s][c].animate_grow(run_time=run_time_per_bar)
                      for c in range(n)],
                    lag_ratio=lag,
                )
                for s in range(k)
            ]
            return LaggedStart(*anims, lag_ratio=lag * 3)
        else:
            col_anims = []
            for c in range(n):
                col_anims.append(AnimationGroup(
                    *[self.bars[s][c].animate_grow(run_time=run_time_per_bar)
                      for s in range(k)]
                ))
            return LaggedStart(*col_anims, lag_ratio=lag)

    def highlight_series(
        self,
        series_index: int,
        color: ManimColor = YELLOW,
        dim_others: bool = True,
        dim_opacity: float = 0.25,
    ) -> "GroupedBarChart3D":
        for s, bars in enumerate(self.bars):
            for bar in bars:
                if s == series_index:
                    bar.highlight(color=color)
                elif dim_others:
                    bar.set_opacity(dim_opacity)
        return self


# ---------------------------------------------------------------------------
# StackedBarChart3D
# ---------------------------------------------------------------------------

class StackedBarChart3D(VGroup):
    """Stacked bar chart: k series stacked vertically per category.

    Each segment is a ``Bar3D`` placed on top of the cumulative height
    of segments below it.  The top face of each segment is visible, and
    edge strokes separate the stacking layers.

    Parameters
    ----------
    values : 2-D sequence, shape (k, N)
        ``values[s][c]`` is the height contribution of series *s*
        for category *c*.  All values must be non-negative.
    category_labels : sequence of str or None
    series_labels : sequence of str or None
    series_colors : sequence of ManimColor or None
        One colour per stacking layer (bottom=0, top=k-1).
    x_start : float
    bar_spacing : float
    bar_width : float
    bar_depth : float
    z_scale : float
    config : BarConfig
    scene : ThreeDScene or None

    Attributes
    ----------
    bars : list of list of Bar3D
        ``bars[s][c]`` — segment s in column c.
    totals : list of float
        Total stacked height per category.
    """

    def __init__(
        self,
        values: Sequence[Sequence[float]],
        category_labels: Optional[Sequence[str]] = None,
        series_labels: Optional[Sequence[str]] = None,
        series_colors: Optional[Sequence[ManimColor]] = None,
        x_start: float = 0.0,
        bar_spacing: float = 1.0,
        bar_width: float = 0.55,
        bar_depth: float = 0.38,
        z_scale: float = 1.0,
        y_position: float = 0.0,
        config: Optional[BarConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._values = [list(row) for row in values]
        k = len(self._values)
        n = len(self._values[0])

        self._cat_labels = list(category_labels) if category_labels else [None] * n
        self._ser_labels = list(series_labels) if series_labels else [f"Layer {i}" for i in range(k)]

        palette = BarColorPalette.SEQUENTIAL_BLUE
        self._ser_colors = (
            list(series_colors) if series_colors
            else BarColorPalette.sequential_ramp(
                ManimColor("#93BDE0"), ManimColor("#0C3D7A"), k
            )
        )

        cfg = config if config is not None else BarConfig(
            gloss_opacity=0.10,
            shadow_opacity=0.0,    # only one shadow per column
            show_value_label=False,
            show_category_label=False,
            edge_opacity=0.35,
        )

        self.bars: List[List[Bar3D]] = [[] for _ in range(k)]
        self.totals: List[float] = []

        for c in range(n):
            z_cursor = 0.0
            for s in range(k):
                seg_h = max(self._values[s][c] * z_scale, 1e-3)
                x = x_start + c * bar_spacing
                pos = np.array([x, y_position, z_cursor])
                bar = Bar3D(
                    width=bar_width,
                    depth=bar_depth,
                    height=seg_h,
                    position=pos,
                    color=self._ser_colors[s],
                    config=cfg,
                    value=self._values[s][c],
                    label=None,
                    scene=scene,
                )
                self.bars[s].append(bar)
                self.add(bar)
                z_cursor += seg_h

            self.totals.append(z_cursor)

        # Category labels
        if any(lbl is not None for lbl in self._cat_labels):
            for c, lbl in enumerate(self._cat_labels):
                if lbl is None:
                    continue
                x = x_start + c * bar_spacing
                t = Text(
                    lbl,
                    font_size=cfg.category_label_font_size,
                    color=cfg.category_label_color,
                )
                t.move_to(np.array([x, y_position, -cfg.category_label_offset]))
                if scene is not None:
                    scene.add_fixed_orientation_mobjects(t)
                self.add(t)

        # Total value labels above each stack
        if config is None or config.show_value_label:
            for c, total in enumerate(self.totals):
                x = x_start + c * bar_spacing
                pos = np.array([x, y_position - bar_depth / 2, total + 0.20])
                lbl = _ValueLabel3D(
                    total / z_scale if z_scale != 1.0 else total,
                    pos,
                    decimals=1,
                    font_size=20,
                    color=WHITE,
                    scene=scene,
                )
                self.add(lbl)

    # ------------------------------------------------------------------

    def animate_grow(
        self,
        lag: float = 0.05,
        run_time_per_segment: float = 0.55,
        column_lag: float = 0.08,
    ) -> AnimationGroup:
        """Grow each column layer by layer, columns staggered."""
        k = len(self.bars)
        n = len(self.bars[0])

        col_anims = []
        for c in range(n):
            layer_anims = [
                self.bars[s][c].animate_grow(run_time=run_time_per_segment)
                for s in range(k)
            ]
            col_anims.append(
                LaggedStart(*layer_anims, lag_ratio=lag)
            )

        return LaggedStart(*col_anims, lag_ratio=column_lag)

    def highlight_layer(
        self,
        layer_index: int,
        color: ManimColor = YELLOW,
        dim_others: bool = True,
        dim_opacity: float = 0.25,
    ) -> "StackedBarChart3D":
        k = len(self.bars)
        for s in range(k):
            for bar in self.bars[s]:
                if s == layer_index:
                    bar.highlight(color=color)
                elif dim_others:
                    bar.set_opacity(dim_opacity)
        return self


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def bar_chart_from_dict(
    data: Dict[str, float],
    config: Optional[BarConfig] = None,
    palette: str = "categorical",
    **kwargs,
) -> BarChart3D:
    """Build a ``BarChart3D`` directly from a ``{label: value}`` dict.

    Parameters
    ----------
    data : dict
        Mapping of category name → numeric value.
    palette : str
        ``"categorical"``, ``"sequential_blue"``, ``"sequential_teal"``,
        ``"value_mapped"``, or ``"diverging"``.

    Example
    -------
    ::

        chart = bar_chart_from_dict(
            {"A": 3.2, "B": 5.1, "C": 2.8, "D": 4.4},
            palette="value_mapped",
        )
    """
    labels = list(data.keys())
    values = list(data.values())

    pal_map = {
        "categorical": BarColorPalette.CATEGORICAL,
        "sequential_blue": BarColorPalette.SEQUENTIAL_BLUE,
        "sequential_teal": BarColorPalette.SEQUENTIAL_TEAL,
        "diverging": BarColorPalette.DIVERGING,
    }

    if palette == "value_mapped":
        colors = BarColorPalette.value_mapped(values)
    else:
        raw = pal_map.get(palette, BarColorPalette.CATEGORICAL)
        colors = [raw[i % len(raw)] for i in range(len(values))]

    return BarChart3D(
        values=values,
        labels=labels,
        colors=colors,
        config=config,
        **kwargs,
    )


def grouped_from_dataframe(
    df,  # pandas DataFrame  — typed loosely to avoid hard dependency
    series_colors: Optional[Sequence[ManimColor]] = None,
    config: Optional[BarConfig] = None,
    **kwargs,
) -> GroupedBarChart3D:
    """Build a ``GroupedBarChart3D`` from a pandas DataFrame.

    The DataFrame should have categories as its index and series as
    columns.  e.g.:

    ::

             Mon   Tue   Wed
        S1   2.1   3.5   4.2
        S2   1.8   2.9   3.6

    Parameters
    ----------
    df : pd.DataFrame
        Rows = categories, columns = series.

    Returns
    -------
    GroupedBarChart3D
    """
    values = df.values.T.tolist()  # shape (k_series, n_categories)
    category_labels = list(df.index.astype(str))
    series_labels = list(df.columns.astype(str))
    return GroupedBarChart3D(
        values=values,
        category_labels=category_labels,
        series_labels=series_labels,
        series_colors=series_colors,
        config=config,
        **kwargs,
    )