"""
manim_stats/charts/box_plot3d.py
=================================
Production-quality 3D box plots for statistical visualizations.

Design philosophy
-----------------
A box plot in 3D is the most geometrically complex of the standard
chart types.  Every visual component is built as proper 3D geometry
rather than 2D shapes placed in 3D space:

    BoxBody3D           – Q1–Q3 prism: 3 shaded faces, optional notch,
                          gloss strip, floor shadow.
    Whisker3D           – three styles: plain line, capped (with end bar),
                          or tapered (narrowing prism toward the fence).
    MedianLine3D        – a flat prism at the median with glow halo.
    MeanMarker3D        – diamond or cross marker for the mean.
    OutlierMarkers3D    – per-point 3D markers outside whisker fences,
                          with optional y-jitter for overplotting.
    SignificanceBracket3D – bracket + p-value annotation between two boxes.

Five-number summary is encapsulated in ``FiveNumberSummary`` which can
be constructed from raw data (``FiveNumberSummary.from_data``) or from
pre-computed values.  Outlier detection uses Tukey's 1.5×IQR rule.

Animation layers build in teaching order:
    shadow → box body → whiskers → median line → mean → outliers

Classes
-------
BoxConfig
FiveNumberSummary
BoxPlot3D
BoxPlotGroup3D
NotchedBoxPlot3D

Helpers / internals
-------------------
BoxBody3D
Whisker3D
MedianLine3D
MeanMarker3D
OutlierMarkers3D
SignificanceBracket3D

Usage examples
--------------
    from manim import *
    from manim_stats.axes.grid3d import FullGrid3D
    from manim_stats.charts.box_plot3d import BoxPlotGroup3D

    class DemoBox(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65*DEGREES, theta=-50*DEGREES)
            grid = FullGrid3D(x_range=[-0.5, 4.5, 1], y_range=[-1, 1, 1],
                              z_range=[0, 8, 1])
            import numpy as np
            np.random.seed(42)
            groups = {
                "Control": np.random.normal(4.0, 1.2, 60),
                "Treatment A": np.random.normal(5.5, 0.9, 60),
                "Treatment B": np.random.normal(3.8, 1.5, 60),
                "Placebo": np.random.normal(4.2, 1.1, 60),
            }
            chart = BoxPlotGroup3D.from_data(groups, show_mean=True,
                                             whisker_style="tapered")
            self.play(grid.animate_build())
            self.play(chart.animate_build(stagger=0.12))
            self.wait()
            chart.add_significance_bracket(0, 1, p_value=0.003)
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
    Dot, Dot3D, Sphere, Arrow, Text, MathTex, Ellipse,
    # Scene
    ThreeDScene,
    # Animations
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, GrowFromEdge, Transform,
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
    rate_functions, smooth,
)

# ---------------------------------------------------------------------------
# Shared colour helpers — self-contained copy for module independence
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
# BoxColorPalette
# ---------------------------------------------------------------------------

class BoxColorPalette:
    """Pre-built colour palettes for box plots.

    Attributes
    ----------
    CATEGORICAL : list[ManimColor]
        Eight high-contrast hues, consistent with bar/line palettes.
    SEQUENTIAL_COOL : list[ManimColor]
        Six shades from pale-cyan → deep-teal, for ordered groups.
    DIVERGING : list[ManimColor]
        Seven steps red → neutral → blue, centred at index 3.
    MONOCHROME : list[ManimColor]
        Single colour repeated (caller sets one theme for all boxes).
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

    SEQUENTIAL_COOL: List[ManimColor] = [
        ManimColor("#B8EAE0"),
        ManimColor("#78CFC0"),
        ManimColor("#38B5A0"),
        ManimColor("#1A9080"),
        ManimColor("#0E6860"),
        ManimColor("#084540"),
    ]

    DIVERGING: List[ManimColor] = [
        ManimColor("#A32D2D"),
        ManimColor("#D05555"),
        ManimColor("#E89090"),
        ManimColor("#888880"),
        ManimColor("#7AABD0"),
        ManimColor("#3A7AB8"),
        ManimColor("#0C4478"),
    ]

    MONOCHROME: List[ManimColor] = [ManimColor("#4A90D9")] * 8

    @staticmethod
    def ramp(lo: ManimColor, hi: ManimColor, n: int) -> List[ManimColor]:
        return [_lerp_color(lo, hi, i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# FiveNumberSummary
# ---------------------------------------------------------------------------

@dataclass
class FiveNumberSummary:
    """The five-number summary of a distribution plus derived statistics.

    Parameters / Attributes
    -----------------------
    minimum : float
        Minimum whisker end (after outlier removal).
    q1 : float
        First quartile (25th percentile).
    median : float
        Second quartile (50th percentile).
    q3 : float
        Third quartile (75th percentile).
    maximum : float
        Maximum whisker end (after outlier removal).
    mean : float or None
        Arithmetic mean (optional, for mean marker).
    outliers : np.ndarray
        Raw data values outside Tukey's 1.5×IQR fences.
    n : int
        Sample size.
    label : str
        Group label.
    """

    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    mean: Optional[float] = None
    outliers: np.ndarray = field(default_factory=lambda: np.array([]))
    n: int = 0
    label: str = ""

    # ------------------------------------------------------------------
    # Derived stats (properties, not stored)
    # ------------------------------------------------------------------

    @property
    def iqr(self) -> float:
        """Interquartile range Q3 − Q1."""
        return self.q3 - self.q1

    @property
    def lower_fence(self) -> float:
        """Tukey lower fence Q1 − 1.5×IQR."""
        return self.q1 - 1.5 * self.iqr

    @property
    def upper_fence(self) -> float:
        """Tukey upper fence Q3 + 1.5×IQR."""
        return self.q3 + 1.5 * self.iqr

    @property
    def box_height(self) -> float:
        return self.q3 - self.q1

    @property
    def lower_whisker_length(self) -> float:
        return self.q1 - self.minimum

    @property
    def upper_whisker_length(self) -> float:
        return self.maximum - self.q3

    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        label: str = "",
        iqr_factor: float = 1.5,
    ) -> "FiveNumberSummary":
        """Compute the five-number summary from a raw data array.

        Outliers are identified using Tukey's rule:
        values outside Q1 − *iqr_factor*×IQR or Q3 + *iqr_factor*×IQR.

        Parameters
        ----------
        data : array-like
            Raw data values.
        label : str
            Group name for display.
        iqr_factor : float
            Multiplier for IQR fence (default 1.5 = Tukey's rule;
            use 3.0 for "far outside" / extreme outliers).
        """
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]

        q1 = float(np.percentile(arr, 25))
        median = float(np.percentile(arr, 50))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1

        lower_fence = q1 - iqr_factor * iqr
        upper_fence = q3 + iqr_factor * iqr

        outlier_mask = (arr < lower_fence) | (arr > upper_fence)
        outliers = arr[outlier_mask]
        inliers = arr[~outlier_mask]

        minimum = float(inliers.min()) if len(inliers) > 0 else q1
        maximum = float(inliers.max()) if len(inliers) > 0 else q3
        mean = float(arr.mean())

        return cls(
            minimum=minimum,
            q1=q1,
            median=median,
            q3=q3,
            maximum=maximum,
            mean=mean,
            outliers=outliers,
            n=len(arr),
            label=label,
        )

    @classmethod
    def from_precomputed(
        cls,
        q1: float,
        median: float,
        q3: float,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        label: str = "",
    ) -> "FiveNumberSummary":
        """Build from pre-computed percentiles.  Whisker ends default to
        Tukey fences if *minimum* / *maximum* are not supplied."""
        iqr = q3 - q1
        lo = minimum if minimum is not None else q1 - 1.5 * iqr
        hi = maximum if maximum is not None else q3 + 1.5 * iqr
        return cls(minimum=lo, q1=q1, median=median, q3=q3, maximum=hi,
                   mean=mean, label=label)


# ---------------------------------------------------------------------------
# BoxConfig
# ---------------------------------------------------------------------------

@dataclass
class BoxConfig:
    """Complete visual specification for ``BoxPlot3D``.

    Box body shading
    ~~~~~~~~~~~~~~~~
    The box body is a prism — three visible faces with independent
    colours derived from the base colour:
    - **Front** face: base colour.
    - **Right** face: darkened by ``side_shade_factor``.
    - **Top** face: lightened by ``top_shade_factor``.
    A gloss strip runs along the top of the front face when
    ``gloss_opacity > 0``.

    Whisker style
    ~~~~~~~~~~~~~
    ``whisker_style`` controls the whisker geometry:
    - ``"line"``    – single thin line from box edge to fence.
    - ``"capped"``  – line with a horizontal crossbar at the fence end.
    - ``"tapered"`` – a flat prism that is full-width at the box edge
                      and narrows to ``whisker_taper_ratio`` × width at
                      the fence end.  Most visually rich.

    Notch
    ~~~~~
    ``notch`` draws a V-shaped cutout on both sides of the box body
    representing the 95% CI around the median.  Notch half-width is
    computed as ``1.58 × IQR / √n``.

    Outlier markers
    ~~~~~~~~~~~~~~~
    ``outlier_style`` selects the marker shape.
    ``outlier_jitter`` adds random y-offset to prevent stacking.

    Attributes
    ----------
    box_color : ManimColor
    side_shade_factor : float
    top_shade_factor : float
    body_opacity : float
    edge_color : ManimColor
    edge_stroke_width : float
    edge_opacity : float
    gloss_opacity : float
    gloss_height_fraction : float
    shadow_opacity : float
    median_color : ManimColor
    median_stroke_width : float
    median_opacity : float
    median_glow_width : float
    median_glow_opacity : float
    mean_marker_style : str
        ``"diamond"``, ``"cross"``, ``"sphere"``, or ``"none"``.
    mean_color : ManimColor
    mean_marker_size : float
    whisker_style : str
        ``"line"``, ``"capped"``, or ``"tapered"``.
    whisker_stroke_width : float
    whisker_opacity : float
    cap_width_fraction : float
        Cap width relative to box width (for ``"capped"`` style).
    whisker_taper_ratio : float
        Ratio of whisker-end width to box width (for ``"tapered"``).
    whisker_taper_shade_factor : float
        Darkening applied to the taper prism side face.
    notch : bool
    notch_color : ManimColor or None
        Fill colour of the notch region (None = background colour).
    outlier_style : str
        ``"sphere"``, ``"diamond"``, ``"cross"``, ``"ring"``.
    outlier_color : ManimColor
    outlier_size : float
    outlier_opacity : float
    outlier_jitter : float
        Maximum y-offset for jittering outlier markers (0 = no jitter).
    show_category_label : bool
    category_label_font_size : int
    category_label_color : ManimColor
    category_label_offset : float
    show_stats_label : bool
        Whether to float a small stats panel (n, median, IQR) near the box.
    stats_label_font_size : int
    """

    box_color: ManimColor = ManimColor("#4A90D9")
    side_shade_factor: float = 0.62
    top_shade_factor: float = 1.30
    body_opacity: float = 0.88
    edge_color: ManimColor = WHITE
    edge_stroke_width: float = 0.7
    edge_opacity: float = 0.30
    gloss_opacity: float = 0.16
    gloss_height_fraction: float = 0.28
    shadow_opacity: float = 0.18
    median_color: ManimColor = WHITE
    median_stroke_width: float = 2.5
    median_opacity: float = 0.95
    median_glow_width: float = 7.0
    median_glow_opacity: float = 0.18
    mean_marker_style: str = "diamond"   # diamond | cross | sphere | none
    mean_color: ManimColor = ManimColor("#E0AA40")
    mean_marker_size: float = 0.10
    whisker_style: str = "tapered"       # line | capped | tapered
    whisker_stroke_width: float = 1.4
    whisker_opacity: float = 0.75
    cap_width_fraction: float = 0.45
    whisker_taper_ratio: float = 0.18
    whisker_taper_shade_factor: float = 0.55
    notch: bool = False
    notch_color: Optional[ManimColor] = None
    outlier_style: str = "ring"          # sphere | diamond | cross | ring
    outlier_color: ManimColor = ManimColor("#E8593C")
    outlier_size: float = 0.08
    outlier_opacity: float = 0.75
    outlier_jitter: float = 0.08
    show_category_label: bool = True
    category_label_font_size: int = 20
    category_label_color: ManimColor = ManimColor("#AABBCC")
    category_label_offset: float = 0.30
    show_stats_label: bool = False
    stats_label_font_size: int = 16


# ── Preset configs ─────────────────────────────────────────────────────────

MINIMAL_BOX = BoxConfig(
    gloss_opacity=0.0,
    shadow_opacity=0.0,
    whisker_style="line",
    notch=False,
    mean_marker_style="none",
    show_stats_label=False,
    edge_opacity=0.20,
)

POLISHED_BOX = BoxConfig(
    side_shade_factor=0.58,
    top_shade_factor=1.40,
    gloss_opacity=0.20,
    shadow_opacity=0.22,
    whisker_style="tapered",
    median_glow_opacity=0.20,
    mean_marker_style="diamond",
    edge_stroke_width=0.9,
    edge_opacity=0.35,
)

NOTCHED_BOX = BoxConfig(
    notch=True,
    whisker_style="capped",
    cap_width_fraction=0.40,
    gloss_opacity=0.14,
    shadow_opacity=0.16,
    mean_marker_style="cross",
)


# ---------------------------------------------------------------------------
# BoxBody3D  — internal
# ---------------------------------------------------------------------------

class BoxBody3D(VGroup):
    """The rectangular prism body of a box plot (Q1 to Q3).

    Geometry
    --------
    Three visible faces:
    - Front  (−y): base colour.
    - Right  (+x): darkened.
    - Top    (+z): lightened.

    Optional:
    - Notch: a V-cut on front and back faces at the median height,
      representing the 95% CI.  Rendered as two triangles clipped
      from the front face.
    - Gloss: semi-transparent strip near the top of the front face.
    - Floor shadow: soft ellipse below the box.

    Parameters
    ----------
    x : float
        X centre of the box.
    width : float
        Box width along x.
    depth : float
        Box depth along y.
    z_lo : float
        Bottom of box (Q1 in data-space, already scaled).
    z_hi : float
        Top of box (Q3 in data-space, already scaled).
    z_median : float
        Median height — used only when ``notch=True``.
    notch_half : float
        Half-width of notch in z units (0 = no notch).
    y_position : float
        Y offset for the entire box.
    config : BoxConfig
    """

    def __init__(
        self,
        x: float,
        width: float,
        depth: float,
        z_lo: float,
        z_hi: float,
        z_median: float,
        notch_half: float = 0.0,
        y_position: float = 0.0,
        config: Optional[BoxConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cfg = config if config is not None else BoxConfig()
        self.x = x
        self.width = width
        self.depth = depth
        self.z_lo = z_lo
        self.z_hi = z_hi
        self.z_median = z_median
        self.notch_half = notch_half
        self.y = y_position

        # Derived face colours
        base = self.cfg.box_color
        self._c_front = base
        self._c_right = _darken(base, self.cfg.side_shade_factor)
        self._c_top   = _lighten(base, self.cfg.top_shade_factor)

        self._build_faces()
        self._build_edges()
        self._build_gloss()
        self._build_shadow()

    # ------------------------------------------------------------------

    def _corners(self, z_lo=None, z_hi=None):
        """8 corners of the prism at the given z extents."""
        zlo = z_lo if z_lo is not None else self.z_lo
        zhi = z_hi if z_hi is not None else self.z_hi
        x, y = self.x, self.y
        hw, hd = self.width / 2, self.depth / 2
        return {
            "BFL": np.array([x - hw, y - hd, zlo]),
            "BFR": np.array([x + hw, y - hd, zlo]),
            "BBR": np.array([x + hw, y + hd, zlo]),
            "BBL": np.array([x - hw, y + hd, zlo]),
            "TFL": np.array([x - hw, y - hd, zhi]),
            "TFR": np.array([x + hw, y - hd, zhi]),
            "TBR": np.array([x + hw, y + hd, zhi]),
            "TBL": np.array([x - hw, y + hd, zhi]),
        }

    def _build_faces(self):
        cfg = self.cfg
        body_op = cfg.body_opacity
        c = self._corners()

        if self.notch_half > 0 and cfg.notch:
            self._build_notched_front(c, body_op)
        else:
            # Plain front face
            self.face_front = Polygon(
                c["BFL"], c["BFR"], c["TFR"], c["TFL"],
                fill_color=_with_opacity(self._c_front, body_op),
                fill_opacity=1.0, stroke_width=0,
            )

        # Right face
        self.face_right = Polygon(
            c["BFR"], c["BBR"], c["TBR"], c["TFR"],
            fill_color=_with_opacity(self._c_right, body_op * 0.95),
            fill_opacity=1.0, stroke_width=0,
        )
        # Top face
        self.face_top = Polygon(
            c["TFL"], c["TFR"], c["TBR"], c["TBL"],
            fill_color=_with_opacity(self._c_top, body_op * 0.90),
            fill_opacity=1.0, stroke_width=0,
        )

        self.faces = VGroup()
        if hasattr(self, "face_front"):
            self.faces.add(self.face_front)
        if hasattr(self, "notch_faces"):
            self.faces.add(self.notch_faces)
        self.faces.add(self.face_right, self.face_top)
        self.add(self.faces)

    def _build_notched_front(self, c, body_op):
        """Build the front face as two trapezoid panels with a V-notch cutout."""
        x, y = self.x, self.y
        hw, hd = self.width / 2, self.depth / 2
        nh = self.notch_half   # notch half-height in z
        zm = self.z_median
        zlo, zhi = self.z_lo, self.z_hi

        # Notch width in x — taper to a point at median
        nw = hw * 0.35

        # Lower trapezoid: bottom → median bottom
        lower_pts = [
            np.array([x - hw, y - hd, zlo]),
            np.array([x + hw, y - hd, zlo]),
            np.array([x + hw, y - hd, zm - nh]),
            np.array([x + nw,  y - hd, zm]),
            np.array([x - nw,  y - hd, zm]),
            np.array([x - hw, y - hd, zm - nh]),
        ]
        upper_pts = [
            np.array([x - nw,  y - hd, zm]),
            np.array([x + nw,  y - hd, zm]),
            np.array([x + hw, y - hd, zm + nh]),
            np.array([x + hw, y - hd, zhi]),
            np.array([x - hw, y - hd, zhi]),
            np.array([x - hw, y - hd, zm + nh]),
        ]

        col = _with_opacity(self._c_front, body_op)
        lower_face = Polygon(*lower_pts, fill_color=col, fill_opacity=1.0, stroke_width=0)
        upper_face = Polygon(*upper_pts, fill_color=col, fill_opacity=1.0, stroke_width=0)

        # Notch fill — slightly darker recess
        notch_col = _with_opacity(
            self.cfg.notch_color if self.cfg.notch_color else _darken(self._c_front, 0.40),
            body_op * 0.85,
        )
        notch_pts = [
            np.array([x - hw, y - hd, zm - nh]),
            np.array([x - nw,  y - hd, zm]),
            np.array([x + nw,  y - hd, zm]),
            np.array([x + hw, y - hd, zm + nh]),
            np.array([x + hw, y - hd, zm - nh]),
            # small rectangle to close: right side
            np.array([x + hw, y - hd, zm - nh]),
            np.array([x - hw, y - hd, zm - nh]),
        ]
        notch_face = Polygon(
            np.array([x - hw, y - hd, zm - nh]),
            np.array([x - nw,  y - hd, zm]),
            np.array([x + nw,  y - hd, zm]),
            np.array([x + hw, y - hd, zm + nh]),
            np.array([x + hw, y - hd, zm - nh]),
            fill_color=notch_col, fill_opacity=1.0, stroke_width=0,
        )

        self.face_front = lower_face   # use lower as primary face reference
        self.notch_faces = VGroup(lower_face, upper_face, notch_face)

    def _build_edges(self):
        c = self._corners()
        col = _with_opacity(self.cfg.edge_color, self.cfg.edge_opacity)
        sw = self.cfg.edge_stroke_width

        # Visible silhouette edges in standard 3/4 view
        edge_pairs = [
            # Front face outline
            (c["BFL"], c["BFR"]), (c["BFR"], c["TFR"]),
            (c["TFR"], c["TFL"]), (c["TFL"], c["BFL"]),
            # Right face extras
            (c["BFR"], c["BBR"]), (c["BBR"], c["TBR"]), (c["TBR"], c["TFR"]),
            # Top face extras
            (c["TFL"], c["TBL"]), (c["TBL"], c["TBR"]),
            # Bottom back
            (c["BBR"], c["BBL"]), (c["BBL"], c["BFL"]),
        ]
        self.edges = VGroup(*[
            Line(a, b, color=col, stroke_width=sw) for a, b in edge_pairs
        ])
        self.add(self.edges)

    def _build_gloss(self):
        if self.cfg.gloss_opacity <= 0:
            self.gloss = VGroup()
            self.add(self.gloss)
            return

        x, y = self.x, self.y
        hw, hd = self.width / 2, self.depth / 2
        gloss_h = (self.z_hi - self.z_lo) * self.cfg.gloss_height_fraction
        z_bot = self.z_hi - gloss_h

        gloss_pts = [
            np.array([x - hw, y - hd, z_bot]),
            np.array([x + hw, y - hd, z_bot]),
            np.array([x + hw, y - hd, self.z_hi]),
            np.array([x - hw, y - hd, self.z_hi]),
        ]
        self.gloss = Polygon(
            *gloss_pts,
            fill_color=_with_opacity(WHITE, self.cfg.gloss_opacity),
            fill_opacity=1.0, stroke_width=0,
        )
        self.add(self.gloss)

    def _build_shadow(self):
        if self.cfg.shadow_opacity <= 0:
            self.shadow = VGroup()
            self.add(self.shadow)
            return

        sx = self.width * 1.10
        sy = self.depth * 0.55
        self.shadow = Ellipse(
            width=sx, height=sy,
            fill_color=_with_opacity(BLACK, self.cfg.shadow_opacity),
            fill_opacity=1.0, stroke_width=0,
        )
        self.shadow.move_to(np.array([self.x, self.y, self.z_lo]))
        self.add(self.shadow)

    # ------------------------------------------------------------------
    # Animation helpers (used by BoxPlot3D.animate_build)
    # ------------------------------------------------------------------

    def animate_grow(self, run_time: float = 0.9) -> UpdateFromAlphaFunc:
        """Grow the box body from z_lo upward."""
        target_hi = self.z_hi
        z_lo = self.z_lo
        x, y = self.x, self.y
        hw, hd = self.width / 2, self.depth / 2
        cfg = self.cfg
        c_front = self._c_front
        c_right = self._c_right
        c_top   = self._c_top

        def updater(mob: BoxBody3D, alpha: float) -> None:
            h = z_lo + smooth(alpha) * (target_hi - z_lo)
            if h <= z_lo + 1e-5:
                mob.set_opacity(0)
                return
            mob.set_opacity(1)

            mob.face_right.set_points_as_corners([
                np.array([x + hw, y - hd, z_lo]),
                np.array([x + hw, y + hd, z_lo]),
                np.array([x + hw, y + hd, h]),
                np.array([x + hw, y - hd, h]),
                np.array([x + hw, y - hd, z_lo]),
            ])
            mob.face_top.set_points_as_corners([
                np.array([x - hw, y - hd, h]),
                np.array([x + hw, y - hd, h]),
                np.array([x + hw, y + hd, h]),
                np.array([x - hw, y + hd, h]),
                np.array([x - hw, y - hd, h]),
            ])
            mob.face_front.set_points_as_corners([
                np.array([x - hw, y - hd, z_lo]),
                np.array([x + hw, y - hd, z_lo]),
                np.array([x + hw, y - hd, h]),
                np.array([x - hw, y - hd, h]),
                np.array([x - hw, y - hd, z_lo]),
            ])
            if cfg.gloss_opacity > 0 and isinstance(mob.gloss, Polygon):
                gh = (target_hi - z_lo) * cfg.gloss_height_fraction
                zb = max(z_lo, h - gh)
                mob.gloss.set_points_as_corners([
                    np.array([x - hw, y - hd, zb]),
                    np.array([x + hw, y - hd, zb]),
                    np.array([x + hw, y - hd, h]),
                    np.array([x - hw, y - hd, h]),
                    np.array([x - hw, y - hd, zb]),
                ])

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)


# ---------------------------------------------------------------------------
# MedianLine3D  — internal
# ---------------------------------------------------------------------------

class MedianLine3D(VGroup):
    """A flat prism at the median height with glow halo.

    Rendered as a thin horizontal rectangle (very small z-thickness)
    spanning the full width and depth of the box, plus an optional
    glow line on the front face.

    Parameters
    ----------
    x, width, depth, z_median, y_position : float
    config : BoxConfig
    """

    def __init__(
        self,
        x: float,
        width: float,
        depth: float,
        z_median: float,
        y_position: float = 0.0,
        config: Optional[BoxConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else BoxConfig()
        hw, hd = width / 2, depth / 2
        y = y_position
        thick = cfg.median_stroke_width * 0.025   # prism z-thickness

        col = _with_opacity(cfg.median_color, cfg.median_opacity)

        # Front line (most visible) — draw as a thick Line
        self.line_front = Line(
            np.array([x - hw, y - hd, z_median]),
            np.array([x + hw, y - hd, z_median]),
            color=col,
            stroke_width=cfg.median_stroke_width,
        )

        # Right-side line
        self.line_right = Line(
            np.array([x + hw, y - hd, z_median]),
            np.array([x + hw, y + hd, z_median]),
            color=_with_opacity(cfg.median_color, cfg.median_opacity * 0.65),
            stroke_width=cfg.median_stroke_width * 0.7,
        )

        # Top surface of the median slab (subtle fill)
        top_pts = [
            np.array([x - hw, y - hd, z_median]),
            np.array([x + hw, y - hd, z_median]),
            np.array([x + hw, y + hd, z_median]),
            np.array([x - hw, y + hd, z_median]),
        ]
        self.top_slab = Polygon(
            *top_pts,
            fill_color=_with_opacity(cfg.median_color, cfg.median_opacity * 0.30),
            fill_opacity=1.0,
            stroke_width=0,
        )

        # Glow halo behind the front line
        if cfg.median_glow_opacity > 0:
            self.glow = Line(
                np.array([x - hw, y - hd, z_median]),
                np.array([x + hw, y - hd, z_median]),
                color=_with_opacity(cfg.median_color, cfg.median_glow_opacity),
                stroke_width=cfg.median_glow_width,
            )
            self.add(self.glow)

        self.add(self.top_slab, self.line_right, self.line_front)


# ---------------------------------------------------------------------------
# Whisker3D  — internal
# ---------------------------------------------------------------------------

class Whisker3D(VGroup):
    """A single whisker (upper or lower) with three style options.

    Styles
    ------
    ``"line"``
        A plain thin line from the box edge to the fence.
    ``"capped"``
        Line + horizontal crossbar at the fence end.  The cap spans
        ``cap_width_fraction × box_width``.
    ``"tapered"``
        A flat prism that is full box-width at the box edge and narrows
        to ``whisker_taper_ratio × box_width`` at the fence end.
        Front and right faces are shaded like the box body.

    Parameters
    ----------
    x, width, depth, y_position : float
        Box geometry.
    z_box_edge : float
        Z where the whisker leaves the box (Q3 for upper, Q1 for lower).
    z_fence : float
        Z of the whisker end (min/max inlier).
    direction : int
        +1 for upper whisker, -1 for lower.
    config : BoxConfig
    """

    def __init__(
        self,
        x: float,
        width: float,
        depth: float,
        z_box_edge: float,
        z_fence: float,
        direction: int,
        y_position: float = 0.0,
        config: Optional[BoxConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else BoxConfig()
        style = cfg.whisker_style
        hw, hd = width / 2, depth / 2
        y = y_position
        col = _with_opacity(cfg.box_color, cfg.whisker_opacity)
        sw = cfg.whisker_stroke_width

        if style == "line":
            self._build_line(x, y, hd, z_box_edge, z_fence, col, sw)

        elif style == "capped":
            self._build_capped(x, y, hw, hd, z_box_edge, z_fence,
                               col, sw, cfg.cap_width_fraction)

        elif style == "tapered":
            self._build_tapered(x, y, hw, hd, z_box_edge, z_fence,
                                col, cfg)
        else:
            self._build_line(x, y, hd, z_box_edge, z_fence, col, sw)

    # ------------------------------------------------------------------

    def _build_line(self, x, y, hd, z0, z1, col, sw):
        self.shaft = Line(
            np.array([x, y - hd, z0]),
            np.array([x, y - hd, z1]),
            color=col, stroke_width=sw,
        )
        # Duplicate on right side for 3D feel
        self.shaft_r = Line(
            np.array([x, y + hd, z0]),
            np.array([x, y + hd, z1]),
            color=_with_opacity(col, col.get_hex_l() if hasattr(col, "get_hex_l") else 0.5),
            stroke_width=sw * 0.5,
        )
        self.add(self.shaft, self.shaft_r)

    def _build_capped(self, x, y, hw, hd, z0, z1, col, sw, cap_frac):
        # Shaft (centre line)
        self.shaft = Line(
            np.array([x, y - hd, z0]),
            np.array([x, y - hd, z1]),
            color=col, stroke_width=sw,
        )
        # Cap bar at fence end
        cap_hw = hw * cap_frac
        self.cap_front = Line(
            np.array([x - cap_hw, y - hd, z1]),
            np.array([x + cap_hw, y - hd, z1]),
            color=col, stroke_width=sw * 1.4,
        )
        # Cap right side
        self.cap_right = Line(
            np.array([x + cap_hw, y - hd, z1]),
            np.array([x + cap_hw, y + hd, z1]),
            color=_with_opacity(col, 0.55),
            stroke_width=sw * 0.7,
        )
        self.add(self.shaft, self.cap_front, self.cap_right)

    def _build_tapered(self, x, y, hw, hd, z0, z1, col, cfg):
        """A flat prism tapered from box width at z0 to narrow at z1."""
        taper_hw = hw * cfg.whisker_taper_ratio
        taper_hd = hd * max(cfg.whisker_taper_ratio, 0.20)
        sw = cfg.whisker_stroke_width
        body_op = cfg.whisker_opacity

        c_front = _with_opacity(cfg.box_color, body_op)
        c_right = _with_opacity(_darken(cfg.box_color, cfg.whisker_taper_shade_factor), body_op)
        c_top   = _with_opacity(_lighten(cfg.box_color, 1.15), body_op * 0.80)

        # Front face of taper prism
        front_pts = [
            np.array([x - hw,     y - hd,     z0]),
            np.array([x + hw,     y - hd,     z0]),
            np.array([x + taper_hw, y - hd,   z1]),
            np.array([x - taper_hw, y - hd,   z1]),
        ]
        self.face_front = Polygon(*front_pts,
                                   fill_color=c_front, fill_opacity=1.0, stroke_width=0)

        # Right face
        right_pts = [
            np.array([x + hw,     y - hd,     z0]),
            np.array([x + hw,     y + hd,     z0]),
            np.array([x + taper_hw, y + taper_hd, z1]),
            np.array([x + taper_hw, y - hd,   z1]),
        ]
        self.face_right = Polygon(*right_pts,
                                   fill_color=c_right, fill_opacity=1.0, stroke_width=0)

        # Top (outer) face — at z1, narrow cap
        top_pts = [
            np.array([x - taper_hw, y - hd,        z1]),
            np.array([x + taper_hw, y - hd,        z1]),
            np.array([x + taper_hw, y + taper_hd,  z1]),
            np.array([x - taper_hw, y + taper_hd,  z1]),
        ]
        self.face_top = Polygon(*top_pts,
                                 fill_color=c_top, fill_opacity=1.0, stroke_width=0)

        # Silhouette edges
        ecol = _with_opacity(cfg.edge_color, cfg.edge_opacity * 0.7)
        self.edges = VGroup(*[
            Line(a, b, color=ecol, stroke_width=sw * 0.8)
            for a, b in [
                (front_pts[0], front_pts[1]),
                (front_pts[1], front_pts[2]),
                (front_pts[2], front_pts[3]),
                (front_pts[3], front_pts[0]),
                (right_pts[1], right_pts[2]),
                (top_pts[0], top_pts[3]),
            ]
        ])

        self.add(self.face_front, self.face_right, self.face_top, self.edges)


# ---------------------------------------------------------------------------
# MeanMarker3D  — internal
# ---------------------------------------------------------------------------

class MeanMarker3D(VGroup):
    """3D marker for the arithmetic mean.

    Rendered as a diamond (rotated square), cross, or sphere.

    Parameters
    ----------
    x, z_mean, y_position : float
    config : BoxConfig
    """

    def __init__(
        self,
        x: float,
        z_mean: float,
        y_position: float = 0.0,
        config: Optional[BoxConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else BoxConfig()
        style = cfg.mean_marker_style
        if style == "none":
            return

        col = _with_opacity(cfg.mean_color, 0.95)
        sz = cfg.mean_marker_size
        pos = np.array([x, y_position, z_mean])

        if style == "diamond":
            pts = [
                pos + np.array([0, 0, sz]),
                pos + np.array([sz, 0, 0]),
                pos + np.array([0, 0, -sz]),
                pos + np.array([-sz, 0, 0]),
            ]
            marker = Polygon(*pts, fill_color=col, fill_opacity=1.0,
                             stroke_color=_darken(cfg.mean_color, 0.65),
                             stroke_width=0.9)

        elif style == "cross":
            h = sz * 1.5
            marker = VGroup(
                Line(pos - h * RIGHT, pos + h * RIGHT, color=col, stroke_width=2.0),
                Line(pos - h * UP,    pos + h * UP,    color=col, stroke_width=2.0),
            )

        elif style == "sphere":
            marker = Dot3D(point=pos, radius=sz, color=col)

        else:
            marker = Dot3D(point=pos, radius=sz, color=col)

        self.add(marker)


# ---------------------------------------------------------------------------
# OutlierMarkers3D  — internal
# ---------------------------------------------------------------------------

class OutlierMarkers3D(VGroup):
    """3D markers for outlier data points.

    Each outlier is placed at its true z-value on the x-column of the box,
    optionally jittered along y to prevent stacking.

    Parameters
    ----------
    x : float
        X column position.
    outliers : np.ndarray
        Array of outlier values (already scaled to scene units).
    y_position : float
    config : BoxConfig
    rng_seed : int
        Seed for jitter reproducibility.
    """

    def __init__(
        self,
        x: float,
        outliers: np.ndarray,
        y_position: float = 0.0,
        config: Optional[BoxConfig] = None,
        rng_seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = config if config is not None else BoxConfig()
        if len(outliers) == 0:
            return

        style = cfg.outlier_style
        col = _with_opacity(cfg.outlier_color, cfg.outlier_opacity)
        sz = cfg.outlier_size
        jitter = cfg.outlier_jitter

        rng = np.random.default_rng(rng_seed)
        y_offsets = rng.uniform(-jitter, jitter, len(outliers)) if jitter > 0 else np.zeros(len(outliers))

        self.individual_markers = VGroup()

        for val, dy in zip(outliers, y_offsets):
            pos = np.array([x, y_position + dy, val])

            if style == "sphere":
                m = Dot3D(point=pos, radius=sz, color=col)

            elif style == "diamond":
                pts = [
                    pos + np.array([0, 0, sz]),
                    pos + np.array([sz, 0, 0]),
                    pos + np.array([0, 0, -sz]),
                    pos + np.array([-sz, 0, 0]),
                ]
                m = Polygon(*pts, fill_color=col, fill_opacity=1.0,
                            stroke_color=_darken(cfg.outlier_color, 0.65),
                            stroke_width=0.8)

            elif style == "cross":
                h = sz * 1.4
                m = VGroup(
                    Line(pos - h * RIGHT, pos + h * RIGHT, color=col, stroke_width=1.5),
                    Line(pos - h * UP,    pos + h * UP,    color=col, stroke_width=1.5),
                )

            elif style == "ring":
                n_pts = 12
                angles = np.linspace(0, TAU, n_pts, endpoint=False)
                ring_pts = [pos + sz * np.array([np.cos(a), 0, np.sin(a)]) for a in angles]
                m = VMobject()
                m.set_points_as_corners(ring_pts + [ring_pts[0]])
                m.set_stroke(color=col, width=1.5)
                m.set_fill(opacity=0)

            else:
                m = Dot3D(point=pos, radius=sz, color=col)

            self.individual_markers.add(m)

        self.add(self.individual_markers)

    def animate_scatter(
        self,
        run_time_per: float = 0.15,
        lag: float = 0.05,
    ) -> LaggedStart:
        """Pop markers in one by one."""
        return LaggedStart(
            *[FadeIn(m, scale=0.1, run_time=run_time_per)
              for m in self.individual_markers],
            lag_ratio=lag,
        )


# ---------------------------------------------------------------------------
# SignificanceBracket3D
# ---------------------------------------------------------------------------

class SignificanceBracket3D(VGroup):
    """Bracket + p-value annotation drawn between two box plots.

    The bracket sits above both boxes at a height just above the taller
    maximum whisker end.  Star notation follows convention:
    - p ≤ 0.001 → ***
    - p ≤ 0.01  → **
    - p ≤ 0.05  → *
    - p > 0.05  → ns

    Parameters
    ----------
    x_left, x_right : float
        X positions of the two boxes.
    z_top : float
        Z height of the bracket bar.
    p_value : float
    bracket_drop : float
        How far the bracket legs drop below the bar.
    color : ManimColor
    stroke_width : float
    label_font_size : int
    scene : ThreeDScene or None
    y_position : float
    """

    def __init__(
        self,
        x_left: float,
        x_right: float,
        z_top: float,
        p_value: float,
        bracket_drop: float = 0.18,
        color: ManimColor = WHITE,
        stroke_width: float = 1.4,
        label_font_size: int = 20,
        y_position: float = 0.0,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        col = _with_opacity(color, 0.85)
        y = y_position

        # Horizontal bar
        bar = Line(
            np.array([x_left, y, z_top]),
            np.array([x_right, y, z_top]),
            color=col, stroke_width=stroke_width,
        )
        # Left drop
        leg_l = Line(
            np.array([x_left, y, z_top]),
            np.array([x_left, y, z_top - bracket_drop]),
            color=col, stroke_width=stroke_width,
        )
        # Right drop
        leg_r = Line(
            np.array([x_right, y, z_top]),
            np.array([x_right, y, z_top - bracket_drop]),
            color=col, stroke_width=stroke_width,
        )
        self.add(bar, leg_l, leg_r)

        # Significance label
        if p_value <= 0.001:
            sig_text = "***"
        elif p_value <= 0.01:
            sig_text = "**"
        elif p_value <= 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"

        x_mid = (x_left + x_right) / 2
        lbl = Text(sig_text, font_size=label_font_size, color=color)
        lbl.move_to(np.array([x_mid, y, z_top + 0.20]))
        self.label = lbl
        self.add(lbl)

        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)

    def animate_draw(self, run_time: float = 0.6) -> AnimationGroup:
        return AnimationGroup(
            Create(self, run_time=run_time),
            lag_ratio=0.2,
        )


# ---------------------------------------------------------------------------
# BoxPlot3D  — single box
# ---------------------------------------------------------------------------

class BoxPlot3D(VGroup):
    """A single richly-detailed 3D box plot.

    Constructed from a ``FiveNumberSummary`` (computed from raw data or
    pre-supplied).  All visual layers are built as separate sub-objects so
    they can be animated independently.

    Parameters
    ----------
    summary : FiveNumberSummary
        The five-number summary to visualize.
    x : float
        X position of the box centre.
    box_width : float
        Width of the box along the x-axis.
    box_depth : float
        Depth of the box along the y-axis.
    z_scale : float
        Multiplier applied to all z-values (value → scene units).
    y_position : float
    config : BoxConfig
    scene : ThreeDScene or None

    Attributes
    ----------
    summary : FiveNumberSummary
    body : BoxBody3D
    upper_whisker : Whisker3D
    lower_whisker : Whisker3D
    median_line : MedianLine3D
    mean_marker : MeanMarker3D
    outliers : OutlierMarkers3D
    category_label : VGroup
    """

    def __init__(
        self,
        summary: FiveNumberSummary,
        x: float = 0.0,
        box_width: float = 0.55,
        box_depth: float = 0.38,
        z_scale: float = 1.0,
        y_position: float = 0.0,
        config: Optional[BoxConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.summary = summary
        self.cfg = config if config is not None else POLISHED_BOX
        self._scene = scene
        self.x = x
        self.box_width = box_width
        self.box_depth = box_depth
        self.z_scale = z_scale
        self.y = y_position

        # Scale all values to scene units
        s = summary
        zs = z_scale
        self._z_q1     = s.q1     * zs
        self._z_q3     = s.q3     * zs
        self._z_median = s.median * zs
        self._z_min    = s.minimum* zs
        self._z_max    = s.maximum* zs
        self._z_mean   = (s.mean  * zs) if s.mean is not None else None
        self._outliers_z = s.outliers * zs

        # Notch half-width: 1.58 × IQR / √n  (95% CI around median)
        notch_half = 0.0
        if self.cfg.notch and s.n > 0:
            notch_half = (1.58 * s.iqr * zs) / max(np.sqrt(s.n), 1.0)
            notch_half = min(notch_half, (self._z_q3 - self._z_q1) * 0.42)

        # Build layers
        self.body = BoxBody3D(
            x=x, width=box_width, depth=box_depth,
            z_lo=self._z_q1, z_hi=self._z_q3,
            z_median=self._z_median,
            notch_half=notch_half,
            y_position=y_position,
            config=self.cfg,
        )

        self.upper_whisker = Whisker3D(
            x=x, width=box_width, depth=box_depth,
            z_box_edge=self._z_q3, z_fence=self._z_max,
            direction=+1, y_position=y_position,
            config=self.cfg,
        )

        self.lower_whisker = Whisker3D(
            x=x, width=box_width, depth=box_depth,
            z_box_edge=self._z_q1, z_fence=self._z_min,
            direction=-1, y_position=y_position,
            config=self.cfg,
        )

        self.median_line = MedianLine3D(
            x=x, width=box_width, depth=box_depth,
            z_median=self._z_median, y_position=y_position,
            config=self.cfg,
        )

        self.mean_marker = MeanMarker3D(
            x=x, z_mean=self._z_mean if self._z_mean is not None else self._z_median,
            y_position=y_position, config=self.cfg,
        ) if self.cfg.mean_marker_style != "none" and self._z_mean is not None else VGroup()

        self.outliers = OutlierMarkers3D(
            x=x, outliers=self._outliers_z,
            y_position=y_position, config=self.cfg,
        )

        # Category label below the box
        if self.cfg.show_category_label and summary.label:
            lbl = Text(
                summary.label,
                font_size=self.cfg.category_label_font_size,
                color=self.cfg.category_label_color,
            )
            lbl.move_to(np.array([
                x, y_position - box_depth / 2,
                self._z_q1 - self.cfg.category_label_offset
            ]))
            self.category_label = VGroup(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)
        else:
            self.category_label = VGroup()

        # Stats panel
        if self.cfg.show_stats_label:
            self._build_stats_label(scene)
        else:
            self.stats_label = VGroup()

        self.add(
            self.body,
            self.upper_whisker, self.lower_whisker,
            self.median_line, self.mean_marker,
            self.outliers, self.category_label, self.stats_label,
        )

    # ------------------------------------------------------------------

    def _build_stats_label(self, scene: Optional[ThreeDScene]) -> None:
        s = self.summary
        lines = [
            f"n = {s.n}",
            f"med = {s.median:.2f}",
            f"IQR = {s.iqr:.2f}",
        ]
        grp = VGroup()
        for i, txt in enumerate(lines):
            t = Text(txt, font_size=self.cfg.stats_label_font_size,
                     color=_with_opacity(self.cfg.category_label_color, 0.70))
            t.move_to(np.array([
                self.x + self.box_width * 0.75,
                self.y,
                self._z_q3 + 0.25 - i * 0.22,
            ]))
            grp.add(t)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(t)
        self.stats_label = grp
        self.add(self.stats_label)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_build(
        self,
        run_time_body: float = 0.85,
        run_time_whiskers: float = 0.50,
        run_time_median: float = 0.30,
        run_time_outliers: float = 0.40,
    ) -> Succession:
        """Build the box plot layer by layer.

        Order: shadow → body grow → whiskers → median line → mean → outliers.
        """
        anims = []

        # Shadow fade in
        if hasattr(self.body, "shadow") and isinstance(self.body.shadow, Ellipse):
            anims.append(FadeIn(self.body.shadow, run_time=0.25))

        # Box body grows upward
        anims.append(self.body.animate_grow(run_time=run_time_body))

        # Whiskers appear together
        anims.append(AnimationGroup(
            Create(self.upper_whisker, run_time=run_time_whiskers),
            Create(self.lower_whisker, run_time=run_time_whiskers),
        ))

        # Median line draws across
        anims.append(Create(self.median_line, run_time=run_time_median))

        # Mean marker pops in
        if len(self.mean_marker) > 0:
            anims.append(FadeIn(self.mean_marker, scale=0.3, run_time=0.25))

        # Outliers scatter in
        if len(self.outliers) > 0:
            anims.append(self.outliers.animate_scatter(run_time_per=0.12))

        # Category label
        if len(self.category_label) > 0:
            anims.append(FadeIn(self.category_label, run_time=0.20))

        return Succession(*anims)

    # ------------------------------------------------------------------
    # Highlighting
    # ------------------------------------------------------------------

    def highlight(
        self,
        color: ManimColor = YELLOW,
        edge_width: float = 2.0,
        edge_opacity: float = 0.90,
    ) -> "BoxPlot3D":
        """Emphasise this box with thick bright edges."""
        edge_col = _with_opacity(color, edge_opacity)
        for edge in self.body.edges:
            edge.set_stroke(color=edge_col, width=edge_width)
        return self

    def unhighlight(self) -> "BoxPlot3D":
        cfg = self.cfg
        edge_col = _with_opacity(cfg.edge_color, cfg.edge_opacity)
        for edge in self.body.edges:
            edge.set_stroke(color=edge_col, width=cfg.edge_stroke_width)
        return self

    def set_box_color(self, color: ManimColor) -> "BoxPlot3D":
        """Recolour the box body preserving shading ratios."""
        cfg = self.cfg
        self.body.face_front.set_fill(_with_opacity(color, cfg.body_opacity))
        self.body.face_right.set_fill(_with_opacity(_darken(color, cfg.side_shade_factor),
                                                    cfg.body_opacity * 0.95))
        self.body.face_top.set_fill(_with_opacity(_lighten(color, cfg.top_shade_factor),
                                                  cfg.body_opacity * 0.90))
        return self

    # ------------------------------------------------------------------
    # Information access
    # ------------------------------------------------------------------

    @property
    def top_z(self) -> float:
        """Z coordinate of the top whisker end (maximum inlier)."""
        return self._z_max

    @property
    def bottom_z(self) -> float:
        """Z coordinate of the bottom whisker end (minimum inlier)."""
        return self._z_min

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        label: str = "",
        **kwargs,
    ) -> "BoxPlot3D":
        """Build directly from raw data array."""
        summary = FiveNumberSummary.from_data(data, label=label)
        return cls(summary=summary, **kwargs)


# ---------------------------------------------------------------------------
# BoxPlotGroup3D  — k boxes side by side
# ---------------------------------------------------------------------------

class BoxPlotGroup3D(VGroup):
    """Multiple ``BoxPlot3D`` objects arranged side-by-side along the x-axis.

    Parameters
    ----------
    summaries : list[FiveNumberSummary]
        One summary per group.
    x_start : float
        X position of the first box.
    spacing : float
        Centre-to-centre distance between adjacent boxes.
    box_width : float
    box_depth : float
    z_scale : float
    y_position : float
    colors : list[ManimColor] or None
        Per-box colours.  If None, cycles through
        ``BoxColorPalette.CATEGORICAL``.
    config : BoxConfig
    show_mean : bool
        Shortcut to enable mean markers on all boxes.
    whisker_style : str
        Shortcut to set whisker style on all boxes (overrides config).
    scene : ThreeDScene or None

    Attributes
    ----------
    boxes : list[BoxPlot3D]
    """

    def __init__(
        self,
        summaries: List[FiveNumberSummary],
        x_start: float = 0.0,
        spacing: float = 1.2,
        box_width: float = 0.55,
        box_depth: float = 0.38,
        z_scale: float = 1.0,
        y_position: float = 0.0,
        colors: Optional[Sequence[ManimColor]] = None,
        config: Optional[BoxConfig] = None,
        show_mean: bool = True,
        whisker_style: Optional[str] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        palette = BoxColorPalette.CATEGORICAL
        n = len(summaries)
        self._colors = (
            list(colors) if colors is not None
            else [palette[i % len(palette)] for i in range(n)]
        )

        self.boxes: List[BoxPlot3D] = []
        self._spacing = spacing
        self._x_start = x_start
        self._scene = scene

        for i, (summary, col) in enumerate(zip(summaries, self._colors)):
            # Per-box config copy
            if config is not None:
                cfg = BoxConfig(**config.__dict__)
            else:
                cfg = BoxConfig(**POLISHED_BOX.__dict__)

            cfg.box_color = col
            if not show_mean:
                cfg.mean_marker_style = "none"
            if whisker_style is not None:
                cfg.whisker_style = whisker_style

            box = BoxPlot3D(
                summary=summary,
                x=x_start + i * spacing,
                box_width=box_width,
                box_depth=box_depth,
                z_scale=z_scale,
                y_position=y_position,
                config=cfg,
                scene=scene,
            )
            self.boxes.append(box)
            self.add(box)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        groups: Dict[str, np.ndarray],
        iqr_factor: float = 1.5,
        **kwargs,
    ) -> "BoxPlotGroup3D":
        """Build from a ``{group_name: data_array}`` dictionary.

        Parameters
        ----------
        groups : dict[str, array-like]
            Mapping of group label → raw data values.
        iqr_factor : float
            IQR multiplier for outlier detection (default 1.5).

        Example
        -------
        ::

            chart = BoxPlotGroup3D.from_data({
                "Control":     np.random.normal(4.0, 1.0, 80),
                "Treatment A": np.random.normal(5.5, 0.8, 80),
            })
        """
        summaries = [
            FiveNumberSummary.from_data(data, label=name, iqr_factor=iqr_factor)
            for name, data in groups.items()
        ]
        return cls(summaries=summaries, **kwargs)

    @classmethod
    def from_precomputed(
        cls,
        stats: List[Dict],
        **kwargs,
    ) -> "BoxPlotGroup3D":
        """Build from a list of precomputed stat dicts.

        Each dict must have keys: ``q1``, ``median``, ``q3``.
        Optional keys: ``minimum``, ``maximum``, ``mean``, ``label``.

        Example
        -------
        ::

            chart = BoxPlotGroup3D.from_precomputed([
                {"q1": 3.1, "median": 4.2, "q3": 5.3, "label": "A"},
                {"q1": 2.8, "median": 3.9, "q3": 5.1, "label": "B"},
            ])
        """
        summaries = [
            FiveNumberSummary.from_precomputed(
                q1=d["q1"], median=d["median"], q3=d["q3"],
                minimum=d.get("minimum"), maximum=d.get("maximum"),
                mean=d.get("mean"), label=d.get("label", ""),
            )
            for d in stats
        ]
        return cls(summaries=summaries, **kwargs)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_build(
        self,
        stagger: float = 0.10,
        run_time_body: float = 0.85,
    ) -> LaggedStart:
        """Build all boxes with a left-to-right stagger.

        Parameters
        ----------
        stagger : float
            Seconds between the start of each box's build animation.
        """
        return LaggedStart(
            *[box.animate_build(run_time_body=run_time_body)
              for box in self.boxes],
            lag_ratio=stagger,
        )

    def animate_update(
        self,
        new_summaries: List[FiveNumberSummary],
        run_time: float = 1.2,
    ) -> AnimationGroup:
        """Morph all box bodies to new five-number summaries.

        Uses ``UpdateFromAlphaFunc`` per box body so heights interpolate
        smoothly.  Whiskers and median are rebuilt via ``FadeOut`` →
        rebuild → ``FadeIn``.
        """
        if len(new_summaries) != len(self.boxes):
            raise ValueError("new_summaries length must match number of boxes")

        anims = []
        for box, new_sum in zip(self.boxes, new_summaries):
            zs = box.z_scale
            old_q1, old_q3 = box._z_q1, box._z_q3
            new_q1 = new_sum.q1 * zs
            new_q3 = new_sum.q3 * zs
            x, y = box.x, box.y
            hw, hd = box.box_width / 2, box.box_depth / 2
            cfg = box.cfg

            def make_updater(b, oq1, oq3, nq1, nq3, bx, by, bhw, bhd, bcfg):
                def updater(mob, alpha):
                    t = rate_functions.ease_in_out_cubic(alpha)
                    q1 = oq1 + (nq1 - oq1) * t
                    q3 = oq3 + (nq3 - oq3) * t

                    mob.body.face_front.set_points_as_corners([
                        np.array([bx - bhw, by - bhd, q1]),
                        np.array([bx + bhw, by - bhd, q1]),
                        np.array([bx + bhw, by - bhd, q3]),
                        np.array([bx - bhw, by - bhd, q3]),
                        np.array([bx - bhw, by - bhd, q1]),
                    ])
                    mob.body.face_right.set_points_as_corners([
                        np.array([bx + bhw, by - bhd, q1]),
                        np.array([bx + bhw, by + bhd, q1]),
                        np.array([bx + bhw, by + bhd, q3]),
                        np.array([bx + bhw, by - bhd, q3]),
                        np.array([bx + bhw, by - bhd, q1]),
                    ])
                    mob.body.face_top.set_points_as_corners([
                        np.array([bx - bhw, by - bhd, q3]),
                        np.array([bx + bhw, by - bhd, q3]),
                        np.array([bx + bhw, by + bhd, q3]),
                        np.array([bx - bhw, by + bhd, q3]),
                        np.array([bx - bhw, by - bhd, q3]),
                    ])
                return updater

            anims.append(UpdateFromAlphaFunc(
                box,
                make_updater(box, old_q1, old_q3, new_q1, new_q3,
                             x, y, hw, hd, cfg),
                run_time=run_time,
            ))
            box._z_q1 = new_q1
            box._z_q3 = new_q3

        return AnimationGroup(*anims)

    # ------------------------------------------------------------------
    # Highlighting / selection
    # ------------------------------------------------------------------

    def highlight_box(
        self,
        index: int,
        color: ManimColor = YELLOW,
        dim_others: bool = True,
        dim_opacity: float = 0.25,
    ) -> "BoxPlotGroup3D":
        for i, box in enumerate(self.boxes):
            if i == index:
                box.highlight(color=color)
            elif dim_others:
                box.set_opacity(dim_opacity)
        return self

    def unhighlight_all(self) -> "BoxPlotGroup3D":
        for box in self.boxes:
            box.unhighlight()
            box.set_opacity(1.0)
        return self

    def add_significance_bracket(
        self,
        left_index: int,
        right_index: int,
        p_value: float,
        height_clearance: float = 0.35,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ) -> SignificanceBracket3D:
        """Draw a significance bracket between boxes *left_index* and *right_index*.

        The bracket is placed above the taller of the two boxes' maximum
        whisker ends plus *height_clearance*.

        Returns the ``SignificanceBracket3D`` so the caller can animate it.
        """
        box_l = self.boxes[left_index]
        box_r = self.boxes[right_index]
        z_top = max(box_l.top_z, box_r.top_z) + height_clearance

        bracket = SignificanceBracket3D(
            x_left=box_l.x,
            x_right=box_r.x,
            z_top=z_top,
            p_value=p_value,
            y_position=box_l.y,
            scene=scene if scene is not None else self._scene,
            **kwargs,
        )
        self.add(bracket)
        return bracket

    def apply_value_coloring(
        self,
        lo_color: ManimColor = ManimColor("#A32D2D"),
        hi_color: ManimColor = ManimColor("#0C4478"),
    ) -> "BoxPlotGroup3D":
        """Recolour boxes based on relative median value."""
        medians = [b.summary.median for b in self.boxes]
        lo, hi = min(medians), max(medians)
        span = hi - lo if hi != lo else 1.0
        for box, med in zip(self.boxes, medians):
            t = (med - lo) / span
            col = _lerp_color(lo_color, hi_color, t)
            box.set_box_color(col)
        return self


# ---------------------------------------------------------------------------
# NotchedBoxPlot3D  — convenience subclass
# ---------------------------------------------------------------------------

class NotchedBoxPlot3D(BoxPlotGroup3D):
    """``BoxPlotGroup3D`` with notches pre-enabled for all boxes.

    Notches represent the 95% confidence interval around the median.
    Non-overlapping notches suggest the medians differ significantly.

    All parameters forwarded to ``BoxPlotGroup3D``; ``notch=True``
    is always set in the config.
    """

    def __init__(self, *args, config: Optional[BoxConfig] = None, **kwargs):
        if config is None:
            config = BoxConfig(**NOTCHED_BOX.__dict__)
        else:
            config.notch = True
        super().__init__(*args, config=config, **kwargs)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def box_from_normal(
    mean: float = 0.0,
    std: float = 1.0,
    n: int = 200,
    label: str = "",
    seed: int = 42,
    **kwargs,
) -> BoxPlot3D:
    """Generate a ``BoxPlot3D`` from a simulated normal distribution.

    Useful for quick demonstrations without real data.

    Parameters
    ----------
    mean, std : float
        Parameters of the normal distribution.
    n : int
        Sample size.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(mean, std, n)
    return BoxPlot3D.from_data(data, label=label, **kwargs)


def comparison_chart(
    means: Sequence[float],
    stds: Sequence[float],
    labels: Sequence[str],
    n: int = 100,
    **kwargs,
) -> BoxPlotGroup3D:
    """Build a ``BoxPlotGroup3D`` from parallel normal distributions.

    Each group is simulated with the given mean and standard deviation.
    Seed is index-based for reproducibility.

    Parameters
    ----------
    means, stds : sequence of float
    labels : sequence of str
    n : int
        Sample size per group.

    Example
    -------
    ::

        chart = comparison_chart(
            means=[3.5, 5.0, 4.2, 6.1],
            stds=[0.9, 1.1, 0.8, 1.3],
            labels=["A", "B", "C", "D"],
        )
    """
    groups = {}
    for i, (m, s, lbl) in enumerate(zip(means, stds, labels)):
        rng = np.random.default_rng(i * 137 + 11)
        groups[lbl] = rng.normal(m, s, n)
    return BoxPlotGroup3D.from_data(groups, **kwargs)