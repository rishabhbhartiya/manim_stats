"""
manim_stats/charts/histogram3d.py
==================================
A production-quality 3D histogram for Manim with:
  - Physically shaded 3D bar prisms (front/side/top faces with distinct shading)
  - Frequency-based color gradient (cold→warm heat mapping across bars)
  - Floor grid plane with perspective depth
  - Full 3D axes: bin-edge ticks on X, value ticks on Y, optional depth on Z
  - KDE/PDF overlay curve (ParametricCurve floating above bars)
  - Statistical markers: mean plane, median plane, ±1σ shaded slab
  - Floating count/frequency labels above each bar
  - Rich animation suite: GrowFromFloor, SweepBins, MorphHistogram, HighlightBin
  - HistogramConfig dataclass for full customization
  - LogScale mode, normalized (density) vs count mode, orientation toggle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

from manim import (
    WHITE, BLACK, BLUE, BLUE_E, BLUE_B, YELLOW, RED, GREEN, ORANGE,
    GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    UP, DOWN, LEFT, RIGHT, OUT, IN,
    PI, TAU,
    VGroup, VMobject, Mobject,
    ThreeDAxes, Arrow3D, Line3D, Polygon, Prism,
    Text, MathTex, DecimalNumber,
    Surface, ParametricFunction,
    FadeIn, FadeOut, GrowFromPoint, Transform,
    Create, Write, AnimationGroup, Succession, LaggedStart,
    always_redraw, ValueTracker,
    interpolate_color, color_to_rgb, rgb_to_color,
    ManimColor, color_gradient,
    DEGREES,
)


# ---------------------------------------------------------------------------
# Color palette helpers
# ---------------------------------------------------------------------------

HIST_COLD    = ManimColor("#2166AC")   # deep blue  – low bars
HIST_MID     = ManimColor("#74ADD1")   # mid blue
HIST_NEUTRAL = ManimColor("#FEE090")   # straw      – mid bars
HIST_WARM    = ManimColor("#F46D43")   # orange     – high bars
HIST_HOT     = ManimColor("#D73027")   # red        – peak bar

MEAN_COLOR   = ManimColor("#E040FB")   # vivid violet
MEDIAN_COLOR = ManimColor("#00BCD4")   # cyan
SIGMA_COLOR  = ManimColor("#FFEB3B")   # yellow

FACE_DARKEN  = 0.40   # how much darker side faces are vs the base color
FACE_LIGHTEN = 0.15   # how much lighter the top face is vs the base color


def _darken(color: ManimColor, factor: float) -> ManimColor:
    """Return a darker version of *color* by blending towards black."""
    return interpolate_color(color, BLACK, factor)


def _lighten(color: ManimColor, factor: float) -> ManimColor:
    """Return a lighter version of *color* by blending towards white."""
    return interpolate_color(color, WHITE, factor)


def _freq_color(
    value: float,
    vmin: float,
    vmax: float,
    palette: Sequence[ManimColor] | None = None,
) -> ManimColor:
    """Map a frequency value to a heatmap color within [vmin, vmax]."""
    palette = palette or [HIST_COLD, HIST_MID, HIST_NEUTRAL, HIST_WARM, HIST_HOT]
    if vmax <= vmin:
        return palette[len(palette) // 2]
    t = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    # Multi-stop interpolation along the palette
    n = len(palette) - 1
    segment = min(int(t * n), n - 1)
    local_t = t * n - segment
    return interpolate_color(palette[segment], palette[segment + 1], local_t)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HistogramConfig:
    """All visual and statistical options for Histogram3D.

    Parameters
    ----------
    bins : int | sequence
        Number of equal-width bins, or an explicit array of bin edges.
    density : bool
        If True, bars show probability density (area sums to 1).
        If False, bars show raw counts.
    log_scale : bool
        Apply log₁₀ transform to bar heights (useful for heavy-tailed data).
    orientation : str
        ``"xy"`` — bars rise along Z (default 3-D look).
        ``"flat"`` — bars rise along Y (classic 2-D histogram in 3-D space).
    bar_width_ratio : float
        Fraction of bin width used by the bar (0 < ratio ≤ 1).
        Gap between bars = ``1 - bar_width_ratio``.
    bar_depth : float
        Depth of each bar along the third axis (visual thickness).
    color_palette : list[ManimColor] | None
        Override the default cold→hot heatmap ramp.
    use_uniform_color : bool
        If True, all bars share a single ``uniform_color`` instead of the heatmap.
    uniform_color : ManimColor
        Used when ``use_uniform_color=True``.
    bar_opacity : float
        Overall fill opacity for bar faces [0, 1].
    edge_stroke_width : float
        Width of bar edge strokes.  0 to disable edges.
    edge_opacity : float
        Opacity of bar edge strokes.
    show_kde : bool
        Overlay a kernel density estimate curve above the histogram.
    kde_bandwidth : float | str
        Bandwidth for KDE; ``"scott"`` or ``"silverman"`` use automatic rules.
    kde_color : ManimColor
        Color of the KDE curve.
    kde_stroke_width : float
        Stroke width of the KDE curve.
    show_floor_grid : bool
        Render a semi-transparent XZ grid plane under the bars.
    floor_grid_lines : int
        Number of lines in each direction on the floor grid.
    floor_color : ManimColor
        Color of the floor grid.
    floor_opacity : float
        Opacity of the floor grid.
    show_mean : bool
        Draw a vertical plane / line at the mean.
    show_median : bool
        Draw a vertical plane / line at the median.
    show_sigma_band : bool
        Shade a translucent slab between mean ± 1 standard deviation.
    show_bin_labels : bool
        Show floating count/density labels above each bar.
    label_font_size : int
        Font size for bar labels.
    label_precision : int
        Decimal places for bar labels.
    axes_x_label : str
        Label for the horizontal (bin) axis.
    axes_y_label : str
        Label for the vertical (frequency) axis.
    axes_color : ManimColor
        Color of axes lines.
    tick_length : float
        Length of axis tick marks.
    x_scale : float
        Total rendered width of the histogram along X.
    y_scale : float
        Total rendered height of the histogram along Y (max bar height).
    z_offset : float
        Offset along Z for the whole histogram (for scene positioning).
    """

    bins: int | Sequence[float] = 20
    density: bool = False
    log_scale: bool = False
    orientation: str = "xy"

    bar_width_ratio: float = 0.80
    bar_depth: float = 0.40
    color_palette: list[ManimColor] | None = None
    use_uniform_color: bool = False
    uniform_color: ManimColor = BLUE_E
    bar_opacity: float = 0.92
    edge_stroke_width: float = 0.8
    edge_opacity: float = 0.45

    show_kde: bool = True
    kde_bandwidth: float | str = "scott"
    kde_color: ManimColor = ManimColor("#00E5FF")
    kde_stroke_width: float = 2.5

    show_floor_grid: bool = True
    floor_grid_lines: int = 10
    floor_color: ManimColor = GRAY_C
    floor_opacity: float = 0.18

    show_mean: bool = True
    show_median: bool = True
    show_sigma_band: bool = True

    show_bin_labels: bool = True
    label_font_size: int = 18
    label_precision: int = 1

    axes_x_label: str = "Value"
    axes_y_label: str = "Frequency"
    axes_color: ManimColor = GRAY_B
    tick_length: float = 0.12

    x_scale: float = 7.0
    y_scale: float = 3.5
    z_offset: float = 0.0


# ---------------------------------------------------------------------------
# Bar prism (3 faces: front, side, top)
# ---------------------------------------------------------------------------

class _BarPrism(VGroup):
    """A single 3D histogram bar rendered as three shaded quads.

    Manim's built-in ``Prism`` does not support per-face coloring for
    proper lighting simulation; we draw three explicit parallelograms
    instead so front / right-side / top faces can carry distinct shades.

    Coordinate system (right-hand, same as Manim's ThreeDAxes):
        x  → left to right  (bin position)
        y  → bottom to top  (frequency / height)
        z  → front to back  (depth axis, perspective)

    The bar occupies:
        x : [x0, x0 + w]
        y : [0,  h     ]
        z : [z0, z0 + d]
    """

    def __init__(
        self,
        x0: float,
        width: float,
        height: float,
        depth: float,
        z0: float,
        base_color: ManimColor,
        opacity: float = 0.92,
        edge_stroke_width: float = 0.8,
        edge_opacity: float = 0.45,
    ):
        super().__init__()
        self.x0     = x0
        self.width  = width
        self.height = height
        self.depth  = depth
        self.z0     = z0

        # Shaded variants
        front_color = base_color
        side_color  = _darken(base_color, FACE_DARKEN)
        top_color   = _lighten(base_color, FACE_LIGHTEN)

        # 8 corners of the box
        #   bottom-front-left  = A,  bottom-front-right = B
        #   top-front-left     = C,  top-front-right    = D
        #   bottom-back-left   = E,  bottom-back-right  = F
        #   top-back-left      = G,  top-back-right     = H
        A = np.array([x0,         0,      z0      ])
        B = np.array([x0 + width, 0,      z0      ])
        C = np.array([x0,         height, z0      ])
        D = np.array([x0 + width, height, z0      ])
        E = np.array([x0,         0,      z0 + depth])
        F = np.array([x0 + width, 0,      z0 + depth])
        G = np.array([x0,         height, z0 + depth])
        H = np.array([x0 + width, height, z0 + depth])

        def _face(pts: list[np.ndarray], color: ManimColor) -> Polygon:
            p = Polygon(*pts, color=color)
            p.set_fill(color=color, opacity=opacity)
            p.set_stroke(
                color=_darken(color, 0.55),
                width=edge_stroke_width,
                opacity=edge_opacity,
            )
            return p

        # Front face  (ABDC – viewed straight on)
        self.front = _face([A, B, D, C], front_color)
        # Right side  (BFHD)
        self.side  = _face([B, F, H, D], side_color)
        # Top face    (CDHG)
        self.top   = _face([C, D, H, G], top_color)

        self.add(self.front, self.side, self.top)

        # Store corners for external animation use
        self.corners = {"A": A, "B": B, "C": C, "D": D,
                        "E": E, "F": F, "G": G, "H": H}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def top_center(self) -> np.ndarray:
        """World position of the center of the top face."""
        C, H = self.corners["C"], self.corners["H"]
        return (C + H) / 2

    def set_bar_height(self, new_height: float) -> None:
        """Resize the bar in-place (used for morphing animations)."""
        # Rebuild the face geometry from stored base params
        self.height = new_height
        C = np.array([self.x0,              new_height, self.z0              ])
        D = np.array([self.x0 + self.width, new_height, self.z0              ])
        G = np.array([self.x0,              new_height, self.z0 + self.depth ])
        H = np.array([self.x0 + self.width, new_height, self.z0 + self.depth ])
        A, B, E, F = (self.corners[k] for k in ("A", "B", "E", "F"))

        new_pts = {
            "front": [A, B, D, C],
            "side":  [B, F, H, D],
            "top":   [C, D, H, G],
        }
        for face_name, pts in new_pts.items():
            face: Polygon = getattr(self, face_name)
            face.set_points_as_corners([*pts, pts[0]])

        self.corners.update({"C": C, "D": D, "G": G, "H": H})


# ---------------------------------------------------------------------------
# Floor grid
# ---------------------------------------------------------------------------

class _FloorGrid(VGroup):
    """Semi-transparent grid plane at y = 0 beneath the histogram bars."""

    def __init__(
        self,
        x_range: tuple[float, float],
        z_range: tuple[float, float],
        n_lines: int = 10,
        color: ManimColor = GRAY_C,
        opacity: float = 0.18,
    ):
        super().__init__()
        x0, x1 = x_range
        z0, z1 = z_range

        for i in range(n_lines + 1):
            t = i / n_lines
            # Lines along x direction
            xv = x0 + t * (x1 - x0)
            line = Line3D(
                start=np.array([xv, 0, z0]),
                end=np.array([xv, 0, z1]),
                color=color,
                stroke_width=0.6,
            ).set_opacity(opacity)
            self.add(line)
            # Lines along z direction
            zv = z0 + t * (z1 - z0)
            line = Line3D(
                start=np.array([x0, 0, zv]),
                end=np.array([x1, 0, zv]),
                color=color,
                stroke_width=0.6,
            ).set_opacity(opacity)
            self.add(line)


# ---------------------------------------------------------------------------
# KDE overlay
# ---------------------------------------------------------------------------

class _KDE3D(VMobject):
    """Kernel Density Estimate curve rendered as a ParametricFunction."""

    def __init__(
        self,
        data: np.ndarray,
        x_range: tuple[float, float],
        x_scale: float,
        y_scale: float,
        max_y: float,
        z_front: float,
        bandwidth: float | str = "scott",
        color: ManimColor = ManimColor("#00E5FF"),
        stroke_width: float = 2.5,
        density: bool = True,
    ):
        super().__init__()
        self.set_stroke(color=color, width=stroke_width)
        self.set_fill(opacity=0)

        kde = scipy_stats.gaussian_kde(data, bw_method=bandwidth)
        data_min, data_max = x_range
        data_span = data_max - data_min

        # KDE area normalisation factor to match bar heights
        if density:
            scale_factor = y_scale
        else:
            # When using counts, integrate and scale to max bar height
            xs_check = np.linspace(data_min, data_max, 500)
            ys_check = kde(xs_check)
            scale_factor = max_y / ys_check.max() if ys_check.max() > 0 else y_scale

        def kde_point(t: float) -> np.ndarray:
            x_data = data_min + t * data_span
            y_val  = float(kde(x_data)[0]) * scale_factor
            x_3d   = (t - 0.5) * x_scale
            return np.array([x_3d, y_val, z_front])

        curve = ParametricFunction(
            kde_point,
            t_range=[0, 1, 0.002],
            color=color,
            stroke_width=stroke_width,
        )
        self.add(curve)


# ---------------------------------------------------------------------------
# Statistical marker objects
# ---------------------------------------------------------------------------

class _StatPlane(VGroup):
    """A thin vertical polygon marking mean or median on the histogram."""

    def __init__(
        self,
        x_pos: float,          # rendered X coordinate
        y_max: float,          # height of the plane
        z_range: tuple[float, float],
        color: ManimColor,
        label_text: str,
        stroke_width: float = 2.0,
        opacity: float = 0.80,
        font_size: int = 24,
    ):
        super().__init__()
        z0, z1 = z_range

        plane = Polygon(
            np.array([x_pos, 0,     z0]),
            np.array([x_pos, 0,     z1]),
            np.array([x_pos, y_max, z1]),
            np.array([x_pos, y_max, z0]),
            color=color,
        )
        plane.set_fill(color=color, opacity=0.12)
        plane.set_stroke(color=color, width=stroke_width, opacity=opacity)
        self.add(plane)

        label = MathTex(label_text, color=color, font_size=font_size)
        label.move_to(np.array([x_pos, y_max + 0.30, (z0 + z1) / 2]))
        self.add(label)


class _SigmaBand(VGroup):
    """A translucent slab between mean ± k*sigma."""

    def __init__(
        self,
        x_left: float,
        x_right: float,
        y_max: float,
        z_range: tuple[float, float],
        color: ManimColor = SIGMA_COLOR,
        opacity: float = 0.10,
    ):
        super().__init__()
        z0, z1 = z_range

        # Six faces of the slab; only render top + front for clarity
        corners = [
            np.array([x_left,  0,     z0]),
            np.array([x_right, 0,     z0]),
            np.array([x_right, y_max, z0]),
            np.array([x_left,  y_max, z0]),
        ]
        front = Polygon(*corners, color=color)
        front.set_fill(color=color, opacity=opacity)
        front.set_stroke(width=0)
        self.add(front)

        # Top cap
        top_corners = [
            np.array([x_left,  y_max, z0]),
            np.array([x_right, y_max, z0]),
            np.array([x_right, y_max, z1]),
            np.array([x_left,  y_max, z1]),
        ]
        top = Polygon(*top_corners, color=color)
        top.set_fill(color=color, opacity=opacity * 0.8)
        top.set_stroke(color=color, width=0.5, opacity=0.3)
        self.add(top)


# ---------------------------------------------------------------------------
# Axis tick marks
# ---------------------------------------------------------------------------

class _AxisTicks(VGroup):
    """Tick marks + numeric labels along a histogram axis."""

    def __init__(
        self,
        positions: Sequence[float],   # world-space coordinates
        axis: str = "x",              # "x" or "y"
        z_pos: float = 0.0,
        tick_length: float = 0.12,
        label_values: Sequence[float] | None = None,
        color: ManimColor = GRAY_B,
        font_size: int = 18,
        precision: int = 1,
    ):
        super().__init__()
        for i, pos in enumerate(positions):
            if axis == "x":
                start = np.array([pos, 0,            z_pos])
                end   = np.array([pos, -tick_length, z_pos])
            else:
                start = np.array([0,             pos, z_pos])
                end   = np.array([-tick_length,  pos, z_pos])

            tick = Line3D(start=start, end=end, color=color, stroke_width=0.8)
            self.add(tick)

            if label_values is not None:
                val  = label_values[i]
                text = f"{val:.{precision}f}"
                lbl  = Text(text, color=color, font_size=font_size)
                if axis == "x":
                    lbl.move_to(end + DOWN * 0.22)
                else:
                    lbl.move_to(end + LEFT * 0.30)
                self.add(lbl)


# ---------------------------------------------------------------------------
# Main Histogram3D class
# ---------------------------------------------------------------------------

class Histogram3D(VGroup):
    """A detailed 3D histogram for Manim statistics animations.

    Usage
    -----
    >>> import numpy as np
    >>> from manim import *
    >>> from manim_stats.charts.histogram3d import Histogram3D, HistogramConfig
    >>>
    >>> class MyScene(ThreeDScene):
    ...     def construct(self):
    ...         data = np.random.normal(loc=0, scale=1, size=500)
    ...         cfg  = HistogramConfig(bins=25, show_kde=True, show_sigma_band=True)
    ...         hist = Histogram3D(data, config=cfg)
    ...         self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
    ...         self.play(hist.animate_grow())

    Parameters
    ----------
    data : array-like
        1-D array of observations.
    config : HistogramConfig, optional
        Visual and statistical configuration.  Defaults to ``HistogramConfig()``.
    """

    def __init__(
        self,
        data: Sequence[float] | np.ndarray,
        config: HistogramConfig | None = None,
    ):
        super().__init__()
        self.cfg  = config or HistogramConfig()
        self.data = np.asarray(data, dtype=float)
        self._build()

    # ------------------------------------------------------------------
    # Private build pipeline
    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg  = self.cfg
        data = self.data

        # ---- 1. Compute histogram counts / density -------------------
        counts, bin_edges = np.histogram(data, bins=cfg.bins, density=False)
        n_bins = len(counts)

        if cfg.density:
            bin_widths = np.diff(bin_edges)
            heights_raw = counts / (counts.sum() * bin_widths)
        else:
            heights_raw = counts.astype(float)

        if cfg.log_scale:
            heights_raw = np.where(heights_raw > 0, np.log10(heights_raw + 1), 0)

        h_max  = heights_raw.max() if heights_raw.max() > 0 else 1.0
        h_norm = heights_raw / h_max  # normalized [0, 1]

        # ---- 2. Coordinate mapping -----------------------------------
        data_min, data_max = bin_edges[0], bin_edges[-1]
        data_span = data_max - data_min

        def data_to_x(val: float) -> float:
            """Map a data-space value to Manim X coordinate."""
            return ((val - data_min) / data_span - 0.5) * cfg.x_scale

        bar_w_data = (bin_edges[1] - bin_edges[0]) * cfg.bar_width_ratio
        bar_w = bar_w_data / data_span * cfg.x_scale

        z0 = cfg.z_offset
        x_left  = data_to_x(data_min)
        x_right = data_to_x(data_max)
        z_range = (z0, z0 + cfg.bar_depth)

        # ---- 3. Floor grid -------------------------------------------
        if cfg.show_floor_grid:
            grid = _FloorGrid(
                x_range=(x_left  - 0.2, x_right + 0.2),
                z_range=(z0 - 0.1, z0 + cfg.bar_depth + 0.1),
                n_lines=cfg.floor_grid_lines,
                color=cfg.floor_color,
                opacity=cfg.floor_opacity,
            )
            self.add(grid)
            self.floor_grid = grid

        # ---- 4. Bars -------------------------------------------------
        self.bars: list[_BarPrism] = []
        palette = cfg.color_palette

        for i in range(n_bins):
            h_world = h_norm[i] * cfg.y_scale
            if h_world < 1e-6:
                continue

            # Bar left edge in data space
            x_center = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            x_bar_center = data_to_x(x_center)
            x_bar_left   = x_bar_center - bar_w / 2

            # Color from heatmap
            if cfg.use_uniform_color:
                color = cfg.uniform_color
            else:
                color = _freq_color(
                    heights_raw[i], heights_raw.min(), heights_raw.max(), palette
                )

            bar = _BarPrism(
                x0=x_bar_left,
                width=bar_w,
                height=h_world,
                depth=cfg.bar_depth,
                z0=z0,
                base_color=color,
                opacity=cfg.bar_opacity,
                edge_stroke_width=cfg.edge_stroke_width,
                edge_opacity=cfg.edge_opacity,
            )
            self.bars.append(bar)
            self.add(bar)

        # ---- 5. Statistical overlays ---------------------------------
        # Compute stats in Manim X space
        mean_x   = data_to_x(np.mean(data))
        median_x = data_to_x(np.median(data))
        std      = np.std(data)
        sigma_x_left  = data_to_x(np.mean(data) - std)
        sigma_x_right = data_to_x(np.mean(data) + std)

        y_stat = cfg.y_scale * 1.05

        if cfg.show_sigma_band:
            band = _SigmaBand(
                x_left=sigma_x_left,
                x_right=sigma_x_right,
                y_max=y_stat,
                z_range=z_range,
                color=SIGMA_COLOR,
                opacity=0.10,
            )
            self.add(band)
            self.sigma_band = band

        if cfg.show_mean:
            mean_plane = _StatPlane(
                x_pos=mean_x,
                y_max=y_stat,
                z_range=z_range,
                color=MEAN_COLOR,
                label_text=r"\mu",
                font_size=cfg.label_font_size + 4,
            )
            self.add(mean_plane)
            self.mean_plane = mean_plane

        if cfg.show_median:
            median_plane = _StatPlane(
                x_pos=median_x,
                y_max=y_stat,
                z_range=z_range,
                color=MEDIAN_COLOR,
                label_text=r"\tilde{x}",
                font_size=cfg.label_font_size + 4,
            )
            self.add(median_plane)
            self.median_plane = median_plane

        # ---- 6. KDE overlay ------------------------------------------
        if cfg.show_kde and len(data) > 1:
            h_max_world = h_norm.max() * cfg.y_scale
            kde_obj = _KDE3D(
                data=data,
                x_range=(data_min, data_max),
                x_scale=cfg.x_scale,
                y_scale=cfg.y_scale,
                max_y=h_max_world,
                z_front=z0,
                bandwidth=cfg.kde_bandwidth,
                color=cfg.kde_color,
                stroke_width=cfg.kde_stroke_width,
                density=cfg.density,
            )
            self.add(kde_obj)
            self.kde = kde_obj

        # ---- 7. Axis ticks + labels ----------------------------------
        n_x_ticks = min(n_bins + 1, 11)
        tick_indices = np.round(np.linspace(0, n_bins, n_x_ticks)).astype(int)
        tick_indices = np.unique(np.clip(tick_indices, 0, n_bins))
        x_tick_positions = [data_to_x(bin_edges[i]) for i in tick_indices]
        x_tick_values    = [bin_edges[i] for i in tick_indices]

        x_ticks = _AxisTicks(
            positions=x_tick_positions,
            axis="x",
            z_pos=z0,
            tick_length=cfg.tick_length,
            label_values=x_tick_values,
            color=cfg.axes_color,
            font_size=cfg.label_font_size,
            precision=cfg.label_precision,
        )
        self.add(x_ticks)

        n_y_ticks = 5
        y_tick_vals_raw = np.linspace(0, h_max, n_y_ticks)
        y_tick_positions = (y_tick_vals_raw / h_max) * cfg.y_scale

        y_ticks = _AxisTicks(
            positions=y_tick_positions,
            axis="y",
            z_pos=z0,
            tick_length=cfg.tick_length,
            label_values=y_tick_vals_raw,
            color=cfg.axes_color,
            font_size=cfg.label_font_size,
            precision=0 if not cfg.density else 3,
        )
        self.add(y_ticks)

        # Axis arrow lines
        x_axis = Arrow3D(
            start=np.array([x_left - 0.3,  0, z0]),
            end  =np.array([x_right + 0.3, 0, z0]),
            color=cfg.axes_color,
            stroke_width=1.5,
        )
        y_axis = Arrow3D(
            start=np.array([x_left - 0.3, 0,                   z0]),
            end  =np.array([x_left - 0.3, cfg.y_scale * 1.15,  z0]),
            color=cfg.axes_color,
            stroke_width=1.5,
        )
        self.add(x_axis, y_axis)

        # Axis text labels
        x_label = Text(cfg.axes_x_label, color=cfg.axes_color, font_size=cfg.label_font_size + 4)
        x_label.move_to(np.array([0, -0.55, z0]))
        y_label = Text(cfg.axes_y_label, color=cfg.axes_color, font_size=cfg.label_font_size + 4)
        y_label.move_to(np.array([x_left - 0.75, cfg.y_scale / 2, z0]))
        self.add(x_label, y_label)

        # ---- 8. Floating bar labels ----------------------------------
        if cfg.show_bin_labels:
            self.bar_labels = VGroup()
            for i, bar in enumerate(self.bars):
                val = heights_raw[np.searchsorted(bin_edges, data_to_x(bar.x0) / cfg.x_scale * data_span + data_min + data_span / n_bins / 2) - 1] \
                    if not cfg.density else heights_raw[i]
                if cfg.log_scale:
                    display_val = 10 ** heights_raw[i] - 1
                else:
                    display_val = heights_raw[i]
                fmt = f"{display_val:.{cfg.label_precision}f}" if cfg.density else f"{int(counts[i])}"
                lbl = Text(fmt, color=WHITE, font_size=cfg.label_font_size - 2)
                top = bar.top_center
                lbl.move_to(top + UP * 0.20)
                self.bar_labels.add(lbl)
            self.add(self.bar_labels)

        # ---- Store metadata for animations --------------------------
        self._data_min   = data_min
        self._data_max   = data_max
        self._data_span  = data_span
        self._data_to_x  = data_to_x
        self._h_norm     = h_norm
        self._h_max      = h_max
        self._counts     = counts
        self._bin_edges  = bin_edges
        self._n_bins     = n_bins
        self._bar_w      = bar_w

    # ------------------------------------------------------------------
    # Public animation helpers (return Animation objects for use in self.play())
    # ------------------------------------------------------------------

    def animate_grow(
        self,
        lag_ratio: float = 0.05,
        run_time: float = 2.5,
    ) -> LaggedStart:
        """Grow all bars from the floor upward with a staggered delay.

        Returns a ``LaggedStart`` animation suitable for::

            self.play(hist.animate_grow())
        """
        anims = []
        for bar in self.bars:
            # Temporarily shrink bar to zero height for the grow-in
            flat_bar = bar.copy()
            flat_bar.set_bar_height(1e-5)
            anims.append(Transform(flat_bar, bar, run_time=run_time))
        return LaggedStart(*[GrowFromPoint(b, point=b.corners["A"]) for b in self.bars],
                           lag_ratio=lag_ratio, run_time=run_time)

    def animate_sweep_bins(
        self,
        run_time: float = 3.0,
        lag_ratio: float = 0.08,
    ) -> LaggedStart:
        """Fade in bars left-to-right, one after another."""
        return LaggedStart(
            *[FadeIn(bar, shift=UP * 0.15) for bar in self.bars],
            lag_ratio=lag_ratio,
            run_time=run_time,
        )

    def animate_highlight_bin(
        self,
        bin_index: int,
        highlight_color: ManimColor = YELLOW,
        run_time: float = 0.4,
    ) -> AnimationGroup:
        """Flash a single bar to draw attention to it.

        Parameters
        ----------
        bin_index : int
            0-based index into ``self.bars``.
        """
        if bin_index < 0 or bin_index >= len(self.bars):
            raise IndexError(f"bin_index {bin_index} out of range [0, {len(self.bars) - 1}]")
        bar = self.bars[bin_index]
        original_colors = [
            bar.front.get_fill_color(),
            bar.side.get_fill_color(),
            bar.top.get_fill_color(),
        ]
        flash_in = AnimationGroup(
            bar.front.animate(run_time=run_time).set_fill(color=highlight_color, opacity=1.0),
            bar.side.animate(run_time=run_time).set_fill(color=_darken(highlight_color, FACE_DARKEN), opacity=1.0),
            bar.top.animate(run_time=run_time).set_fill(color=_lighten(highlight_color, FACE_LIGHTEN), opacity=1.0),
        )
        flash_out = AnimationGroup(
            bar.front.animate(run_time=run_time).set_fill(color=original_colors[0], opacity=self.cfg.bar_opacity),
            bar.side.animate(run_time=run_time).set_fill(color=original_colors[1], opacity=self.cfg.bar_opacity),
            bar.top.animate(run_time=run_time).set_fill(color=original_colors[2], opacity=self.cfg.bar_opacity),
        )
        return Succession(flash_in, flash_out)

    def animate_morph_to(
        self,
        new_data: np.ndarray,
        run_time: float = 2.0,
    ) -> AnimationGroup:
        """Morph bar heights to a new dataset (same bin structure).

        Use when updating the histogram live during a scene, e.g. to show
        convergence of the CLT::

            new_samples = np.concatenate([old_data, np.random.normal(size=100)])
            self.play(hist.animate_morph_to(new_samples))
        """
        new_counts, _ = np.histogram(new_data, bins=self._bin_edges, density=False)
        if self.cfg.density:
            bw = np.diff(self._bin_edges)
            new_heights_raw = new_counts / (new_counts.sum() * bw)
        else:
            new_heights_raw = new_counts.astype(float)

        if self.cfg.log_scale:
            new_heights_raw = np.where(new_heights_raw > 0, np.log10(new_heights_raw + 1), 0)

        new_h_max  = new_heights_raw.max() if new_heights_raw.max() > 0 else 1.0
        new_h_norm = new_heights_raw / new_h_max

        anims = []
        for i, bar in enumerate(self.bars):
            if i >= len(new_h_norm):
                continue
            target_height = new_h_norm[i] * self.cfg.y_scale
            # We use the bar's faces individually to animate height transitions
            C_new = bar.corners["A"] + UP * target_height
            D_new = bar.corners["B"] + UP * target_height
            G_new = bar.corners["E"] + UP * target_height
            H_new = bar.corners["F"] + UP * target_height

            new_front = Polygon(bar.corners["A"], bar.corners["B"], D_new, C_new)
            new_front.match_style(bar.front)
            new_side  = Polygon(bar.corners["B"], bar.corners["F"], H_new, D_new)
            new_side.match_style(bar.side)
            new_top   = Polygon(C_new, D_new, H_new, G_new)
            new_top.match_style(bar.top)

            anims.extend([
                Transform(bar.front, new_front, run_time=run_time),
                Transform(bar.side,  new_side,  run_time=run_time),
                Transform(bar.top,   new_top,   run_time=run_time),
            ])

        return AnimationGroup(*anims)

    def animate_reveal_stats(
        self,
        run_time: float = 1.5,
    ) -> LaggedStart:
        """Fade in the statistical overlays (sigma band → mean → median)."""
        targets = []
        if hasattr(self, "sigma_band"):
            targets.append(self.sigma_band)
        if hasattr(self, "mean_plane"):
            targets.append(self.mean_plane)
        if hasattr(self, "median_plane"):
            targets.append(self.median_plane)
        if hasattr(self, "kde"):
            targets.append(self.kde)
        return LaggedStart(
            *[FadeIn(t, run_time=run_time * 0.8) for t in targets],
            lag_ratio=0.30,
            run_time=run_time,
        )

    # ------------------------------------------------------------------
    # Convenience class methods (pre-configured common distributions)
    # ------------------------------------------------------------------

    @classmethod
    def from_normal(
        cls,
        n: int = 500,
        loc: float = 0.0,
        scale: float = 1.0,
        seed: int | None = None,
        config: HistogramConfig | None = None,
    ) -> "Histogram3D":
        """Return a histogram of Gaussian samples."""
        rng  = np.random.default_rng(seed)
        data = rng.normal(loc=loc, scale=scale, size=n)
        return cls(data, config=config)

    @classmethod
    def from_exponential(
        cls,
        n: int = 500,
        scale: float = 1.0,
        seed: int | None = None,
        config: HistogramConfig | None = None,
    ) -> "Histogram3D":
        """Return a histogram of Exponential samples."""
        rng  = np.random.default_rng(seed)
        data = rng.exponential(scale=scale, size=n)
        return cls(data, config=config)

    @classmethod
    def from_bimodal(
        cls,
        n: int = 500,
        loc1: float = -2.0,
        loc2: float = 2.0,
        scale: float = 0.8,
        seed: int | None = None,
        config: HistogramConfig | None = None,
    ) -> "Histogram3D":
        """Return a histogram from a bimodal (mixture of two normals) distribution."""
        rng  = np.random.default_rng(seed)
        n1   = n // 2
        n2   = n - n1
        data = np.concatenate([
            rng.normal(loc=loc1, scale=scale, size=n1),
            rng.normal(loc=loc2, scale=scale, size=n2),
        ])
        return cls(data, config=config)

    @classmethod
    def from_uniform(
        cls,
        n: int = 500,
        low: float = -3.0,
        high: float = 3.0,
        seed: int | None = None,
        config: HistogramConfig | None = None,
    ) -> "Histogram3D":
        """Return a histogram of Uniform samples."""
        rng  = np.random.default_rng(seed)
        data = rng.uniform(low=low, high=high, size=n)
        return cls(data, config=config)