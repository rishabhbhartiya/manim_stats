"""
manim_stats/distributions/base_dist.py
========================================
Master base layer for every distribution asset in the Manim Statistics Extension.

All 18+ distribution classes (Normal, t, Chi², Binomial, Poisson, …) inherit
from either ``ContinuousDistribution3D`` or ``DiscreteDistribution3D``, both
of which inherit from ``BaseDistribution3D``.

Architecture
------------
BaseDistribution3D
    ├── ParameterTrackerSystem    — ValueTracker per param, live-updater wiring
    ├── DistributionGeometryLayer — body curve / bars + baseline
    ├── FillRegionSystem          — shaded probability regions with area labels
    ├── MomentMarkerSystem        — mean / median / mode arrows + σ bands
    ├── StatsAnnotationPanel      — floating panel: μ, σ, skew, kurt, support
    ├── PercentileProbe3D         — vertical probe with CDF readout
    ├── ComparisonOverlay         — KL divergence, overlay second distribution
    └── DistributionDrawingLayer  — extra curves added by user (e.g. fitted KDE)

    ├── ContinuousDistribution3D  — smooth ParametricFunction curve + fill polygon
    └── DiscreteDistribution3D    — PMF bar forest + stem lines + bar labels

RepresentationMode
------------------
Every distribution can be rendered as any of:
    PDF / PMF   — primary density / mass
    CDF         — cumulative distribution function
    SF          — survival function (1 - CDF)
    LOG_PDF     — log-density (useful for comparing tails)
    HAZARD      — hazard rate h(x) = f(x)/S(x)

Switching modes triggers a smooth ``Transform`` animation.

Parameter animation
-------------------
Every distribution parameter is a ``ValueTracker``.  Attach Manim updaters or
use ``animate_param(name, target, run_time)`` to smoothly vary parameters::

    normal = NormalDistribution3D(axes, mu=0, sigma=1)
    scene.play(normal.animate_param("sigma", 2.0, run_time=3))

Shading API
-----------
::

    dist.shade_region("reject_right", lo=1.96,  hi=np.inf)
    dist.shade_region("reject_left",  lo=-np.inf, hi=-1.96)
    dist.shade_central("ci_95", coverage=0.95)
    dist.shade_by_sigma("1_sigma", n_sigma=1)
    dist.remove_shade("reject_right")

Probing API
-----------
::

    dist.probe_at("x0", x=1.645)          # vertical line + CDF label
    dist.probe_quantile("q95", p=0.95)    # find x such that CDF(x) = p
"""

from __future__ import annotations

import math
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Literal,
    Optional, Sequence, Tuple, Union,
)

import numpy as np
from numpy.typing import ArrayLike

from manim import (
    VGroup, VMobject, Mobject,
    Line3D, Arrow3D, Dot3D,
    Line, DashedLine, Polygon,
    Text, MathTex, DecimalNumber,
    RoundedRectangle, Rectangle, SurroundingRectangle,
    ParametricFunction,
    Animation, AnimationGroup, Succession,
    Create, FadeIn, FadeOut, Write, Transform,
    DrawBorderThenFill, GrowArrow, GrowFromCenter,
    UpdateFromAlphaFunc,
    ORIGIN, UP, DOWN, LEFT, RIGHT, IN, OUT,
    DEGREES, PI, TAU,
    WHITE, BLACK, GRAY,
    interpolate_color, smooth, there_and_back,
    ValueTracker, always_redraw, rate_functions,
)

from ..core.base import (
    StatsObject3D, StatsChart3D,
    StatsTheme, StatsColorPalette,
    MaterialConfig, MaterialApplicator,
    LabelConfig, LabelAnchor,
    AnimationConfig, BuildStyle, DataUpdateMode,
    HighlightStyle, HighlightSystem,
    BoundData, ThemeMode,
)
from ..core.math_utils import (
    DistributionFunction, DistributionResult,
    compute_descriptive, area_under_curve, cdf_from_pdf,
    entropy, kl_divergence, format_stat_value,
    FloatArray,
)
from ..axes.axes3d import (
    StatsAxes3D, AxesConfig, AxisID, GridStyle,
    ReferenceLineConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  ENUMERATIONS & CONFIG DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationMode(Enum):
    PDF    = auto()   # probability density (or PMF for discrete)
    CDF    = auto()   # cumulative distribution function
    SF     = auto()   # survival function  1 - CDF
    LOG_PDF = auto()  # log-density
    HAZARD = auto()   # hazard rate f(x) / S(x)


class ShadeFillStyle(Enum):
    SOLID    = auto()   # flat fill
    GRADIENT = auto()   # colour gradient along the x-axis
    STRIPED  = auto()   # diagonal stripes


class MomentMarkerStyle(Enum):
    ARROW    = auto()   # Arrow3D pointing down to axis
    DASHED   = auto()   # dashed vertical line
    BOTH     = auto()   # arrow + dashed line


@dataclass
class DistributionCurveConfig:
    """Visual style for the main curve / bar body."""
    # Curve
    stroke_color:       Optional[str]  = None      # None → theme.primary
    stroke_width:       float          = 3.0
    stroke_opacity:     float          = 1.0
    n_sample_points:    int            = 400        # curve resolution
    use_smooth:         bool           = True

    # Fill under curve
    show_fill:          bool           = True
    fill_color:         Optional[str]  = None      # None → stroke_color
    fill_opacity:       float          = 0.15

    # Baseline
    show_baseline:      bool           = True
    baseline_color:     Optional[str]  = None      # None → theme.neutral
    baseline_width:     float          = 1.2
    baseline_opacity:   float          = 0.6

    # Z-depth for layering
    depth_offset:       float          = 0.01


@dataclass
class ShadeRegionConfig:
    """Style for one shaded probability region."""
    lo:              float              = -math.inf
    hi:              float              =  math.inf
    color:           Optional[str]      = None    # None → theme.positive
    fill_opacity:    float              = 0.38
    fill_style:      ShadeFillStyle     = ShadeFillStyle.SOLID
    stroke_width:    float              = 1.5
    stroke_opacity:  float              = 0.7
    # Label showing P(lo ≤ X ≤ hi)
    show_label:      bool               = True
    label_is_math:   bool               = True
    label_font_size: float              = 22
    label_pos:       Literal["auto", "inside", "above"] = "auto"
    label_decimals:  int                = 4
    gradient_colors: Optional[Tuple[str, str]] = None   # for GRADIENT style


@dataclass
class MomentMarkerConfig:
    """Style for mean / median / mode / sigma markers."""
    show_mean:        bool            = True
    show_median:      bool            = False
    show_mode:        bool            = False
    show_sigma_bands: bool            = True
    n_sigma_bands:    int             = 2        # show ±1σ, ±2σ, …
    marker_style:     MomentMarkerStyle = MomentMarkerStyle.DASHED
    mean_color:       Optional[str]   = None    # None → theme.accent
    median_color:     Optional[str]   = None    # None → theme.secondary
    mode_color:       Optional[str]   = None    # None → theme.positive
    sigma_colors:     Optional[List[str]] = None  # per band; cycles if short
    marker_opacity:   float           = 0.80
    marker_width:     float           = 1.8
    label_font_size:  float           = 20


@dataclass
class StatsAnnotationConfig:
    """Style for the floating stats annotation panel."""
    visible:          bool            = True
    show_mean:        bool            = True
    show_variance:    bool            = True
    show_std:         bool            = True
    show_skewness:    bool            = True
    show_kurtosis:    bool            = True
    show_entropy:     bool            = False
    show_support:     bool            = True
    font_size:        float           = 20
    panel_width:      float           = 2.8
    panel_color:      Optional[str]   = None   # None → theme.surface
    text_color:       Optional[str]   = None   # None → theme.text_primary
    corner_radius:    float           = 0.14
    padding:          float           = 0.20
    border_opacity:   float           = 0.5
    position:         np.ndarray      = field(
        default_factory=lambda: np.array([3.5, 1.5, 0.0]))


@dataclass
class ProbeConfig:
    """Configuration for a percentile / quantile probe."""
    color:            Optional[str]   = None   # None → theme.accent
    line_width:       float           = 2.0
    line_opacity:     float           = 0.9
    dot_radius:       float           = 0.07
    show_cdf_label:   bool            = True
    show_x_label:     bool            = True
    show_pdf_dot:     bool            = True
    label_font_size:  float           = 22
    label_decimals:   int             = 4
    drop_to_axis:     bool            = True   # dashed drop to x-axis


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETER TRACKER SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class ParameterTrackerSystem:
    """
    Manages one ``ValueTracker`` per distribution parameter.

    Each parameter name maps to a tracker + optional bounds + optional
    post-update callback.  When a tracker value changes, registered
    geometry-rebuild callbacks fire automatically via Manim updaters.

    Usage (inside a distribution class)
    ------------------------------------
    ::

        self.params = ParameterTrackerSystem()
        self.params.register("mu",    0.0, bounds=(-10, 10))
        self.params.register("sigma", 1.0, bounds=(0.001, 20))
        self.params.on_any_change(self._rebuild_callback)
    """

    def __init__(self) -> None:
        self._trackers:  Dict[str, ValueTracker] = {}
        self._bounds:    Dict[str, Tuple[float, float]] = {}
        self._callbacks: List[Callable]           = []

    # ── registration ──────────────────────────────────────────────────────

    def register(
        self,
        name:   str,
        value:  float,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> ValueTracker:
        """Register a new parameter and return its ``ValueTracker``."""
        tracker = ValueTracker(value)
        self._trackers[name] = tracker
        if bounds:
            self._bounds[name] = bounds
        return tracker

    # ── callbacks ─────────────────────────────────────────────────────────

    def on_any_change(self, callback: Callable[[], None]) -> None:
        """Register *callback* to fire whenever any tracked value changes."""
        self._callbacks.append(callback)

    def _fire_callbacks(self) -> None:
        for cb in self._callbacks:
            cb()

    # ── value access ──────────────────────────────────────────────────────

    def get(self, name: str) -> float:
        """Get the current value of parameter *name*."""
        if name not in self._trackers:
            raise KeyError(f"Unknown parameter: {name!r}")
        return self._trackers[name].get_value()

    def set(self, name: str, value: float) -> None:
        """Set parameter *name* to *value* with bounds clamping."""
        if name not in self._trackers:
            raise KeyError(f"Unknown parameter: {name!r}")
        if name in self._bounds:
            lo, hi = self._bounds[name]
            value  = float(np.clip(value, lo, hi))
        self._trackers[name].set_value(value)
        self._fire_callbacks()

    def tracker(self, name: str) -> ValueTracker:
        """Return the raw ``ValueTracker`` for *name*."""
        return self._trackers[name]

    def all_values(self) -> Dict[str, float]:
        """Return a snapshot dict {name: current_value}."""
        return {k: v.get_value() for k, v in self._trackers.items()}

    def all_trackers(self) -> Dict[str, ValueTracker]:
        return dict(self._trackers)

    # ── constraint validation ─────────────────────────────────────────────

    def validate(self, constraints: Dict[str, Callable[[float], bool]]) -> List[str]:
        """
        Run *constraints* against current values.
        Returns a list of error messages (empty if all pass).

        Example::

            errors = self.params.validate({
                "sigma": lambda v: v > 0,
                "p":     lambda v: 0 < v < 1,
            })
        """
        errors = []
        for name, predicate in constraints.items():
            val = self.get(name)
            if not predicate(val):
                errors.append(f"Parameter '{name}' = {val} violates constraint.")
        return errors


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FILL REGION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class FillRegionSystem(VGroup):
    """
    Manages named shaded probability regions drawn under a curve.

    Each region is a ``Polygon`` built from the curve points within
    [lo, hi] plus a bottom edge along the x-axis.  Gradient fills
    are achieved by stacking many thin vertical strips.

    Regions are keyed by name so they can be individually added,
    removed, or transformed.
    """

    def __init__(
        self,
        axes:    StatsAxes3D,
        theme:   StatsColorPalette,
    ) -> None:
        super().__init__()
        self._axes    = axes
        self._theme   = theme
        self._regions: Dict[str, VGroup] = {}

    # ── public API ────────────────────────────────────────────────────────

    def add_region(
        self,
        key:        str,
        result:     DistributionResult,
        cfg:        ShadeRegionConfig,
        mode:       RepresentationMode = RepresentationMode.PDF,
    ) -> VGroup:
        """
        Build and register a shaded region for *key*.
        Returns the VGroup for external animation.
        """
        if key in self._regions:
            self.remove_region(key)

        grp = self._build_region(result, cfg, mode)
        self._regions[key] = grp
        self.add(grp)
        return grp

    def remove_region(self, key: str) -> None:
        if key in self._regions:
            self.remove(self._regions.pop(key))

    def update_region(
        self,
        key:    str,
        result: DistributionResult,
        cfg:    ShadeRegionConfig,
        mode:   RepresentationMode = RepresentationMode.PDF,
    ) -> VGroup:
        """Rebuild and return an updated region."""
        self.remove_region(key)
        return self.add_region(key, result, cfg, mode)

    def get_region(self, key: str) -> Optional[VGroup]:
        return self._regions.get(key)

    def clear_all(self) -> None:
        for key in list(self._regions):
            self.remove_region(key)

    # ── geometry builders ─────────────────────────────────────────────────

    def _build_region(
        self,
        result: DistributionResult,
        cfg:    ShadeRegionConfig,
        mode:   RepresentationMode,
    ) -> VGroup:
        x  = result.x
        y  = self._select_y(result, mode)
        t  = self._theme
        c  = cfg.color or t.positive

        # Clip x to [lo, hi]
        lo = cfg.lo if not math.isinf(cfg.lo) else x.min()
        hi = cfg.hi if not math.isinf(cfg.hi) else x.max()
        lo = max(lo, x.min());  hi = min(hi, x.max())
        mask = (x >= lo) & (x <= hi)
        xc, yc = x[mask], y[mask]

        if len(xc) < 2:
            return VGroup()

        grp = VGroup()

        if cfg.fill_style == ShadeFillStyle.SOLID:
            poly = self._solid_polygon(xc, yc, c, cfg)
            grp.add(poly)

        elif cfg.fill_style == ShadeFillStyle.GRADIENT:
            colors = cfg.gradient_colors or (t.gradient_low, t.gradient_high)
            strips = self._gradient_strips(xc, yc, colors, cfg)
            grp.add(strips)

        elif cfg.fill_style == ShadeFillStyle.STRIPED:
            poly = self._solid_polygon(xc, yc, c, cfg)
            grp.add(poly)
            stripes = self._stripe_overlay(xc, yc, c, cfg)
            grp.add(stripes)

        # Probability label
        if cfg.show_label:
            prob = self._compute_prob(result, lo, hi)
            lbl  = self._build_prob_label(xc, yc, prob, cfg, c)
            if lbl is not None:
                grp.add(lbl)

        return grp

    def _solid_polygon(
        self,
        x:   FloatArray,
        y:   FloatArray,
        c:   str,
        cfg: ShadeRegionConfig,
    ) -> VMobject:
        """Build a filled polygon: curve points + bottom edge."""
        pts_top = [self._axes.c2p(xi, yi) for xi, yi in zip(x, y)]
        pts_bot = [self._axes.c2p(x[-1], 0.0), self._axes.c2p(x[0], 0.0)]
        all_pts  = pts_top + pts_bot

        poly = Polygon(*all_pts)
        poly.set_fill(c, opacity=cfg.fill_opacity)
        poly.set_stroke(c, width=cfg.stroke_width, opacity=cfg.stroke_opacity)
        return poly

    def _gradient_strips(
        self,
        x:      FloatArray,
        y:      FloatArray,
        colors: Tuple[str, str],
        cfg:    ShadeRegionConfig,
    ) -> VGroup:
        """
        Approximate gradient fill as N thin vertical strips, each with
        its own interpolated color.
        """
        n    = max(2, len(x) - 1)
        grp  = VGroup()
        for i in range(n):
            t     = i / max(n - 1, 1)
            c     = interpolate_color(colors[0], colors[1], t)
            strip_x = [x[i], x[i+1], x[i+1], x[i]]
            strip_y = [0.0,  0.0,    y[i+1], y[i]]
            pts   = [self._axes.c2p(xi, yi) for xi, yi in zip(strip_x, strip_y)]
            poly  = Polygon(*pts)
            poly.set_fill(c, opacity=cfg.fill_opacity)
            poly.set_stroke(width=0)
            grp.add(poly)
        return grp

    def _stripe_overlay(
        self,
        x:   FloatArray,
        y:   FloatArray,
        c:   str,
        cfg: ShadeRegionConfig,
    ) -> VGroup:
        """Diagonal stripe overlay (every other strip is filled)."""
        step = (x[-1] - x[0]) / max(len(x) // 4, 1)
        grp  = VGroup()
        i    = 0
        xp   = x[0]
        while xp < x[-1]:
            x0, x1 = xp, min(xp + step * 0.5, x[-1])
            mask0 = np.searchsorted(x, x0)
            mask1 = np.searchsorted(x, x1)
            xs  = x[mask0:mask1+1]
            ys  = y[mask0:mask1+1]
            if len(xs) >= 2:
                pts = ([self._axes.c2p(xi, yi) for xi, yi in zip(xs, ys)]
                       + [self._axes.c2p(xs[-1], 0), self._axes.c2p(xs[0], 0)])
                poly = Polygon(*pts)
                poly.set_fill(c, opacity=cfg.fill_opacity * 0.7)
                poly.set_stroke(width=0)
                grp.add(poly)
            xp += step
            i  += 1
        return grp

    def _compute_prob(
        self,
        result: DistributionResult,
        lo:     float,
        hi:     float,
    ) -> float:
        """Numerically integrate PDF over [lo, hi] for the label."""
        from scipy.interpolate import interp1d
        x, y = result.x, result.pdf
        if y is None:
            return float("nan")
        mask   = (x >= lo) & (x <= hi)
        xm, ym = x[mask], y[mask]
        if len(xm) < 2:
            return 0.0
        return float(np.trapezoid(ym, xm))

    def _build_prob_label(
        self,
        x:    FloatArray,
        y:    FloatArray,
        prob: float,
        cfg:  ShadeRegionConfig,
        c:    str,
    ) -> Optional[VMobject]:
        """Build a small text label showing "P = {prob:.4f}" inside the region."""
        x_mid = float(x.mean())
        y_mid = float(y.mean()) * 0.5
        pos   = self._axes.c2p(x_mid, y_mid)

        if math.isnan(prob):
            return None

        txt = f"P = {prob:.{cfg.label_decimals}f}"
        lbl = Text(txt, font_size=cfg.label_font_size, color=c)
        lbl.move_to(pos + UP * 0.15)
        return lbl

    @staticmethod
    def _select_y(
        result: DistributionResult,
        mode:   RepresentationMode,
    ) -> FloatArray:
        """Extract the correct y-array based on representation mode."""
        if mode == RepresentationMode.PDF and result.pdf is not None:
            return result.pdf
        elif mode == RepresentationMode.CDF and result.cdf is not None:
            return result.cdf
        elif mode == RepresentationMode.SF and result.sf is not None:
            return result.sf
        elif mode == RepresentationMode.LOG_PDF and result.log_pdf is not None:
            return result.log_pdf
        elif mode == RepresentationMode.HAZARD:
            sf   = np.clip(result.sf,  1e-300, None)
            pdf  = np.clip(result.pdf, 0.0,    None) if result.pdf is not None \
                   else np.zeros_like(result.x)
            return pdf / sf
        return result.pdf if result.pdf is not None else np.zeros_like(result.x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MOMENT MARKER SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class MomentMarkerSystem(VGroup):
    """
    Visual markers for distributional moments:
        • Mean   (labelled "μ")
        • Median (labelled "Md")
        • Mode   (labelled "Mo")
        • ±1σ, ±2σ, ±3σ bands

    All markers lie at the curve height (PDF(x)) with a drop line to
    the x-axis.  Sigma bands are semi-transparent vertical spans.
    """

    def __init__(
        self,
        axes:   StatsAxes3D,
        result: DistributionResult,
        cfg:    MomentMarkerConfig,
        theme:  StatsColorPalette,
    ) -> None:
        super().__init__()
        self._axes   = axes
        self._theme  = theme
        self._cfg    = cfg
        self._mobs:  Dict[str, VMobject] = {}
        self._build(result)

    # ── public API ────────────────────────────────────────────────────────

    def rebuild(self, result: DistributionResult) -> None:
        """Rebuild all markers from a new DistributionResult."""
        for mob in list(self._mobs.values()):
            self.remove(mob)
        self._mobs.clear()
        self._build(result)

    # ── internal build ────────────────────────────────────────────────────

    def _build(self, result: DistributionResult) -> None:
        cfg   = self._cfg
        theme = self._theme
        axes  = self._axes

        y_arr = result.pdf if result.pdf is not None else np.zeros_like(result.x)

        def _pdf_at(xv: float) -> float:
            """Interpolate PDF at xv."""
            if result.pdf is None:
                return 0.0
            return float(np.interp(xv, result.x, y_arr,
                                    left=0.0, right=0.0))

        # Mean
        if cfg.show_mean and result.mean is not None:
            c    = cfg.mean_color or theme.accent
            grp  = self._make_marker(result.mean, _pdf_at(result.mean), c,
                                     r"\mu", cfg)
            self._mobs["mean"] = grp
            self.add(grp)

        # Median
        if cfg.show_median:
            med   = float(np.interp(0.5, result.cdf, result.x)) \
                    if result.cdf is not None else result.mean or 0.0
            c     = cfg.median_color or theme.secondary
            grp   = self._make_marker(med, _pdf_at(med), c,
                                      r"\text{Md}", cfg)
            self._mobs["median"] = grp
            self.add(grp)

        # Mode
        if cfg.show_mode and result.pdf is not None:
            idx  = int(np.argmax(result.pdf))
            mode = float(result.x[idx])
            c    = cfg.mode_color or theme.positive
            grp  = self._make_marker(mode, float(result.pdf[idx]), c,
                                     r"\text{Mo}", cfg)
            self._mobs["mode"] = grp
            self.add(grp)

        # σ bands
        if cfg.show_sigma_bands and result.mean is not None \
                and result.variance is not None:
            sigma  = math.sqrt(max(result.variance, 0.0))
            mu     = result.mean
            s_cols = cfg.sigma_colors or [
                theme.distribution_palette[1],
                theme.distribution_palette[4],
                theme.distribution_palette[7],
            ]
            for k in range(1, cfg.n_sigma_bands + 1):
                lo = mu - k * sigma
                hi = mu + k * sigma
                c  = s_cols[(k - 1) % len(s_cols)]
                band = self._make_sigma_band(lo, hi, c, k, y_arr, result, axes)
                self._mobs[f"sigma_{k}"] = band
                self.add(band)

    def _make_marker(
        self,
        x_data:  float,
        y_data:  float,
        color:   str,
        label:   str,
        cfg:     MomentMarkerConfig,
    ) -> VGroup:
        """Build one vertical marker at (x_data, y_data)."""
        axes  = self._axes
        grp   = VGroup()

        scene_top = axes.c2p(x_data, y_data)
        scene_bot = axes.c2p(x_data, 0.0)

        if cfg.marker_style in (MomentMarkerStyle.DASHED, MomentMarkerStyle.BOTH):
            line = DashedLine(scene_bot, scene_top,
                              color=color,
                              stroke_width=cfg.marker_width * 1.3,
                              dash_length=0.10)
            line.set_opacity(cfg.marker_opacity)
            grp.add(line)

        if cfg.marker_style in (MomentMarkerStyle.ARROW, MomentMarkerStyle.BOTH):
            arr = Arrow3D(scene_top + UP * 0.3, scene_top,
                          color=color,
                          thickness=0.012,
                          tip_length=0.12)
            grp.add(arr)

        # Dot at curve height
        dot = Dot3D(scene_top, radius=0.055, color=color)
        grp.add(dot)

        # Label
        lbl = MathTex(label,
                      font_size=cfg.label_font_size,
                      color=color)
        lbl.move_to(scene_top + UP * 0.30 + RIGHT * 0.10)
        grp.add(lbl)

        return grp

    def _make_sigma_band(
        self,
        lo:     float,
        hi:     float,
        color:  str,
        k:      int,
        y_arr:  FloatArray,
        result: DistributionResult,
        axes:   StatsAxes3D,
    ) -> VGroup:
        """
        Semi-transparent vertical band between lo and hi,
        bounded above by the PDF curve.
        """
        x  = result.x
        mask = (x >= lo) & (x <= hi)
        xc = x[mask]
        yc = y_arr[mask]

        if len(xc) < 2:
            return VGroup()

        pts_top = [axes.c2p(xi, yi) for xi, yi in zip(xc, yc)]
        pts_bot = [axes.c2p(xc[-1], 0.0), axes.c2p(xc[0], 0.0)]
        poly    = Polygon(*(pts_top + pts_bot))
        poly.set_fill(color, opacity=0.12 + 0.04 / k)
        poly.set_stroke(color, width=1.0, opacity=0.35)

        # Tick lines at lo and hi
        grp = VGroup(poly)
        for xv in [lo, hi]:
            scene_bot = axes.c2p(xv, 0.0)
            scene_top = axes.c2p(xv, float(np.interp(xv, x, y_arr)))
            tick = DashedLine(scene_bot, scene_top,
                              color=color,
                              stroke_width=1.2,
                              dash_length=0.07)
            tick.set_opacity(0.50)
            grp.add(tick)

        # Label: "1σ", "2σ", …
        x_lbl = axes.c2p(hi + (result.x.max() - hi) * 0.15,
                          float(y_arr.max()) * 0.55)
        label = MathTex(f"{k}\\sigma",
                        font_size=16, color=color)
        label.move_to(x_lbl)
        grp.add(label)

        return grp


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STATS ANNOTATION PANEL
# ─────────────────────────────────────────────────────────────────────────────

class StatsAnnotationPanel(VGroup):
    """
    A floating info panel listing key distributional statistics:
    μ, σ², σ, skewness, kurtosis, entropy, support.

    The panel updates live when the underlying distribution changes.
    """

    def __init__(
        self,
        result: DistributionResult,
        cfg:    StatsAnnotationConfig,
        theme:  StatsColorPalette,
    ) -> None:
        super().__init__()
        self._cfg   = cfg
        self._theme = theme
        self._rows: Dict[str, VMobject] = {}
        self._bg:   Optional[VMobject]  = None
        if cfg.visible:
            self._build(result)

    # ── public API ────────────────────────────────────────────────────────

    def rebuild(self, result: DistributionResult) -> None:
        """Rebuild the entire panel from fresh data."""
        self.submobjects.clear()
        self._rows.clear()
        self._bg = None
        if self._cfg.visible:
            self._build(result)

    # ── internal ──────────────────────────────────────────────────────────

    def _build(self, result: DistributionResult) -> None:
        cfg   = self._cfg
        theme = self._theme
        tc    = cfg.text_color or theme.text_primary
        pc    = cfg.panel_color or theme.surface

        rows_data = self._collect_rows(result, cfg)
        row_mobs  = []

        for label_str, value_str in rows_data:
            lbl = Text(label_str, font_size=cfg.font_size, color=tc)
            val = Text(value_str, font_size=cfg.font_size, color=theme.accent)
            pair = VGroup(lbl, val)
            val.next_to(lbl, RIGHT, buff=0.15)
            row_mobs.append(pair)

        if not row_mobs:
            return

        # Stack rows vertically
        for i, mob in enumerate(row_mobs):
            mob.move_to(cfg.position + DOWN * i * (cfg.font_size / 60.0 + 0.05))
            self.add(mob)

        # Background panel
        if row_mobs:
            union_w = max(m.width  for m in row_mobs) + cfg.padding * 2
            union_h = sum(m.height for m in row_mobs) + \
                      len(row_mobs) * 0.05 + cfg.padding * 2
            mid     = row_mobs[len(row_mobs) // 2].get_center()
            bg = RoundedRectangle(
                corner_radius=cfg.corner_radius,
                width=max(union_w, cfg.panel_width),
                height=union_h,
            )
            bg.set_fill(pc, opacity=0.80)
            bg.set_stroke(theme.border, width=1.2, opacity=cfg.border_opacity)
            bg.move_to(mid)
            self.add_to_back(bg)
            self._bg = bg

    @staticmethod
    def _collect_rows(
        result: DistributionResult,
        cfg:    StatsAnnotationConfig,
    ) -> List[Tuple[str, str]]:
        rows = [("Distribution:", result.name)]
        if cfg.show_mean and result.mean is not None:
            rows.append(("μ =", format_stat_value(result.mean)))
        if cfg.show_variance and result.variance is not None:
            rows.append(("σ² =", format_stat_value(result.variance)))
        if cfg.show_std and result.variance is not None:
            rows.append(("σ =",  format_stat_value(math.sqrt(max(result.variance,0)))))
        if cfg.show_skewness and result.skewness is not None:
            rows.append(("Skew =", format_stat_value(result.skewness)))
        if cfg.show_kurtosis and result.kurtosis is not None:
            rows.append(("Kurt =", format_stat_value(result.kurtosis)))
        if cfg.show_support and result.support is not None:
            lo, hi = result.support
            lo_s   = f"{lo:.2g}" if not math.isinf(lo) else "-∞"
            hi_s   = f"{hi:.2g}" if not math.isinf(hi) else "+∞"
            rows.append(("Supp:", f"[{lo_s}, {hi_s}]"))
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PERCENTILE PROBE
# ─────────────────────────────────────────────────────────────────────────────

class PercentileProbe3D(VGroup):
    """
    An interactive probe that drops a vertical line at x = *value*,
    marks the PDF height with a dot, and annotates the CDF value.

    Two probe modes:
        • ``probe_at(x)``         — drop at x, read off CDF(x)
        • ``probe_quantile(p)``   — find x such that CDF(x) = p
    """

    def __init__(
        self,
        axes:    StatsAxes3D,
        result:  DistributionResult,
        x_value: float,
        cfg:     ProbeConfig,
        theme:   StatsColorPalette,
        key:     str = "probe",
    ) -> None:
        super().__init__()
        self._axes   = axes
        self._theme  = theme
        self._cfg    = cfg
        self._key    = key
        self._build(result, x_value)

    # ── build ─────────────────────────────────────────────────────────────

    def _build(self, result: DistributionResult, xv: float) -> None:
        axes  = self._axes
        cfg   = self._cfg
        theme = self._theme
        c     = cfg.color or theme.accent

        y_pdf  = float(np.interp(xv, result.x, result.pdf)) \
                 if result.pdf  is not None else 0.0
        y_cdf  = float(np.interp(xv, result.x, result.cdf)) \
                 if result.cdf  is not None else 0.0

        scene_pdf = axes.c2p(xv, y_pdf)
        scene_cdf = axes.c2p(xv, y_cdf)
        scene_bot = axes.c2p(xv, 0.0)
        y_axis_pt = axes.c2p(axes.x_range[0], y_cdf)

        # Vertical probe line
        probe_line = DashedLine(scene_bot, scene_pdf,
                                color=c,
                                stroke_width=cfg.line_width * 1.3,
                                dash_length=0.10)
        probe_line.set_opacity(cfg.line_opacity)
        self.add(probe_line)

        # PDF dot
        if cfg.show_pdf_dot:
            dot = Dot3D(scene_pdf, radius=cfg.dot_radius, color=c)
            dot.set_opacity(0.95)
            self.add(dot)

        # CDF dot
        cdf_dot = Dot3D(scene_cdf, radius=cfg.dot_radius * 0.8, color=c)
        cdf_dot.set_opacity(0.70)
        self.add(cdf_dot)

        # Drop to x-axis label
        if cfg.drop_to_axis:
            drop = DashedLine(scene_bot, scene_bot + DOWN * 0.25,
                              color=c, stroke_width=1.0,
                              dash_length=0.05)
            drop.set_opacity(0.5)
            self.add(drop)

        # X label
        if cfg.show_x_label:
            x_str = format_stat_value(xv, decimals=cfg.label_decimals)
            x_lbl = Text(f"x = {x_str}",
                          font_size=cfg.label_font_size, color=c)
            x_lbl.move_to(scene_bot + DOWN * 0.45)
            self.add(x_lbl)

        # CDF label
        if cfg.show_cdf_label:
            cdf_str = format_stat_value(y_cdf, decimals=cfg.label_decimals)
            cdf_lbl = Text(f"F(x) = {cdf_str}",
                            font_size=cfg.label_font_size, color=c)
            cdf_lbl.move_to(scene_cdf + RIGHT * 0.9 + UP * 0.05)
            # Horizontal leader from cdf_dot to label
            h_line = DashedLine(scene_cdf,
                                scene_cdf + RIGHT * 0.85,
                                color=c, stroke_width=1.0,
                                dash_length=0.06)
            h_line.set_opacity(0.55)
            self.add(h_line, cdf_lbl)

    # ── animation ─────────────────────────────────────────────────────────

    def animate_build(self, cfg: Optional[AnimationConfig] = None) -> Animation:
        cfg = cfg or AnimationConfig()
        return Succession(
            Create(self.submobjects[0] if self.submobjects else self,
                   run_time=cfg.run_time * 0.5),
            FadeIn(VGroup(*self.submobjects[1:]),
                   run_time=cfg.run_time * 0.5),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  COMPARISON OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonConfig:
    """Config for overlaying a second distribution."""
    show_kl_divergence:  bool  = True
    show_js_divergence:  bool  = False
    show_overlap_area:   bool  = True
    kl_label_pos:        np.ndarray = field(
        default_factory=lambda: np.array([3.5, -1.5, 0.0]))
    font_size:           float = 22


class ComparisonOverlay(VGroup):
    """
    Overlays a second distribution on the same axes and displays
    divergence statistics in a small panel.

    Usage
    -----
    ::

        comp = ComparisonOverlay(axes, result_A, result_B,
                                 color_A=BLUE, color_B=RED)
        scene.add(comp)
    """

    def __init__(
        self,
        axes:     StatsAxes3D,
        result_a: DistributionResult,
        result_b: DistributionResult,
        cfg:      ComparisonConfig,
        theme:    StatsColorPalette,
        color_a:  Optional[str] = None,
        color_b:  Optional[str] = None,
    ) -> None:
        super().__init__()
        self._axes  = axes
        self._theme = theme
        ca = color_a or theme.primary
        cb = color_b or theme.secondary

        # Overlap area
        if cfg.show_overlap_area and result_a.pdf is not None \
                and result_b.pdf is not None:
            x_common = np.linspace(
                max(result_a.x.min(), result_b.x.min()),
                min(result_a.x.max(), result_b.x.max()),
                300)
            pa = np.interp(x_common, result_a.x, result_a.pdf)
            pb = np.interp(x_common, result_b.x, result_b.pdf)
            y_min = np.minimum(pa, pb)

            pts_top = [axes.c2p(xi, yi) for xi, yi in zip(x_common, y_min)]
            pts_bot = [axes.c2p(x_common[-1], 0.0), axes.c2p(x_common[0], 0.0)]
            poly    = Polygon(*(pts_top + pts_bot))
            poly.set_fill(theme.positive, opacity=0.25)
            poly.set_stroke(theme.positive, width=1.0)
            self.add(poly)

        # Divergence panel
        lines = []
        if cfg.show_kl_divergence and result_a.pdf is not None \
                and result_b.pdf is not None:
            x_ev = np.linspace(
                max(result_a.x.min(), result_b.x.min()),
                min(result_a.x.max(), result_b.x.max()), 300)
            pa   = np.interp(x_ev, result_a.x, result_a.pdf) + 1e-300
            pb   = np.interp(x_ev, result_b.x, result_b.pdf) + 1e-300
            kl   = float(kl_divergence(pa / pa.sum(), pb / pb.sum()))
            lines.append(f"KL(A||B) = {kl:.4f}")

        if cfg.show_js_divergence and result_a.pdf is not None \
                and result_b.pdf is not None:
            from ..core.math_utils import js_divergence
            x_ev = np.linspace(
                max(result_a.x.min(), result_b.x.min()),
                min(result_a.x.max(), result_b.x.max()), 300)
            pa   = np.interp(x_ev, result_a.x, result_a.pdf) + 1e-300
            pb   = np.interp(x_ev, result_b.x, result_b.pdf) + 1e-300
            js   = float(js_divergence(pa, pb))
            lines.append(f"JSD = {js:.4f}")

        if lines:
            for i, line in enumerate(lines):
                lbl = Text(line, font_size=cfg.font_size,
                           color=theme.text_primary)
                lbl.move_to(cfg.kl_label_pos + DOWN * i * 0.45)
                self.add(lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BASE DISTRIBUTION 3D  (master abstract base)
# ─────────────────────────────────────────────────────────────────────────────

class BaseDistribution3D(StatsObject3D):
    """
    Master abstract base for every distribution asset.

    Owns all five sub-systems and wires them together.
    Subclasses implement ``_build_body()`` (the actual curve or bar geometry)
    and call ``super().__init__()`` which drives the full build sequence.

    Constructor parameters
    ----------------------
    axes          : StatsAxes3D      — coordinate system to draw on
    dist_fn       : DistributionFunction — mathematical backend
    params        : dict[str, float] — initial parameter values
    mode          : RepresentationMode — PDF / CDF / SF / LOG_PDF / HAZARD
    curve_cfg     : DistributionCurveConfig
    shade_cfgs    : dict[str, ShadeRegionConfig] — pre-built shade regions
    moment_cfg    : MomentMarkerConfig
    stats_cfg     : StatsAnnotationConfig
    """

    def __init__(
        self,
        axes:         StatsAxes3D,
        dist_fn:      DistributionFunction,
        params:       Optional[Dict[str, float]]       = None,
        mode:         RepresentationMode                = RepresentationMode.PDF,
        curve_cfg:    Optional[DistributionCurveConfig] = None,
        shade_cfgs:   Optional[Dict[str, ShadeRegionConfig]] = None,
        moment_cfg:   Optional[MomentMarkerConfig]      = None,
        stats_cfg:    Optional[StatsAnnotationConfig]   = None,
        **kwargs,
    ) -> None:
        self._axes       = axes
        self._dist_fn    = dist_fn
        self._mode       = mode
        self._curve_cfg  = curve_cfg  or DistributionCurveConfig()
        self._moment_cfg = moment_cfg or MomentMarkerConfig()
        self._stats_cfg  = stats_cfg  or StatsAnnotationConfig()

        # Parameter tracker system
        self.params = ParameterTrackerSystem()
        for name, val in (params or {}).items():
            self.params.register(name, val)

        # Cached evaluation result
        self._result: Optional[DistributionResult] = None

        # Sub-system VGroups (populated in _build_geometry)
        self._body:    VGroup = VGroup()    # curve or bars
        self._fill:    FillRegionSystem = None   # type: ignore[assignment]
        self._moments: MomentMarkerSystem = None # type: ignore[assignment]
        self._panel:   StatsAnnotationPanel = None  # type: ignore[assignment]
        self._probes:  Dict[str, PercentileProbe3D] = {}
        self._compare: Optional[ComparisonOverlay] = None

        # Stored shade configs for deferred build
        self._shade_cfgs_initial: Dict[str, ShadeRegionConfig] = shade_cfgs or {}

        # Register auto-rebuild when any parameter changes
        self.params.on_any_change(self._on_param_change)

        super().__init__(**kwargs)

    # ── abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def _build_body(self, result: DistributionResult) -> VGroup:
        """
        Build and return the primary visual body (curve or bars).
        Called by ``_build_geometry`` and ``_rebuild``.
        """
        ...

    @abstractmethod
    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        """
        Return a dict {param_name: predicate} for validation.
        E.g. {"sigma": lambda v: v > 0}
        """
        ...

    # ── full geometry build ───────────────────────────────────────────────

    def _build_geometry(self) -> None:
        theme = self._palette
        axes  = self._axes

        # Evaluate distribution
        self._result = self._evaluate()

        # Body
        self._body = self._build_body(self._result)
        self.add(self._body)

        # Fill regions
        self._fill = FillRegionSystem(axes, theme)
        for key, cfg in self._shade_cfgs_initial.items():
            self._fill.add_region(key, self._result, cfg, self._mode)
        self.add(self._fill)

        # Moment markers
        self._moments = MomentMarkerSystem(axes, self._result,
                                            self._moment_cfg, theme)
        self.add(self._moments)

        # Stats annotation panel
        self._panel = StatsAnnotationPanel(self._result, self._stats_cfg, theme)
        if self._panel.submobjects:
            self.add(self._panel)

    # ── evaluation ────────────────────────────────────────────────────────

    def _evaluate(self) -> DistributionResult:
        """
        Re-evaluate the distribution with current parameter values.
        Subclasses call ``_rebuild_dist_fn()`` before calling this.
        """
        xlo, xhi = self._axes.x_range[:2]
        n        = self._curve_cfg.n_sample_points
        x        = np.linspace(xlo, xhi, n)
        return self._dist_fn.evaluate(x=x)

    def _rebuild_dist_fn(self) -> None:
        """
        Override in subclasses to reconstruct ``_dist_fn`` from
        current ``self.params`` values before each evaluation.

        Example (Normal)::

            def _rebuild_dist_fn(self):
                self._dist_fn = DistributionFunction.normal(
                    mu=self.params.get("mu"),
                    sigma=self.params.get("sigma"),
                )
        """
        pass

    # ── parameter change callback ─────────────────────────────────────────

    def _on_param_change(self) -> None:
        """
        Fired whenever any parameter tracker changes.
        Validates, rebuilds the dist_fn, re-evaluates, and rebuilds geometry.

        Note: this fires synchronously.  In a live Manim scene, trigger
        geometry updates via ``animate_param()`` instead, which wraps
        this in a proper ``UpdateFromAlphaFunc`` or ``Transform``.
        """
        errors = self.params.validate(self._param_constraints())
        if errors:
            for e in errors:
                warnings.warn(e, stacklevel=3)
            return
        self._rebuild_dist_fn()
        self._result = self._evaluate()
        self._rebuild()

    def _rebuild(self) -> None:
        """
        In-place geometry rebuild after a parameter change.
        Replaces body, fill regions, moment markers, and stats panel.
        """
        # Body
        self.remove(self._body)
        self._body = self._build_body(self._result)
        self.add(self._body)

        # Fill regions — rebuild all with updated result
        for key, cfg in list(self._shade_cfgs_initial.items()):
            self._fill.update_region(key, self._result, cfg, self._mode)

        # Moment markers
        self.remove(self._moments)
        self._moments = MomentMarkerSystem(
            self._axes, self._result, self._moment_cfg, self._palette)
        self.add(self._moments)

        # Stats panel
        self._panel.rebuild(self._result)

    # ── representation mode switching ─────────────────────────────────────

    def set_mode(self, mode: RepresentationMode) -> "BaseDistribution3D":
        """Switch representation mode immediately (no animation)."""
        self._mode = mode
        self._rebuild()
        return self

    def animate_mode_switch(
        self,
        new_mode: RepresentationMode,
        cfg:      Optional[AnimationConfig] = None,
    ) -> Animation:
        """Smoothly transform geometry to a new representation mode."""
        cfg      = cfg or self._anim_cfg
        old_copy = self.copy()
        self.set_mode(new_mode)
        return Transform(old_copy, self,
                         run_time=cfg.run_time,
                         rate_func=cfg.rate_func)

    # ── parameter animation ───────────────────────────────────────────────

    def animate_param(
        self,
        param_name:  str,
        target:      float,
        run_time:    float = 2.0,
        rate_func:   Callable = smooth,
    ) -> Animation:
        """
        Smoothly animate parameter *param_name* from its current value
        to *target*, live-updating all geometry via an updater.
        """
        tracker  = self.params.tracker(param_name)
        return tracker.animate(
    run_time=run_time,
    rate_func=rate_func,
).set_value(target)

    def animate_params(
        self,
        targets:   Dict[str, float],
        run_time:  float = 2.0,
        rate_func: Callable = smooth,
    ) -> AnimationGroup:
        """Animate multiple parameters simultaneously."""
        anims = [
            self.animate_param(name, val, run_time, rate_func)
            for name, val in targets.items()
        ]
        return AnimationGroup(*anims)

    # ── shading API ───────────────────────────────────────────────────────

    def shade_region(
        self,
        key:     str,
        lo:      float = -math.inf,
        hi:      float =  math.inf,
        color:   Optional[str]  = None,
        opacity: float          = 0.38,
        style:   ShadeFillStyle = ShadeFillStyle.SOLID,
        label:   bool           = True,
    ) -> VGroup:
        """Shade the area under the curve between *lo* and *hi*."""
        cfg = ShadeRegionConfig(lo=lo, hi=hi, color=color,
                                fill_opacity=opacity,
                                fill_style=style,
                                show_label=label)
        self._shade_cfgs_initial[key] = cfg
        return self._fill.add_region(key, self._result, cfg, self._mode)

    def shade_tail_left(
        self, key: str, x: float, **kwargs
    ) -> VGroup:
        """Shade the left tail P(X ≤ x)."""
        return self.shade_region(key, lo=-math.inf, hi=x, **kwargs)

    def shade_tail_right(
        self, key: str, x: float, **kwargs
    ) -> VGroup:
        """Shade the right tail P(X ≥ x)."""
        return self.shade_region(key, lo=x, hi=math.inf, **kwargs)

    def shade_central(
        self,
        key:      str,
        coverage: float          = 0.95,
        color:    Optional[str]  = None,
        **kwargs,
    ) -> VGroup:
        """Shade the central *coverage* probability mass (e.g. 95% CI region)."""
        alpha = 1.0 - coverage
        lo    = float(self._dist_fn.ppf(alpha / 2))
        hi    = float(self._dist_fn.ppf(1 - alpha / 2))
        return self.shade_region(key, lo=lo, hi=hi,
                                 color=color or self._palette.positive,
                                 **kwargs)

    def shade_by_sigma(
        self,
        key:     str,
        n_sigma: float = 1.0,
        **kwargs,
    ) -> VGroup:
        """Shade μ ± n_sigma * σ region."""
        if self._result is None or self._result.mean is None:
            return VGroup()
        mu    = self._result.mean
        sigma = math.sqrt(max(self._result.variance or 0, 0))
        return self.shade_region(key,
                                 lo=mu - n_sigma * sigma,
                                 hi=mu + n_sigma * sigma,
                                 **kwargs)

    def remove_shade(self, key: str) -> None:
        """Remove a named shaded region."""
        self._fill.remove_region(key)
        self._shade_cfgs_initial.pop(key, None)

    def clear_shades(self) -> None:
        """Remove all shaded regions."""
        self._fill.clear_all()
        self._shade_cfgs_initial.clear()

    # ── probing API ───────────────────────────────────────────────────────

    def probe_at(
        self,
        key:   str,
        x:     float,
        cfg:   Optional[ProbeConfig] = None,
    ) -> PercentileProbe3D:
        """Drop a probe at *x*, showing PDF(x) and CDF(x)."""
        if key in self._probes:
            self.remove(self._probes[key])
        pcfg  = cfg or ProbeConfig()
        probe = PercentileProbe3D(self._axes, self._result,
                                   x, pcfg, self._palette, key)
        self._probes[key] = probe
        self.add(probe)
        return probe

    def probe_quantile(
        self,
        key:   str,
        p:     float,
        cfg:   Optional[ProbeConfig] = None,
    ) -> PercentileProbe3D:
        """Probe at x such that CDF(x) = p."""
        x = float(self._dist_fn.ppf(p))
        return self.probe_at(key, x, cfg)

    def remove_probe(self, key: str) -> None:
        if key in self._probes:
            self.remove(self._probes.pop(key))

    # ── CI builder ────────────────────────────────────────────────────────

    def show_confidence_interval(
        self,
        alpha:    float         = 0.05,
        key:      str           = "ci",
        color:    Optional[str] = None,
        opacity:  float         = 0.30,
        label:    bool          = True,
    ) -> Tuple[VGroup, VGroup, VGroup]:
        """
        Shade the central (1-α) probability region and add critical
        value probes at the two tails.

        Returns (shade_grp, probe_lo, probe_hi).
        """
        shade = self.shade_central(key, coverage=1.0 - alpha,
                                    color=color, opacity=opacity)
        lo_x  = float(self._dist_fn.ppf(alpha / 2))
        hi_x  = float(self._dist_fn.ppf(1 - alpha / 2))
        p_lo  = self.probe_at(f"{key}_lo", lo_x)
        p_hi  = self.probe_at(f"{key}_hi", hi_x)
        return shade, p_lo, p_hi

    # ── comparison overlay ────────────────────────────────────────────────

    def add_comparison(
        self,
        other_dist:  "BaseDistribution3D",
        cfg:         Optional[ComparisonConfig] = None,
        color_self:  Optional[str] = None,
        color_other: Optional[str] = None,
    ) -> ComparisonOverlay:
        """Overlay *other_dist* on the same axes with divergence stats."""
        if self._compare is not None:
            self.remove(self._compare)
        ccfg = cfg or ComparisonConfig()
        self._compare = ComparisonOverlay(
            self._axes,
            self._result,
            other_dist._result,
            ccfg, self._palette,
            color_a=color_self or self._palette.primary,
            color_b=color_other or self._palette.secondary,
        )
        self.add(self._compare)
        return self._compare

    # ── animation protocol ────────────────────────────────────────────────

    def animate_build(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        """
        Layered reveal:
            1. Axes (if embedded)
            2. Body curve / bars grow in
            3. Fill regions fade in
            4. Moment markers appear
            5. Stats panel slides in
        """
        cfg = cfg or self._anim_cfg
        t   = cfg.run_time

        body_anim    = self._animate_body_build(cfg)
        fill_anim    = FadeIn(self._fill,    run_time=t * 0.35, rate_func=smooth)
        moments_anim = FadeIn(self._moments, run_time=t * 0.30, rate_func=smooth)
        panel_anim   = FadeIn(self._panel,   run_time=t * 0.25, rate_func=smooth) \
                       if self._panel.submobjects else FadeIn(VGroup(), run_time=0.01)

        return Succession(
            body_anim,
            AnimationGroup(fill_anim, moments_anim, lag_ratio=0.2),
            panel_anim,
        )

    def _animate_body_build(self, cfg: AnimationConfig) -> Animation:
        """Default body build animation — subclasses can override."""
        return Create(self._body, run_time=cfg.run_time * 0.5,
                      rate_func=smooth)

    def animate_update(
        self, new_data: Any, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        cfg      = cfg or self._anim_cfg
        old_copy = self.copy()
        if isinstance(new_data, dict):
            for k, v in new_data.items():
                self.params.set(k, v)
        return Transform(old_copy, self,
                         run_time=cfg.run_time,
                         rate_func=cfg.rate_func)

    def animate_highlight(
        self,
        style: HighlightStyle = HighlightStyle.GLOW,
        cfg:   Optional[AnimationConfig] = None,
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return HighlightSystem.glow(self._body, cfg)

    def animate_exit(
        self, cfg: Optional[AnimationConfig] = None
    ) -> Animation:
        cfg = cfg or self._anim_cfg
        return FadeOut(self, run_time=cfg.run_time * 0.6)

    # ── CDF build animation ───────────────────────────────────────────────

    def animate_cdf_build(
        self,
        run_time:  float = 3.0,
        rate_func: Callable = smooth,
    ) -> Animation:
        """
        Animate the CDF curve filling in left-to-right,
        as if probability is accumulating.

        Switches to CDF mode first, then gradually reveals the curve.
        """
        old_mode = self._mode
        self.set_mode(RepresentationMode.CDF)
        return Create(self._body, run_time=run_time, rate_func=rate_func)

    # ── property accessors ─────────────────────────────────────────────────

    @property
    def result(self) -> Optional[DistributionResult]:
        return self._result

    @property
    def mode(self) -> RepresentationMode:
        return self._mode

    @property
    def mean(self) -> Optional[float]:
        return self._result.mean if self._result else None

    @property
    def std(self) -> Optional[float]:
        if self._result and self._result.variance is not None:
            return math.sqrt(max(self._result.variance, 0))
        return None

    def __repr__(self) -> str:
        p = self.params.all_values()
        return f"{self.__class__.__name__}({p})"


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CONTINUOUS DISTRIBUTION 3D
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousDistribution3D(BaseDistribution3D):
    """
    Sub-base for continuous distributions (Normal, t, Gamma, etc.).

    Body geometry
    -------------
    The primary body is a ``ParametricFunction`` curve
    (smooth, camera-facing) backed by dense sample points.
    An optional filled polygon sits beneath the curve.

    Subclasses only need to implement:
        ``_rebuild_dist_fn()`` — reconstruct ``_dist_fn`` from current params
        ``_param_constraints()`` — validation predicates
    """

    def _build_body(self, result: DistributionResult) -> VGroup:
        """Build the smooth 3-D curve + optional fill polygon."""
        cfg   = self._curve_cfg
        theme = self._palette
        axes  = self._axes
        c     = cfg.stroke_color or theme.primary

        y_arr = FillRegionSystem._select_y(result, self._mode)
        grp   = VGroup()

        # ── main curve ────────────────────────────────────────────────────
        # Build as a sequence of Line3D segments for reliable 3-D rendering
        pts = [axes.c2p(float(x), float(y))
               for x, y in zip(result.x, y_arr)
               if not (math.isnan(y) or math.isinf(y))]

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                seg = Line3D(pts[i], pts[i + 1],
                              color=c,
                              thickness=cfg.stroke_width * 0.006)
                seg.set_opacity(cfg.stroke_opacity)
                grp.add(seg)

        # ── fill polygon ──────────────────────────────────────────────────
        if cfg.show_fill and result.pdf is not None:
            fc   = cfg.fill_color or c
            x_f  = result.x[np.isfinite(y_arr)]
            y_f  = y_arr[np.isfinite(y_arr)]
            if len(x_f) >= 2:
                pts_top = [axes.c2p(xi, yi) for xi, yi in zip(x_f, y_f)]
                pts_bot = [axes.c2p(x_f[-1], 0.0), axes.c2p(x_f[0], 0.0)]
                poly    = Polygon(*(pts_top + pts_bot))
                poly.set_fill(fc, opacity=cfg.fill_opacity)
                poly.set_stroke(width=0)
                grp.add(poly)

        # ── baseline ──────────────────────────────────────────────────────
        if cfg.show_baseline:
            bc    = cfg.baseline_color or theme.neutral
            xlo   = axes.x_range[0]
            xhi   = axes.x_range[1]
            bl    = Line3D(axes.c2p(xlo, 0.0), axes.c2p(xhi, 0.0),
                           color=bc,
                           thickness=cfg.baseline_width * 0.005)
            bl.set_opacity(cfg.baseline_opacity)
            grp.add(bl)

        return grp

    def _animate_body_build(self, cfg: AnimationConfig) -> Animation:
        """ContinuousDistribution grows curve from left to right."""
        curve_segs = [m for m in self._body.submobjects
                      if isinstance(m, Line3D)]
        fills      = [m for m in self._body.submobjects
                      if isinstance(m, Polygon)]
        baselines   = [m for m in self._body.submobjects
                       if m not in curve_segs and m not in fills]

        base_in  = FadeIn(VGroup(*baselines), run_time=cfg.run_time * 0.15)
        curve_in = AnimationGroup(
            *[Create(s) for s in curve_segs],
            lag_ratio=1.0 / max(len(curve_segs), 1),
            run_time=cfg.run_time * 0.55,
        )
        fill_in  = FadeIn(VGroup(*fills), run_time=cfg.run_time * 0.30,
                          rate_func=smooth)

        return Succession(base_in, curve_in, fill_in)

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {}   # Override in each distribution subclass


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DISCRETE DISTRIBUTION 3D
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BarConfig:
    """Visual config for PMF bar geometry."""
    bar_width_ratio:    float          = 0.72   # fraction of bin width
    bar_color:          Optional[str]  = None   # None → theme.primary
    bar_opacity:        float          = 0.85
    bar_stroke_color:   Optional[str]  = None   # None → theme.border
    bar_stroke_width:   float          = 1.2
    bar_depth:          float          = 0.18   # 3-D extrusion depth
    bar_depth_color:    Optional[str]  = None   # side-face color
    bar_depth_opacity:  float          = 0.55
    show_bar_labels:    bool           = True
    label_font_size:    float          = 18
    label_decimals:     int            = 3
    label_offset:       float          = 0.15   # above bar top
    stem_color:         Optional[str]  = None
    stem_width:         float          = 1.5
    stem_opacity:       float          = 0.60
    show_stems:         bool           = True
    highlight_bar_idx:  Optional[int]  = None   # highlight one bar
    highlight_color:    Optional[str]  = None


class DiscreteDistribution3D(BaseDistribution3D):
    """
    Sub-base for discrete distributions (Binomial, Poisson, etc.).

    Body geometry
    -------------
    The body is a forest of 3-D bars (Prism-like constructions):
        • Each bar represents P(X = k) for integer k
        • Bars have a slight 3-D depth for visual quality
        • Stem lines connect each bar to the x-axis
        • Optional value labels float above each bar

    Subclasses only need to implement:
        ``_rebuild_dist_fn()``
        ``_param_constraints()``
    """

    def __init__(
        self,
        *args,
        bar_cfg: Optional[BarConfig] = None,
        **kwargs,
    ) -> None:
        self._bar_cfg = bar_cfg or BarConfig()
        super().__init__(*args, **kwargs)

    def _build_body(self, result: DistributionResult) -> VGroup:
        """Build the PMF bar forest."""
        bcfg  = self._bar_cfg
        theme = self._palette
        axes  = self._axes
        c     = bcfg.bar_color or theme.primary
        sc    = bcfg.bar_stroke_color or theme.border
        dc    = bcfg.bar_depth_color or theme.secondary

        y_arr = FillRegionSystem._select_y(result, self._mode)
        grp   = VGroup()

        x_vals = np.round(result.x).astype(int)
        # Deduplicate
        seen   = set()
        pairs  = []
        for xv, yv in zip(x_vals, y_arr):
            if xv not in seen and not math.isnan(yv) and not math.isinf(yv):
                seen.add(xv); pairs.append((int(xv), float(yv)))

        if not pairs:
            return grp

        # Bin width in scene units
        if len(pairs) >= 2:
            dx_scene = abs(axes.data_to_scene_length(1.0, AxisID.X))
        else:
            dx_scene = axes.data_to_scene_length(1.0, AxisID.X)

        bar_w = dx_scene * bcfg.bar_width_ratio
        depth = bcfg.bar_depth

        for idx, (k, prob) in enumerate(pairs):
            if prob <= 0:
                continue

            bot    = axes.c2p(k, 0.0)
            top    = axes.c2p(k, prob)
            height = float(np.linalg.norm(top - bot))
            mid    = (bot + top) / 2

            is_highlighted = (bcfg.highlight_bar_idx is not None and
                              idx == bcfg.highlight_bar_idx)
            bar_c = bcfg.highlight_color if is_highlighted else c

            bar_grp = self._make_bar(
                centre=mid,
                height=height,
                width=bar_w,
                depth=depth,
                face_color=bar_c,
                face_opacity=bcfg.bar_opacity,
                stroke_color=sc,
                stroke_width=bcfg.bar_stroke_width,
                depth_color=dc,
                depth_opacity=bcfg.bar_depth_opacity,
            )
            grp.add(bar_grp)

            # Stem
            if bcfg.show_stems:
                stem_c = bcfg.stem_color or theme.neutral
                stem   = Line3D(bot, top, color=stem_c,
                                thickness=bcfg.stem_width * 0.005)
                stem.set_opacity(bcfg.stem_opacity)
                grp.add(stem)

            # Value label above bar
            if bcfg.show_bar_labels:
                lbl_str = f"{prob:.{bcfg.label_decimals}f}"
                lbl     = Text(lbl_str,
                               font_size=bcfg.label_font_size,
                               color=theme.text_secondary)
                lbl.move_to(top + UP * bcfg.label_offset)
                grp.add(lbl)

        # Baseline
        cfg = self._curve_cfg
        if cfg.show_baseline:
            bc  = cfg.baseline_color or theme.neutral
            xlo = axes.x_range[0]
            xhi = axes.x_range[1]
            bl  = Line3D(axes.c2p(xlo, 0.0), axes.c2p(xhi, 0.0),
                         color=bc, thickness=cfg.baseline_width * 0.005)
            bl.set_opacity(cfg.baseline_opacity)
            grp.add(bl)

        return grp

    @staticmethod
    def _make_bar(
        centre:        np.ndarray,
        height:        float,
        width:         float,
        depth:         float,
        face_color:    str,
        face_opacity:  float,
        stroke_color:  str,
        stroke_width:  float,
        depth_color:   str,
        depth_opacity: float,
    ) -> VGroup:
        """
        Build a 3-D bar as a VGroup of polygons:
            front face + top face + right side face
        """
        grp = VGroup()
        hw  = width / 2
        hh  = height / 2
        hd  = depth / 2

        # Front face corners (in scene space, centred at 'centre')
        bl = centre + np.array([-hw, -hh, +hd])
        br = centre + np.array([+hw, -hh, +hd])
        tr = centre + np.array([+hw, +hh, +hd])
        tl = centre + np.array([-hw, +hh, +hd])

        front = Polygon(bl, br, tr, tl)
        front.set_fill(face_color, opacity=face_opacity)
        front.set_stroke(stroke_color, width=stroke_width)
        grp.add(front)

        # Top face
        bl_b = centre + np.array([-hw, +hh, -hd])
        br_b = centre + np.array([+hw, +hh, -hd])
        top  = Polygon(tl, tr, br_b, bl_b)
        top.set_fill(depth_color, opacity=depth_opacity)
        top.set_stroke(stroke_color, width=stroke_width * 0.7)
        grp.add(top)

        # Right side face
        br_b2 = centre + np.array([+hw, -hh, -hd])
        right = Polygon(br, tr, br_b, br_b2)
        right.set_fill(depth_color, opacity=depth_opacity * 0.8)
        right.set_stroke(stroke_color, width=stroke_width * 0.7)
        grp.add(right)

        return grp

    def _animate_body_build(self, cfg: AnimationConfig) -> Animation:
        """Bars grow from the baseline upward with a stagger."""
        bars = self._body.submobjects
        return AnimationGroup(
            *[GrowFromCenter(b) for b in bars],
            lag_ratio=0.06,
            run_time=cfg.run_time * 0.7,
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {}

    # ── PMF → PDF morph ───────────────────────────────────────────────────

    def animate_morph_to_continuous(
        self,
        target_dist:  ContinuousDistribution3D,
        run_time:     float = 3.0,
    ) -> Animation:
        """
        Morph discrete bars into a continuous PDF curve.
        Used for Binomial → Normal CLT demonstrations.
        """
        return Transform(self, target_dist,
                         run_time=run_time, rate_func=smooth)

    def highlight_bar(
        self,
        k:     int,
        color: Optional[str] = None,
        cfg:   Optional[AnimationConfig] = None,
    ) -> Animation:
        """Pulse-highlight the bar at X = k."""
        acfg = cfg or self._anim_cfg
        self._bar_cfg.highlight_bar_idx = k
        self._bar_cfg.highlight_color   = color or self._palette.accent
        self._rebuild()
        return HighlightSystem.pulse(self._body, acfg)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enumerations
    "RepresentationMode",
    "ShadeFillStyle",
    "MomentMarkerStyle",
    # Config dataclasses
    "DistributionCurveConfig",
    "ShadeRegionConfig",
    "MomentMarkerConfig",
    "StatsAnnotationConfig",
    "ProbeConfig",
    "ComparisonConfig",
    "BarConfig",
    # Sub-systems
    "ParameterTrackerSystem",
    "FillRegionSystem",
    "MomentMarkerSystem",
    "StatsAnnotationPanel",
    "PercentileProbe3D",
    "ComparisonOverlay",
    # Base classes
    "BaseDistribution3D",
    "ContinuousDistribution3D",
    "DiscreteDistribution3D",
]