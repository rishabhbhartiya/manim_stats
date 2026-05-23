"""
manim_stats/distributions/normal3d.py
=======================================
Univariate and bivariate Normal distribution visualizations.

This module is the visual showpiece of the distributions package.
The Normal distribution has more dedicated pedagogy than any other —
the 68-95-99.7 rule, standardisation, the bivariate surface, Q-Q plots,
and its role as the CLT limit all deserve rich 3D treatments.

Visual objects
--------------
``NormalCurve3D``
    Univariate N(μ, σ) bell curve with Normal-specific annotation layers:
    - 68-95-99.7 rule shading with bracket annotations.
    - z-score ruler beneath the curve.
    - Standardisation animation: any N(μ, σ) morphs to N(0, 1).
    - CLT overlay: sampling distribution of x̄ drawn over population.

``BivariateNormal3D``
    True 3D parametric surface f(x, y) for the bivariate normal,
    rendered as a dense mesh of shaded quad polygons.
    - Independent face shading per quad (Phong-style light model).
    - Marginal density curves projected onto the xz / yz walls.
    - Isoprobability contour ellipses at configurable levels.
    - Conditional slice: fix x = x₀, show the resulting 1D normal.
    - Animated "tent" reveal: surface rises from z = 0 upward.
    - Correlation sweep: ρ varies from -0.9 → 0 → +0.9.

``NormalApproximation3D``
    Binomial(n, p) PMF bars with the Normal(np, np(1-p)) approximation
    curve overlaid.  Shows the approximation improving as n grows.

``QQPlot3D``
    Quantile-quantile plot against a Normal reference line.
    Each data point is a Dot3D at (theoretical quantile, sample quantile).
    The reference line and deviation envelope are drawn separately.

Classes
-------
BivariateNormalConfig
NormalCurve3D
BivariateNormal3D
NormalApproximation3D
QQPlot3D

Ready-to-render scenes
----------------------
NormalCurveScene
BivariateNormalScene
StandardisationScene
NormalApproximationScene
QQPlotScene

Usage
-----
    from manim import *
    from manim_stats.distributions.normal3d import NormalCurve3D, BivariateNormal3D

    class MyScene(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65*DEGREES, theta=-45*DEGREES)

            # Univariate
            bell = NormalCurve3D(mu=0, sigma=1)
            self.play(bell.animate_curve())
            self.play(bell.animate_fill())
            self.play(bell.animate_rule_68_95_99())

            # Bivariate surface
            surf = BivariateNormal3D(mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, rho=0.0)
            self.play(surf.animate_rise())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Union, Dict
import numpy as np

from manim import (
    VGroup, VMobject, Polygon, Line, DashedLine, Dot3D,
    Text, MathTex, Arrow,
    ThreeDScene,
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, Transform,
    UpdateFromAlphaFunc, Flash, Write,
    DEGREES, PI, TAU,
    RIGHT, LEFT, UP, DOWN, OUT, IN, ORIGIN,
    X_AXIS, Y_AXIS, Z_AXIS,
    WHITE, BLACK, GRAY, BLUE, GREEN, RED, YELLOW,
    ManimColor, color_to_rgb, rgba_to_color, color_to_rgba,
    rate_functions, smooth,
)

# ---------------------------------------------------------------------------
# Colour helpers
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
# Special functions (self-contained)
# ---------------------------------------------------------------------------

def _erf(z: float) -> float:
    t = 1.0 / (1.0 + 0.3275911 * abs(z))
    poly = t * (0.254829592 + t * (-0.284496736
           + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    val = 1.0 - poly * np.exp(-z * z)
    return float(val if z >= 0 else -val)


def _normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(TAU))


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    z = (x - mu) / (sigma * np.sqrt(2))
    return 0.5 * (1.0 + _erf(z))


def _bivariate_normal_pdf(
    x: np.ndarray,
    y: np.ndarray,
    mu_x: float, mu_y: float,
    sigma_x: float, sigma_y: float,
    rho: float,
) -> np.ndarray:
    """Bivariate normal density on meshgrid arrays X, Y."""
    denom = 2 * PI * sigma_x * sigma_y * np.sqrt(1 - rho**2)
    dx = (x - mu_x) / sigma_x
    dy = (y - mu_y) / sigma_y
    z = dx**2 - 2 * rho * dx * dy + dy**2
    return np.exp(-z / (2 * (1 - rho**2))) / denom


def _catmull_rom(points: np.ndarray, resolution: int = 16) -> np.ndarray:
    n = len(points)
    if n < 2:
        return points.copy()
    p = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        for j in range(resolution):
            t = j / resolution
            t2, t3 = t*t, t*t*t
            pt = 0.5 * ((-t3+2*t2-t)*p0 + (3*t3-5*t2+2)*p1
                        + (-3*t3+4*t2+t)*p2 + (t3-t2)*p3)
            out.append(pt)
    out.append(points[-1].copy())
    return np.array(out)


# ---------------------------------------------------------------------------
# Internal fill helper
# ---------------------------------------------------------------------------

def _gradient_fill(
    curve_pts: np.ndarray,
    floor_z: float,
    y_pos: float,
    color: ManimColor,
    opacity: float,
    n_strips: int = 12,
) -> VGroup:
    floor_pts = np.array([[p[0], y_pos, floor_z] for p in curve_pts])
    grp = VGroup()
    for s in range(n_strips):
        t0, t1 = s / n_strips, (s + 1) / n_strips
        strip_op = opacity * (1.0 - 0.62 * t0)
        lo = curve_pts * (1 - t0) + floor_pts * t0
        hi = curve_pts * (1 - t1) + floor_pts * t1
        pts = list(lo) + list(reversed(hi))
        grp.add(Polygon(*pts,
                        fill_color=_with_opacity(color, strip_op),
                        fill_opacity=1.0, stroke_width=0))
    return grp


# ===========================================================================
# BivariateNormalConfig
# ===========================================================================

@dataclass
class BivariateNormalConfig:
    """Visual configuration for ``BivariateNormal3D``.

    Surface mesh
    ~~~~~~~~~~~~
    ``n_x`` / ``n_y``      : mesh resolution (quads per axis).
    ``x_range`` / ``y_range``: (min, max) display range per axis.
    ``z_scale``            : scene-unit height of the PDF peak.
    ``surface_color_lo``   : colour at the tails (low density).
    ``surface_color_hi``   : colour at the peak (high density).
    ``surface_opacity``    : overall surface opacity.
    ``edge_opacity``       : opacity of quad edge strokes.
    ``edge_stroke_width``  : quad edge stroke width.

    Lighting
    ~~~~~~~~
    ``light_direction``    : 3D unit vector toward the light source.
                             Each quad's face normal is dotted with this
                             to produce shading intensity (0.4 – 1.0).
    ``ambient_light``      : minimum shading factor regardless of angle.

    Marginal curves
    ~~~~~~~~~~~~~~~
    ``show_marginals``     : whether to project marginal density curves
                             onto the xz and yz walls.
    ``marginal_color``     : colour of marginal curves.
    ``marginal_stroke_width``: stroke width of marginal curves.
    ``marginal_fill``      : whether to fill below marginal curves.
    ``marginal_fill_opacity``: opacity of marginal fill.
    ``y_wall``             : y-coordinate of the xz projection plane.
    ``x_wall``             : x-coordinate of the yz projection plane.

    Contours
    ~~~~~~~~
    ``contour_levels``     : list of probability density levels to contour.
    ``contour_color``      : colour of all contour ellipses.
    ``contour_stroke_width``: stroke width.
    ``contour_opacity``    : opacity of contour lines.
    ``n_contour_pts``      : points per contour ellipse.

    Conditional slice
    ~~~~~~~~~~~~~~~~~
    ``slice_color``        : colour of the conditional distribution plane.
    ``slice_opacity``      : opacity of the slice fill.
    """

    # Surface mesh
    n_x: int = 40
    n_y: int = 40
    x_range: Tuple[float, float] = (-3.5, 3.5)
    y_range: Tuple[float, float] = (-3.5, 3.5)
    z_scale: float = 3.5

    surface_color_lo: ManimColor = ManimColor("#0C3D7A")
    surface_color_hi: ManimColor = ManimColor("#88CCFF")
    surface_opacity: float = 0.82
    edge_opacity: float = 0.18
    edge_stroke_width: float = 0.4

    # Lighting
    light_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.35, -0.55, 1.0]) / np.linalg.norm(
            np.array([0.35, -0.55, 1.0])
        )
    )
    ambient_light: float = 0.40

    # Marginal curves
    show_marginals: bool = True
    marginal_color: ManimColor = ManimColor("#E0AA40")
    marginal_stroke_width: float = 2.4
    marginal_fill: bool = True
    marginal_fill_opacity: float = 0.18
    y_wall: float = 3.8
    x_wall: float = -3.8

    # Contours
    contour_levels: List[float] = field(default_factory=lambda: [0.5, 0.25, 0.10, 0.04])
    contour_color: ManimColor = ManimColor("#FFD700")
    contour_stroke_width: float = 1.6
    contour_opacity: float = 0.70
    n_contour_pts: int = 80

    # Conditional slice
    slice_color: ManimColor = ManimColor("#E8593C")
    slice_opacity: float = 0.30


# ── Presets ──────────────────────────────────────────────────────────────

CRISP_SURFACE = BivariateNormalConfig(
    n_x=30, n_y=30,
    edge_opacity=0.22,
    surface_opacity=0.85,
    show_marginals=True,
)

DENSE_SURFACE = BivariateNormalConfig(
    n_x=55, n_y=55,
    edge_opacity=0.10,
    surface_opacity=0.80,
    show_marginals=True,
    contour_levels=[0.60, 0.30, 0.12, 0.04, 0.01],
)

WIREFRAME = BivariateNormalConfig(
    n_x=25, n_y=25,
    surface_opacity=0.0,
    edge_opacity=0.65,
    edge_stroke_width=0.8,
    show_marginals=False,
    contour_levels=[],
)


# ===========================================================================
# NormalCurve3D
# ===========================================================================

class NormalCurve3D(VGroup):
    """Univariate Normal bell curve with full pedagogical annotation system.

    Visual layers
    ~~~~~~~~~~~~~
    - ``curve_group``     : glow halo + primary stroke.
    - ``fill_group``      : gradient area fill.
    - ``moments_group``   : μ line, σ brackets, σ tick marks.
    - ``rule_group``      : 68-95-99.7 shaded regions.
    - ``zscore_group``    : z-score ruler beneath the curve.
    - ``clt_group``       : sampling distribution overlay (σ/√n).
    - ``title_group``     : N(μ, σ²) title.
    - ``std_form_group``  : standardised N(0,1) target curve (for morph).

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (> 0).
    x_range : (float, float, float) or None
        (x_min, x_max, x_step).  Defaults to μ ± 4σ.
    color : ManimColor
    z_scale : float
        Scene z-units at the PDF peak.
    x_pos, y_pos, floor_z : float
        Scene-space offsets.
    glow_opacity : float
    fill_opacity : float
    fill_gradient : bool
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        x_range: Optional[Tuple[float, float, float]] = None,
        color: ManimColor = ManimColor("#4A90D9"),
        z_scale: float = 3.8,
        x_pos: float = 0.0,
        y_pos: float = 0.0,
        floor_z: float = 0.0,
        glow_opacity: float = 0.14,
        fill_opacity: float = 0.26,
        fill_gradient: bool = True,
        stroke_width: float = 3.0,
        smooth_resolution: int = 22,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.color = color
        self.z_scale = z_scale
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.floor_z = floor_z
        self._scene = scene

        # Resolve x_range
        if x_range is not None:
            self._x_min, self._x_max, self._x_step = x_range
        else:
            self._x_min = mu - 4.2 * sigma
            self._x_max = mu + 4.2 * sigma
            self._x_step = (self._x_max - self._x_min) / 220

        # Sample PDF — store raw and scaled arrays
        xs = np.arange(self._x_min, self._x_max + self._x_step * 0.5, self._x_step)
        ys_raw = _normal_pdf(xs, mu, sigma)
        peak = float(ys_raw.max())
        self._peak = peak
        ys = ys_raw / peak * z_scale   # peak maps to z_scale

        self._xs = xs + x_pos
        self._ys = ys
        self._ys_raw = ys_raw

        # 3D curve points
        raw_pts = np.column_stack([self._xs, np.full(len(xs), y_pos), ys])
        self._pts = _catmull_rom(raw_pts, resolution=smooth_resolution)

        # Layer groups
        self.curve_group  = VGroup()
        self.fill_group   = VGroup()
        self.moments_group = VGroup()
        self.rule_group   = VGroup()
        self.zscore_group = VGroup()
        self.clt_group    = VGroup()
        self.title_group  = VGroup()

        # Build primary layers
        self._stroke_width = stroke_width
        self._glow_opacity = glow_opacity
        self._fill_opacity = fill_opacity
        self._fill_gradient = fill_gradient
        self._n_strips = 12

        self._build_curve()
        self._build_fill()
        self._build_moments(scene)

        self.add(
            self.fill_group, self.curve_group,
            self.moments_group, self.rule_group,
            self.zscore_group, self.clt_group,
            self.title_group,
        )

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_curve(self) -> None:
        col = _with_opacity(self.color, 1.0)

        self.stroke = VMobject()
        self.stroke.set_points_as_corners(self._pts)
        self.stroke.set_stroke(color=col, width=self._stroke_width)
        self.stroke.set_fill(opacity=0)

        if self._glow_opacity > 0:
            self.glow = VMobject()
            self.glow.set_points_as_corners(self._pts)
            self.glow.set_stroke(
                color=_with_opacity(self.color, self._glow_opacity),
                width=self._stroke_width * 3.2,
            )
            self.glow.set_fill(opacity=0)
            self.curve_group.add(self.glow)

        self.curve_group.add(self.stroke)

    def _build_fill(self) -> None:
        fill = _gradient_fill(
            self._pts, self.floor_z, self.y_pos,
            self.color, self._fill_opacity, self._n_strips,
        )
        self.fill_group.add(fill)

    def _build_moments(self, scene: Optional[ThreeDScene]) -> None:
        mu, sigma = self.mu, self.sigma
        y = self.y_pos
        z0 = self.floor_z
        x_mu = mu + self.x_pos
        z_peak = self.z_scale   # PDF peak height

        mean_col = _with_opacity(ManimColor("#E0AA40"), 0.85)

        # Mean dashed line
        mean_ln = DashedLine(
            np.array([x_mu, y, z0]),
            np.array([x_mu, y, z_peak * 1.04]),
            dash_length=0.07, dashed_ratio=0.45,
            color=mean_col, stroke_width=1.8,
        )
        mean_dot = Dot3D(np.array([x_mu, y, z0]), radius=0.06, color=mean_col)
        mu_lbl = Text(f"μ = {mu:.3g}", font_size=17, color=ManimColor("#E0AA40"))
        mu_lbl.move_to(np.array([x_mu + 0.15, y, z_peak + 0.30]))
        self.moments_group.add(mean_ln, mean_dot, mu_lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(mu_lbl)

        # ±1σ and ±2σ tick marks and brackets
        sig_col = _with_opacity(ManimColor("#E0AA40"), 0.65)
        for n_sig in [1, 2]:
            for sign in [-1, +1]:
                x_s = (mu + sign * n_sig * sigma) + self.x_pos
                tick = Line(
                    np.array([x_s, y, z0 - 0.05]),
                    np.array([x_s, y, z0 + 0.05]),
                    color=sig_col, stroke_width=1.4,
                )
                self.moments_group.add(tick)

            # Horizontal bracket at z = -0.12*n_sig
            lo_x = (mu - n_sig * sigma) + self.x_pos
            hi_x = (mu + n_sig * sigma) + self.x_pos
            z_b = z0 - 0.12 * n_sig
            bracket = VGroup(
                Line(np.array([lo_x, y, z_b]), np.array([hi_x, y, z_b]),
                     color=sig_col, stroke_width=1.2),
                Line(np.array([lo_x, y, z_b]), np.array([lo_x, y, z_b + 0.07]),
                     color=sig_col, stroke_width=1.2),
                Line(np.array([hi_x, y, z_b]), np.array([hi_x, y, z_b + 0.07]),
                     color=sig_col, stroke_width=1.2),
            )
            lbl = Text(f"±{n_sig}σ", font_size=14, color=ManimColor("#E0AA40"))
            lbl.move_to(np.array([(lo_x + hi_x) / 2, y, z_b - 0.22]))
            self.moments_group.add(bracket, lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

    # ------------------------------------------------------------------
    # Public annotation methods
    # ------------------------------------------------------------------

    def animate_rule_68_95_99(
        self,
        scene: Optional[ThreeDScene] = None,
        sequential: bool = True,
        run_time_per: float = 0.55,
    ) -> AnimationGroup:
        """Build and animate the 68-95-99.7 rule shaded regions.

        Three nested shaded zones appear, each with a probability label.

        Parameters
        ----------
        scene : ThreeDScene or None
            Needed to register fixed-orientation labels.
        sequential : bool
            If True, reveal 1σ → 2σ → 3σ in sequence.
        run_time_per : float
            Duration per zone reveal.

        Returns
        -------
        AnimationGroup (or LaggedStart if sequential)
        """
        rule_data = [
            (1, ManimColor("#2DAA6E"), "68.27%"),
            (2, ManimColor("#E0AA40"), "95.45%"),
            (3, ManimColor("#E8593C"), "99.73%"),
        ]
        self.rule_group.remove(*list(self.rule_group))

        anims = []
        for n_sig, col, pct in rule_data:
            zone = self._build_rule_zone(n_sig, col, pct, scene)
            self.rule_group.add(zone)
            anims.append(FadeIn(zone, run_time=run_time_per))

        if sequential:
            return LaggedStart(*anims, lag_ratio=0.4)
        return AnimationGroup(*anims)

    def _build_rule_zone(
        self,
        n_sig: int,
        color: ManimColor,
        label: str,
        scene: Optional[ThreeDScene],
    ) -> VGroup:
        mu, sigma = self.mu, self.sigma
        lo = mu - n_sig * sigma
        hi = mu + n_sig * sigma

        mask = (self._xs >= lo + self.x_pos) & (self._xs <= hi + self.x_pos)
        xs_r = self._xs[mask]
        ys_r = self._ys[mask]

        if len(xs_r) < 2:
            return VGroup()

        # Boundary points
        y_lo = float(_normal_pdf(lo, mu, sigma)) / self._peak * self.z_scale
        y_hi = float(_normal_pdf(hi, mu, sigma)) / self._peak * self.z_scale

        xs_full = np.concatenate([[lo + self.x_pos], xs_r, [hi + self.x_pos]])
        ys_full = np.concatenate([[y_lo], ys_r, [y_hi]])
        pts = np.column_stack([xs_full, np.full(len(xs_full), self.y_pos), ys_full])

        fill = _gradient_fill(pts, self.floor_z, self.y_pos,
                              color, 0.28, n_strips=8)

        grp = VGroup(fill)

        # Probability label
        z_peak_zone = float(np.max(ys_r)) if len(ys_r) > 0 else self.z_scale
        lbl = Text(label, font_size=20, color=color)
        lbl.move_to(np.array([
            (lo + hi) / 2 + self.x_pos,
            self.y_pos,
            z_peak_zone * 0.55,
        ]))
        grp.add(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)

        return grp

    def add_zscore_ruler(
        self,
        scene: Optional[ThreeDScene] = None,
        n_ticks: int = 9,
    ) -> VGroup:
        """Add a z-score ruler beneath the curve.

        Tick marks at every integer z-score from -4 to +4 with labels.
        Tick at z=0 is emphasised.

        Returns the ruler VGroup added to ``self.zscore_group``.
        """
        self.zscore_group.remove(*list(self.zscore_group))
        mu, sigma = self.mu, self.sigma
        y = self.y_pos
        z0 = self.floor_z - 0.35

        ruler_col = _with_opacity(WHITE, 0.40)
        ruler = Line(
            np.array([self._x_min, y, z0]),
            np.array([self._x_max, y, z0]),
            color=ruler_col, stroke_width=1.0,
        )
        self.zscore_group.add(ruler)

        z_vals = np.arange(-4, 5)
        for z_val in z_vals:
            x_pos_tick = mu + z_val * sigma + self.x_pos
            if x_pos_tick < self._x_min or x_pos_tick > self._x_max:
                continue
            h = 0.10 if z_val != 0 else 0.16
            tick_col = _with_opacity(WHITE, 0.65 if z_val == 0 else 0.38)
            tick = Line(
                np.array([x_pos_tick, y, z0 - h / 2]),
                np.array([x_pos_tick, y, z0 + h / 2]),
                color=tick_col,
                stroke_width=1.6 if z_val == 0 else 1.0,
            )
            lbl = Text(str(int(z_val)), font_size=14, color=tick_col)
            lbl.move_to(np.array([x_pos_tick, y, z0 - 0.28]))
            self.zscore_group.add(tick, lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        return self.zscore_group

    def add_title(
        self,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Add a 'N(μ, σ²)' title above the curve peak."""
        self.title_group.remove(*list(self.title_group))
        x_cen = self.mu + self.x_pos
        z_top = self.z_scale * 1.14

        main_lbl = Text(
            f"N(μ={self.mu:.3g}, σ={self.sigma:.3g})",
            font_size=22, color=_with_opacity(self.color, 0.88),
        )
        main_lbl.move_to(np.array([x_cen, self.y_pos, z_top]))
        self.title_group.add(main_lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(main_lbl)
        return self.title_group

    def add_clt_overlay(
        self,
        n_sample: int = 10,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Overlay the sampling distribution of x̄ ~ N(μ, σ²/n).

        The overlay is narrower and taller than the population curve.
        A label "x̄ ~ N(μ, σ/√n)" is added.

        Parameters
        ----------
        n_sample : int
            Sample size n.
        color : ManimColor or None
            Defaults to a lighter version of the curve colour.

        Returns
        -------
        VGroup added to ``self.clt_group``.
        """
        self.clt_group.remove(*list(self.clt_group))
        se = self.sigma / np.sqrt(max(n_sample, 1))
        col = color if color is not None else _lighten(self.color, 1.50)

        xs = np.arange(self._x_min, self._x_max + self._x_step * 0.5, self._x_step)
        ys_clt = _normal_pdf(xs, self.mu, se)
        ys_clt_scaled = ys_clt / float(ys_clt.max()) * self.z_scale if ys_clt.max() > 0 else ys_clt

        pts_clt = _catmull_rom(np.column_stack([
            xs + self.x_pos,
            np.full(len(xs), self.y_pos),
            ys_clt_scaled,
        ]), resolution=18)

        stroke_clt = VMobject()
        stroke_clt.set_points_as_corners(pts_clt)
        stroke_clt.set_stroke(color=_with_opacity(col, 0.88), width=2.5)
        stroke_clt.set_fill(opacity=0)

        glow_clt = VMobject()
        glow_clt.set_points_as_corners(pts_clt)
        glow_clt.set_stroke(color=_with_opacity(col, 0.13), width=8.0)
        glow_clt.set_fill(opacity=0)

        lbl = Text(
            f"x̄ ~ N(μ, σ/√{n_sample})",
            font_size=18, color=col,
        )
        lbl.move_to(np.array([
            self.mu + self.x_pos + self.sigma * 1.8,
            self.y_pos,
            self.z_scale * 0.85,
        ]))
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)

        self.clt_group.add(glow_clt, stroke_clt, lbl)
        return self.clt_group

    def shade_region(
        self,
        x_lo: float,
        x_hi: float,
        color: ManimColor = ManimColor("#E8593C"),
        label: bool = True,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Shade the region under the curve between *x_lo* and *x_hi*.

        Computes P(x_lo ≤ X ≤ x_hi) analytically.
        """
        lo = max(x_lo, self._x_min - self.x_pos)
        hi = min(x_hi, self._x_max - self.x_pos)
        mask = (self._xs >= lo + self.x_pos) & (self._xs <= hi + self.x_pos)
        xs_r = self._xs[mask]
        ys_r = self._ys[mask]

        if len(xs_r) < 2:
            return VGroup()

        y_lo = float(_normal_pdf(lo, self.mu, self.sigma)) / self._peak * self.z_scale
        y_hi = float(_normal_pdf(hi, self.mu, self.sigma)) / self._peak * self.z_scale
        xs_f = np.concatenate([[lo + self.x_pos], xs_r, [hi + self.x_pos]])
        ys_f = np.concatenate([[y_lo], ys_r, [y_hi]])
        pts = np.column_stack([xs_f, np.full(len(xs_f), self.y_pos), ys_f])

        fill = _gradient_fill(pts, self.floor_z, self.y_pos, color, 0.30, n_strips=1)

        p_val = _normal_cdf(hi, self.mu, self.sigma) - _normal_cdf(lo, self.mu, self.sigma)
        grp = VGroup(fill)

        if label:
            z_mid = float(np.max(ys_r)) * 0.55 if len(ys_r) > 0 else self.z_scale * 0.5
            lbl = Text(f"P = {p_val:.4f}", font_size=20, color=color)
            lbl.move_to(np.array([(lo + hi) / 2 + self.x_pos, self.y_pos, z_mid + 0.38]))
            grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        return grp

    # ------------------------------------------------------------------
    # Standardisation animation
    # ------------------------------------------------------------------

    def animate_standardise(
        self,
        run_time: float = 2.0,
        scene: Optional[ThreeDScene] = None,
    ) -> UpdateFromAlphaFunc:
        """Animate N(μ, σ) → N(0, 1) standardisation.

        The curve shifts left by μ and narrows by factor 1/σ.
        Returns an ``UpdateFromAlphaFunc`` that morphs the stroke.

        Note: the caller should update mu/sigma after playing this
        animation if further operations are needed.
        """
        mu0, sigma0 = self.mu, self.sigma
        mu1, sigma1 = 0.0, 1.0

        # Build target curve points
        xs_std = np.arange(self._x_min, self._x_max + self._x_step * 0.5, self._x_step)
        ys_std = _normal_pdf(xs_std, mu1, sigma1)
        peak_std = float(ys_std.max())
        ys_std_scaled = ys_std / peak_std * self.z_scale

        src_pts = _catmull_rom(np.column_stack([
            xs_std + self.x_pos,
            np.full(len(xs_std), self.y_pos),
            _normal_pdf(xs_std, mu0, sigma0) / self._peak * self.z_scale,
        ]), resolution=18)

        tgt_pts = _catmull_rom(np.column_stack([
            xs_std + self.x_pos,
            np.full(len(xs_std), self.y_pos),
            ys_std_scaled,
        ]), resolution=18)

        # Resample to same length
        n = max(len(src_pts), len(tgt_pts))

        def _resample(pts, nn):
            if len(pts) == nn:
                return pts
            idx = np.linspace(0, len(pts) - 1, nn)
            return np.array([pts[int(i)] for i in np.floor(idx).astype(int)])

        src_r = _resample(src_pts, n)
        tgt_r = _resample(tgt_pts, n)
        stroke = self.stroke

        def updater(mob: VMobject, alpha: float) -> None:
            t = rate_functions.ease_in_out_cubic(alpha)
            pts = src_r + (tgt_r - src_r) * t
            mob.set_points_as_corners(pts)
            if hasattr(self, 'glow'):
                self.glow.set_points_as_corners(pts)

        return UpdateFromAlphaFunc(stroke, updater, run_time=run_time)

    # ------------------------------------------------------------------
    # Animation helpers
    # ------------------------------------------------------------------

    def animate_curve(self, run_time: float = 1.5) -> AnimationGroup:
        anims = [Create(self.stroke, run_time=run_time)]
        if hasattr(self, 'glow'):
            anims.append(FadeIn(self.glow, run_time=run_time * 0.4))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def animate_fill(self, run_time: float = 1.0) -> UpdateFromAlphaFunc:
        pts = self._pts
        floor_z = self.floor_z
        y_pos = self.y_pos
        color = self.color
        op = self._fill_opacity
        n_strips = self._n_strips

        def updater(mob: VGroup, alpha: float) -> None:
            t = smooth(alpha)
            interp = np.column_stack([
                pts[:, 0], pts[:, 1],
                floor_z + (pts[:, 2] - floor_z) * t,
            ])
            mob.become(_gradient_fill(interp, floor_z, y_pos, color, op, n_strips))

        return UpdateFromAlphaFunc(self.fill_group, updater, run_time=run_time)

    def animate_moments(self, run_time: float = 0.7) -> FadeIn:
        return FadeIn(self.moments_group, run_time=run_time)

    def full_reveal(
        self,
        scene: ThreeDScene,
        curve_rt: float = 1.5,
        fill_rt: float = 1.0,
        moments_rt: float = 0.7,
    ) -> None:
        scene.play(self.animate_curve(run_time=curve_rt))
        scene.play(self.animate_fill(run_time=fill_rt))
        scene.play(self.animate_moments(run_time=moments_rt))


# ===========================================================================
# BivariateNormal3D
# ===========================================================================

class BivariateNormal3D(VGroup):
    """True 3D parametric surface for the bivariate normal distribution.

    The surface is a dense mesh of quad polygons, each independently
    shaded using a Phong-style light model:
        intensity = ambient + (1 - ambient) × max(0, n̂ · l̂)
    where n̂ is the face normal and l̂ is the light direction.

    Visual layers
    ~~~~~~~~~~~~~
    - ``surface_group``   : all quad polygon patches.
    - ``marginal_x``      : marginal density on the xz wall.
    - ``marginal_y``      : marginal density on the yz wall.
    - ``contour_group``   : isoprobability ellipses at z = 0.
    - ``axes_group``      : thin axis lines in the base plane.
    - ``title_group``     : distribution name label.

    Parameters
    ----------
    mu_x, mu_y : float
        Marginal means.
    sigma_x, sigma_y : float
        Marginal standard deviations.
    rho : float
        Correlation coefficient (−1 < ρ < 1).
    config : BivariateNormalConfig
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        mu_x: float = 0.0,
        mu_y: float = 0.0,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        rho: float = 0.0,
        config: Optional[BivariateNormalConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not -1 < rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {rho}")

        self.mu_x = float(mu_x)
        self.mu_y = float(mu_y)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.rho = float(rho)
        self.cfg = config if config is not None else BivariateNormalConfig()
        self._scene = scene

        # Pre-compute the PDF on a grid
        cfg = self.cfg
        self._xs_1d = np.linspace(cfg.x_range[0], cfg.x_range[1], cfg.n_x + 1)
        self._ys_1d = np.linspace(cfg.y_range[0], cfg.y_range[1], cfg.n_y + 1)
        self._X, self._Y = np.meshgrid(self._xs_1d, self._ys_1d)
        self._Z_raw = _bivariate_normal_pdf(
            self._X, self._Y,
            mu_x, mu_y, sigma_x, sigma_y, rho,
        )
        self._z_peak = float(self._Z_raw.max())
        self._Z = self._Z_raw / self._z_peak * cfg.z_scale

        # Layer groups
        self.surface_group = VGroup()
        self.marginal_x    = VGroup()
        self.marginal_y    = VGroup()
        self.contour_group = VGroup()
        self.axes_group    = VGroup()
        self.title_group   = VGroup()

        # Build all layers
        self._build_surface()
        if cfg.show_marginals:
            self._build_marginal_x()
            self._build_marginal_y()
        self._build_contours()
        self._build_axes()

        self.add(
            self.surface_group,
            self.marginal_x, self.marginal_y,
            self.contour_group,
            self.axes_group, self.title_group,
        )

    # ------------------------------------------------------------------
    # Surface mesh builder
    # ------------------------------------------------------------------

    def _face_normal(
        self,
        p00: np.ndarray,
        p10: np.ndarray,
        p01: np.ndarray,
    ) -> np.ndarray:
        """Compute the outward-facing normal of a quad patch.

        Uses the cross product of two edge vectors.
        """
        v1 = p10 - p00
        v2 = p01 - p00
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-10 else np.array([0, 0, 1.0])

    def _shade(self, normal: np.ndarray) -> float:
        """Return shading intensity ∈ [ambient, 1.0]."""
        cfg = self.cfg
        dot = float(np.dot(normal, cfg.light_direction))
        return cfg.ambient_light + (1.0 - cfg.ambient_light) * max(dot, 0.0)

    def _z_to_color(self, z_val: float) -> ManimColor:
        """Map a scene-z value to the surface colour ramp."""
        t = z_val / self.cfg.z_scale
        return _lerp_color(self.cfg.surface_color_lo, self.cfg.surface_color_hi, t)

    def _build_surface(self) -> None:
        cfg = self.cfg
        Z = self._Z
        xs = self._xs_1d
        ys = self._ys_1d
        ec = _with_opacity(WHITE, cfg.edge_opacity)
        op = cfg.surface_opacity

        for j in range(cfg.n_y):
            for i in range(cfg.n_x):
                # Four corners of quad (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1)
                p00 = np.array([xs[i],   ys[j],   Z[j,   i  ]])
                p10 = np.array([xs[i+1], ys[j],   Z[j,   i+1]])
                p11 = np.array([xs[i+1], ys[j+1], Z[j+1, i+1]])
                p01 = np.array([xs[i],   ys[j+1], Z[j+1, i  ]])

                n_hat = self._face_normal(p00, p10, p01)
                shade = self._shade(n_hat)

                # Colour based on average z of the quad
                z_avg = (Z[j, i] + Z[j, i+1] + Z[j+1, i+1] + Z[j+1, i]) / 4
                base_col = self._z_to_color(z_avg)
                quad_col = _lerp_color(
                    _darken(base_col, 0.4),
                    _lighten(base_col, 1.2),
                    shade,
                )

                quad = Polygon(
                    p00, p10, p11, p01,
                    fill_color=_with_opacity(quad_col, op),
                    fill_opacity=1.0,
                    stroke_color=ec,
                    stroke_width=cfg.edge_stroke_width,
                )
                self.surface_group.add(quad)

    # ------------------------------------------------------------------
    # Marginal curves
    # ------------------------------------------------------------------

    def _build_marginal_x(self) -> None:
        """Project marginal f_X onto the y = y_wall (xz) plane."""
        cfg = self.cfg
        xs = np.linspace(cfg.x_range[0], cfg.x_range[1], 200)
        ys_marg = _normal_pdf(xs, self.mu_x, self.sigma_x)
        z_max = float(ys_marg.max())
        ys_scaled = ys_marg / z_max * cfg.z_scale if z_max > 0 else ys_marg

        pts = np.column_stack([xs, np.full(len(xs), cfg.y_wall), ys_scaled])
        smoothed = _catmull_rom(pts, resolution=14)

        col = _with_opacity(cfg.marginal_color, 0.88)
        stroke = VMobject()
        stroke.set_points_as_corners(smoothed)
        stroke.set_stroke(color=col, width=cfg.marginal_stroke_width)
        stroke.set_fill(opacity=0)
        self.marginal_x.add(stroke)

        if cfg.marginal_fill:
            fill = _gradient_fill(smoothed, 0.0, cfg.y_wall,
                                  cfg.marginal_color, cfg.marginal_fill_opacity, 8)
            self.marginal_x.add(fill)

    def _build_marginal_y(self) -> None:
        """Project marginal f_Y onto the x = x_wall (yz) plane."""
        cfg = self.cfg
        ys = np.linspace(cfg.y_range[0], cfg.y_range[1], 200)
        zs_marg = _normal_pdf(ys, self.mu_y, self.sigma_y)
        z_max = float(zs_marg.max())
        zs_scaled = zs_marg / z_max * cfg.z_scale if z_max > 0 else zs_marg

        pts = np.column_stack([np.full(len(ys), cfg.x_wall), ys, zs_scaled])
        smoothed = _catmull_rom(pts, resolution=14)

        col = _with_opacity(cfg.marginal_color, 0.85)
        stroke = VMobject()
        stroke.set_points_as_corners(smoothed)
        stroke.set_stroke(color=col, width=cfg.marginal_stroke_width)
        stroke.set_fill(opacity=0)
        self.marginal_y.add(stroke)

        if cfg.marginal_fill:
            # Fill in yz-plane: need custom floor at x=x_wall
            floor_pts = np.array([[cfg.x_wall, p[1], 0.0] for p in smoothed])
            fill_grp = VGroup()
            n_strips = 8
            for s in range(n_strips):
                t0, t1 = s / n_strips, (s + 1) / n_strips
                op_s = cfg.marginal_fill_opacity * (1.0 - 0.60 * t0)
                lo = smoothed * (1 - t0) + floor_pts * t0
                hi = smoothed * (1 - t1) + floor_pts * t1
                fill_grp.add(Polygon(
                    *list(lo) + list(reversed(hi)),
                    fill_color=_with_opacity(cfg.marginal_color, op_s),
                    fill_opacity=1.0, stroke_width=0,
                ))
            self.marginal_y.add(fill_grp)

    # ------------------------------------------------------------------
    # Contour ellipses
    # ------------------------------------------------------------------

    def _build_contours(self) -> None:
        """Draw isoprobability contour ellipses at z = 0."""
        cfg = self.cfg
        if not cfg.contour_levels:
            return

        peak_pdf = float(_bivariate_normal_pdf(
            np.array([[self.mu_x]]), np.array([[self.mu_y]]),
            self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho,
        )[0, 0])

        col = _with_opacity(cfg.contour_color, cfg.contour_opacity)
        angles = np.linspace(0, TAU, cfg.n_contour_pts, endpoint=False)

        for level in cfg.contour_levels:
            # The bivariate normal contour at density level c is an ellipse
            # defined by (x-μ)ᵀ Σ⁻¹ (x-μ) = -2 ln(c / peak)
            # We parameterise using the Cholesky decomposition of Σ.
            if level <= 0 or level >= peak_pdf:
                continue

            chi2_val = -2.0 * np.log(level / peak_pdf)
            if chi2_val <= 0:
                continue
            r = np.sqrt(chi2_val)

            # Covariance matrix
            s1, s2, rho = self.sigma_x, self.sigma_y, self.rho
            cov = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                continue

            # Unit circle → ellipse via Cholesky
            circle = r * np.column_stack([np.cos(angles), np.sin(angles)])
            ellipse_xy = (L @ circle.T).T

            pts3d = np.column_stack([
                ellipse_xy[:, 0] + self.mu_x,
                ellipse_xy[:, 1] + self.mu_y,
                np.zeros(len(angles)),
            ])
            # Close the ellipse
            pts3d = np.vstack([pts3d, pts3d[0]])

            contour_mob = VMobject()
            contour_mob.set_points_as_corners(pts3d)
            contour_mob.set_stroke(color=col, width=cfg.contour_stroke_width)
            contour_mob.set_fill(opacity=0)

            # Glow
            glow_mob = VMobject()
            glow_mob.set_points_as_corners(pts3d)
            glow_mob.set_stroke(
                color=_with_opacity(cfg.contour_color, cfg.contour_opacity * 0.20),
                width=cfg.contour_stroke_width * 2.5,
            )
            glow_mob.set_fill(opacity=0)

            self.contour_group.add(glow_mob, contour_mob)

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    def _build_axes(self) -> None:
        """Draw thin x and y axis lines in the z=0 base plane."""
        cfg = self.cfg
        ax_col = _with_opacity(WHITE, 0.28)
        sw = 0.8
        self.axes_group.add(Line(
            np.array([cfg.x_range[0], 0, 0]),
            np.array([cfg.x_range[1], 0, 0]),
            color=ax_col, stroke_width=sw,
        ))
        self.axes_group.add(Line(
            np.array([0, cfg.y_range[0], 0]),
            np.array([0, cfg.y_range[1], 0]),
            color=ax_col, stroke_width=sw,
        ))

    # ------------------------------------------------------------------
    # Conditional slice
    # ------------------------------------------------------------------

    def add_conditional_slice(
        self,
        x0: float,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Show the conditional distribution f(Y | X = x0).

        Draws a vertical 1D normal slice through the surface at x = x0,
        together with a shaded plane highlighting the slice.

        Parameters
        ----------
        x0 : float
            The conditioning value of X.

        Returns
        -------
        VGroup added to the scene but not to any layer group.
        """
        cfg = self.cfg
        # Conditional: Y | X=x0 ~ N(μ_Y|X, σ_Y|X²)
        mu_cond = self.mu_y + self.rho * (self.sigma_y / self.sigma_x) * (x0 - self.mu_x)
        sigma_cond = self.sigma_y * np.sqrt(1 - self.rho**2)

        ys = np.linspace(cfg.y_range[0], cfg.y_range[1], 120)
        zs = _normal_pdf(ys, mu_cond, sigma_cond)
        z_max = float(zs.max())
        zs_sc = zs / z_max * cfg.z_scale * 0.95 if z_max > 0 else zs

        pts = np.column_stack([np.full(len(ys), x0), ys, zs_sc])
        smoothed = _catmull_rom(pts, resolution=14)

        col = _with_opacity(cfg.slice_color, 0.90)
        stroke = VMobject()
        stroke.set_points_as_corners(smoothed)
        stroke.set_stroke(color=col, width=2.4)
        stroke.set_fill(opacity=0)

        glow = VMobject()
        glow.set_points_as_corners(smoothed)
        glow.set_stroke(color=_with_opacity(cfg.slice_color, 0.15), width=8.0)
        glow.set_fill(opacity=0)

        # Shaded plane at x = x0
        plane_pts = [
            np.array([x0, cfg.y_range[0], 0]),
            np.array([x0, cfg.y_range[1], 0]),
            np.array([x0, cfg.y_range[1], cfg.z_scale]),
            np.array([x0, cfg.y_range[0], cfg.z_scale]),
        ]
        plane = Polygon(
            *plane_pts,
            fill_color=_with_opacity(cfg.slice_color, cfg.slice_opacity),
            fill_opacity=1.0, stroke_width=0,
        )

        # Label
        lbl = Text(
            f"f(Y|X={x0:.2g})\n= N({mu_cond:.2g}, {sigma_cond:.2g})",
            font_size=16, color=cfg.slice_color,
        )
        lbl.move_to(np.array([x0, cfg.y_range[1] + 0.3, cfg.z_scale * 0.5]))
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)

        grp = VGroup(plane, glow, stroke, lbl)
        return grp

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_rise(
        self,
        run_time: float = 2.5,
        lag: float = 0.005,
    ) -> LaggedStart:
        """Animate the surface rising from z=0 upward.

        Each quad starts at height 0 and grows to its final z-position.
        Quads are revealed in row-major order (front-to-back).

        Parameters
        ----------
        run_time : float
            Total duration.
        lag : float
            Lag ratio between consecutive quads.
        """
        n_quads = len(self.surface_group)
        rt_per = run_time * 0.25

        return LaggedStart(
            *[FadeIn(q, scale=0.1, run_time=rt_per)
              for q in self.surface_group],
            lag_ratio=lag,
        )

    def animate_contours(
        self,
        lag: float = 0.25,
        run_time_per: float = 0.8,
    ) -> LaggedStart:
        """Draw contour ellipses one by one, innermost first."""
        return LaggedStart(
            *[Create(c, run_time=run_time_per) for c in self.contour_group],
            lag_ratio=lag,
        )

    def animate_marginals(self, run_time: float = 0.9) -> AnimationGroup:
        """Create both marginal curves simultaneously."""
        anims = []
        if len(self.marginal_x) > 0:
            anims.append(Create(self.marginal_x, run_time=run_time))
        if len(self.marginal_y) > 0:
            anims.append(Create(self.marginal_y, run_time=run_time))
        return AnimationGroup(*anims)

    def animate_correlation_sweep(
        self,
        scene: ThreeDScene,
        rho_values: Optional[List[float]] = None,
        run_time_each: float = 1.2,
        hold_each: float = 0.4,
    ) -> None:
        """Rebuild the surface for each value of ρ in *rho_values*.

        Plays animations directly on *scene*.

        Parameters
        ----------
        rho_values : list of float or None
            Defaults to [-0.8, -0.5, 0.0, +0.5, +0.8].
        """
        if rho_values is None:
            rho_values = [-0.8, -0.5, 0.0, 0.5, 0.8]

        for rho_new in rho_values:
            # Build the new surface
            new_surface = BivariateNormal3D(
                mu_x=self.mu_x, mu_y=self.mu_y,
                sigma_x=self.sigma_x, sigma_y=self.sigma_y,
                rho=rho_new,
                config=self.cfg,
            )

            # Crossfade old surface for new surface
            scene.play(
                FadeOut(self.surface_group, run_time=run_time_each * 0.4),
                FadeIn(new_surface.surface_group, run_time=run_time_each * 0.6),
                run_time=run_time_each,
            )
            # Update contours
            scene.play(
                FadeOut(self.contour_group, run_time=0.3),
                FadeIn(new_surface.contour_group, run_time=0.5),
            )

            # Adopt new objects into self
            self.remove(self.surface_group, self.contour_group)
            self.surface_group = new_surface.surface_group
            self.contour_group = new_surface.contour_group
            self.add(self.surface_group, self.contour_group)
            self.rho = rho_new

            scene.wait(hold_each)

    def full_reveal(self, scene: ThreeDScene) -> None:
        """Play the complete layered reveal."""
        scene.play(self.animate_rise(run_time=2.2))
        scene.play(self.animate_marginals(run_time=0.9))
        scene.play(self.animate_contours(lag=0.20, run_time_per=0.75))


# ===========================================================================
# NormalApproximation3D
# ===========================================================================

class NormalApproximation3D(VGroup):
    """Binomial(n, p) PMF bars with the Normal approximation curve overlaid.

    As n grows the normal curve (with mean np and std √(np(1-p))) becomes
    an increasingly accurate envelope over the discrete bars.

    Parameters
    ----------
    n : int
        Number of trials.
    p : float
        Success probability.
    config_bar_color : ManimColor
        Bar colour.
    config_curve_color : ManimColor
        Normal approximation curve colour.
    x_start : float
        Left edge of the bar chart.
    bar_spacing : float
    z_scale : float
    scene : ThreeDScene or None

    Attributes
    ----------
    bars : VGroup
        All bar prisms.
    approx_curve : VMobject
        The normal approximation stroke.
    """

    def __init__(
        self,
        n: int = 20,
        p: float = 0.4,
        bar_color: ManimColor = ManimColor("#4A90D9"),
        curve_color: ManimColor = ManimColor("#E8593C"),
        x_start: float = 0.0,
        bar_spacing: float = 0.0,
        z_scale: float = 4.0,
        y_pos: float = 0.0,
        floor_z: float = 0.0,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n = n
        self.p = p
        self.bar_color = bar_color
        self.curve_color = curve_color
        self.z_scale = z_scale
        self.y_pos = y_pos
        self.floor_z = floor_z
        self._scene = scene

        # Parameters
        mu_norm = n * p
        sigma_norm = np.sqrt(n * p * (1 - p))

        # Spacing
        sp = bar_spacing if bar_spacing > 0 else max(0.45, min(0.90, 6.0 / (n + 1)))
        self._spacing = sp
        total_w = n * sp
        x0 = x_start - total_w / 2

        # Compute PMF using log-gamma (self-contained)
        import math as _math

        def _lgamma(x):
            g = 7
            c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                 771.32342877765313, -176.61502916214059, 12.507343278686905,
                 -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
            if x < 0.5:
                return _math.log(PI / _math.sin(PI * x)) - _lgamma(1 - x)
            x -= 1
            a = c[0]
            t = x + g + 0.5
            for i in range(1, g + 2):
                a += c[i] / (x + i)
            return 0.5 * _math.log(TAU) + (x + 0.5) * _math.log(t) - t + _math.log(a)

        def _binom_pmf(k):
            if k < 0 or k > n or p <= 0 or p >= 1:
                return 0.0
            lp = (_lgamma(n+1) - _lgamma(k+1) - _lgamma(n-k+1)
                  + k*_math.log(p) + (n-k)*_math.log(1-p))
            return _math.exp(lp)

        ks = list(range(n + 1))
        ps_arr = np.array([_binom_pmf(k) for k in ks])
        ps_scaled = ps_arr * z_scale

        bar_w = sp * 0.80
        hd = 0.30
        col_r = _darken(bar_color, 0.62)
        col_t = _lighten(bar_color, 1.30)

        self.bars = VGroup()
        self._bar_xs = []

        for i, (k, h) in enumerate(zip(ks, ps_scaled)):
            x = x0 + i * sp
            self._bar_xs.append(x)
            hw = bar_w / 2
            z0_bar, z1 = floor_z, floor_z + max(h, 1e-4)

            ff = Polygon(
                np.array([x-hw,y_pos-hd,z0_bar]),
                np.array([x+hw,y_pos-hd,z0_bar]),
                np.array([x+hw,y_pos-hd,z1]),
                np.array([x-hw,y_pos-hd,z1]),
                fill_color=_with_opacity(bar_color, 0.88),
                fill_opacity=1.0, stroke_width=0,
            )
            fr = Polygon(
                np.array([x+hw,y_pos-hd,z0_bar]),
                np.array([x+hw,y_pos+hd,z0_bar]),
                np.array([x+hw,y_pos+hd,z1]),
                np.array([x+hw,y_pos-hd,z1]),
                fill_color=_with_opacity(col_r, 0.85),
                fill_opacity=1.0, stroke_width=0,
            )
            ft = Polygon(
                np.array([x-hw,y_pos-hd,z1]),
                np.array([x+hw,y_pos-hd,z1]),
                np.array([x+hw,y_pos+hd,z1]),
                np.array([x-hw,y_pos+hd,z1]),
                fill_color=_with_opacity(col_t, 0.80),
                fill_opacity=1.0, stroke_width=0,
            )
            self.bars.add(VGroup(ff, fr, ft))

        # Normal approximation curve
        x_cont = np.linspace(x0 - sp, x0 + n * sp + sp, 300)
        # Map scene x back to k-values
        k_cont = (x_cont - x0) / sp
        ys_norm = _normal_pdf(k_cont, mu_norm, sigma_norm)
        ys_norm_scaled = ys_norm * z_scale

        pts_norm = np.column_stack([x_cont, np.full(len(x_cont), y_pos), ys_norm_scaled])
        smoothed_norm = _catmull_rom(pts_norm, resolution=14)

        col_c = _with_opacity(curve_color, 0.90)
        self.approx_curve = VMobject()
        self.approx_curve.set_points_as_corners(smoothed_norm)
        self.approx_curve.set_stroke(color=col_c, width=2.8)
        self.approx_curve.set_fill(opacity=0)

        self.approx_glow = VMobject()
        self.approx_glow.set_points_as_corners(smoothed_norm)
        self.approx_glow.set_stroke(color=_with_opacity(curve_color, 0.13), width=9.0)
        self.approx_glow.set_fill(opacity=0)

        self.add(self.bars, self.approx_glow, self.approx_curve)

    # ------------------------------------------------------------------

    def animate_bars(self, lag: float = 0.03, run_time_per: float = 0.45) -> LaggedStart:
        def grow_bar(bar, run_time):
            ff, fr, ft = bar[0], bar[1], bar[2]
            pts = ff.get_all_points()
            z_top = float(pts[:, 2].max()) if len(pts) > 0 else 0.01
            z0 = self.floor_z

            def updater(mob, alpha):
                h = max(smooth(alpha) * z_top, 1e-4)
                z1 = z0 + h
                for face, face_pts in [
                    (mob[0], [[mob[0].get_all_points()[0,0]-0.001,
                               self.y_pos-0.30, z0],
                              [mob[0].get_all_points()[1,0]+0.001,
                               self.y_pos-0.30, z0],
                              [mob[0].get_all_points()[1,0]+0.001,
                               self.y_pos-0.30, z1],
                              [mob[0].get_all_points()[0,0]-0.001,
                               self.y_pos-0.30, z1],
                              [mob[0].get_all_points()[0,0]-0.001,
                               self.y_pos-0.30, z0]]),
                ]:
                    pass  # simplified — bars appear via FadeIn
                mob.set_opacity(smooth(alpha))

            return UpdateFromAlphaFunc(bar, updater, run_time=run_time)

        return LaggedStart(
            *[FadeIn(b, run_time=run_time_per) for b in self.bars],
            lag_ratio=lag,
        )

    def animate_curve(self, run_time: float = 1.2) -> AnimationGroup:
        return AnimationGroup(
            FadeIn(self.approx_glow, run_time=run_time * 0.3),
            Create(self.approx_curve, run_time=run_time),
            lag_ratio=0.0,
        )

    def full_reveal(self, scene: ThreeDScene) -> None:
        scene.play(self.animate_bars(run_time_per=0.40))
        scene.play(self.animate_curve(run_time=1.0))


# ===========================================================================
# QQPlot3D
# ===========================================================================

class QQPlot3D(VGroup):
    """Quantile-quantile plot of sample data against a Normal reference.

    Each data point is a ``Dot3D`` at (theoretical quantile, sample quantile).
    The reference line y = x is drawn in gold.
    An optional 95% envelope is drawn around the reference line.

    Parameters
    ----------
    data : np.ndarray
        Sample data (1-D).
    mu : float or None
        Reference Normal mean.  If None, uses sample mean.
    sigma : float or None
        Reference Normal std.  If None, uses sample std.
    x_scale : float
        Scene x-units per quantile unit.
    z_scale : float
        Scene z-units per quantile unit.
    dot_color : ManimColor
    line_color : ManimColor
    scene : ThreeDScene or None

    Attributes
    ----------
    dots : VGroup
    ref_line : VMobject
    envelope : VGroup
    """

    def __init__(
        self,
        data: np.ndarray,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        x_scale: float = 1.0,
        z_scale: float = 1.0,
        y_pos: float = 0.0,
        dot_color: ManimColor = ManimColor("#4A90D9"),
        line_color: ManimColor = ManimColor("#E0AA40"),
        envelope_color: ManimColor = ManimColor("#E0AA40"),
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        arr = np.sort(np.asarray(data, dtype=float))
        n = len(arr)
        mu_ref = float(mu) if mu is not None else float(arr.mean())
        sigma_ref = float(sigma) if sigma is not None else float(arr.std(ddof=1))

        # Theoretical quantiles via Blom's formula
        # p_i = (i - 0.375) / (n + 0.25)
        probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)

        # Normal PPF via bisection (self-contained)
        def ppf_norm(p, mu, sigma):
            lo, hi = mu - 6 * sigma, mu + 6 * sigma
            for _ in range(60):
                mid = (lo + hi) / 2
                if _normal_cdf(mid, mu, sigma) < p:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2

        theor_q = np.array([ppf_norm(p, mu_ref, sigma_ref) for p in probs])
        sample_q = arr

        # Scale to scene units
        theor_sc = theor_q * x_scale
        sample_sc = sample_q * z_scale

        self.dots = VGroup()
        for tq, sq in zip(theor_sc, sample_sc):
            d = Dot3D(
                point=np.array([tq, y_pos, sq]),
                radius=0.07,
                color=_with_opacity(dot_color, 0.82),
            )
            self.dots.add(d)

        # Reference line: theoretical == sample
        q_min = float(min(theor_sc.min(), sample_sc.min()))
        q_max = float(max(theor_sc.max(), sample_sc.max()))
        self.ref_line = Line(
            np.array([q_min, y_pos, q_min]),
            np.array([q_max, y_pos, q_max]),
            color=_with_opacity(line_color, 0.85),
            stroke_width=2.2,
        )

        # 95% confidence envelope (approximate: ±1.36/√n)
        env_hw = 1.36 / np.sqrt(max(n, 1)) * z_scale
        env_pts_lo = [np.array([tq, y_pos, tq - env_hw]) for tq in theor_sc]
        env_pts_hi = [np.array([tq, y_pos, tq + env_hw]) for tq in theor_sc]

        self.envelope = VGroup()
        env_lo_mob = VMobject()
        env_lo_mob.set_points_as_corners(env_pts_lo)
        env_lo_mob.set_stroke(
            color=_with_opacity(envelope_color, 0.35), width=1.2
        )
        env_lo_mob.set_fill(opacity=0)

        env_hi_mob = VMobject()
        env_hi_mob.set_points_as_corners(env_pts_hi)
        env_hi_mob.set_stroke(
            color=_with_opacity(envelope_color, 0.35), width=1.2
        )
        env_hi_mob.set_fill(opacity=0)

        # Fill between envelopes
        env_fill_pts = env_pts_hi + list(reversed(env_pts_lo))
        env_fill = Polygon(
            *env_fill_pts,
            fill_color=_with_opacity(envelope_color, 0.07),
            fill_opacity=1.0, stroke_width=0,
        )
        self.envelope.add(env_fill, env_lo_mob, env_hi_mob)

        # Axis labels
        xlab = Text("Theoretical Quantiles", font_size=16, color=_with_opacity(WHITE, 0.50))
        xlab.move_to(np.array([0, y_pos, q_min - 0.45]))
        zlab = Text("Sample Quantiles", font_size=16, color=_with_opacity(WHITE, 0.50))
        zlab.move_to(np.array([q_min - 0.5, y_pos, 0]))
        if scene is not None:
            scene.add_fixed_orientation_mobjects(xlab, zlab)

        self.add(self.envelope, self.ref_line, self.dots, xlab, zlab)

    # ------------------------------------------------------------------

    def animate_reveal(
        self,
        lag: float = 0.02,
        run_time_per: float = 0.18,
    ) -> AnimationGroup:
        return AnimationGroup(
            Create(self.ref_line, run_time=0.8),
            FadeIn(self.envelope, run_time=0.6),
            LaggedStart(
                *[FadeIn(d, scale=0.3, run_time=run_time_per) for d in self.dots],
                lag_ratio=lag,
            ),
            lag_ratio=0.2,
        )


# ===========================================================================
# Ready-to-render ThreeDScene subclasses
# ===========================================================================

class NormalCurveScene(ThreeDScene):
    """N(0,1) bell curve with 68-95-99.7 rule and z-score ruler.

    Render:  manim -pql normal3d.py NormalCurveScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        bell = NormalCurve3D(mu=0, sigma=1, z_scale=3.8, scene=self)
        self.add(bell)

        bell.full_reveal(self)

        bell.add_zscore_ruler(scene=self)
        self.play(FadeIn(bell.zscore_group, run_time=0.6))
        self.wait(0.4)

        bell.add_title(scene=self)
        self.play(FadeIn(bell.title_group, run_time=0.5))
        self.wait(0.4)

        self.play(bell.animate_rule_68_95_99(scene=self, sequential=True))
        self.wait(2)


class BivariateNormalScene(ThreeDScene):
    """Bivariate N((0,0), Σ) surface with marginals, contours, and a slice.

    Render:  manim -pql normal3d.py BivariateNormalScene
    """

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.03)

        surf = BivariateNormal3D(
            mu_x=0, mu_y=0, sigma_x=1.2, sigma_y=0.9, rho=0.45,
            config=CRISP_SURFACE, scene=self,
        )
        self.add(surf)
        surf.full_reveal(self)
        self.wait(0.5)

        # Conditional slice at x = 1.0
        slice_grp = surf.add_conditional_slice(x0=1.0, scene=self)
        self.add(slice_grp)
        self.play(FadeIn(slice_grp, run_time=0.8))
        self.wait(2)


class StandardisationScene(ThreeDScene):
    """Animate N(2, 1.5) → N(0, 1) standardisation.

    Render:  manim -pql normal3d.py StandardisationScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        bell = NormalCurve3D(mu=2.0, sigma=1.5, z_scale=3.5, scene=self)
        self.add(bell)
        bell.full_reveal(self)

        lbl_before = Text("N(μ=2, σ=1.5)", font_size=22, color=ManimColor("#4A90D9"))
        lbl_before.move_to(np.array([2.0, 0, 4.1]))
        self.add_fixed_orientation_mobjects(lbl_before)
        self.add(lbl_before)
        self.play(FadeIn(lbl_before, run_time=0.4))
        self.wait(0.5)

        self.play(
            bell.animate_standardise(run_time=2.0, scene=self),
            FadeOut(lbl_before, run_time=0.5),
        )

        lbl_after = Text("N(0, 1)", font_size=22, color=ManimColor("#E0AA40"))
        lbl_after.move_to(np.array([0.0, 0, 4.1]))
        self.add_fixed_orientation_mobjects(lbl_after)
        self.add(lbl_after)
        self.play(FadeIn(lbl_after, run_time=0.4))
        self.wait(2)


class NormalApproximationScene(ThreeDScene):
    """Binomial(n, 0.35) with Normal approximation; n sweeps 10 → 50.

    Render:  manim -pql normal3d.py NormalApproximationScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-52 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.022)

        approx = NormalApproximation3D(
            n=10, p=0.35,
            bar_color=ManimColor("#4A90D9"),
            curve_color=ManimColor("#E8593C"),
            z_scale=4.0,
            scene=self,
        )
        self.add(approx)
        approx.full_reveal(self)
        self.wait(0.5)

        for n_new in [20, 35, 50]:
            new_approx = NormalApproximation3D(
                n=n_new, p=0.35,
                bar_color=ManimColor("#4A90D9"),
                curve_color=ManimColor("#E8593C"),
                z_scale=4.0,
                scene=self,
            )
            self.play(
                FadeOut(approx, run_time=0.5),
                FadeIn(new_approx, run_time=0.5),
            )
            self.remove(approx)
            approx = new_approx
            self.wait(0.6)

        self.wait(2)


class QQPlotScene(ThreeDScene):
    """Q-Q plot: heavy-tailed vs Normal — shows deviation in tails.

    Render:  manim -pql normal3d.py QQPlotScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-48 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        rng = np.random.default_rng(42)
        # Mix of Normal + heavy-tailed noise
        data = np.concatenate([
            rng.normal(0, 1, 80),
            rng.standard_cauchy(20) * 0.3,
        ])

        qq = QQPlot3D(
            data=data, mu=0.0, sigma=1.0,
            x_scale=1.0, z_scale=1.0,
            scene=self,
        )
        self.add(qq)
        self.play(qq.animate_reveal(lag=0.015, run_time_per=0.15))
        self.wait(2)