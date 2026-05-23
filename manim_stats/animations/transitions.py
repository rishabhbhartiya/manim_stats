"""
manim_stats/animations/transitions.py
=======================================
Reusable transition and morphing animations for statistical visualizations.

Design philosophy
-----------------
This module owns animations that *cross the boundary* between chart types —
motions that none of the chart-specific modules can own alone.  Every
function or class here is composable: it takes existing VGroup / VMobject
instances from other modules (``BarChart3D``, ``LineSeries3D``, etc.) and
returns Manim animation objects that the caller plays.

No visual geometry is built here from scratch.  ``transitions.py`` is a
pure animation layer.

Organisation
------------
The module is split into seven conceptual groups:

    1. Curve morphing      – ``DistMorph3D``, ``HistMorph3D``
       Interpolate one distribution shape into another point-by-point.

    2. CDF / PDF coupling  – ``CDFBuild3D``, ``HistToCurve3D``
       Show the relationship between density and cumulative functions.

    3. Parameter sweeps    – ``ParameterSweep3D``
       Animate a distribution changing as μ, σ, λ, p, … vary.

    4. Scene transitions   – ``CurtainReveal3D``, ``SceneWipe3D``
       Structured wipes and reveals for transitioning between topics.

    5. Camera transitions  – ``FocusZoom3D``, ``OrbitTransition``
       Smooth camera repositioning to emphasise a region of interest.

    6. Statistical reveals – ``CIBuild3D``, ``RippleUpdate3D``
       Animations that teach a specific statistical concept through motion.

    7. Cloud / scatter     – ``CollapseToMean3D``, ``ScatterToRegression3D``
       Dot-cloud animations for teaching expectation and regression.

All public classes follow the same interface:
    ``build(*args) → Animation``  or  ``run(scene, *args) → None``
When a transition needs to modify the scene directly (e.g. camera moves)
it uses ``run``; otherwise it returns a composable animation.

Classes
-------
DistMorph3D
HistMorph3D
CDFBuild3D
HistToCurve3D
ParameterSweep3D
CurtainReveal3D
SceneWipe3D
FocusZoom3D
OrbitTransition
CIBuild3D
RippleUpdate3D
CollapseToMean3D
ScatterToRegression3D

Helpers / internals
-------------------
_resample_vmobject_points
_bar_heights_from_group
_build_normal_curve_points
_build_kde_curve_points

Module-level convenience functions
-----------------------------------
dist_morph          – quick DistMorph3D.build() call
hist_morph          – quick HistMorph3D.build() call
cdf_build           – quick CDFBuild3D.build() call
parameter_sweep     – quick ParameterSweep3D.run() call
curtain_reveal      – quick CurtainReveal3D.build() call
ci_build            – quick CIBuild3D.build() call
collapse_to_mean    – quick CollapseToMean3D.build() call

Usage
-----
    from manim_stats.animations.transitions import DistMorph3D, ParameterSweep3D
    from manim_stats.charts.line_plot3d import line_from_function, PDF_LINE
    import numpy as np

    class MyScene(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65*DEGREES, theta=-45*DEGREES)

            # Morph a normal PDF into a t-distribution
            normal_curve = line_from_function(
                lambda x: np.exp(-0.5*x**2) / np.sqrt(2*np.pi),
                x_range=(-4, 4, 0.05), config=PDF_LINE,
            )
            t3_curve = line_from_function(
                lambda x: (1 + x**2/3)**(-2) * 0.608,
                x_range=(-4, 4, 0.05), config=PDF_LINE,
            )
            self.add(normal_curve)
            self.play(
                DistMorph3D(normal_curve.stroke, t3_curve.stroke).build(run_time=1.5)
            )

            # Sweep σ from 0.5 to 2.0
            ParameterSweep3D(
                func=lambda x, s: np.exp(-0.5*(x/s)**2)/(s*np.sqrt(2*np.pi)),
                x_range=(-4, 4, 0.05),
                param_range=(0.5, 2.0, 0.1),
                param_name="σ",
                existing_curve=normal_curve.stroke,
            ).run(self, run_time_total=3.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    List, Sequence, Tuple, Optional, Callable, Union, Dict, Iterator
)
import numpy as np

from manim import (
    VGroup, VMobject, Polygon, Line, DashedLine, Dot3D,
    Text, MathTex, Arrow, Rectangle,
    ThreeDScene,
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, Transform,
    UpdateFromAlphaFunc, Flash,
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
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _resample_vmobject_points(mob: VMobject, n: int) -> np.ndarray:
    """Return *n* evenly-spaced points along *mob*'s path.

    Used to normalise two curves to the same point count so they can be
    interpolated point-by-point without topology jumps.

    Parameters
    ----------
    mob : VMobject
        The source mobject.  Must have at least 2 points.
    n : int
        Number of output points.

    Returns
    -------
    np.ndarray, shape (n, 3)
    """
    raw = mob.get_all_points()
    if len(raw) == 0:
        return np.zeros((n, 3))
    if len(raw) == 1:
        return np.tile(raw[0], (n, 1))

    # Cumulative arc-length parameterisation
    deltas = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    total = cum[-1]
    if total < 1e-9:
        return np.tile(raw[0], (n, 1))

    ts_uniform = np.linspace(0.0, total, n)
    out = np.zeros((n, 3))
    for i, t in enumerate(ts_uniform):
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = np.clip(idx, 0, len(raw) - 2)
        seg_len = deltas[idx] if deltas[idx] > 1e-12 else 1e-12
        frac = (t - cum[idx]) / seg_len
        out[i] = raw[idx] + frac * (raw[idx + 1] - raw[idx])
    return out


def _build_normal_curve_points(
    mu: float,
    sigma: float,
    x_range: Tuple[float, float],
    n_pts: int,
    y_pos: float,
    scale: float,
) -> np.ndarray:
    """Sample a Normal(μ, σ) PDF curve in 3D scene coordinates.

    Parameters
    ----------
    mu, sigma : float
    x_range : (x_min, x_max)
    n_pts : int
    y_pos : float
        Y coordinate (depth).
    scale : float
        Vertical scale factor (z units per PDF unit).

    Returns
    -------
    np.ndarray, shape (n_pts, 3)
    """
    xs = np.linspace(x_range[0], x_range[1], n_pts)
    zs = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(TAU))
    zs = zs * scale
    return np.column_stack([xs, np.full(n_pts, y_pos), zs])


def _build_kde_curve_points(
    data: np.ndarray,
    x_range: Tuple[float, float],
    n_pts: int,
    y_pos: float,
    scale: float,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Evaluate a manual Gaussian KDE over *data* in 3D scene coordinates.

    No scipy dependency.  Uses Silverman's rule-of-thumb for bandwidth
    if *bandwidth* is None.

    Returns
    -------
    np.ndarray, shape (n_pts, 3)
    """
    n = len(data)
    if n == 0:
        return np.zeros((n_pts, 3))

    if bandwidth is None:
        std = float(np.std(data))
        bw = 1.06 * std * n ** (-0.2) if std > 0 else 0.1
    else:
        bw = bandwidth

    xs = np.linspace(x_range[0], x_range[1], n_pts)
    zs = np.array([
        np.mean(np.exp(-0.5 * ((x - data) / bw) ** 2) / (bw * np.sqrt(TAU)))
        for x in xs
    ])
    z_max = zs.max()
    if z_max > 0:
        zs = zs / z_max * scale
    return np.column_stack([xs, np.full(n_pts, y_pos), zs])


def _bar_heights_from_group(bar_group: VGroup) -> List[float]:
    """Extract the z-height of each bar in a BarChart3D bar_group.

    Works by reading the z-coordinate of the top-left corner of each
    bar's front face polygon.  Assumes bars are ``VGroup(front, right, top)``.

    Parameters
    ----------
    bar_group : VGroup
        The ``bar_group`` attribute of a ``BarChart3D``.

    Returns
    -------
    list of float
        One height per bar.
    """
    heights = []
    for bar in bar_group:
        # bar is VGroup(face_front, face_right, face_top)
        # face_front corners: BL, BR, TR, TL — z of TR = bar height
        try:
            pts = bar[0].get_all_points()
            if len(pts) >= 3:
                heights.append(float(pts[2][2]))   # third point = top-right
            else:
                heights.append(0.0)
        except (IndexError, AttributeError):
            heights.append(0.0)
    return heights


# ============================================================================
# 1. Curve morphing
# ============================================================================

class DistMorph3D:
    """Morph one distribution curve into another point-by-point.

    Both curves are resampled to *n_interp_pts* points via arc-length
    parameterisation so the interpolation is smooth and free of
    topology jumps (no sudden re-routing of the path).

    The morph works on the raw ``VMobject`` (the stroke line), not on any
    wrapper class, so it is agnostic of chart type.  Pass
    ``some_lineseries.stroke`` directly.

    Parameters
    ----------
    source : VMobject
        The curve to morph *from*.  Modified in place during animation.
    target : VMobject
        The curve to morph *to*.  Not added to the scene; used only
        for its point geometry.
    n_interp_pts : int
        Number of resampled points.  Higher = smoother but heavier.
        Default 300 is adequate for most distribution curves.
    color_start : ManimColor or None
        Starting stroke colour.  ``None`` preserves *source*'s current colour.
    color_end : ManimColor or None
        Ending stroke colour.  ``None`` preserves *source*'s current colour.
    rate_func : Callable
        Manim rate function for the morph.
    """

    def __init__(
        self,
        source: VMobject,
        target: VMobject,
        n_interp_pts: int = 300,
        color_start: Optional[ManimColor] = None,
        color_end: Optional[ManimColor] = None,
        rate_func: Callable = rate_functions.ease_in_out_cubic,
    ):
        self.source = source
        self._src_pts = _resample_vmobject_points(source, n_interp_pts)
        self._tgt_pts = _resample_vmobject_points(target, n_interp_pts)
        self._n = n_interp_pts
        self._c0 = color_start
        self._c1 = color_end
        self._rate = rate_func

    def build(self, run_time: float = 1.5) -> UpdateFromAlphaFunc:
        """Return an animation that morphs the source curve.

        Parameters
        ----------
        run_time : float

        Returns
        -------
        UpdateFromAlphaFunc
            Play with ``scene.play(morph.build())``.
        """
        src = self._src_pts
        tgt = self._tgt_pts
        c0 = self._c0
        c1 = self._c1
        rate = self._rate

        def updater(mob: VMobject, alpha: float) -> None:
            t = rate(alpha)
            pts = src + (tgt - src) * t
            mob.set_points_as_corners(pts)
            if c0 is not None and c1 is not None:
                col = _lerp_color(c0, c1, t)
                mob.set_stroke(color=col)

        return UpdateFromAlphaFunc(self.source, updater, run_time=run_time)


class HistMorph3D:
    """Morph a bar chart from one set of heights to another.

    Works on the ``bar_group`` VGroup of a ``BarChart3D`` object.
    Each bar's three faces (front, right, top) are reshaped by
    direct point manipulation — the same technique used by
    ``BarChart3D.animate_update``, but generalized here as a
    standalone transition.

    Parameters
    ----------
    bar_group : VGroup
        The ``bar_group`` attribute of a ``BarChart3D``.
    new_heights : sequence of float
        Target z-height for each bar (already in scene units — multiply
        by ``z_scale`` before passing if needed).
    bar_positions : list of (float, float, float, float, float)
        Per-bar geometry: ``(x, y, hw, hd, z_floor)`` where *hw* is
        half-width, *hd* is half-depth.  If None, inferred from
        bar face points.
    color_map : list of ManimColor or None
        Per-bar target colour (optional colour morph).
    rate_func : Callable
    """

    def __init__(
        self,
        bar_group: VGroup,
        new_heights: Sequence[float],
        bar_positions: Optional[List[Tuple[float, float, float, float, float]]] = None,
        color_map: Optional[List[ManimColor]] = None,
        rate_func: Callable = rate_functions.ease_in_out_cubic,
    ):
        self.bar_group = bar_group
        self.new_heights = list(new_heights)
        self._rate = rate_func
        self._color_map = color_map

        # Extract current heights and geometry
        self._old_heights = _bar_heights_from_group(bar_group)

        # Extract bar geometry (x, y, hw, hd, z_floor) per bar
        if bar_positions is not None:
            self._geom = bar_positions
        else:
            self._geom = self._infer_geometry()

    def _infer_geometry(self) -> List[Tuple[float, float, float, float, float]]:
        """Read (x, y, hw, hd, z_floor) from each bar's front face."""
        result = []
        for bar in self.bar_group:
            try:
                pts = bar[0].get_all_points()   # front face
                x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
                y0 = float(pts[:, 1].min())
                z0 = float(pts[:, 2].min())
                x = (x0 + x1) / 2
                hw = (x1 - x0) / 2
                # hd from right face
                try:
                    rpts = bar[1].get_all_points()
                    hd = (float(rpts[:, 1].max()) - float(rpts[:, 1].min())) / 2
                    y = (float(rpts[:, 1].max()) + float(rpts[:, 1].min())) / 2
                except Exception:
                    hd, y = 0.2, y0
                result.append((x, y, hw, hd, z0))
            except Exception:
                result.append((0.0, 0.0, 0.25, 0.2, 0.0))
        return result

    def build(self, run_time: float = 1.2) -> UpdateFromAlphaFunc:
        """Return an animation that reshapes all bars simultaneously.

        Returns
        -------
        UpdateFromAlphaFunc
        """
        old_h = list(self._old_heights)
        new_h = list(self.new_heights)
        geom = self._geom
        rate = self._rate
        bar_group = self.bar_group
        color_map = self._color_map

        def updater(mob: VGroup, alpha: float) -> None:
            t = rate(alpha)
            for i, bar in enumerate(mob):
                if i >= len(old_h) or i >= len(new_h):
                    break
                h = old_h[i] + (new_h[i] - old_h[i]) * t
                h = max(h, 1e-3)
                x, y, hw, hd, z0 = geom[i]
                z1 = z0 + h

                # Front face
                try:
                    bar[0].set_points_as_corners([
                        np.array([x-hw, y-hd, z0]),
                        np.array([x+hw, y-hd, z0]),
                        np.array([x+hw, y-hd, z1]),
                        np.array([x-hw, y-hd, z1]),
                        np.array([x-hw, y-hd, z0]),
                    ])
                    bar[1].set_points_as_corners([
                        np.array([x+hw, y-hd, z0]),
                        np.array([x+hw, y+hd, z0]),
                        np.array([x+hw, y+hd, z1]),
                        np.array([x+hw, y-hd, z1]),
                        np.array([x+hw, y-hd, z0]),
                    ])
                    bar[2].set_points_as_corners([
                        np.array([x-hw, y-hd, z1]),
                        np.array([x+hw, y-hd, z1]),
                        np.array([x+hw, y+hd, z1]),
                        np.array([x-hw, y+hd, z1]),
                        np.array([x-hw, y-hd, z1]),
                    ])
                except (IndexError, AttributeError):
                    pass

                if color_map is not None and i < len(color_map):
                    try:
                        bar[0].set_fill(color=_with_opacity(color_map[i], 0.90))
                    except Exception:
                        pass

        return UpdateFromAlphaFunc(bar_group, updater, run_time=run_time)


# ============================================================================
# 2. CDF / PDF coupling
# ============================================================================

class CDFBuild3D:
    """Animate a CDF curve assembling from the left as a running integral.

    Simultaneously shades the area under the PDF curve with a growing
    fill region.  The two curves must share the same x-axis.

    Parameters
    ----------
    pdf_curve : VMobject
        The already-rendered PDF stroke (e.g. ``LineSeries3D.stroke``).
        Used only for reading point positions — not modified.
    cdf_curve : VMobject
        The CDF stroke to animate building left-to-right.  Modified in place.
    area_fill : VMobject or None
        An ``AreaFill3D`` or plain ``Polygon`` beneath the PDF.  If given,
        it will grow in sync with the CDF.
    x_range : (float, float)
        The data x-range used to align the sweep.
    y_pos : float
        Scene y-coordinate of both curves.
    accent_color : ManimColor
        Colour of the sweep-line indicator that tracks the integral front.
    """

    def __init__(
        self,
        pdf_curve: VMobject,
        cdf_curve: VMobject,
        area_fill: Optional[VMobject] = None,
        x_range: Tuple[float, float] = (-4.0, 4.0),
        y_pos: float = 0.0,
        accent_color: ManimColor = ManimColor("#E0AA40"),
    ):
        self.pdf_curve = pdf_curve
        self.cdf_curve = cdf_curve
        self.area_fill = area_fill
        self.x_range = x_range
        self.y_pos = y_pos
        self.accent_color = accent_color

        # Sample the full CDF curve points
        self._full_cdf_pts = _resample_vmobject_points(cdf_curve, 300)
        self._full_area_pts: Optional[np.ndarray] = None
        if area_fill is not None:
            self._full_area_pts = _resample_vmobject_points(area_fill, 300)

    def build(self, run_time: float = 2.0) -> AnimationGroup:
        """Return an animation that builds the CDF from left to right.

        Returns
        -------
        AnimationGroup
            Contains the CDF draw animation and an optional area-fill grow.
        """
        full_cdf = self._full_cdf_pts
        cdf_mob = self.cdf_curve
        area_mob = self.area_fill
        full_area = self._full_area_pts
        ac = self.accent_color

        # Sweep tracker dot along the CDF
        tracker = Dot3D(
            point=full_cdf[0] if len(full_cdf) > 0 else ORIGIN,
            radius=0.09,
            color=_with_opacity(ac, 0.90),
        )
        tracker_glow = Dot3D(
            point=full_cdf[0] if len(full_cdf) > 0 else ORIGIN,
            radius=0.22,
            color=_with_opacity(ac, 0.18),
        )

        def cdf_updater(mob: VMobject, alpha: float) -> None:
            t = smooth(alpha)
            n_show = max(2, int(t * len(full_cdf)))
            pts = full_cdf[:n_show]
            mob.set_points_as_corners(pts)

            # Move tracker to current front
            front = full_cdf[n_show - 1]
            tracker.move_to(front)
            tracker_glow.move_to(front)

        def area_updater(mob: VMobject, alpha: float) -> None:
            if full_area is None:
                return
            t = smooth(alpha)
            n_show = max(2, int(t * len(full_area)))
            pts = full_area[:n_show]
            mob.set_points_as_corners(pts)

        anims: List = [UpdateFromAlphaFunc(cdf_mob, cdf_updater, run_time=run_time)]

        if area_mob is not None:
            anims.append(UpdateFromAlphaFunc(area_mob, area_updater, run_time=run_time))

        return AnimationGroup(
            FadeIn(tracker, run_time=0.2),
            FadeIn(tracker_glow, run_time=0.2),
            AnimationGroup(*anims),
            lag_ratio=0.1,
        )

    def build_with_scene(
        self,
        scene: ThreeDScene,
        run_time: float = 2.0,
    ) -> None:
        """Add tracker dots and play the animation directly on *scene*."""
        full_cdf = self._full_cdf_pts
        ac = self.accent_color

        tracker = Dot3D(
            point=full_cdf[0] if len(full_cdf) > 0 else ORIGIN,
            radius=0.09, color=_with_opacity(ac, 0.90),
        )
        tracker_glow = Dot3D(
            point=full_cdf[0] if len(full_cdf) > 0 else ORIGIN,
            radius=0.22, color=_with_opacity(ac, 0.18),
        )
        scene.add(tracker, tracker_glow)

        anim = self.build(run_time=run_time)
        scene.play(anim)

        scene.play(FadeOut(tracker, tracker_glow, run_time=0.3))


class HistToCurve3D:
    """Dissolve a bar histogram into a smooth distribution curve.

    Simulates the effect of n → ∞ (more data, finer bins) by:
    1. Simultaneously fading out bar edges (strokes).
    2. Fading out bar right/top faces.
    3. Fading in a smooth KDE/theoretical curve in their place.

    Parameters
    ----------
    bar_group : VGroup
        Bar chart's ``bar_group``.
    target_curve : VMobject
        The smooth curve that emerges.  Must be already constructed but
        not yet added to the scene.
    run_time : float
    overlap_fraction : float
        Fraction of run_time over which bars and curve overlap (crossfade).
    """

    def __init__(
        self,
        bar_group: VGroup,
        target_curve: VMobject,
        run_time: float = 1.8,
        overlap_fraction: float = 0.4,
    ):
        self.bar_group = bar_group
        self.target_curve = target_curve
        self.run_time = run_time
        self.overlap = overlap_fraction

    def build(self) -> AnimationGroup:
        """Return the crossfade animation."""
        bars_fade_rt = self.run_time * (1.0 - self.overlap * 0.5)
        curve_build_rt = self.run_time * (1.0 - self.overlap * 0.5)
        lag = self.run_time * self.overlap

        def bar_fade_updater(mob: VGroup, alpha: float) -> None:
            t = smooth(alpha)
            for bar in mob:
                try:
                    # Fade front face only; keep ghost outlines briefly
                    bar[0].set_fill(opacity=max(0.0, 0.88 * (1 - t)))
                    bar[1].set_fill(opacity=max(0.0, 0.85 * (1 - t)))
                    bar[2].set_fill(opacity=max(0.0, 0.80 * (1 - t)))
                except (IndexError, AttributeError):
                    pass

        return AnimationGroup(
            UpdateFromAlphaFunc(self.bar_group, bar_fade_updater, run_time=bars_fade_rt),
            Succession(
                FadeIn(VGroup(), run_time=lag),   # delay slot
                Create(self.target_curve, run_time=curve_build_rt),
            ),
            lag_ratio=0.0,
        )


# ============================================================================
# 3. Parameter sweeps
# ============================================================================

class ParameterSweep3D:
    """Animate a distribution curve as a single parameter varies.

    On each step the curve is re-evaluated from ``func(x, param)`` and
    the existing stroke VMobject is updated in place via
    ``set_points_as_corners``.  A live annotation shows the current
    parameter value.

    Parameters
    ----------
    func : Callable[[float, float], float]
        ``func(x, param) → z`` — the distribution function.
    x_range : (float, float, float)
        ``(x_min, x_max, x_step)`` for sampling x.
    param_range : (float, float, float)
        ``(param_min, param_max, param_step)`` — the sweep range.
    param_name : str
        Human-readable parameter name shown in the annotation.
    existing_curve : VMobject
        The stroke to update in place.  Must already be in the scene.
    y_pos : float
        Scene y-coordinate of the curve.
    z_scale : float
        Vertical scale factor (scene z per PDF unit).
    annotation_position : np.ndarray or None
        Where to place the live parameter readout.
    annotation_color : ManimColor
    annotation_font_size : int
    """

    def __init__(
        self,
        func: Callable[[float, float], float],
        x_range: Tuple[float, float, float],
        param_range: Tuple[float, float, float],
        param_name: str,
        existing_curve: VMobject,
        y_pos: float = 0.0,
        z_scale: float = 3.5,
        annotation_position: Optional[np.ndarray] = None,
        annotation_color: ManimColor = ManimColor("#E0AA40"),
        annotation_font_size: int = 22,
    ):
        self.func = func
        self.x_range = x_range
        self.param_range = param_range
        self.param_name = param_name
        self.curve = existing_curve
        self.y_pos = y_pos
        self.z_scale = z_scale
        self.ann_pos = (
            annotation_position
            if annotation_position is not None
            else np.array([x_range[1] + 0.5, y_pos, z_scale * 1.1])
        )
        self.ann_color = annotation_color
        self.ann_font_size = annotation_font_size

        # Build the parameter sequence
        p_min, p_max, p_step = param_range
        self._params: np.ndarray = np.arange(p_min, p_max + p_step * 0.5, p_step)

        # Build all curve point arrays up front
        x_min, x_max, x_step = x_range
        self._xs = np.arange(x_min, x_max + x_step * 0.5, x_step)
        self._all_pts: List[np.ndarray] = []
        for p in self._params:
            zs = np.array([max(func(x, p), 0.0) for x in self._xs])
            z_max = zs.max()
            if z_max > 0:
                zs = zs / z_max * z_scale
            pts = np.column_stack([self._xs, np.full(len(self._xs), y_pos), zs])
            self._all_pts.append(pts)

    # ------------------------------------------------------------------

    def run(
        self,
        scene: ThreeDScene,
        run_time_total: float = 3.0,
        hold_at_ends: float = 0.5,
        show_annotation: bool = True,
    ) -> None:
        """Sweep the parameter and update the curve in real time.

        Plays animations directly on *scene*.

        Parameters
        ----------
        run_time_total : float
            Total duration of the entire sweep.
        hold_at_ends : float
            Extra hold time at the first and last parameter values.
        show_annotation : bool
            Whether to display and update the parameter readout.
        """
        n_steps = len(self._params)
        rt_per_step = (run_time_total - 2 * hold_at_ends) / max(n_steps - 1, 1)

        # Build annotation label
        ann_label: Optional[Text] = None
        if show_annotation:
            ann_label = Text(
                f"{self.param_name} = {self._params[0]:.3f}",
                font_size=self.ann_font_size,
                color=self.ann_color,
            )
            ann_label.move_to(self.ann_pos)
            scene.add_fixed_orientation_mobjects(ann_label)
            scene.add(ann_label)
            scene.play(FadeIn(ann_label, run_time=0.2))

        # Hold at first parameter
        scene.wait(hold_at_ends)

        for i in range(n_steps - 1):
            src_pts = self._all_pts[i]
            tgt_pts = self._all_pts[i + 1]
            next_param = self._params[i + 1]
            curve_ref = self.curve

            def make_step_updater(sp, tp):
                def updater(mob: VMobject, alpha: float) -> None:
                    t = smooth(alpha)
                    pts = sp + (tp - sp) * t
                    mob.set_points_as_corners(pts)
                return updater

            step_anim = UpdateFromAlphaFunc(
                curve_ref,
                make_step_updater(src_pts, tgt_pts),
                run_time=rt_per_step,
            )

            if ann_label is not None:
                # Build new label for next param step
                new_lbl = Text(
                    f"{self.param_name} = {next_param:.3f}",
                    font_size=self.ann_font_size,
                    color=self.ann_color,
                )
                new_lbl.move_to(self.ann_pos)
                scene.add_fixed_orientation_mobjects(new_lbl)

                scene.play(
                    step_anim,
                    Transform(ann_label, new_lbl, run_time=rt_per_step),
                )
            else:
                scene.play(step_anim)

        # Hold at final parameter
        scene.wait(hold_at_ends)

        if ann_label is not None:
            scene.play(FadeOut(ann_label, run_time=0.25))

    def build_single_step(
        self,
        from_index: int,
        to_index: int,
        run_time: float = 0.5,
    ) -> UpdateFromAlphaFunc:
        """Return an animation for one step of the sweep.

        Useful when the caller wants manual control over the sequence.
        """
        src = self._all_pts[from_index]
        tgt = self._all_pts[to_index]
        curve = self.curve

        def updater(mob: VMobject, alpha: float) -> None:
            t = smooth(alpha)
            mob.set_points_as_corners(src + (tgt - src) * t)

        return UpdateFromAlphaFunc(curve, updater, run_time=run_time)


# ============================================================================
# 4. Scene transitions
# ============================================================================

class CurtainReveal3D:
    """A translucent plane sweeps across a VGroup, revealing it.

    The curtain is a flat polygon in the yz-plane that travels from
    ``x_start`` to ``x_end``.  As it passes each element, that element
    fades in, creating a left-to-right reveal.

    Parameters
    ----------
    target : VGroup
        The group to reveal.
    x_start : float
        Starting x position of the curtain (usually left edge of target).
    x_end : float
        Ending x position of the curtain (usually right edge of target).
    y_range : (float, float)
        Vertical (y) extent of the curtain plane.
    z_range : (float, float)
        Height (z) extent of the curtain plane.
    curtain_color : ManimColor
    curtain_opacity : float
    reveal_mode : str
        ``"left_to_right"`` | ``"right_to_left"`` | ``"bottom_to_top"``
        | ``"top_to_bottom"``.
    """

    def __init__(
        self,
        target: VGroup,
        x_start: float = -4.0,
        x_end: float = 4.0,
        y_range: Tuple[float, float] = (-1.0, 1.0),
        z_range: Tuple[float, float] = (0.0, 5.0),
        curtain_color: ManimColor = ManimColor("#4A90D9"),
        curtain_opacity: float = 0.18,
        reveal_mode: str = "left_to_right",
    ):
        self.target = target
        self.x_start = x_start
        self.x_end = x_end
        self.y_range = y_range
        self.z_range = z_range
        self.curtain_color = curtain_color
        self.curtain_opacity = curtain_opacity
        self.mode = reveal_mode

    def build(self, run_time: float = 1.5) -> AnimationGroup:
        """Return the curtain reveal animation.

        The target is set to opacity 0 initially; the animation fades in
        each sub-element as the curtain passes over it.

        Returns
        -------
        AnimationGroup
        """
        target = self.target
        x0, x1 = self.x_start, self.x_end
        y0, y1 = self.y_range
        z0, z1 = self.z_range

        # Build the curtain plane
        curtain = Polygon(
            np.array([x0, y0, z0]),
            np.array([x0, y1, z0]),
            np.array([x0, y1, z1]),
            np.array([x0, y0, z1]),
            fill_color=_with_opacity(self.curtain_color, self.curtain_opacity),
            fill_opacity=1.0,
            stroke_width=0,
        )

        # Hide target initially
        target.set_opacity(0)

        def curtain_updater(mob: Polygon, alpha: float) -> None:
            t = smooth(alpha)
            x_cur = x0 + (x1 - x0) * t
            mob.set_points_as_corners([
                np.array([x_cur, y0, z0]),
                np.array([x_cur, y1, z0]),
                np.array([x_cur, y1, z1]),
                np.array([x_cur, y0, z1]),
                np.array([x_cur, y0, z0]),
            ])
            # Reveal target elements that the curtain has passed
            for sub in target:
                try:
                    sub_x = sub.get_center()[0]
                    revealed = sub_x <= x_cur
                    sub.set_opacity(1.0 if revealed else 0.0)
                except Exception:
                    pass

        return AnimationGroup(
            FadeIn(curtain, run_time=0.2),
            UpdateFromAlphaFunc(curtain, curtain_updater, run_time=run_time),
            FadeOut(curtain, run_time=0.25),
            lag_ratio=0.05,
        )


class SceneWipe3D:
    """Wipe between two VGroups using a sweeping divider plane.

    The divider plane travels from one side of the scene to the other.
    Group A fades out behind the wipe; Group B fades in ahead of it.
    Useful for transitioning between two different distribution views.

    Parameters
    ----------
    group_out : VGroup
        The current scene content (fades out behind the wipe).
    group_in : VGroup
        The new scene content (fades in ahead of the wipe).
    direction : str
        ``"right"`` | ``"left"`` | ``"up"`` | ``"down"``.
    divider_color : ManimColor
    divider_opacity : float
    scene_width : float
        Total scene width in scene units.
    scene_height : float
        Total scene height in scene units.
    """

    def __init__(
        self,
        group_out: VGroup,
        group_in: VGroup,
        direction: str = "right",
        divider_color: ManimColor = ManimColor("#E0AA40"),
        divider_opacity: float = 0.55,
        scene_width: float = 14.0,
        scene_height: float = 8.0,
    ):
        self.group_out = group_out
        self.group_in = group_in
        self.direction = direction
        self.div_color = divider_color
        self.div_opacity = divider_opacity
        self.sw = scene_width
        self.sh = scene_height

    def build(self, run_time: float = 1.4) -> AnimationGroup:
        """Return the wipe transition animation."""
        g_out = self.group_out
        g_in = self.group_in
        sw, sh = self.sw, self.sh
        col = _with_opacity(self.div_color, self.div_opacity)

        # Build divider as a thin vertical rectangle
        if self.direction in ("right", "left"):
            x_start = -sw / 2 if self.direction == "right" else sw / 2
            x_end = sw / 2 if self.direction == "right" else -sw / 2
            divider = Polygon(
                np.array([x_start, -sh / 2, 0]),
                np.array([x_start,  sh / 2, 0]),
                np.array([x_start,  sh / 2, sh]),
                np.array([x_start, -sh / 2, sh]),
                fill_color=col, fill_opacity=1.0, stroke_width=0,
            )
        else:
            x_start = -sw / 2
            x_end = sw / 2
            divider = Polygon(
                np.array([-sw/2, -sh/2, 0]),
                np.array([ sw/2, -sh/2, 0]),
                fill_color=col, fill_opacity=1.0, stroke_width=0,
            )

        g_in.set_opacity(0)
        sign = 1 if self.direction == "right" else -1

        def wipe_updater(mob: Polygon, alpha: float) -> None:
            t = smooth(alpha)
            x_cur = x_start + (x_end - x_start) * t
            mob.set_points_as_corners([
                np.array([x_cur, -sh/2, 0]),
                np.array([x_cur,  sh/2, 0]),
                np.array([x_cur,  sh/2, sh]),
                np.array([x_cur, -sh/2, sh]),
                np.array([x_cur, -sh/2, 0]),
            ])
            # Out fades behind wipe, in reveals ahead
            for sub in g_out:
                try:
                    sx = sub.get_center()[0]
                    sub.set_opacity(max(0.0, 1.0 - max(0, sign * (sx - x_cur)) * 4.0))
                except Exception:
                    pass
            for sub in g_in:
                try:
                    sx = sub.get_center()[0]
                    sub.set_opacity(max(0.0, min(1.0, sign * (x_cur - sx) * 4.0)))
                except Exception:
                    pass

        return AnimationGroup(
            FadeIn(divider, run_time=0.15),
            UpdateFromAlphaFunc(divider, wipe_updater, run_time=run_time),
            FadeOut(divider, run_time=0.20),
            lag_ratio=0.05,
        )


# ============================================================================
# 5. Camera transitions
# ============================================================================

class FocusZoom3D:
    """Smoothly reposition the camera to focus on a region of interest.

    Moves the camera to a new ``(phi, theta)`` orientation while also
    adjusting the focal point so a target 3D position is centred.

    Parameters
    ----------
    target_phi : float
        Destination polar angle (radians).
    target_theta : float
        Destination azimuthal angle (radians).
    target_zoom : float
        Destination zoom level (1.0 = default).
    focus_point : np.ndarray or None
        3D point to centre the camera on.  If None, camera stays centred.
    run_time : float

    Notes
    -----
    Camera moves in ThreeDScene use ``scene.move_camera``.  Call
    ``focus.run(scene)`` rather than ``scene.play(focus.build())``.
    """

    def __init__(
        self,
        target_phi: float = 65 * DEGREES,
        target_theta: float = -45 * DEGREES,
        target_zoom: float = 1.0,
        focus_point: Optional[np.ndarray] = None,
        run_time: float = 1.5,
    ):
        self.phi = target_phi
        self.theta = target_theta
        self.zoom = target_zoom
        self.focus = focus_point
        self.run_time = run_time

    def run(self, scene: ThreeDScene, added_anims: Optional[List] = None) -> None:
        """Reposition the camera directly on *scene*.

        Parameters
        ----------
        added_anims : list or None
            Additional animations to play simultaneously with the camera move.
        """
        move_kwargs: Dict = dict(
            phi=self.phi,
            theta=self.theta,
            zoom=self.zoom,
            run_time=self.run_time,
        )
        if self.focus is not None:
            move_kwargs["frame_center"] = self.focus

        scene.move_camera(
            added_anims=added_anims or [],
            **move_kwargs,
        )


class OrbitTransition:
    """Orbit the camera around the scene by a given angular displacement.

    Parameters
    ----------
    delta_theta : float
        Change in azimuthal angle (radians).  Positive = orbit right.
    delta_phi : float
        Change in polar angle (radians).  Positive = tilt down.
    run_time : float
    rate_func : Callable
    """

    def __init__(
        self,
        delta_theta: float = PI / 4,
        delta_phi: float = 0.0,
        run_time: float = 2.0,
        rate_func: Callable = rate_functions.ease_in_out_sine,
    ):
        self.d_theta = delta_theta
        self.d_phi = delta_phi
        self.run_time = run_time
        self.rate_func = rate_func

    def run(
        self,
        scene: ThreeDScene,
        added_anims: Optional[List] = None,
    ) -> None:
        """Orbit directly on *scene*, optionally with other simultaneous anims."""
        try:
            cur_phi = scene.camera.phi
            cur_theta = scene.camera.theta
        except AttributeError:
            cur_phi = 65 * DEGREES
            cur_theta = -45 * DEGREES

        scene.move_camera(
            phi=cur_phi + self.d_phi,
            theta=cur_theta + self.d_theta,
            run_time=self.run_time,
            rate_func=self.rate_func,
            added_anims=added_anims or [],
        )


# ============================================================================
# 6. Statistical reveals
# ============================================================================

class CIBuild3D:
    """Animate a confidence interval extending symmetrically from the mean.

    Builds a CI visualisation in three beats:
        Beat 1 – Point estimate appears (dot at x̄).
        Beat 2 – Error bars extend outward left and right.
        Beat 3 – Shaded interval band fades in between the bars.

    Parameters
    ----------
    mean_pos : np.ndarray
        3D position of the point estimate (x̄).
    half_width : float
        Half-width of the CI in scene units.
    y_pos : float
        Scene y-coordinate.
    ci_color : ManimColor
    point_color : ManimColor
    bar_stroke_width : float
    band_opacity : float
    ci_label : str or None
        Text label (e.g. ``"95% CI"``).
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        mean_pos: np.ndarray,
        half_width: float,
        y_pos: float = 0.0,
        ci_color: ManimColor = ManimColor("#2DAA6E"),
        point_color: ManimColor = WHITE,
        bar_stroke_width: float = 2.5,
        band_opacity: float = 0.20,
        ci_label: Optional[str] = None,
        scene: Optional[ThreeDScene] = None,
    ):
        self.mean_pos = np.array(mean_pos, dtype=float)
        self.half_width = half_width
        self.y_pos = y_pos
        self.ci_color = ci_color
        self.point_color = point_color
        self.bar_sw = bar_stroke_width
        self.band_opacity = band_opacity
        self.ci_label = ci_label
        self._scene = scene

        mx, my, mz = self.mean_pos
        col = _with_opacity(ci_color, 0.85)
        col_band = _with_opacity(ci_color, band_opacity)

        # Point estimate dot
        self.dot = Dot3D(
            point=self.mean_pos, radius=0.10,
            color=_with_opacity(point_color, 0.95),
        )
        self.dot_glow = Dot3D(
            point=self.mean_pos, radius=0.25,
            color=_with_opacity(point_color, 0.15),
        )

        # Left error bar: vertical cap + horizontal arm
        left_x = mx - half_width
        right_x = mx + half_width
        cap_h = 0.18

        self.left_cap = Line(
            np.array([left_x, y_pos, mz - cap_h]),
            np.array([left_x, y_pos, mz + cap_h]),
            color=col, stroke_width=bar_stroke_width,
        )
        self.right_cap = Line(
            np.array([right_x, y_pos, mz - cap_h]),
            np.array([right_x, y_pos, mz + cap_h]),
            color=col, stroke_width=bar_stroke_width,
        )
        self.arm = Line(
            np.array([left_x, y_pos, mz]),
            np.array([right_x, y_pos, mz]),
            color=col, stroke_width=bar_stroke_width * 0.7,
        )

        # Shaded band
        hd = 0.30
        self.band = Polygon(
            np.array([left_x,  y_pos - hd, mz - cap_h]),
            np.array([right_x, y_pos - hd, mz - cap_h]),
            np.array([right_x, y_pos - hd, mz + cap_h]),
            np.array([left_x,  y_pos - hd, mz + cap_h]),
            fill_color=col_band,
            fill_opacity=1.0,
            stroke_width=0,
        )

        # Label
        self.label = VGroup()
        if ci_label is not None:
            lbl = Text(ci_label, font_size=18, color=ci_color)
            lbl.move_to(np.array([mx, y_pos, mz + cap_h + 0.28]))
            self.label.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

    def build(self, run_time: float = 1.2) -> Succession:
        """Return a three-beat CI build animation.

        Returns
        -------
        Succession
        """
        rt = run_time / 3

        # Beat 1: point estimate
        beat1 = AnimationGroup(
            FadeIn(self.dot_glow, scale=0.3, run_time=rt * 0.6),
            FadeIn(self.dot, scale=0.3, run_time=rt * 0.6),
        )

        # Beat 2: arms extend
        left_arm = Line(
            self.mean_pos,
            np.array([self.mean_pos[0] - self.half_width,
                      self.y_pos, self.mean_pos[2]]),
            color=_with_opacity(self.ci_color, 0.85),
            stroke_width=self.bar_sw * 0.7,
        )
        right_arm = Line(
            self.mean_pos,
            np.array([self.mean_pos[0] + self.half_width,
                      self.y_pos, self.mean_pos[2]]),
            color=_with_opacity(self.ci_color, 0.85),
            stroke_width=self.bar_sw * 0.7,
        )

        beat2 = AnimationGroup(
            Create(left_arm, run_time=rt * 0.8),
            Create(right_arm, run_time=rt * 0.8),
            Create(self.left_cap, run_time=rt),
            Create(self.right_cap, run_time=rt),
            lag_ratio=0.3,
        )

        # Beat 3: band fades in + label
        beat3 = AnimationGroup(
            FadeIn(self.band, run_time=rt),
            FadeIn(self.label, run_time=rt * 0.7),
        )

        return Succession(beat1, beat2, beat3)


class RippleUpdate3D:
    """When a bar or curve point is updated, emit a ripple through neighbours.

    A colour pulse radiates outward from the updated bar, dimming as it
    travels.  Used to visually signal which part of a distribution changed.

    Parameters
    ----------
    bar_group : VGroup
        Bar chart's ``bar_group``.
    center_index : int
        Index of the bar that was updated (ripple origin).
    ripple_color : ManimColor
    ripple_width : int
        Number of bars the ripple reaches.
    """

    def __init__(
        self,
        bar_group: VGroup,
        center_index: int,
        ripple_color: ManimColor = ManimColor("#FFD700"),
        ripple_width: int = 5,
    ):
        self.bar_group = bar_group
        self.center = center_index
        self.ripple_color = ripple_color
        self.ripple_width = ripple_width

    def build(self, run_time: float = 0.8) -> UpdateFromAlphaFunc:
        """Return the ripple animation."""
        bg = self.bar_group
        c = self.center
        rw = self.ripple_width
        col = self.ripple_color
        n_bars = len(bg)

        # Pre-compute original face colours
        original_fills: List[Optional[ManimColor]] = []
        for bar in bg:
            try:
                original_fills.append(bar[0].get_fill_color())
            except Exception:
                original_fills.append(None)

        def updater(mob: VGroup, alpha: float) -> None:
            # Ripple front position (fractional bar index)
            ripple_front = alpha * (rw + 1)
            for i, bar in enumerate(mob):
                dist = abs(i - c)
                # Pulse intensity: peaks as front passes, then decays
                diff = ripple_front - dist
                if 0 <= diff <= 1.0:
                    intensity = smooth(diff) * (1.0 - alpha * 0.5)
                else:
                    intensity = 0.0
                try:
                    if intensity > 0.02:
                        bar[0].set_fill(
                            color=_lerp_color(
                                original_fills[i] or WHITE,
                                col,
                                intensity,
                            )
                        )
                    else:
                        if original_fills[i] is not None:
                            bar[0].set_fill(color=original_fills[i])
                except Exception:
                    pass

        return UpdateFromAlphaFunc(bg, updater, run_time=run_time)


# ============================================================================
# 7. Cloud / scatter transitions
# ============================================================================

class CollapseToMean3D:
    """Animate all dots in a cloud converging to the mean position.

    Each dot travels along a Bézier arc toward the mean.  Used to
    illustrate E[X] visually — all individual values collapse to a
    single expected value.

    Parameters
    ----------
    dots : list of VGroup or Dot3D
        Each element must respond to ``move_to`` and ``get_center``.
    mean_position : np.ndarray
        3D position to converge toward.
    arc_height : float
        Arc height above the direct path to the mean.
    trail_color : ManimColor
        Colour of the arc trail (optional ghosting effect).
    """

    def __init__(
        self,
        dots: List,
        mean_position: np.ndarray,
        arc_height: float = 0.6,
        trail_color: ManimColor = ManimColor("#4A90D9"),
    ):
        self.dots = dots
        self.mean = np.array(mean_position, dtype=float)
        self.arc_height = arc_height
        self.trail_color = trail_color

        # Pre-compute arc control points
        self._arcs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for dot in dots:
            try:
                start = dot.get_center().copy()
            except AttributeError:
                start = ORIGIN.copy()
            end = self.mean.copy()
            mid = (start + end) / 2 + np.array([0, 0, arc_height])
            self._arcs.append((start, mid, end))

    def build(
        self,
        run_time: float = 1.5,
        lag: float = 0.02,
    ) -> LaggedStart:
        """Return a staggered animation of all dots converging.

        Parameters
        ----------
        run_time : float
            Duration of each individual dot's animation.
        lag : float
            Time between dot start times.

        Returns
        -------
        LaggedStart
        """
        anims = []
        for dot, (start, mid, end) in zip(self.dots, self._arcs):
            def make_updater(s, m, e, d):
                def updater(mob, alpha: float) -> None:
                    t = smooth(alpha)
                    pos = (1 - t)**2 * s + 2 * (1 - t) * t * m + t**2 * e
                    mob.move_to(pos)
                return updater

            anims.append(UpdateFromAlphaFunc(dot, make_updater(start, mid, end, dot),
                                              run_time=run_time))

        return LaggedStart(*anims, lag_ratio=lag)

    def build_flash_at_mean(
        self,
        color: Optional[ManimColor] = None,
    ) -> Flash:
        """Return a Flash at the mean position for use after collapse."""
        col = color if color is not None else self.trail_color
        dot = Dot3D(point=self.mean, radius=0.12, color=col)
        return Flash(dot, color=col, flash_radius=0.5, run_time=0.5)


class ScatterToRegression3D:
    """Animate scatter-plot dots rearranging to lie on a regression line.

    Each dot travels from its current position to its predicted value
    on the regression line y_hat = a + b*x.  The residual distance is
    preserved visually before and after.

    Parameters
    ----------
    dots : list of VGroup or Dot3D
        One dot per data point.  Each must respond to ``move_to``.
    x_values : np.ndarray
        X coordinate of each data point (parallel to *dots*).
    y_values : np.ndarray
        Actual Y (z in scene) coordinate of each data point.
    a : float
        Regression intercept.
    b : float
        Regression slope.
    x_to_scene : Callable[[float], float]
        Map data-x to scene-x.
    z_to_scene : Callable[[float], float]
        Map data-y to scene-z.
    y_pos : float
        Scene y-coordinate.
    residual_color : ManimColor
        Colour of residual lines drawn from fitted to actual position.
    show_residuals : bool
        Whether to show residual lines before collapsing.
    """

    def __init__(
        self,
        dots: List,
        x_values: np.ndarray,
        y_values: np.ndarray,
        a: float,
        b: float,
        x_to_scene: Callable[[float], float],
        z_to_scene: Callable[[float], float],
        y_pos: float = 0.0,
        residual_color: ManimColor = ManimColor("#E8593C"),
        show_residuals: bool = True,
    ):
        self.dots = dots
        self.xs = np.asarray(x_values, dtype=float)
        self.ys = np.asarray(y_values, dtype=float)
        self.a = a
        self.b = b
        self.x_to_scene = x_to_scene
        self.z_to_scene = z_to_scene
        self.y_pos = y_pos
        self.residual_color = residual_color
        self.show_residuals = show_residuals

        # Compute fitted positions
        y_hat = a + b * self.xs
        self._fitted_positions: List[np.ndarray] = [
            np.array([x_to_scene(xi), y_pos, z_to_scene(yhi)])
            for xi, yhi in zip(self.xs, y_hat)
        ]

    def build_residual_lines(self) -> VGroup:
        """Build residual line segments (actual → fitted).

        Returns
        -------
        VGroup of Lines — add to scene before animating.
        """
        residuals = VGroup()
        col = _with_opacity(self.residual_color, 0.55)
        for dot, fitted_pos in zip(self.dots, self._fitted_positions):
            try:
                actual_pos = dot.get_center()
            except AttributeError:
                actual_pos = fitted_pos
            resid = DashedLine(
                actual_pos, fitted_pos,
                dash_length=0.06, dashed_ratio=0.5,
                color=col, stroke_width=1.2,
            )
            residuals.add(resid)
        return residuals

    def build_collapse(
        self,
        run_time: float = 1.4,
        lag: float = 0.02,
    ) -> LaggedStart:
        """Animate all dots moving to their fitted positions.

        Returns
        -------
        LaggedStart
        """
        anims = []
        for dot, fitted in zip(self.dots, self._fitted_positions):
            try:
                start = dot.get_center().copy()
            except AttributeError:
                start = fitted.copy()

            def make_updater(s, f):
                def updater(mob, alpha: float) -> None:
                    t = smooth(alpha)
                    mob.move_to(s + (f - s) * t)
                return updater

            anims.append(UpdateFromAlphaFunc(dot, make_updater(start, fitted),
                                              run_time=run_time))

        return LaggedStart(*anims, lag_ratio=lag)

    def run(
        self,
        scene: ThreeDScene,
        run_time_residuals: float = 0.7,
        run_time_collapse: float = 1.4,
        hold: float = 0.5,
    ) -> None:
        """Show residuals, then collapse dots to regression line.

        Parameters
        ----------
        run_time_residuals : float
            Duration of residual line appearance.
        run_time_collapse : float
            Duration of dot collapse.
        hold : float
            Pause between residuals and collapse.
        """
        if self.show_residuals:
            residuals = self.build_residual_lines()
            scene.add(residuals)
            scene.play(Create(residuals, run_time=run_time_residuals))
            scene.wait(hold)

        scene.play(self.build_collapse(run_time=run_time_collapse))

        if self.show_residuals:
            scene.play(FadeOut(residuals, run_time=0.35))


# ============================================================================
# Module-level convenience functions
# ============================================================================

def dist_morph(
    source: VMobject,
    target: VMobject,
    run_time: float = 1.5,
    n_interp_pts: int = 300,
    color_start: Optional[ManimColor] = None,
    color_end: Optional[ManimColor] = None,
    rate_func: Callable = rate_functions.ease_in_out_cubic,
) -> UpdateFromAlphaFunc:
    """One-liner wrapper for ``DistMorph3D.build()``.

    Parameters
    ----------
    source : VMobject
        Curve to morph from (modified in place).
    target : VMobject
        Curve to morph to (read-only).
    run_time : float
    n_interp_pts : int
    color_start, color_end : ManimColor or None
    rate_func : Callable

    Returns
    -------
    UpdateFromAlphaFunc — ready for ``scene.play()``.

    Example
    -------
    ::

        scene.play(dist_morph(normal_stroke, t3_stroke, run_time=1.5))
    """
    return DistMorph3D(
        source, target,
        n_interp_pts=n_interp_pts,
        color_start=color_start,
        color_end=color_end,
        rate_func=rate_func,
    ).build(run_time=run_time)


def hist_morph(
    bar_group: VGroup,
    new_heights: Sequence[float],
    run_time: float = 1.2,
    color_map: Optional[List[ManimColor]] = None,
    rate_func: Callable = rate_functions.ease_in_out_cubic,
) -> UpdateFromAlphaFunc:
    """One-liner wrapper for ``HistMorph3D.build()``.

    Parameters
    ----------
    bar_group : VGroup
        ``bar_group`` of a ``BarChart3D``.
    new_heights : sequence of float
        Target heights in scene units.
    run_time : float
    color_map : list of ManimColor or None
    rate_func : Callable

    Returns
    -------
    UpdateFromAlphaFunc

    Example
    -------
    ::

        scene.play(hist_morph(chart.bar_group, [1.5, 2.8, 3.1], run_time=1.0))
    """
    return HistMorph3D(
        bar_group, new_heights,
        color_map=color_map, rate_func=rate_func,
    ).build(run_time=run_time)


def cdf_build(
    pdf_curve: VMobject,
    cdf_curve: VMobject,
    area_fill: Optional[VMobject] = None,
    x_range: Tuple[float, float] = (-4.0, 4.0),
    run_time: float = 2.0,
) -> AnimationGroup:
    """One-liner wrapper for ``CDFBuild3D.build()``.

    Example
    -------
    ::

        scene.play(cdf_build(pdf.stroke, cdf.stroke, run_time=2.0))
    """
    return CDFBuild3D(
        pdf_curve, cdf_curve,
        area_fill=area_fill,
        x_range=x_range,
    ).build(run_time=run_time)


def parameter_sweep(
    scene: ThreeDScene,
    func: Callable[[float, float], float],
    x_range: Tuple[float, float, float],
    param_range: Tuple[float, float, float],
    param_name: str,
    existing_curve: VMobject,
    run_time_total: float = 3.0,
    **kwargs,
) -> None:
    """One-liner wrapper for ``ParameterSweep3D.run()``.

    Example
    -------
    ::

        parameter_sweep(
            self,
            func=lambda x, s: np.exp(-0.5*(x/s)**2)/(s*np.sqrt(TAU)),
            x_range=(-4, 4, 0.05),
            param_range=(0.5, 2.5, 0.1),
            param_name="σ",
            existing_curve=normal_curve.stroke,
            run_time_total=4.0,
        )
    """
    ParameterSweep3D(
        func=func,
        x_range=x_range,
        param_range=param_range,
        param_name=param_name,
        existing_curve=existing_curve,
        **kwargs,
    ).run(scene, run_time_total=run_time_total)


def curtain_reveal(
    target: VGroup,
    x_start: float = -4.0,
    x_end: float = 4.0,
    run_time: float = 1.5,
    **kwargs,
) -> AnimationGroup:
    """One-liner wrapper for ``CurtainReveal3D.build()``.

    Example
    -------
    ::

        scene.play(curtain_reveal(chart.bar_group, x_start=-3.5, x_end=3.5))
    """
    return CurtainReveal3D(
        target, x_start=x_start, x_end=x_end, **kwargs
    ).build(run_time=run_time)


def ci_build(
    mean_pos: np.ndarray,
    half_width: float,
    run_time: float = 1.2,
    **kwargs,
) -> Succession:
    """One-liner wrapper for ``CIBuild3D.build()``.

    Example
    -------
    ::

        scene.play(ci_build(np.array([0, 0, 2.5]), half_width=0.8))
    """
    return CIBuild3D(mean_pos=mean_pos, half_width=half_width, **kwargs).build(run_time=run_time)


def collapse_to_mean(
    dots: List,
    mean_position: np.ndarray,
    run_time: float = 1.5,
    lag: float = 0.02,
    **kwargs,
) -> LaggedStart:
    """One-liner wrapper for ``CollapseToMean3D.build()``.

    Example
    -------
    ::

        scene.play(collapse_to_mean(population.dots, mean_pos))
    """
    return CollapseToMean3D(
        dots=dots, mean_position=mean_position, **kwargs
    ).build(run_time=run_time, lag=lag)