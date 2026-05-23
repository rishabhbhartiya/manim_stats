"""
manim_stats/regression/regression_plane.py
==========================================
Three-dimensional regression plane visualisation for multiple linear regression
with two predictors.  Extends the 2-D work in ``correlation.py`` into full 3-D.

Scope
-----
This file covers **everything a statistics educator needs to visualise a
regression plane**:

  * The plane itself — a smooth, tilted ``Surface`` coloured by fitted value,
    residual magnitude, or leverage.
  * The scatter cloud — 3-D ``Dot`` objects at (x1, x2, y) with colour maps.
  * Residual lines — vertical sticks from each point down to the plane.
  * Confidence and prediction shells — semi-transparent upper/lower surfaces
    bounding the 95 % CI or PI region.
  * Animated coefficient sliders — morph the plane live as beta changes.
  * Projection arrows — drop perpendiculars from a point to the plane.
  * A complete ``ThreeDAxes`` scaffold with axis labels and tick formatters.

Architecture
------------
  Layer A  Pure-math helpers         — plane_predict, plane_grid, ci_surface,
                                       pi_surface, plane_normal, rotation_angles
  Layer B  Geometry dataclasses      — PlaneGeometry, SurfaceColorMap
  Layer C  Manim mobjects            — RegressionPlane3D, ScatterCloud3D,
                                       PlaneResiduals3D, CIShell3D, PIShell3D,
                                       ProjectionArrow3D, CoefficientSlider3D
  Layer D  Scene-level animations    — build_plane_scene, morph_beta,
                                       sweep_x1, sweep_x2, rotate_view
  Layer E  Formula registry bridge   — PLANE_FORMULAS with derivation chains

Design notes
------------
* Surface resolution is configurable: default 30 × 30 for smooth rendering
  while keeping Manim's vertex count manageable.

* All colour work uses ``core.colors`` families so the plane respects whatever
  ``StatsTheme`` the caller sets.  The default ``colouring`` is
  ``'fitted'`` (hue = fitted value, sequential REGRESSION palette).
  Alternatives: ``'residual'`` (diverging, teal↔coral),
                ``'leverage'`` (sequential amber), ``'flat'`` (solid colour).

* The CI surface is the locus of pointwise 95 % confidence bounds for E[Y|x]:
      y_hat(x) ± t_{n-p,0.025} · s · sqrt(x' (X'X)^{-1} x)
  The PI surface adds the individual prediction variance:
      y_hat(x) ± t_{n-p,0.025} · s · sqrt(1 + x' (X'X)^{-1} x)

* ``ThreeDAxes`` is created by ``build_plane_scene()`` — callers get back the
  axes and all mobjects so they can compose their own animations around them.

* No hard Manim dependency at import time — all Layers A and B work in a
  plain Python / NumPy environment; Layers C-E raise ``ImportError`` cleanly.

* ``RegressionResult`` (from correlation.py) is accepted everywhere a fit is
  required; the module re-exports ``ols_fit`` for convenience so callers only
  need to import from this file.

Typical usage
-------------
::

    from manim_stats.regression.regression_plane import (
        RegressionPlane3D, ScatterCloud3D, PlaneResiduals3D, build_plane_scene,
    )
    from manim_stats.regression.correlation import ols_fit
    import numpy as np

    class MyScene(ThreeDScene):
        def construct(self):
            rng = np.random.default_rng(0)
            x1  = rng.normal(0, 1, 60)
            x2  = rng.normal(0, 1, 60)
            y   = 1 + 0.8*x1 - 0.5*x2 + rng.normal(0, 0.4, 60)

            result = ols_fit(np.column_stack([x1, x2]), y)

            axes, cloud, plane, residuals = build_plane_scene(
                result, x1, x2, y, self
            )
            self.play(Create(axes))
            self.play(cloud.animate_appear())
            self.play(plane.animate_grow())
            self.play(residuals.animate_appear())
            self.wait(2)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional SciPy
# ---------------------------------------------------------------------------
try:
    from scipy import stats as _sp
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    _sp = None  # type: ignore

# ---------------------------------------------------------------------------
# Graceful Manim import — full 3-D toolkit
# ---------------------------------------------------------------------------
try:
    import manim as mn
    from manim import (
        # Core
        VGroup, VMobject, Mobject,
        # 3-D scene
        ThreeDScene, ThreeDAxes,
        # 3-D surfaces
        Surface,
        # 2-D primitives re-used in 3-D
        Line, Arrow, DashedLine, Dot, Sphere,
        Text, MathTex,
        # Colours
        ManimColor, WHITE, BLACK, GRAY, DARK_GRAY, LIGHT_GRAY,
        RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE,
        DARK_BLUE, PURE_RED,
        # Direction constants
        UP, DOWN, LEFT, RIGHT, OUT, IN, ORIGIN,
        TAU, PI,
        # Animations
        Write, Create, FadeIn, FadeOut,
        Transform, ReplacementTransform,
        Rotate, Flash, Indicate,
        AnimationGroup, Succession, LaggedStart,
        rate_functions,
        interpolate_color,
        # Camera
        GrowFromCenter,
    )
    _MANIM_AVAILABLE = True
except ImportError:
    _MANIM_AVAILABLE = False
    # Stubs so type annotations don't crash
    VGroup = VMobject = Surface = object  # type: ignore
    ThreeDAxes = object  # type: ignore

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from manim_stats.regression.correlation import (
        RegressionResult,
        ols_fit,
        influence_measures,
        InfluenceMeasures,
    )
    _CORR_AVAILABLE = True
except ImportError:
    _CORR_AVAILABLE = False
    RegressionResult = object  # type: ignore

try:
    from manim_stats.core.colors import (
        REGRESSION_FAMILY, NORMAL_FAMILY, INFERENCE_FAMILY,
        DISCRETE_FAMILY, NEUTRAL_FAMILY,
        TEAL_600, TEAL_200, TEAL_800,
        CORAL_600, CORAL_200, CORAL_800,
        PURPLE_600, PURPLE_200, PURPLE_800,
        BLUE_600, BLUE_200,
        AMBER_600, AMBER_200,
        GRAY_400, GRAY_200,
        StatColor, diverging_map, gradient_ramp,
    )
    _COLORS_AVAILABLE = True
except ImportError:
    _COLORS_AVAILABLE = False

try:
    from manim_stats.core.tex_utils import (
        TexFormula, TexDerivationStep,
        _frac, _sqrt, _sum, _exp,
        FORMULAS, register_formula,
    )
    _TEX_AVAILABLE = True
except ImportError:
    _TEX_AVAILABLE = False


def _require_manim(name: str) -> None:
    if not _MANIM_AVAILABLE:
        raise ImportError(
            f"{name} requires Manim.  Install with: pip install manim"
        )


def _require_corr(name: str) -> None:
    if not _CORR_AVAILABLE:
        raise ImportError(
            f"{name} requires manim_stats.regression.correlation."
        )


# ===========================================================================
# LAYER A — Pure-math helpers
# All functions operate on plain NumPy arrays.  No Manim dependency.
# ===========================================================================

def plane_predict(
    beta:   np.ndarray,
    x1:     Union[float, np.ndarray],
    x2:     Union[float, np.ndarray],
) -> np.ndarray:
    """
    Evaluate the fitted plane  z = b0 + b1*x1 + b2*x2.

    Parameters
    ----------
    beta : ndarray, shape (3,)
        Coefficients [b0, b1, b2].  If the RegressionResult was fitted with
        ``fit_intercept=True``, pass ``result.beta`` directly.
    x1, x2 : float or ndarray
        Predictor values.  Arrays may be any broadcastable shape.

    Returns
    -------
    ndarray
        Fitted values, same shape as the broadcast of x1 and x2.
    """
    b0, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])
    return b0 + b1 * np.asarray(x1) + b2 * np.asarray(x2)


def plane_grid(
    beta:       np.ndarray,
    x1_range:   Tuple[float, float],
    x2_range:   Tuple[float, float],
    resolution: int = 30,
    padding:    float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular grid of fitted-plane values for surface rendering.

    Parameters
    ----------
    beta : ndarray, shape (3,)
        Regression coefficients [b0, b1, b2].
    x1_range, x2_range : (float, float)
        Data range for each predictor.
    resolution : int
        Number of grid points per axis.  Default 30.
    padding : float
        Fractional padding beyond the data range.  Default 0.15 (15 %).

    Returns
    -------
    (X1, X2, Z) — each ndarray of shape (resolution, resolution).
        X1, X2 are the meshgrid matrices; Z = plane_predict(beta, X1, X2).
    """
    def _padded(lo, hi, p):
        span = max(hi - lo, 1e-6)
        return lo - p * span, hi + p * span

    u1_lo, u1_hi = _padded(*x1_range, padding)
    u2_lo, u2_hi = _padded(*x2_range, padding)

    u1 = np.linspace(u1_lo, u1_hi, resolution)
    u2 = np.linspace(u2_lo, u2_hi, resolution)
    X1, X2 = np.meshgrid(u1, u2)
    Z = plane_predict(beta, X1, X2)
    return X1, X2, Z


def ci_surface(
    beta:      np.ndarray,
    XtX_inv:   np.ndarray,
    s:         float,
    n:         int,
    x1_range:  Tuple[float, float],
    x2_range:  Tuple[float, float],
    resolution: int   = 25,
    level:     float  = 0.95,
    padding:   float  = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute upper and lower pointwise confidence interval surfaces for E[Y|x].

    Formula (pointwise, for a grid point x0):
        CI = y_hat(x0) ± t_{n-p, alpha/2} · s · sqrt(x0' (X'X)^{-1} x0)

    Parameters
    ----------
    beta : ndarray (3,)
        Coefficients [b0, b1, b2].
    XtX_inv : ndarray (3, 3)
        Inverse of the design matrix Gram matrix (X'X)^{-1}.
    s : float
        Residual standard deviation (sigma_hat).
    n : int
        Sample size.
    x1_range, x2_range : (float, float)
    resolution : int
    level : float
        Confidence level (default 0.95).
    padding : float

    Returns
    -------
    (X1, X2, Z_upper, Z_lower)
        Each ndarray of shape (resolution, resolution).
    """
    df = n - 3   # degrees of freedom for two-predictor model
    if _SCIPY_AVAILABLE:
        t_crit = float(_sp.t.ppf((1 + level) / 2, df=max(df, 1)))
    else:
        t_crit = 1.959963985   # fallback to z-crit

    X1, X2, Z_hat = plane_grid(beta, x1_range, x2_range, resolution, padding)
    margin = np.zeros_like(Z_hat)

    for i in range(resolution):
        for j in range(resolution):
            x0 = np.array([1.0, X1[i, j], X2[i, j]])
            var_fit = float(x0 @ XtX_inv @ x0)
            margin[i, j] = t_crit * s * math.sqrt(max(var_fit, 0.0))

    return X1, X2, Z_hat + margin, Z_hat - margin


def pi_surface(
    beta:      np.ndarray,
    XtX_inv:   np.ndarray,
    s:         float,
    n:         int,
    x1_range:  Tuple[float, float],
    x2_range:  Tuple[float, float],
    resolution: int  = 25,
    level:     float = 0.95,
    padding:   float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute upper and lower pointwise prediction interval surfaces for Y_new.

    Formula (pointwise, for a grid point x0):
        PI = y_hat(x0) ± t_{n-p, alpha/2} · s · sqrt(1 + x0' (X'X)^{-1} x0)

    The PI is wider than the CI because it accounts for individual-observation
    variance (the extra ``1`` inside the square root).

    Parameters are identical to ``ci_surface()``.

    Returns
    -------
    (X1, X2, Z_upper, Z_lower)
        Each ndarray of shape (resolution, resolution).
    """
    df = n - 3
    if _SCIPY_AVAILABLE:
        t_crit = float(_sp.t.ppf((1 + level) / 2, df=max(df, 1)))
    else:
        t_crit = 1.959963985

    X1, X2, Z_hat = plane_grid(beta, x1_range, x2_range, resolution, padding)
    margin = np.zeros_like(Z_hat)

    for i in range(resolution):
        for j in range(resolution):
            x0      = np.array([1.0, X1[i, j], X2[i, j]])
            var_pred = float(1.0 + x0 @ XtX_inv @ x0)
            margin[i, j] = t_crit * s * math.sqrt(max(var_pred, 0.0))

    return X1, X2, Z_hat + margin, Z_hat - margin


def plane_normal(beta: np.ndarray) -> np.ndarray:
    """
    Return the unit normal vector of the plane  z = b0 + b1*x + b2*y.

    The plane can be written as  -b1*x - b2*y + z = b0,
    so the normal is  (-b1, -b2, 1)  (unnormalised).

    Parameters
    ----------
    beta : ndarray (3,)
        [b0, b1, b2]

    Returns
    -------
    ndarray (3,) — unit normal vector.
    """
    b1, b2 = float(beta[1]), float(beta[2])
    n = np.array([-b1, -b2, 1.0])
    return n / np.linalg.norm(n)


def tilt_angle_deg(beta: np.ndarray) -> float:
    """
    Return the tilt angle of the fitted plane from horizontal, in degrees.

    A plane with b1 = b2 = 0 has tilt = 0.  Steeper slopes give larger angles.
    """
    n    = plane_normal(beta)
    horiz = np.array([0.0, 0.0, 1.0])
    cos_a = abs(float(np.dot(n, horiz)))
    return math.degrees(math.acos(min(cos_a, 1.0)))


def leverage_grid(
    XtX_inv:   np.ndarray,
    x1_range:  Tuple[float, float],
    x2_range:  Tuple[float, float],
    resolution: int  = 30,
    padding:   float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute leverage (x0' (X'X)^{-1} x0) on a grid — used to colour the
    regression plane by leverage when ``colouring='leverage'``.

    Returns
    -------
    (X1, X2, H) where H[i,j] is the leverage at grid point (X1[i,j], X2[i,j]).
    """
    def _padded(lo, hi, p):
        span = max(hi - lo, 1e-6)
        return lo - p * span, hi + p * span

    u1 = np.linspace(*_padded(*x1_range, padding), resolution)
    u2 = np.linspace(*_padded(*x2_range, padding), resolution)
    X1, X2 = np.meshgrid(u1, u2)
    H = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            x0 = np.array([1.0, X1[i, j], X2[i, j]])
            H[i, j] = float(x0 @ XtX_inv @ x0)
    return X1, X2, H


def standardise_grid(Z: np.ndarray) -> np.ndarray:
    """
    Map Z values to [0, 1] for use as colormap parameter t.
    Handles degenerate (constant) grids by returning 0.5 everywhere.
    """
    lo, hi = float(Z.min()), float(Z.max())
    if hi - lo < 1e-10:
        return np.full_like(Z, 0.5)
    return (Z - lo) / (hi - lo)


# ===========================================================================
# LAYER B — Geometry and colouring configuration dataclasses
# ===========================================================================

class SurfaceColouring(Enum):
    """
    Colour strategy for the regression plane surface.

    FITTED
        Sequential hue proportional to the fitted value z_hat.
        Uses REGRESSION_FAMILY gradient (light → dark purple).

    RESIDUAL
        Diverging hue: coral for over-prediction, teal for under-prediction.
        Colour is applied to each *data point* and residual line;
        the plane itself is shown in a neutral flat colour.

    LEVERAGE
        Sequential amber hue proportional to x0'(X'X)^{-1}x0 at each grid point.
        Highlights corners (high-leverage regions) in dark amber.

    FLAT
        Solid single colour (REGRESSION_FAMILY.base), maximum readability.
    """
    FITTED    = "fitted"
    RESIDUAL  = "residual"
    LEVERAGE  = "leverage"
    FLAT      = "flat"


@dataclass
class PlaneGeometry:
    """
    Physical dimensions and sampling parameters for a 3-D regression plane.

    All lengths are in Manim world-units.  A ThreeDAxes with default
    ``x_length=6, y_length=6, z_length=5`` fits well on a standard
    1920×1080 scene frame.

    Attributes
    ----------
    x1_length, x2_length, z_length : float
        Lengths of the three axes in world-units.
    resolution : int
        Number of grid vertices per axis on the regression surface.
        Default 30 balances smoothness with Manim vertex budget.
    ci_resolution : int
        Resolution for CI / PI shells (can be coarser — default 20).
    dot_radius : float
        Radius of each scatter-cloud data point sphere.
    residual_stroke_width : float
        Stroke width of residual lines.
    axis_label_font_size : int
        Font size for x1, x2, y axis labels.
    ci_opacity, pi_opacity : float
        Fill opacity of the CI and PI shells.
    plane_opacity : float
        Fill opacity of the regression plane surface.
    """
    x1_length:              float = 6.0
    x2_length:              float = 6.0
    z_length:               float = 5.0
    resolution:             int   = 30
    ci_resolution:          int   = 20
    dot_radius:             float = 0.055
    residual_stroke_width:  float = 1.8
    axis_label_font_size:   int   = 24
    ci_opacity:             float = 0.22
    pi_opacity:             float = 0.12
    plane_opacity:          float = 0.72

    @property
    def axes_kwargs(self) -> dict:
        """
        Ready-to-unpack kwargs for ``manim.ThreeDAxes``.

        Includes axis lengths, range auto-scaled later by ``build_plane_scene``.
        """
        return {
            "x_length": self.x1_length,
            "y_length": self.x2_length,
            "z_length": self.z_length,
            "axis_config": {
                "include_tip":      True,
                "tip_width":        0.12,
                "tip_height":       0.12,
                "stroke_width":     2.0,
                "include_numbers":  False,
            },
        }


#: Sensible defaults for a standard scene
DEFAULT_GEOMETRY = PlaneGeometry()

#: Smaller geometry preset for inset / thumbnail panels
COMPACT_GEOMETRY = PlaneGeometry(
    x1_length=4.0, x2_length=4.0, z_length=3.5,
    resolution=22, ci_resolution=14,
    dot_radius=0.045,
    axis_label_font_size=18,
)

#: High-resolution preset for close-up detail scenes
HIRES_GEOMETRY = PlaneGeometry(
    x1_length=7.0, x2_length=7.0, z_length=6.0,
    resolution=45, ci_resolution=30,
    dot_radius=0.065,
)


# ===========================================================================
# LAYER C — Manim mobjects
# ===========================================================================

class RegressionPlane3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    The fitted regression plane for  y = b0 + b1*x1 + b2*x2  as a Manim
    ``Surface`` wrapped in a ``VGroup`` with axis helpers.

    The plane surface is built from Manim's ``Surface`` class with a
    parametric function  f(u, v) = [u, v, b0 + b1*u + b2*v].

    Colour strategies (``colouring`` parameter)
    -------------------------------------------
    ``'fitted'``   Sequential gradient along z_hat (purple light → dark).
    ``'leverage'`` Sequential gradient along h(x0) (amber).
    ``'flat'``     Solid single colour.

    Parameters
    ----------
    result : RegressionResult
        A two-predictor OLS fit (k=2).  ``result.beta`` must have 3 elements
        (intercept + 2 slopes).
    x1_range, x2_range : (float, float)
        Data ranges for each predictor (used to set the surface extent).
    geometry : PlaneGeometry
    colouring : SurfaceColouring or str
    plane_color : ManimColor, optional
        Override the base colour (used for ``FLAT`` or as the gradient anchor).
    opacity : float
        Surface fill opacity.
    show_grid_lines : bool
        Whether to render the surface with wireframe grid lines.
    grid_stroke_width : float
        Width of wireframe lines (if ``show_grid_lines=True``).
    axes : ThreeDAxes, optional
        If provided, the plane is scaled to this axes object's coordinate system.

    Key sub-mobjects
    ----------------
    .surface       : Surface — the parametric plane
    .equation_label: MathTex — fitted equation displayed near the plane
    .axes_ref      : ThreeDAxes or None

    Animations
    ----------
    .animate_grow(run_time)            — scale plane from zero at centroid
    .animate_tilt(target_beta, run_time)— morph plane to new coefficients
    .flash_plane(color, run_time)       — Indicate the entire surface
    .update_coefficients(beta, run_time)— morph plane + equation to new beta
    """

    def __init__(
        self,
        result:           "RegressionResult",
        x1_range:         Tuple[float, float],
        x2_range:         Tuple[float, float],
        geometry:         PlaneGeometry       = DEFAULT_GEOMETRY,
        colouring:        Union[SurfaceColouring, str] = SurfaceColouring.FITTED,
        plane_color                            = None,
        opacity:          float                = None,
        show_grid_lines:  bool                 = True,
        grid_stroke_width: float               = 0.4,
        axes:             Optional["ThreeDAxes"] = None,
        show_equation:    bool                 = True,
        **kwargs,
    ) -> None:
        _require_manim("RegressionPlane3D.__init__")
        _require_corr("RegressionPlane3D.__init__")

        if result.k != 2:
            raise ValueError(
                f"RegressionPlane3D requires a 2-predictor fit (k=2), "
                f"got k={result.k}.  Use RegressionLine3D for k=1."
            )

        super().__init__(**kwargs)

        self._result    = result
        self._x1_range  = x1_range
        self._x2_range  = x2_range
        self._geometry  = geometry
        self._axes_ref  = axes

        beta            = result.beta        # [b0, b1, b2]
        self._beta      = beta.copy()

        if isinstance(colouring, str):
            colouring = SurfaceColouring(colouring)
        self._colouring = colouring

        opacity = opacity if opacity is not None else geometry.plane_opacity

        # Resolve colours
        if _COLORS_AVAILABLE:
            self._plane_color  = plane_color or ManimColor(REGRESSION_FAMILY.base.hex)
            self._light_color  = ManimColor(REGRESSION_FAMILY.light.hex)
            self._dark_color   = ManimColor(REGRESSION_FAMILY.dark.hex)
            self._amber_color  = ManimColor(AMBER_600.hex)
            self._amber_light  = ManimColor(AMBER_200.hex)
        else:
            self._plane_color  = plane_color or PURPLE
            self._light_color  = BLUE
            self._dark_color   = DARK_BLUE
            self._amber_color  = ORANGE
            self._amber_light  = YELLOW

        self._opacity         = opacity
        self._show_grid_lines = show_grid_lines
        self._grid_sw         = grid_stroke_width
        self._resolution      = geometry.resolution

        # Pre-compute colouring grids
        self._X1, self._X2, self._Z = plane_grid(
            beta, x1_range, x2_range, geometry.resolution
        )
        if colouring == SurfaceColouring.LEVERAGE:
            try:
                X_design = result.x
                XtX_inv  = np.linalg.inv(X_design.T @ X_design)
                _, _, self._H_grid = leverage_grid(
                    XtX_inv, x1_range, x2_range, geometry.resolution
                )
            except (np.linalg.LinAlgError, AttributeError):
                self._H_grid = None
        else:
            self._H_grid = None

        # Build surface
        self._build_surface(beta, x1_range, x2_range, axes)

        # Equation label
        if show_equation:
            self._build_equation_label(beta)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_surface(
        self,
        beta:     np.ndarray,
        x1_range: Tuple[float, float],
        x2_range: Tuple[float, float],
        axes:     Optional["ThreeDAxes"],
    ) -> None:
        """
        Construct the Manim ``Surface`` parametric object.

        The Surface's parametric function maps (u, v) in the data coordinate
        ranges to 3-D world coordinates.  When ``axes`` is provided, we use
        ``axes.c2p`` to map data → scene coordinates; otherwise we map
        data → world-units directly (caller must position the axes themselves).
        """
        b0, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])

        # Build padded data ranges for the surface extent
        def _pad(lo, hi, p=0.15):
            span = max(hi - lo, 1e-6)
            return lo - p * span, hi + p * span

        u_lo, u_hi = _pad(*x1_range)
        v_lo, v_hi = _pad(*x2_range)

        # Pre-compute Z range for colour mapping
        z_corners = [b0 + b1*u + b2*v
                     for u in (u_lo, u_hi) for v in (v_lo, v_hi)]
        z_lo, z_hi = min(z_corners), max(z_corners)
        z_span     = max(z_hi - z_lo, 1e-6)

        # Pre-compute H range for leverage colouring
        if self._H_grid is not None:
            h_lo  = float(self._H_grid.min())
            h_span = max(float(self._H_grid.max()) - h_lo, 1e-6)
        else:
            h_lo = h_span = 1.0

        colouring   = self._colouring
        light_color = self._light_color
        dark_color  = self._dark_color
        amber_color = self._amber_color
        amber_light = self._amber_light
        plane_color = self._plane_color
        H_grid      = self._H_grid
        res         = self._resolution

        if axes is not None:
            def _func(u: float, v: float) -> np.ndarray:
                z = b0 + b1 * u + b2 * v
                return axes.c2p(u, v, z)
        else:
            # Direct world-space mapping (u, v) → (u, v, z) unchanged
            def _func(u: float, v: float) -> np.ndarray:
                z = b0 + b1 * u + b2 * v
                return np.array([u, v, z])

        def _color_func(u: float, v: float) -> "ManimColor":
            """
            Determine the colour of the surface at parameter (u, v).

            For FITTED:  t = (z - z_lo) / z_span  → interpolate light→dark
            For LEVERAGE: t = (h - h_lo) / h_span → interpolate amber_light→amber
            For FLAT:    constant plane_color
            """
            if colouring == SurfaceColouring.FITTED:
                z = b0 + b1 * u + b2 * v
                t = max(0.0, min(1.0, (z - z_lo) / z_span))
                return interpolate_color(light_color, dark_color, t)

            elif colouring == SurfaceColouring.LEVERAGE:
                if H_grid is not None:
                    # Nearest-grid-point lookup (simple, no interp)
                    ui = int((u - u_lo) / (u_hi - u_lo) * (res - 1) + 0.5)
                    vi = int((v - v_lo) / (v_hi - v_lo) * (res - 1) + 0.5)
                    ui = max(0, min(res - 1, ui))
                    vi = max(0, min(res - 1, vi))
                    h  = float(H_grid[vi, ui])
                    t  = max(0.0, min(1.0, (h - h_lo) / h_span))
                    return interpolate_color(amber_light, amber_color, t)
                return plane_color

            else:   # FLAT
                return plane_color

        self.surface = Surface(
            func        = _func,
            u_range     = [u_lo, u_hi],
            v_range     = [v_lo, v_hi],
            resolution  = (self._resolution, self._resolution),
            fill_opacity = self._opacity,
            checkerboard_colors = None,
        )
        self.surface.set_fill_by_value(
            axes   = axes,
            colors = [
                (self._light_color, float(z_lo)),
                (self._dark_color,  float(z_hi)),
            ],
        ) if colouring == SurfaceColouring.FITTED and axes is not None else None

        if self._show_grid_lines:
            self.surface.set_stroke(
                color = self._dark_color,
                width = self._grid_sw,
                opacity = 0.35,
            )
        else:
            self.surface.set_stroke(width=0)

        self.add(self.surface)

    def _build_equation_label(self, beta: np.ndarray) -> None:
        """Build a MathTex label showing the fitted equation."""
        b0, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])

        def _signed(val: float, var: str, first: bool = False) -> str:
            if first:
                return rf"{val:.3f}"
            sign = "+" if val >= 0 else "-"
            return rf"{sign} {abs(val):.3f}\,{var}"

        eq = (
            rf"\hat{{y}} = {_signed(b0,'',True)}"
            rf"\;{_signed(b1, 'x_1')}"
            rf"\;{_signed(b2, 'x_2')}"
            rf"\quad R^2\!=\!{self._result.r_squared:.3f}"
        )
        self.equation_label = MathTex(
            eq,
            font_size = 26,
            color     = self._plane_color,
        )
        self.add(self.equation_label)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def beta(self) -> np.ndarray:
        return self._beta.copy()

    @property
    def r_squared(self) -> float:
        return self._result.r_squared

    @property
    def tilt_degrees(self) -> float:
        return tilt_angle_deg(self._beta)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_grow(self, run_time: float = 1.4) -> "mn.Animation":
        """
        Grow the plane from zero scale at its geometric centroid.

        Implementation: scale to 0.001, then animate back to 1.0 with
        ease_out_back so it slightly overshoots and settles.
        """
        _require_manim("animate_grow")
        centre = self.surface.get_center()
        self.surface.scale(0.001, about_point=centre)
        return self.surface.animate(
            run_time  = run_time,
            rate_func = rate_functions.ease_out_back,
        ).scale(1000, about_point=centre)

    def animate_tilt(
        self,
        target_beta: np.ndarray,
        run_time:    float = 1.5,
    ) -> "mn.Animation":
        """
        Animate the plane tilting from its current orientation to the one
        implied by ``target_beta``.

        This is implemented as a ``Transform`` to a freshly constructed
        ``RegressionPlane3D`` with the target coefficients.  The equation
        label morphs simultaneously.

        Parameters
        ----------
        target_beta : ndarray (3,)
            New [b0, b1, b2] coefficients.
        run_time : float

        Returns
        -------
        manim.Transform
        """
        _require_manim("animate_tilt")

        # Build a temporary result-like object so we can re-use the constructor
        class _FakeResult:
            def __init__(self, result, beta):
                self.beta          = np.asarray(beta, dtype=float)
                self.k             = result.k
                self.x             = result.x
                self.r_squared     = result.r_squared
                self.fit_intercept = result.fit_intercept

        fake = _FakeResult(self._result, target_beta)
        target_plane = RegressionPlane3D(
            result        = fake,
            x1_range      = self._x1_range,
            x2_range      = self._x2_range,
            geometry      = self._geometry,
            colouring     = self._colouring,
            opacity       = self._opacity,
            show_equation = hasattr(self, "equation_label"),
            axes          = self._axes_ref,
        ).move_to(self.get_center())

        return Transform(
            self, target_plane,
            run_time  = run_time,
            rate_func = rate_functions.ease_in_out_sine,
        )

    def flash_plane(
        self,
        color     = None,
        run_time: float = 0.7,
    ) -> "mn.Animation":
        """Flash the entire plane surface with Indicate."""
        _require_manim("flash_plane")
        color = color or (self._plane_color if _COLORS_AVAILABLE else YELLOW)
        return Indicate(
            self.surface,
            color        = color,
            scale_factor = 1.04,
            run_time     = run_time,
        )

    def update_coefficients(
        self,
        new_beta: np.ndarray,
        run_time: float = 1.2,
    ) -> "mn.Animation":
        """
        Morph both the plane and its equation label to reflect ``new_beta``.

        Returns a combined ``AnimationGroup``.
        """
        _require_manim("update_coefficients")
        return AnimationGroup(
            self.animate_tilt(new_beta, run_time=run_time),
        )

    def __repr__(self) -> str:
        b = self._beta
        return (
            f"RegressionPlane3D("
            f"y = {b[0]:.3f} + {b[1]:.3f}x1 + {b[2]:.3f}x2, "
            f"tilt={self.tilt_degrees:.1f}°, "
            f"R²={self.r_squared:.3f})"
        )


# ---------------------------------------------------------------------------
# ScatterCloud3D
# ---------------------------------------------------------------------------

class ScatterCloud3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    A 3-D scatter cloud of data points for a two-predictor regression scene.

    Each observation is rendered as a coloured ``Sphere`` (or ``Dot`` in 2-D
    fallback mode) at position (x1_i, x2_i, y_i).

    Colour strategies (``colouring`` parameter)
    -------------------------------------------
    ``'residual'``
        Diverging teal ↔ coral based on the sign and magnitude of the residual.
        Matches ``PlaneResiduals3D`` colour convention so colours are consistent.
    ``'leverage'``
        Sequential amber, brighter = higher leverage.
    ``'cooks_d'``
        Sequential coral; large Cook's D = bright coral.
    ``'flat'``
        Uniform NORMAL_FAMILY.base colour.

    Parameters
    ----------
    x1, x2, y : array-like (n,)
        Observation coordinates.
    result : RegressionResult
        Used to extract residuals, leverage, Cook's D.
    colouring : str or SurfaceColouring
    dot_radius : float
    axes : ThreeDAxes, optional
    highlight_indices : list[int], optional
        Observation indices to mark with a larger dot and bright ring.

    Key sub-mobjects
    ----------------
    .dots          : VGroup of Sphere / Dot objects, indexed like the data.
    .highlight_rings: VGroup of Circle rings around flagged points.

    Animations
    ----------
    .animate_appear(run_time, stagger)    — LaggedStart FadeIn
    .highlight_obs(idx, color, run_time)  — Indicate + scale up one dot
    .dim_all_except(indices, run_time)    — fade non-selected dots
    .undim_all(run_time)                  — restore full opacity
    """

    def __init__(
        self,
        x1:                  np.ndarray,
        x2:                  np.ndarray,
        y:                   np.ndarray,
        result:              "RegressionResult",
        colouring:           str = "residual",
        dot_radius:          float = None,
        geometry:            PlaneGeometry = DEFAULT_GEOMETRY,
        axes:                Optional["ThreeDAxes"] = None,
        highlight_indices:   Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        _require_manim("ScatterCloud3D.__init__")
        _require_corr("ScatterCloud3D.__init__")
        super().__init__(**kwargs)

        self._x1     = np.asarray(x1, dtype=float)
        self._x2     = np.asarray(x2, dtype=float)
        self._y      = np.asarray(y,  dtype=float)
        self._result = result
        self._axes   = axes
        self._radius = dot_radius or geometry.dot_radius

        resid = result.residuals

        # Compute per-point colour
        colors = self._compute_colors(colouring, resid, result)

        # Build dots
        self.dots = VGroup()
        n = len(x1)
        for i in range(n):
            pt = self._data_to_world(float(self._x1[i]),
                                     float(self._x2[i]),
                                     float(self._y[i]), axes)
            dot = Sphere(radius=self._radius, color=colors[i])
            dot.move_to(pt)
            self.dots.add(dot)

        self.add(self.dots)

        # Highlight rings
        self.highlight_rings = VGroup()
        if highlight_indices:
            for idx in highlight_indices:
                if 0 <= idx < n:
                    pt = self._data_to_world(
                        float(self._x1[idx]),
                        float(self._x2[idx]),
                        float(self._y[idx]), axes,
                    )
                    ring = Sphere(
                        radius  = self._radius * 2.0,
                        color   = YELLOW if _MANIM_AVAILABLE else None,
                    )
                    ring.set_fill(opacity=0)
                    ring.set_stroke(color=YELLOW, width=2.0)
                    ring.move_to(pt)
                    self.highlight_rings.add(ring)
        self.add(self.highlight_rings)

    @staticmethod
    def _data_to_world(
        x1: float, x2: float, y: float,
        axes: Optional["ThreeDAxes"],
    ) -> np.ndarray:
        if axes is not None:
            return np.array(axes.c2p(x1, x2, y))
        return np.array([x1, x2, y])

    def _compute_colors(
        self,
        colouring: str,
        resid:     np.ndarray,
        result:    "RegressionResult",
    ) -> List["ManimColor"]:
        """Compute one ManimColor per observation."""
        n = len(resid)
        if _COLORS_AVAILABLE:
            pos_color    = ManimColor(TEAL_600.hex)
            neg_color    = ManimColor(CORAL_600.hex)
            base_color   = ManimColor(NORMAL_FAMILY.base.hex)
            amber_lo     = ManimColor(AMBER_200.hex)
            amber_hi     = ManimColor(AMBER_600.hex)
            coral_lo     = ManimColor(CORAL_200.hex)
            coral_hi     = ManimColor(CORAL_600.hex)
        else:
            pos_color = neg_color = base_color = BLUE
            amber_lo  = amber_hi = ORANGE
            coral_lo  = coral_hi = RED

        if colouring == "residual":
            max_abs = max(float(np.max(np.abs(resid))), 1e-6)
            colors  = []
            for r in resid:
                t     = min(abs(float(r)) / max_abs, 1.0)
                color = interpolate_color(base_color,
                                          pos_color if r > 0 else neg_color,
                                          t * 0.9)
                colors.append(color)
            return colors

        elif colouring == "leverage":
            try:
                infl = influence_measures(result)
                h    = infl.leverage
            except Exception:
                h = np.ones(n) / n
            h_norm = (h - h.min()) / max(h.max() - h.min(), 1e-6)
            return [interpolate_color(amber_lo, amber_hi, float(t))
                    for t in h_norm]

        elif colouring == "cooks_d":
            try:
                infl  = influence_measures(result)
                cooks = infl.cooks_d
            except Exception:
                cooks = np.zeros(n)
            c_norm = (cooks - cooks.min()) / max(cooks.max() - cooks.min(), 1e-6)
            return [interpolate_color(coral_lo, coral_hi, float(t))
                    for t in c_norm]

        else:   # flat
            return [base_color] * n

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_appear(
        self, run_time: float = 1.8, stagger: float = 0.025
    ) -> "mn.Animation":
        """Staggered FadeIn of all dots."""
        _require_manim("animate_appear")
        return LaggedStart(
            *[FadeIn(d, scale=0.4, run_time=run_time * 0.4)
              for d in self.dots],
            lag_ratio = stagger,
        )

    def highlight_obs(
        self,
        idx:      int,
        color     = None,
        run_time: float = 0.6,
    ) -> "mn.Animation":
        """Indicate + scale a single observation."""
        _require_manim("highlight_obs")
        if idx < 0 or idx >= len(self.dots):
            return AnimationGroup()
        color = color or (YELLOW if _MANIM_AVAILABLE else None)
        return AnimationGroup(
            Indicate(self.dots[idx], color=color,
                     scale_factor=1.8, run_time=run_time),
            Flash(self.dots[idx], color=color,
                  flash_radius=self._radius * 3, run_time=run_time),
        )

    def dim_all_except(
        self,
        indices:  Sequence[int],
        run_time: float = 0.5,
        dim_opacity: float = 0.12,
    ) -> "mn.Animation":
        """Fade all dots except those in ``indices`` to ``dim_opacity``."""
        _require_manim("dim_all_except")
        keep  = set(indices)
        anims = []
        for i, dot in enumerate(self.dots):
            if i not in keep:
                anims.append(dot.animate(run_time=run_time).set_opacity(dim_opacity))
        return AnimationGroup(*anims) if anims else AnimationGroup()

    def undim_all(self, run_time: float = 0.4) -> "mn.Animation":
        """Restore all dots to full opacity."""
        _require_manim("undim_all")
        return AnimationGroup(
            *[dot.animate(run_time=run_time).set_opacity(1.0)
              for dot in self.dots]
        )


# ---------------------------------------------------------------------------
# PlaneResiduals3D
# ---------------------------------------------------------------------------

class PlaneResiduals3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Vertical line segments from each observed point (x1_i, x2_i, y_i) down
    (or up) to its fitted value (x1_i, x2_i, y_hat_i) on the regression plane.

    Colour convention (matching ``ScatterCloud3D``):
        Positive residual (y > y_hat) → teal line going UP from plane.
        Negative residual (y < y_hat) → coral line going DOWN from plane.
    Stroke width scales with |residual| so large errors are visually prominent.

    Parameters
    ----------
    x1, x2 : array-like (n,)
    result : RegressionResult
    geometry : PlaneGeometry
    axes : ThreeDAxes, optional
    pos_color, neg_color : ManimColor, optional
    max_stroke_width : float
    min_stroke_width : float
    outlier_threshold : float
        |externally studentized residual| above which a circle is added.
    show_zero_line : bool
        Draw a faint circle on the plane at y_hat for each obs (expensive).

    Key sub-mobjects
    ----------------
    .positive_lines : VGroup
    .negative_lines : VGroup
    .outlier_markers : VGroup

    Animations
    ----------
    .animate_appear(run_time, stagger)
    .flash_outliers(run_time)
    .fade_to_zero(run_time)          — shrink residuals to zero (model fit improves)
    .update_fit(new_result, run_time) — morph to a new model
    """

    def __init__(
        self,
        x1:                   np.ndarray,
        x2:                   np.ndarray,
        result:               "RegressionResult",
        geometry:             PlaneGeometry = DEFAULT_GEOMETRY,
        axes:                 Optional["ThreeDAxes"] = None,
        pos_color             = None,
        neg_color             = None,
        max_stroke_width:     float = 3.2,
        min_stroke_width:     float = 0.6,
        outlier_threshold:    float = 2.5,
        **kwargs,
    ) -> None:
        _require_manim("PlaneResiduals3D.__init__")
        _require_corr("PlaneResiduals3D.__init__")
        super().__init__(**kwargs)

        self._x1      = np.asarray(x1, dtype=float)
        self._x2      = np.asarray(x2, dtype=float)
        self._result  = result
        self._axes    = axes
        self._geo     = geometry

        if _COLORS_AVAILABLE:
            pos_color = pos_color or ManimColor(TEAL_600.hex)
            neg_color = neg_color or ManimColor(CORAL_600.hex)
        else:
            pos_color = pos_color or GREEN
            neg_color = neg_color or RED

        self._pos_color = pos_color
        self._neg_color = neg_color

        self.positive_lines  = VGroup()
        self.negative_lines  = VGroup()
        self.outlier_markers = VGroup()

        self._build(result, x1, x2, axes,
                    pos_color, neg_color,
                    max_stroke_width, min_stroke_width,
                    outlier_threshold)

    def _build(
        self,
        result:            "RegressionResult",
        x1:                np.ndarray,
        x2:                np.ndarray,
        axes:              Optional["ThreeDAxes"],
        pos_color,
        neg_color,
        max_sw:            float,
        min_sw:            float,
        outlier_threshold: float,
    ) -> None:
        y_hat  = result.fitted
        y      = result.y
        resid  = result.residuals
        n      = result.n
        max_abs = max(float(np.max(np.abs(resid))), 1e-6)

        # Influence measures for outlier detection
        try:
            infl     = influence_measures(result)
            ext_stud = infl.externally_studentized
        except Exception:
            ext_stud = np.zeros(n)

        def _pt(x1v, x2v, yv):
            if axes is not None:
                return np.array(axes.c2p(float(x1v), float(x2v), float(yv)))
            return np.array([float(x1v), float(x2v), float(yv)])

        for i in range(n):
            ri     = float(resid[i])
            if abs(ri) < 1e-10:
                continue
            stroke = min_sw + (max_sw - min_sw) * abs(ri) / max_abs
            color  = pos_color if ri > 0 else neg_color

            plane_pt = _pt(x1[i], x2[i], y_hat[i])
            obs_pt   = _pt(x1[i], x2[i], y[i])

            line = Line(
                start        = plane_pt,
                end          = obs_pt,
                color        = color,
                stroke_width = stroke,
            )
            if ri > 0:
                self.positive_lines.add(line)
            else:
                self.negative_lines.add(line)

            # Outlier circle at the observation point
            if abs(float(ext_stud[i])) > outlier_threshold:
                ring = Sphere(
                    radius = self._geo.dot_radius * 2.5,
                    color  = YELLOW if _MANIM_AVAILABLE else None,
                )
                ring.set_fill(opacity=0)
                ring.set_stroke(color=YELLOW, width=2.0)
                ring.move_to(obs_pt)
                self.outlier_markers.add(ring)

        self.add(self.positive_lines, self.negative_lines, self.outlier_markers)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_appear(
        self,
        run_time: float = 1.8,
        stagger:  float = 0.03,
    ) -> "mn.Animation":
        """Grow residual lines from their plane endpoints."""
        _require_manim("animate_appear")
        all_lines = list(self.positive_lines) + list(self.negative_lines)
        return LaggedStart(
            *[Create(line, run_time=run_time * 0.45) for line in all_lines],
            lag_ratio = stagger,
        )

    def flash_outliers(self, run_time: float = 1.0) -> "mn.Animation":
        """Flash rings around outlier observations."""
        _require_manim("flash_outliers")
        if not self.outlier_markers:
            return AnimationGroup()
        return LaggedStart(
            *[Flash(m, color=YELLOW,
                    flash_radius=self._geo.dot_radius * 4,
                    run_time=run_time)
              for m in self.outlier_markers],
            lag_ratio = 0.12,
        )

    def fade_to_zero(self, run_time: float = 1.5) -> "mn.Animation":
        """
        Animate residual lines shrinking to zero — useful for showing
        how a better model reduces RSS.
        """
        _require_manim("fade_to_zero")
        all_lines = list(self.positive_lines) + list(self.negative_lines)
        return AnimationGroup(
            *[line.animate(run_time=run_time).scale(
                0.001, about_point=line.get_start()
              )
              for line in all_lines]
        )

    def update_fit(
        self,
        new_result: "RegressionResult",
        run_time:   float = 1.0,
    ) -> "mn.Animation":
        """Morph residual lines to match a re-fitted model."""
        _require_manim("update_fit")
        target = PlaneResiduals3D(
            x1       = self._x1,
            x2       = self._x2,
            result   = new_result,
            geometry = self._geo,
            axes     = self._axes,
            pos_color= self._pos_color,
            neg_color= self._neg_color,
        ).move_to(self.get_center())
        return Transform(self, target, run_time=run_time,
                         rate_func=rate_functions.ease_in_out_quad)


# ---------------------------------------------------------------------------
# CIShell3D — Confidence interval surface pair
# ---------------------------------------------------------------------------

class CIShell3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Upper and lower 95 % confidence interval surfaces bounding E[Y|x].

    The CI shell is two semi-transparent ``Surface`` objects rendered above
    and below the regression plane.  The gap between them widens toward the
    corners where leverage (and thus uncertainty) is highest.

    Parameters
    ----------
    result : RegressionResult (k=2)
    x1_range, x2_range : (float, float)
    geometry : PlaneGeometry
    axes : ThreeDAxes, optional
    level : float
        Confidence level.  Default 0.95.
    color : ManimColor, optional
        Color of both shell surfaces (default REGRESSION_FAMILY.light).
    opacity : float

    Key sub-mobjects
    ----------------
    .upper_surface : Surface
    .lower_surface : Surface

    Animations
    ----------
    .animate_appear(run_time)     — FadeIn both surfaces
    .morph_level(new_level, run_time) — morph to a different CI level
    """

    def __init__(
        self,
        result:       "RegressionResult",
        x1_range:     Tuple[float, float],
        x2_range:     Tuple[float, float],
        geometry:     PlaneGeometry = DEFAULT_GEOMETRY,
        axes:         Optional["ThreeDAxes"] = None,
        level:        float = 0.95,
        color                    = None,
        opacity:      float = None,
        **kwargs,
    ) -> None:
        _require_manim("CIShell3D.__init__")
        _require_corr("CIShell3D.__init__")
        super().__init__(**kwargs)

        if result.k != 2:
            raise ValueError(
                f"CIShell3D requires k=2 predictor fit, got k={result.k}."
            )

        self._result   = result
        self._x1_range = x1_range
        self._x2_range = x2_range
        self._level    = level
        self._geo      = geometry
        self._axes     = axes

        if _COLORS_AVAILABLE:
            color = color or ManimColor(REGRESSION_FAMILY.light.hex)
        else:
            color = color or BLUE
        opacity = opacity if opacity is not None else geometry.ci_opacity

        # Compute CI surfaces
        try:
            X_design = result.x
            XtX_inv  = np.linalg.inv(X_design.T @ X_design)
        except (np.linalg.LinAlgError, AttributeError):
            XtX_inv  = np.eye(3)

        X1, X2, Z_upper, Z_lower = ci_surface(
            beta       = result.beta,
            XtX_inv    = XtX_inv,
            s          = result.sigma_hat,
            n          = result.n,
            x1_range   = x1_range,
            x2_range   = x2_range,
            resolution = geometry.ci_resolution,
            level      = level,
        )

        self.upper_surface = self._make_surface(
            result.beta, XtX_inv, result.sigma_hat, result.n,
            x1_range, x2_range, geometry.ci_resolution,
            level, upper=True, axes=axes, color=color, opacity=opacity,
        )
        self.lower_surface = self._make_surface(
            result.beta, XtX_inv, result.sigma_hat, result.n,
            x1_range, x2_range, geometry.ci_resolution,
            level, upper=False, axes=axes, color=color, opacity=opacity,
        )
        self.add(self.upper_surface, self.lower_surface)

    @staticmethod
    def _make_surface(
        beta, XtX_inv, s, n,
        x1_range, x2_range, resolution,
        level, upper, axes, color, opacity,
    ) -> "Surface":
        df = n - 3
        if _SCIPY_AVAILABLE:
            t_crit = float(_sp.t.ppf((1 + level) / 2, df=max(df, 1)))
        else:
            t_crit = 1.959963985

        b0, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])
        sign        = 1.0 if upper else -1.0

        def _pad(lo, hi, p=0.15):
            span = max(hi - lo, 1e-6)
            return lo - p * span, hi + p * span

        u_lo, u_hi = _pad(*x1_range)
        v_lo, v_hi = _pad(*x2_range)

        def _func(u: float, v: float) -> np.ndarray:
            x0      = np.array([1.0, u, v])
            var_fit = float(x0 @ XtX_inv @ x0)
            margin  = t_crit * s * math.sqrt(max(var_fit, 0.0))
            z       = b0 + b1 * u + b2 * v + sign * margin
            if axes is not None:
                return np.array(axes.c2p(u, v, z))
            return np.array([u, v, z])

        surf = Surface(
            func         = _func,
            u_range      = [u_lo, u_hi],
            v_range      = [v_lo, v_hi],
            resolution   = (resolution, resolution),
            fill_opacity = opacity,
            checkerboard_colors = None,
        )
        surf.set_color(color)
        surf.set_stroke(width=0)
        return surf

    def animate_appear(self, run_time: float = 1.0) -> "mn.Animation":
        """Fade both CI surfaces in."""
        _require_manim("animate_appear")
        return AnimationGroup(
            FadeIn(self.upper_surface, run_time=run_time),
            FadeIn(self.lower_surface, run_time=run_time),
        )

    def morph_level(
        self,
        new_level: float,
        run_time:  float = 1.2,
    ) -> "mn.Animation":
        """Morph CI shells to a different confidence level."""
        _require_manim("morph_level")
        target = CIShell3D(
            result   = self._result,
            x1_range = self._x1_range,
            x2_range = self._x2_range,
            geometry = self._geo,
            axes     = self._axes,
            level    = new_level,
        ).move_to(self.get_center())
        return Transform(self, target, run_time=run_time,
                         rate_func=rate_functions.ease_in_out_sine)


# ---------------------------------------------------------------------------
# PIShell3D — Prediction interval surface pair
# ---------------------------------------------------------------------------

class PIShell3D(CIShell3D if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Upper and lower 95 % prediction interval surfaces.

    Identical to ``CIShell3D`` except uses the prediction variance formula:
        PI = y_hat ± t * s * sqrt(1 + x0'(X'X)^{-1}x0)

    The PI shell is always wider than the CI shell.  Rendering it in the same
    scene clearly illustrates the distinction between confidence in the mean
    vs. prediction of a new individual.

    Inherits all parameters and animations from ``CIShell3D``.
    """

    def __init__(
        self,
        result:   "RegressionResult",
        x1_range: Tuple[float, float],
        x2_range: Tuple[float, float],
        geometry: PlaneGeometry = DEFAULT_GEOMETRY,
        axes:     Optional["ThreeDAxes"] = None,
        level:    float = 0.95,
        color             = None,
        opacity:  float   = None,
        **kwargs,
    ) -> None:
        _require_manim("PIShell3D.__init__")

        # Bypass CIShell3D.__init__ — rebuild with PI formula
        VGroup.__init__(self, **kwargs)

        if result.k != 2:
            raise ValueError(
                f"PIShell3D requires k=2 predictor fit, got k={result.k}."
            )

        self._result   = result
        self._x1_range = x1_range
        self._x2_range = x2_range
        self._level    = level
        self._geo      = geometry
        self._axes     = axes

        if _COLORS_AVAILABLE:
            color = color or ManimColor(REGRESSION_FAMILY.muted.hex)
        else:
            color = color or BLUE
        opacity = opacity if opacity is not None else geometry.pi_opacity

        try:
            XtX_inv = np.linalg.inv(result.x.T @ result.x)
        except (np.linalg.LinAlgError, AttributeError):
            XtX_inv = np.eye(3)

        df = result.n - 3
        if _SCIPY_AVAILABLE:
            t_crit = float(_sp.t.ppf((1 + level) / 2, df=max(df, 1)))
        else:
            t_crit = 1.959963985

        beta = result.beta
        b0, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])
        s   = result.sigma_hat

        def _pad(lo, hi, p=0.15):
            span = max(hi - lo, 1e-6)
            return lo - p * span, hi + p * span

        u_lo, u_hi = _pad(*x1_range)
        v_lo, v_hi = _pad(*x2_range)
        res        = geometry.ci_resolution

        def _make(sign):
            def _func(u, v):
                x0       = np.array([1.0, u, v])
                var_pred = float(1.0 + x0 @ XtX_inv @ x0)
                margin   = t_crit * s * math.sqrt(max(var_pred, 0.0))
                z        = b0 + b1*u + b2*v + sign * margin
                if axes is not None:
                    return np.array(axes.c2p(u, v, z))
                return np.array([u, v, z])
            surf = Surface(
                func         = _func,
                u_range      = [u_lo, u_hi],
                v_range      = [v_lo, v_hi],
                resolution   = (res, res),
                fill_opacity = opacity,
                checkerboard_colors = None,
            )
            surf.set_color(color)
            surf.set_stroke(width=0)
            return surf

        self.upper_surface = _make(+1.0)
        self.lower_surface = _make(-1.0)
        self.add(self.upper_surface, self.lower_surface)


# ---------------------------------------------------------------------------
# ProjectionArrow3D
# ---------------------------------------------------------------------------

class ProjectionArrow3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Visual showing how a single point (x1, x2, y) projects onto the fitted plane.

    Draws three components:
      1. A vertical dashed line from (x1, x2, y_hat) straight down/up to (x1, x2, y).
      2. A horizontal dashed line from (x1_bar, x2_bar, y_hat) to (x1, x2, y_hat)
         on the plane surface.
      3. A label showing the residual value.

    Parameters
    ----------
    x1_val, x2_val, y_val : float
        Coordinates of the focal point.
    result : RegressionResult
    axes : ThreeDAxes, optional
    color : ManimColor, optional
    label : str, optional
        Custom label text.  Defaults to showing the residual value.

    Key sub-mobjects
    ----------------
    .vertical_line   : DashedLine
    .horizontal_line : DashedLine
    .plane_dot       : Sphere (the foot of the perpendicular on the plane)
    .obs_dot         : Sphere (the observation itself)
    .residual_label  : MathTex

    Animations
    ----------
    .animate_draw(run_time)   — draw lines in sequence
    .flash(run_time)          — Flash both line endpoints
    """

    def __init__(
        self,
        x1_val:   float,
        x2_val:   float,
        y_val:    float,
        result:   "RegressionResult",
        axes:     Optional["ThreeDAxes"] = None,
        color              = None,
        label:    Optional[str] = None,
        dot_radius: float  = 0.07,
        **kwargs,
    ) -> None:
        _require_manim("ProjectionArrow3D.__init__")
        _require_corr("ProjectionArrow3D.__init__")
        super().__init__(**kwargs)

        if _COLORS_AVAILABLE:
            color = color or ManimColor(DISCRETE_FAMILY.base.hex)
        else:
            color = color or YELLOW

        # Fitted value at this point
        beta  = result.beta
        y_hat = float(plane_predict(beta, x1_val, x2_val))
        resid = y_val - y_hat

        def _pt(x1, x2, y):
            if axes is not None:
                return np.array(axes.c2p(float(x1), float(x2), float(y)))
            return np.array([float(x1), float(x2), float(y)])

        plane_pt = _pt(x1_val, x2_val, y_hat)
        obs_pt   = _pt(x1_val, x2_val, y_val)

        # Vertical residual line
        self.vertical_line = DashedLine(
            start        = plane_pt,
            end          = obs_pt,
            color        = color,
            stroke_width = 2.2,
            dash_length  = 0.10,
        )

        # Plane-foot dot
        self.plane_dot = Sphere(radius=dot_radius, color=color)
        self.plane_dot.move_to(plane_pt)

        # Observation dot
        self.obs_dot = Sphere(radius=dot_radius * 1.3, color=color)
        self.obs_dot.move_to(obs_pt)

        # Residual label
        resid_str = label or rf"e = {resid:+.3f}"
        self.residual_label = MathTex(
            resid_str,
            font_size = 22,
            color     = color,
        )
        # Place label beside the midpoint of the residual line
        mid = (plane_pt + obs_pt) / 2
        self.residual_label.move_to(mid + np.array([0.35, 0.0, 0.0]))

        self.add(self.vertical_line, self.plane_dot,
                 self.obs_dot, self.residual_label)

    def animate_draw(self, run_time: float = 1.2) -> "mn.Animation":
        """Draw the vertical line, then flash the dots."""
        _require_manim("animate_draw")
        return Succession(
            Create(self.vertical_line, run_time=run_time * 0.55),
            AnimationGroup(
                GrowFromCenter(self.plane_dot, run_time=run_time * 0.20),
                GrowFromCenter(self.obs_dot,   run_time=run_time * 0.20),
                Write(self.residual_label,     run_time=run_time * 0.25),
            ),
        )

    def flash(self, run_time: float = 0.7) -> "mn.Animation":
        """Flash both endpoint dots simultaneously."""
        _require_manim("flash")
        return AnimationGroup(
            Flash(self.plane_dot, run_time=run_time),
            Flash(self.obs_dot,   run_time=run_time),
        )


# ===========================================================================
# LAYER D — Scene-level animation factories
# ===========================================================================

def build_plane_scene(
    result:        "RegressionResult",
    x1:            np.ndarray,
    x2:            np.ndarray,
    y:             np.ndarray,
    x1_name:       str  = "x_1",
    x2_name:       str  = "x_2",
    y_name:        str  = "y",
    geometry:      PlaneGeometry = DEFAULT_GEOMETRY,
    colouring:     Union[SurfaceColouring, str] = SurfaceColouring.FITTED,
    show_ci:       bool = True,
    show_residuals: bool = True,
    cloud_colouring: str = "residual",
) -> Tuple[
    "ThreeDAxes",
    "ScatterCloud3D",
    "RegressionPlane3D",
    Optional["PlaneResiduals3D"],
    Optional["CIShell3D"],
]:
    """
    Build all mobjects needed for a canonical two-predictor regression scene.

    Creates and positions:
    - ``ThreeDAxes`` spanning the data range with axis labels
    - ``ScatterCloud3D`` of data points
    - ``RegressionPlane3D``
    - ``PlaneResiduals3D`` (optional)
    - ``CIShell3D`` (optional)

    The function does **not** call any ``scene.play()`` — it only creates
    and positions the mobjects.  The caller controls what to display and when.

    Parameters
    ----------
    result : RegressionResult
        k=2 OLS fit.
    x1, x2, y : ndarray
        Original data arrays.
    x1_name, x2_name, y_name : str
        Variable names for axis labels.
    geometry : PlaneGeometry
    colouring : SurfaceColouring
        Plane colouring strategy.
    show_ci : bool
        Build a CIShell3D.
    show_residuals : bool
        Build a PlaneResiduals3D.
    cloud_colouring : str
        Colour strategy for scatter cloud ('residual', 'leverage', 'cooks_d', 'flat').

    Returns
    -------
    (axes, cloud, plane, residuals, ci_shell)
        ``residuals`` and ``ci_shell`` are ``None`` when their flags are False.
    """
    _require_manim("build_plane_scene")
    _require_corr("build_plane_scene")

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y  = np.asarray(y,  dtype=float)

    x1_range = (float(x1.min()), float(x1.max()))
    x2_range = (float(x2.min()), float(x2.max()))
    y_range  = (float(y.min()),  float(y.max()))

    # ---- ThreeDAxes ----
    x1_pad  = (x1_range[1] - x1_range[0]) * 0.18
    x2_pad  = (x2_range[1] - x2_range[0]) * 0.18
    y_pad   = (y_range[1]  - y_range[0])  * 0.22

    axes = ThreeDAxes(
        x_range = [x1_range[0] - x1_pad, x1_range[1] + x1_pad, (x1_range[1]-x1_range[0])/4],
        y_range = [x2_range[0] - x2_pad, x2_range[1] + x2_pad, (x2_range[1]-x2_range[0])/4],
        z_range = [y_range[0]  - y_pad,  y_range[1]  + y_pad,  (y_range[1] -y_range[0]) /4],
        **geometry.axes_kwargs,
    )

    # Axis labels
    x1_label = MathTex(x1_name, font_size=geometry.axis_label_font_size)
    x2_label = MathTex(x2_name, font_size=geometry.axis_label_font_size)
    y_label  = MathTex(y_name,  font_size=geometry.axis_label_font_size)

    x1_label.next_to(axes.x_axis, RIGHT, buff=0.15)
    x2_label.next_to(axes.y_axis, UP,    buff=0.15)
    y_label.next_to(axes.z_axis,  OUT,   buff=0.15)

    axes_group = VGroup(axes, x1_label, x2_label, y_label)

    # ---- Scatter cloud ----
    cloud = ScatterCloud3D(
        x1        = x1,
        x2        = x2,
        y         = y,
        result    = result,
        colouring = cloud_colouring,
        geometry  = geometry,
        axes      = axes,
    )

    # ---- Regression plane ----
    plane = RegressionPlane3D(
        result    = result,
        x1_range  = x1_range,
        x2_range  = x2_range,
        geometry  = geometry,
        colouring = colouring,
        axes      = axes,
    )

    # ---- Residuals ----
    residuals = None
    if show_residuals:
        residuals = PlaneResiduals3D(
            x1     = x1,
            x2     = x2,
            result = result,
            geometry = geometry,
            axes   = axes,
        )

    # ---- CI shell ----
    ci_shell = None
    if show_ci:
        ci_shell = CIShell3D(
            result   = result,
            x1_range = x1_range,
            x2_range = x2_range,
            geometry = geometry,
            axes     = axes,
        )

    return axes_group, cloud, plane, residuals, ci_shell


def morph_beta(
    plane:     RegressionPlane3D,
    scene,
    beta_sequence: Sequence[np.ndarray],
    run_time_each: float = 1.2,
    wait_each:     float = 0.5,
) -> None:
    """
    Animate a ``RegressionPlane3D`` morphing through a sequence of coefficient
    vectors, playing each transition on ``scene`` and waiting briefly between.

    Parameters
    ----------
    plane : RegressionPlane3D
    scene : ThreeDScene
    beta_sequence : sequence of ndarray (3,)
        Each array is [b0, b1, b2].
    run_time_each : float
    wait_each : float
    """
    _require_manim("morph_beta")
    for beta in beta_sequence:
        scene.play(plane.animate_tilt(np.asarray(beta), run_time=run_time_each))
        scene.wait(wait_each)


def sweep_x1(
    plane:    RegressionPlane3D,
    scene,
    x1_vals:  Sequence[float],
    x2_fixed: float,
    marker:   Optional["ProjectionArrow3D"] = None,
    run_time: float = 2.5,
) -> None:
    """
    Animate a vertical marker sweeping along x1 at fixed x2,
    tracing the regression plane's prediction curve for X2 = x2_fixed.

    If ``marker`` is not provided, one is created from the first x1 value.

    Parameters
    ----------
    plane : RegressionPlane3D
    scene : ThreeDScene
    x1_vals : sequence of float
    x2_fixed : float
    marker : ProjectionArrow3D, optional
    run_time : float
        Total sweep duration.
    """
    _require_manim("sweep_x1")
    result = plane._result
    beta   = plane._beta

    dt = run_time / max(len(x1_vals) - 1, 1)
    for i, x1v in enumerate(x1_vals[1:], 1):
        y_hat = float(plane_predict(beta, x1v, x2_fixed))
        axes  = plane._axes_ref
        if axes is not None:
            target = np.array(axes.c2p(x1v, x2_fixed, y_hat))
        else:
            target = np.array([x1v, x2_fixed, y_hat])

        if marker is not None:
            scene.play(
                marker.animate(run_time=dt).move_to(target),
                rate_func=rate_functions.linear,
            )
        else:
            scene.wait(dt)


def rotate_view(
    scene,
    phi:      float = 70 * (3.14159 / 180),
    theta:    float = -45 * (3.14159 / 180),
    run_time: float = 2.0,
) -> None:
    """
    Animate the camera rotating to a new (phi, theta) viewpoint
    for a ``ThreeDScene``.

    Parameters
    ----------
    scene : ThreeDScene
    phi : float
        Polar angle in radians (0 = top-down, pi/2 = side-on).
    theta : float
        Azimuthal angle in radians.
    run_time : float
    """
    _require_manim("rotate_view")
    scene.move_camera(phi=phi, theta=theta, run_time=run_time)


# ===========================================================================
# LAYER E — Formula registry bridge
# ===========================================================================

def _build_multiple_regression_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \varepsilon_i"
        r",\quad \varepsilon_i \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2)"
    )
    return TexFormula(
        name        = "multiple_regression_model",
        raw         = raw,
        description = "Multiple linear regression model (k=2 predictors)",
        parts       = {
            "response":    r"y_i",
            "intercept":   r"\beta_0",
            "slope_x1":    r"\beta_1 x_{1i}",
            "slope_x2":    r"\beta_2 x_{2i}",
            "error":       r"\varepsilon_i",
            "error_dist":  r"\mathcal{N}(0, \sigma^2)",
        },
        tags        = ["regression", "multiple", "model"],
    )


def _build_plane_ols_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"\hat{\boldsymbol{\beta}} = "
        r"\left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}"
    )
    rss_raw = (
        r"\mathrm{RSS} = "
        r"\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2 = "
        r"\left\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\right\|^2"
    )
    return TexFormula(
        name        = "ols_plane_estimator",
        raw         = raw,
        description = "OLS estimator for the regression plane coefficients",
        parts       = {
            "beta_hat":    r"\hat{\boldsymbol{\beta}}",
            "gram":        r"\mathbf{X}^\top \mathbf{X}",
            "gram_inv":    r"\left(\mathbf{X}^\top \mathbf{X}\right)^{-1}",
            "projection":  r"\mathbf{X}^\top \mathbf{y}",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"\mathrm{RSS}(\boldsymbol{\beta})",
                rhs = r"\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2",
                annotation = "Residual sum of squares",
            ),
            TexDerivationStep(
                lhs = r"\nabla_{\boldsymbol{\beta}}\,\mathrm{RSS}",
                rhs = r"-2\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}) = \mathbf{0}",
                annotation = "Set gradient to zero",
            ),
            TexDerivationStep(
                lhs = r"\hat{\boldsymbol{\beta}}",
                rhs = raw.split("=", 1)[1].strip(),
                annotation = "Normal equations solved",
            ),
        ],
        tags        = ["regression", "ols", "plane", "matrix"],
    )


def _build_plane_ci_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    ci_margin = (
        r"t_{n-p,\,\alpha/2} \cdot \hat{\sigma}"
        r"\sqrt{\mathbf{x}_0^\top \left(\mathbf{X}^\top\mathbf{X}\right)^{-1} \mathbf{x}_0}"
    )
    raw = rf"\hat{{y}}(\mathbf{{x}}_0) \pm {ci_margin}"
    return TexFormula(
        name        = "plane_ci_formula",
        raw         = raw,
        description = "Pointwise 95% CI for E[Y|x0] on the regression plane",
        parts       = {
            "fitted":      r"\hat{y}(\mathbf{x}_0)",
            "t_crit":      r"t_{n-p,\,\alpha/2}",
            "sigma_hat":   r"\hat{\sigma}",
            "leverage_sq": (
                r"\mathbf{x}_0^\top \left(\mathbf{X}^\top\mathbf{X}\right)^{-1}"
                r"\mathbf{x}_0"
            ),
            "margin":      ci_margin,
        },
        tags        = ["regression", "confidence_interval", "plane"],
    )


def _build_plane_pi_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    pi_margin = (
        r"t_{n-p,\,\alpha/2} \cdot \hat{\sigma}"
        r"\sqrt{1 + \mathbf{x}_0^\top \left(\mathbf{X}^\top\mathbf{X}\right)^{-1} \mathbf{x}_0}"
    )
    raw = rf"\hat{{y}}(\mathbf{{x}}_0) \pm {pi_margin}"
    return TexFormula(
        name        = "plane_pi_formula",
        raw         = raw,
        description = "Pointwise 95% PI for a new observation Y_new at x0",
        parts       = {
            "fitted":   r"\hat{y}(\mathbf{x}_0)",
            "margin":   pi_margin,
            "extra_1":  r"1 +",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"\mathrm{Var}\!\left(Y_{\mathrm{new}} - \hat{y}\right)",
                rhs = (
                    r"\sigma^2\!\left(1 + "
                    r"\mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0\right)"
                ),
                annotation = "Individual + estimation variance",
            ),
            TexDerivationStep(
                lhs = "PI",
                rhs = raw.split("\\pm", 1)[0].strip()
                      + r" \pm " + pi_margin,
                annotation = "Plug in estimated sigma",
            ),
        ],
        tags        = ["regression", "prediction_interval", "plane"],
    )


def _build_rsq_decomp_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"\underbrace{\sum(y_i-\bar{y})^2}_{\mathrm{SST}} = "
        r"\underbrace{\sum(\hat{y}_i-\bar{y})^2}_{\mathrm{SSR}} + "
        r"\underbrace{\sum(y_i-\hat{y}_i)^2}_{\mathrm{SSE}}"
    )
    return TexFormula(
        name        = "sst_decomposition",
        raw         = raw,
        description = "Total = Regression + Error sum-of-squares decomposition",
        parts       = {
            "SST": r"\sum(y_i-\bar{y})^2",
            "SSR": r"\sum(\hat{y}_i-\bar{y})^2",
            "SSE": r"\sum(y_i-\hat{y}_i)^2",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"y_i - \bar{y}",
                rhs = r"(\hat{y}_i - \bar{y}) + (y_i - \hat{y}_i)",
                annotation = "Add and subtract y_hat",
            ),
            TexDerivationStep(
                lhs = r"\sum(y_i-\bar{y})^2",
                rhs = (
                    r"\sum(\hat{y}_i-\bar{y})^2"
                    r" + \sum(y_i-\hat{y}_i)^2"
                    r" + 2\sum(\hat{y}_i-\bar{y})(y_i-\hat{y}_i)"
                ),
                annotation = "Square and expand",
            ),
            TexDerivationStep(
                lhs = "",
                rhs = (
                    r"= \mathrm{SSR} + \mathrm{SSE}"
                    r"\quad \text{(cross-term = 0 by OLS)}"
                ),
                annotation = "Cross-term vanishes",
            ),
        ],
        tags        = ["regression", "r_squared", "anova"],
    )


# Build PLANE_FORMULAS registry
PLANE_FORMULAS: Dict[str, "TexFormula"] = {}

if _TEX_AVAILABLE:
    _plane_formula_builders = [
        _build_multiple_regression_formula,
        _build_plane_ols_formula,
        _build_plane_ci_formula,
        _build_plane_pi_formula,
        _build_rsq_decomp_formula,
    ]
    for _builder in _plane_formula_builders:
        _f = _builder()
        if _f is not None:
            PLANE_FORMULAS[_f.name] = _f
            try:
                register_formula(_f)
            except (ValueError, KeyError):
                pass


# ===========================================================================
# Re-export convenience
# ===========================================================================

# Make ols_fit available from this module so callers only need one import
try:
    from manim_stats.regression.correlation import ols_fit as ols_fit
except ImportError:
    pass


# ===========================================================================
# __all__
# ===========================================================================

__all__ = [
    # Layer A — math helpers
    "plane_predict",
    "plane_grid",
    "ci_surface",
    "pi_surface",
    "plane_normal",
    "tilt_angle_deg",
    "leverage_grid",
    "standardise_grid",

    # Layer B — config dataclasses
    "SurfaceColouring",
    "PlaneGeometry",
    "DEFAULT_GEOMETRY",
    "COMPACT_GEOMETRY",
    "HIRES_GEOMETRY",

    # Layer C — Manim mobjects
    "RegressionPlane3D",
    "ScatterCloud3D",
    "PlaneResiduals3D",
    "CIShell3D",
    "PIShell3D",
    "ProjectionArrow3D",

    # Layer D — scene-level animations
    "build_plane_scene",
    "morph_beta",
    "sweep_x1",
    "rotate_view",

    # Layer E — formula registry
    "PLANE_FORMULAS",

    # Re-export
    "ols_fit",
]