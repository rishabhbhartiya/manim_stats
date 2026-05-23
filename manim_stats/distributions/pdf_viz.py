"""
manim_stats/distributions/pdf_viz.py
======================================
Production-quality probability density function visualizer.

Design philosophy
-----------------
A PDF visualizer in a statistics course must do far more than draw a
curve.  Every annotatable feature of a distribution — its moments,
tails, specific probability regions, percentiles, comparison to data —
must be expressible as an independent visual layer that can be added,
removed, and animated independently.

``PDFVisualizer3D`` is built from these independent layers:

    Curve layer     – stroke + glow halo (Catmull-Rom smoothed).
    Fill layer      – gradient area fill under the full curve.
    Moment layer    – μ, σ markers with bracket annotations.
    Region layer    – arbitrary [a, b] probability regions with P(·) labels.
    Percentile layer– vertical markers at specified quantiles.
    KDE layer       – empirical density from observed data.
    Overlay layer   – second (or third) PDF curve for comparison.
    Divergence layer– shaded region between two overlapping curves.

Each layer is an independently animated VGroup, so a teacher can
``Create`` the curve first, then ``FadeIn`` the moment annotations,
then shade a tail region — all as separate ``scene.play`` calls.

Distribution support
--------------------
All distributions are self-contained (no scipy dependency).  Supported
families with their parameterisations:

    Normal(μ, σ)
    StudentT(df)
    ChiSquared(df)
    Gamma(shape=k, rate=λ)            mean = k/λ
    Beta(α, β)                        support [0, 1]
    Exponential(rate=λ)               mean = 1/λ
    Cauchy(x0, γ)                     heavy-tailed
    Laplace(μ, b)                     double-exponential
    LogNormal(μ_log, σ_log)
    Weibull(shape=k, scale=λ)
    Uniform(a, b)

Classes
-------
PDFConfig
PDFVisualizer3D
MultiplePDFComparison3D
PDFDistribution            (base)

Distribution helpers
--------------------
NormalDist
StudentTDist
ChiSquaredDist
GammaDist
BetaDist
ExponentialDist
CauchyDist
LaplaceDist
LogNormalDist
WeibullDist
UniformDist

Ready-to-render scenes
----------------------
NormalPDFScene
TDistScene
GammaPDFScene
PDFComparisonScene
PDFParameterSweepScene

Usage
-----
    from manim import *
    from manim_stats.distributions.pdf_viz import PDFVisualizer3D, NormalDist

    class MyScene(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65*DEGREES, theta=-45*DEGREES)

            viz = PDFVisualizer3D(NormalDist(mu=0, sigma=1),
                                  x_range=(-4, 4, 0.04))
            self.play(viz.animate_curve())
            self.play(viz.animate_fill())
            self.play(viz.animate_moments())
            viz.shade_tail_right(z_crit=1.645, label="5%")
            self.play(FadeIn(viz.regions))
            self.wait()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Union, Dict
from abc import ABC, abstractmethod
import numpy as np

from manim import (
    VGroup, VMobject, Polygon, Line, DashedLine, Dot3D,
    Text, MathTex, Arrow, Brace,
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
# Shared colour helpers
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

def _catmull_rom(points: np.ndarray, resolution: int = 20) -> np.ndarray:
    """Catmull-Rom spline through *points*, returning smoothed path."""
    n = len(points)
    if n < 2:
        return points.copy()
    p = np.vstack([2 * points[0] - points[1],
                   points,
                   2 * points[-1] - points[-2]])
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


def _build_fill_polygon(
    curve_pts: np.ndarray,
    floor_z: float,
    y_pos: float,
    n_gradient_strips: int = 10,
    color: ManimColor = ManimColor("#4A90D9"),
    opacity: float = 0.22,
    gradient: bool = True,
) -> VGroup:
    """Build a gradient area fill polygon below *curve_pts*.

    Returns a VGroup of thin horizontal strips (gradient mode) or a
    single Polygon (flat mode).
    """
    floor_pts = np.array([[p[0], y_pos, floor_z] for p in curve_pts])
    grp = VGroup()

    if not gradient or n_gradient_strips <= 1:
        poly_pts = list(curve_pts) + list(reversed(floor_pts))
        grp.add(Polygon(
            *poly_pts,
            fill_color=_with_opacity(color, opacity),
            fill_opacity=1.0, stroke_width=0,
        ))
        return grp

    for s in range(n_gradient_strips):
        t_lo = s / n_gradient_strips
        t_hi = (s + 1) / n_gradient_strips
        strip_op = opacity * (1.0 - 0.60 * t_lo)
        upper_lo = curve_pts * (1 - t_lo) + floor_pts * t_lo
        upper_hi = curve_pts * (1 - t_hi) + floor_pts * t_hi
        strip_pts = list(upper_lo) + list(reversed(upper_hi))
        grp.add(Polygon(
            *strip_pts,
            fill_color=_with_opacity(color, strip_op),
            fill_opacity=1.0, stroke_width=0,
        ))
    return grp


# ===========================================================================
# Distribution hierarchy
# ===========================================================================

class PDFDistribution(ABC):
    """Abstract base class for all continuous distributions.

    Subclasses implement ``pdf(x)``, ``cdf(x)``, ``ppf(p)``, and
    provide ``default_x_range``, ``mean``, ``variance``, and ``name``.

    No scipy is used.  All calculations are implemented analytically or
    via numerical integration where needed.
    """

    @abstractmethod
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density at *x*."""

    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function at *x*."""

    def ppf(self, p: float) -> float:
        """Percent-point function (quantile) via bisection."""
        lo, hi = self.default_x_range[0], self.default_x_range[1]
        for _ in range(64):
            mid = (lo + hi) / 2
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def param_string(self) -> str:
        """Short parameter description, e.g. 'μ=0, σ=1'."""

    @property
    @abstractmethod
    def default_x_range(self) -> Tuple[float, float]:
        """Sensible (x_min, x_max) for plotting."""

    @property
    @abstractmethod
    def mean(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def variance(self) -> Optional[float]: ...

    @property
    def std(self) -> Optional[float]:
        v = self.variance
        return float(np.sqrt(v)) if v is not None else None

    def pdf_array(self, x_range: Tuple[float, float], n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Return (xs, ys) evaluated over *x_range* at *n* points."""
        xs = np.linspace(x_range[0], x_range[1], n)
        ys = np.array([max(float(self.pdf(x)), 0.0) for x in xs])
        return xs, ys


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

class NormalDist(PDFDistribution):
    """Normal (Gaussian) distribution  N(μ, σ).

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (> 0).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def pdf(self, x):
        z = (np.asarray(x, dtype=float) - self.mu) / self.sigma
        return np.exp(-0.5 * z**2) / (self.sigma * np.sqrt(TAU))

    def cdf(self, x):
        z = (float(x) - self.mu) / (self.sigma * np.sqrt(2))
        return 0.5 * (1 + _erf(z))

    @property
    def name(self) -> str:
        return "Normal"

    @property
    def param_string(self) -> str:
        return f"μ = {self.mu:.3g},  σ = {self.sigma:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        return (self.mu - 4 * self.sigma, self.mu + 4 * self.sigma)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma**2


# ---------------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------------

class StudentTDist(PDFDistribution):
    """Student's t-distribution with *df* degrees of freedom.

    Parameters
    ----------
    df : float
        Degrees of freedom (> 0).
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    """

    def __init__(self, df: float = 5.0, mu: float = 0.0, sigma: float = 1.0):
        self.df = float(df)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._log_norm = _lgamma((df + 1) / 2) - _lgamma(df / 2) - 0.5 * np.log(df * PI)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sigma
        nu = self.df
        return (np.exp(self._log_norm)
                * (1 + z**2 / nu) ** (-(nu + 1) / 2)
                / self.sigma)

    def cdf(self, x):
        # Regularised incomplete beta: use numerical integration
        z = (float(x) - self.mu) / self.sigma
        nu = self.df
        if z == 0:
            return 0.5
        x_b = nu / (nu + z**2)
        ib = _reg_inc_beta(nu / 2, 0.5, x_b)
        return 0.5 * ib if z < 0 else 1 - 0.5 * ib

    @property
    def name(self) -> str:
        return "Student's t"

    @property
    def param_string(self) -> str:
        return f"df = {self.df:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        return (-5.0 * self.sigma + self.mu, 5.0 * self.sigma + self.mu)

    @property
    def mean(self) -> Optional[float]:
        return self.mu if self.df > 1 else None

    @property
    def variance(self) -> Optional[float]:
        if self.df > 2:
            return self.sigma**2 * self.df / (self.df - 2)
        return None


# ---------------------------------------------------------------------------
# Chi-squared
# ---------------------------------------------------------------------------

class ChiSquaredDist(PDFDistribution):
    """Chi-squared distribution  χ²(df).

    Parameters
    ----------
    df : int or float
        Degrees of freedom (> 0).
    """

    def __init__(self, df: float = 5.0):
        if df <= 0:
            raise ValueError("df must be positive")
        self.df = float(df)
        self._k2 = df / 2
        self._log_norm = self._k2 * np.log(2) + _lgamma(self._k2)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.where(
            x > 0,
            np.exp((self._k2 - 1) * np.log(np.where(x > 0, x, 1))
                   - x / 2 - self._log_norm),
            0.0,
        )
        return out

    def cdf(self, x):
        x = float(x)
        if x <= 0:
            return 0.0
        return _reg_lower_gamma(self._k2, x / 2)

    @property
    def name(self) -> str:
        return "Chi-squared"

    @property
    def param_string(self) -> str:
        return f"df = {self.df:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        hi = max(self.df + 5 * np.sqrt(2 * self.df), 3.0)
        return (0.0, hi)

    @property
    def mean(self) -> float:
        return self.df

    @property
    def variance(self) -> float:
        return 2 * self.df


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class GammaDist(PDFDistribution):
    """Gamma distribution  Gamma(shape=k, rate=λ).  Mean = k/λ.

    Parameters
    ----------
    shape : float
        Shape parameter k (> 0).
    rate : float
        Rate parameter λ (> 0).  Scale = 1/rate.
    """

    def __init__(self, shape: float = 2.0, rate: float = 1.0):
        self.k = float(shape)
        self.lam = float(rate)
        self._log_norm = _lgamma(self.k) - self.k * np.log(self.lam)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.where(
            x > 0,
            np.exp((self.k - 1) * np.log(np.where(x > 0, x, 1))
                   - self.lam * x - self._log_norm),
            0.0,
        )
        return out

    def cdf(self, x):
        x = float(x)
        if x <= 0:
            return 0.0
        return _reg_lower_gamma(self.k, self.lam * x)

    @property
    def name(self) -> str:
        return "Gamma"

    @property
    def param_string(self) -> str:
        return f"k = {self.k:.3g},  λ = {self.lam:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        mu = self.k / self.lam
        sig = np.sqrt(self.k) / self.lam
        return (0.0, mu + 5 * sig)

    @property
    def mean(self) -> float:
        return self.k / self.lam

    @property
    def variance(self) -> float:
        return self.k / self.lam**2


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class BetaDist(PDFDistribution):
    """Beta distribution  Beta(α, β).  Support [0, 1].

    Parameters
    ----------
    alpha, beta : float
        Shape parameters (> 0).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 3.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._log_norm = _lgamma(alpha) + _lgamma(beta) - _lgamma(alpha + beta)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, b = self.alpha, self.beta
        out = np.where(
            (x > 0) & (x < 1),
            np.exp(
                (a - 1) * np.log(np.where(x > 0, x, 1e-300))
                + (b - 1) * np.log(np.where(x < 1, 1 - x, 1e-300))
                - self._log_norm
            ),
            0.0,
        )
        return out

    def cdf(self, x):
        x = float(x)
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        return _reg_inc_beta(self.alpha, self.beta, x)

    @property
    def name(self) -> str:
        return "Beta"

    @property
    def param_string(self) -> str:
        return f"α = {self.alpha:.3g},  β = {self.beta:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b)**2 * (a + b + 1))


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------

class ExponentialDist(PDFDistribution):
    """Exponential distribution  Exp(rate=λ).  Mean = 1/λ.

    Parameters
    ----------
    rate : float
        Rate parameter λ (> 0).
    """

    def __init__(self, rate: float = 1.0):
        self.lam = float(rate)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.lam * np.exp(-self.lam * x), 0.0)

    def cdf(self, x):
        x = float(x)
        return max(0.0, 1.0 - np.exp(-self.lam * x))

    @property
    def name(self) -> str:
        return "Exponential"

    @property
    def param_string(self) -> str:
        return f"λ = {self.lam:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        return (0.0, 5.0 / self.lam)

    @property
    def mean(self) -> float:
        return 1.0 / self.lam

    @property
    def variance(self) -> float:
        return 1.0 / self.lam**2


# ---------------------------------------------------------------------------
# Cauchy
# ---------------------------------------------------------------------------

class CauchyDist(PDFDistribution):
    """Cauchy distribution  Cauchy(x₀, γ).  Heavy-tailed.

    Parameters
    ----------
    x0 : float
        Location parameter.
    gamma : float
        Scale parameter (half-width at half-maximum).
    """

    def __init__(self, x0: float = 0.0, gamma: float = 1.0):
        self.x0 = float(x0)
        self.gamma = float(gamma)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (PI * self.gamma * (1 + ((x - self.x0) / self.gamma)**2))

    def cdf(self, x):
        return 0.5 + np.arctan((float(x) - self.x0) / self.gamma) / PI

    @property
    def name(self) -> str:
        return "Cauchy"

    @property
    def param_string(self) -> str:
        return f"x₀ = {self.x0:.3g},  γ = {self.gamma:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        return (self.x0 - 6 * self.gamma, self.x0 + 6 * self.gamma)

    @property
    def mean(self) -> Optional[float]:
        return None   # undefined

    @property
    def variance(self) -> Optional[float]:
        return None   # undefined


# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------

class LaplaceDist(PDFDistribution):
    """Laplace (double-exponential) distribution  Laplace(μ, b).

    Parameters
    ----------
    mu : float
        Location (mean).
    b : float
        Scale (diversity); std = b√2.
    """

    def __init__(self, mu: float = 0.0, b: float = 1.0):
        self.mu = float(mu)
        self.b = float(b)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-np.abs(x - self.mu) / self.b) / (2 * self.b)

    def cdf(self, x):
        z = (float(x) - self.mu) / self.b
        return 0.5 * (1 + np.sign(z) * (1 - np.exp(-abs(z))))

    @property
    def name(self) -> str:
        return "Laplace"

    @property
    def param_string(self) -> str:
        return f"μ = {self.mu:.3g},  b = {self.b:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        return (self.mu - 6 * self.b, self.mu + 6 * self.b)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return 2 * self.b**2


# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------

class LogNormalDist(PDFDistribution):
    """Log-normal distribution  LN(μ_log, σ_log).

    X = exp(Y) where Y ~ N(μ_log, σ_log).

    Parameters
    ----------
    mu_log : float
        Mean of the underlying normal (log-space).
    sigma_log : float
        Std of the underlying normal (log-space).
    """

    def __init__(self, mu_log: float = 0.0, sigma_log: float = 0.5):
        self.mu_log = float(mu_log)
        self.sigma_log = float(sigma_log)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.where(
            x > 0,
            np.exp(-0.5 * ((np.log(np.where(x > 0, x, 1)) - self.mu_log)
                           / self.sigma_log)**2)
            / (np.where(x > 0, x, 1) * self.sigma_log * np.sqrt(TAU)),
            0.0,
        )
        return out

    def cdf(self, x):
        x = float(x)
        if x <= 0:
            return 0.0
        z = (np.log(x) - self.mu_log) / (self.sigma_log * np.sqrt(2))
        return 0.5 * (1 + _erf(z))

    @property
    def name(self) -> str:
        return "Log-Normal"

    @property
    def param_string(self) -> str:
        return f"μ_log = {self.mu_log:.3g},  σ_log = {self.sigma_log:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        hi = np.exp(self.mu_log + 4 * self.sigma_log)
        return (0.0, hi)

    @property
    def mean(self) -> float:
        return np.exp(self.mu_log + 0.5 * self.sigma_log**2)

    @property
    def variance(self) -> float:
        return (np.exp(self.sigma_log**2) - 1) * np.exp(2*self.mu_log + self.sigma_log**2)


# ---------------------------------------------------------------------------
# Weibull
# ---------------------------------------------------------------------------

class WeibullDist(PDFDistribution):
    """Weibull distribution  Weibull(shape=k, scale=λ).

    Parameters
    ----------
    shape : float
        Shape parameter k (> 0).  k < 1 = decreasing; k = 1 = Exp.
    scale : float
        Scale parameter λ (> 0).  Characteristic life.
    """

    def __init__(self, shape: float = 2.0, scale: float = 1.0):
        self.k = float(shape)
        self.lam = float(scale)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        k, lam = self.k, self.lam
        return np.where(
            x >= 0,
            (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k),
            0.0,
        )

    def cdf(self, x):
        x = float(x)
        if x < 0:
            return 0.0
        return 1.0 - np.exp(-(x / self.lam)**self.k)

    @property
    def name(self) -> str:
        return "Weibull"

    @property
    def param_string(self) -> str:
        return f"k = {self.k:.3g},  λ = {self.lam:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        hi = self.lam * (-np.log(0.001)) ** (1 / self.k)
        return (0.0, hi)

    @property
    def mean(self) -> float:
        return self.lam * np.exp(_lgamma(1 + 1 / self.k))

    @property
    def variance(self) -> float:
        g1 = np.exp(_lgamma(1 + 1 / self.k))
        g2 = np.exp(_lgamma(1 + 2 / self.k))
        return self.lam**2 * (g2 - g1**2)


# ---------------------------------------------------------------------------
# Uniform (continuous)
# ---------------------------------------------------------------------------

class UniformDist(PDFDistribution):
    """Continuous Uniform distribution  U(a, b).

    Parameters
    ----------
    a, b : float
        Support endpoints.
    """

    def __init__(self, a: float = 0.0, b: float = 1.0):
        if b <= a:
            raise ValueError("b must be > a")
        self.a = float(a)
        self.b = float(b)
        self._height = 1.0 / (b - a)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.a) & (x <= self.b), self._height, 0.0)

    def cdf(self, x):
        x = float(x)
        if x < self.a:
            return 0.0
        if x > self.b:
            return 1.0
        return (x - self.a) / (self.b - self.a)

    @property
    def name(self) -> str:
        return "Uniform"

    @property
    def param_string(self) -> str:
        return f"a = {self.a:.3g},  b = {self.b:.3g}"

    @property
    def default_x_range(self) -> Tuple[float, float]:
        pad = (self.b - self.a) * 0.15
        return (self.a - pad, self.b + pad)

    @property
    def mean(self) -> float:
        return (self.a + self.b) / 2

    @property
    def variance(self) -> float:
        return (self.b - self.a)**2 / 12


# ---------------------------------------------------------------------------
# Special function implementations (no scipy)
# ---------------------------------------------------------------------------

def _erf(z: float) -> float:
    """Error function via Horner-form polynomial approximation (max err < 1.5e-7)."""
    t = 1.0 / (1.0 + 0.3275911 * abs(z))
    poly = t * (0.254829592 + t * (-0.284496736
           + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    val = 1.0 - poly * np.exp(-z * z)
    return val if z >= 0 else -val


def _lgamma(x: float) -> float:
    """Natural log of the gamma function via Lanczos approximation."""
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if x < 0.5:
        return np.log(PI / np.sin(PI * x)) - _lgamma(1 - x)
    x -= 1
    a = c[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += c[i] / (x + i)
    return 0.5 * np.log(TAU) + (x + 0.5) * np.log(t) - t + np.log(a)


def _reg_lower_gamma(a: float, x: float, n_terms: int = 150) -> float:
    """Regularised lower incomplete gamma P(a, x) via series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    log_x = np.log(x)
    term = np.exp(a * log_x - x - _lgamma(a)) / a
    total = term
    for k in range(1, n_terms):
        term *= x / (a + k)
        total += term
        if abs(term) < 1e-14 * abs(total):
            break
    return min(1.0, max(0.0, total))


def _reg_inc_beta(a: float, b: float, x: float, n_terms: int = 200) -> float:
    """Regularised incomplete beta I_x(a, b) via continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_norm = _lgamma(a + b) - _lgamma(a) - _lgamma(b)
    log_factor = log_norm + a * np.log(x) + b * np.log(1 - x)

    # Use symmetry for faster convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _reg_inc_beta(b, a, 1.0 - x)

    # Continued fraction via Lentz's method
    def beta_cf(aa, bb, xx):
        fpmin = 1e-300
        qab = aa + bb
        qap = aa + 1
        qam = aa - 1
        c = 1.0
        d = 1.0 - qab * xx / qap
        d = fpmin if abs(d) < fpmin else d
        d = 1.0 / d
        h = d
        for m in range(1, n_terms + 1):
            m2 = 2 * m
            aa_m = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + aa_m * d
            c = 1.0 + aa_m / c
            d = fpmin if abs(d) < fpmin else d
            c = fpmin if abs(c) < fpmin else c
            d = 1.0 / d
            h *= d * c
            aa_m = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + aa_m * d
            c = 1.0 + aa_m / c
            d = fpmin if abs(d) < fpmin else d
            c = fpmin if abs(c) < fpmin else c
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        return h

    cf = beta_cf(a, b, x)
    return np.exp(log_factor) * cf / a


# ===========================================================================
# PDFConfig
# ===========================================================================

@dataclass
class PDFConfig:
    """Complete visual configuration for a single ``PDFVisualizer3D``.

    Curve layers
    ~~~~~~~~~~~~
    ``curve_color``        : main stroke colour.
    ``curve_stroke_width`` : stroke width.
    ``curve_opacity``      : stroke opacity.
    ``glow_width``         : wider halo behind the stroke (0 = off).
    ``glow_opacity``       : opacity of glow halo.
    ``smooth_resolution``  : Catmull-Rom points per segment.

    Fill layers
    ~~~~~~~~~~~
    ``fill_color``         : area fill colour (defaults to curve_color).
    ``fill_opacity``       : area fill opacity.
    ``fill_gradient``      : True = darker near floor, lighter near curve.
    ``n_gradient_strips``  : number of gradient strips.
    ``floor_z``            : z-coordinate of the density baseline.

    Moment layer
    ~~~~~~~~~~~~
    ``show_mean_line``     : vertical line at μ.
    ``show_sigma_brackets``: horizontal bracket ±1σ, ±2σ around μ.
    ``mean_color``         : colour of mean line and annotation.
    ``sigma_color``        : colour of σ bracket.
    ``moment_stroke_width``: stroke width of moment markers.

    Region layer
    ~~~~~~~~~~~~
    ``region_colors``      : cycle of colours for successive shade regions.
    ``region_opacity``     : opacity of shaded regions.
    ``region_label_font``  : font size for P(·) probability labels.
    ``region_label_decimals`` : decimal places in P(·) labels.

    Percentile layer
    ~~~~~~~~~~~~~~~~
    ``percentile_color``   : colour of percentile marker lines.
    ``percentile_font``    : font size of percentile labels.

    Annotation
    ~~~~~~~~~~
    ``show_title``         : show distribution name above the curve.
    ``title_font``         : font size of title.
    ``show_param_label``   : show parameter string beneath the title.
    ``param_label_font``   : font size of parameter label.

    Layout
    ~~~~~~
    ``x_pos``              : x offset of the entire visualiser in scene.
    ``y_pos``              : y (depth) position.
    ``z_scale``            : vertical (z) scale factor for the PDF.
    """

    curve_color: ManimColor = ManimColor("#4A90D9")
    curve_stroke_width: float = 3.0
    curve_opacity: float = 1.0
    glow_width: float = 9.0
    glow_opacity: float = 0.14
    smooth_resolution: int = 20

    fill_color: Optional[ManimColor] = None     # None → inherit curve_color
    fill_opacity: float = 0.26
    fill_gradient: bool = True
    n_gradient_strips: int = 12
    floor_z: float = 0.0

    show_mean_line: bool = True
    show_sigma_brackets: bool = True
    mean_color: ManimColor = ManimColor("#E0AA40")
    sigma_color: ManimColor = ManimColor("#E0AA40")
    moment_stroke_width: float = 1.8

    region_colors: List[ManimColor] = field(default_factory=lambda: [
        ManimColor("#E8593C"),  # red — tails / rejection
        ManimColor("#2DAA6E"),  # green — acceptance
        ManimColor("#9B59B6"),  # purple — third region
        ManimColor("#1ABC9C"),  # teal — fourth
    ])
    region_opacity: float = 0.32
    region_label_font: int = 20
    region_label_decimals: int = 4

    percentile_color: ManimColor = ManimColor("#9B59B6")
    percentile_font: int = 17

    show_title: bool = True
    title_font: int = 22
    show_param_label: bool = True
    param_label_font: int = 18

    x_pos: float = 0.0
    y_pos: float = 0.0
    z_scale: float = 3.5


# ── Presets ──────────────────────────────────────────────────────────────

MINIMAL_PDF = PDFConfig(
    glow_opacity=0.0,
    fill_gradient=False,
    fill_opacity=0.18,
    show_mean_line=False,
    show_sigma_brackets=False,
    show_title=False,
    show_param_label=False,
)

TEACHING_PDF = PDFConfig(
    glow_opacity=0.16,
    fill_gradient=True,
    fill_opacity=0.28,
    show_mean_line=True,
    show_sigma_brackets=True,
    show_title=True,
    show_param_label=True,
    smooth_resolution=24,
)

COMPARISON_PDF = PDFConfig(
    fill_opacity=0.14,
    fill_gradient=True,
    glow_opacity=0.10,
    show_mean_line=False,
    show_sigma_brackets=False,
    show_title=False,
    show_param_label=False,
)


# ===========================================================================
# PDFVisualizer3D
# ===========================================================================

class PDFVisualizer3D(VGroup):
    """Full-featured 3D PDF visualizer for a single distribution.

    Visual layers (all public VGroup attributes):
    - ``curve_group``   : glow + stroke VMobject.
    - ``fill_group``    : gradient area fill strips.
    - ``moments_group`` : mean line + σ bracket annotations.
    - ``regions``       : all shaded probability regions + labels.
    - ``percentiles``   : all percentile marker lines + labels.
    - ``kde_group``     : empirical KDE curve (if added).
    - ``title_group``   : distribution name + parameter label.

    Parameters
    ----------
    distribution : PDFDistribution
        The distribution to visualize.
    x_range : (float, float, float) or None
        ``(x_min, x_max, x_step)`` for curve sampling.  If None, uses
        ``distribution.default_x_range`` with a step of
        ``(x_max - x_min) / 200``.
    config : PDFConfig
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        distribution: PDFDistribution,
        x_range: Optional[Tuple[float, float, float]] = None,
        config: Optional[PDFConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dist = distribution
        self.cfg = config if config is not None else PDFConfig()
        self._scene = scene

        # Resolve x_range
        if x_range is not None:
            self._x_min, self._x_max, self._x_step = x_range
        else:
            lo, hi = distribution.default_x_range
            step = (hi - lo) / 200
            self._x_min, self._x_max, self._x_step = lo, hi, step

        # Sample the PDF
        xs = np.arange(self._x_min, self._x_max + self._x_step * 0.5, self._x_step)
        ys_raw = np.array([max(float(distribution.pdf(x)), 0.0) for x in xs])
        y_max = ys_raw.max() if ys_raw.max() > 0 else 1.0
        # Scale so peak reaches z_scale
        self._y_max_raw = y_max
        ys = ys_raw / y_max * self.cfg.z_scale

        self._xs = xs
        self._ys = ys
        self._ys_raw = ys_raw

        # Convert to 3D scene coordinates
        cfg = self.cfg
        self._curve_pts: np.ndarray = np.column_stack([
            xs + cfg.x_pos,
            np.full(len(xs), cfg.y_pos),
            ys,
        ])
        self._smoothed_pts: np.ndarray = _catmull_rom(
            self._curve_pts, resolution=cfg.smooth_resolution
        )

        # Public VGroup layers
        self.curve_group   = VGroup()
        self.fill_group    = VGroup()
        self.moments_group = VGroup()
        self.regions       = VGroup()
        self.percentiles   = VGroup()
        self.kde_group     = VGroup()
        self.title_group   = VGroup()

        # Internal state
        self._region_count: int = 0
        self._percentile_count: int = 0

        # Build all layers
        self._build_curve()
        self._build_fill()
        if cfg.show_mean_line or cfg.show_sigma_brackets:
            self._build_moments(scene)
        if cfg.show_title:
            self._build_title(scene)

        self.add(
            self.fill_group, self.curve_group,
            self.moments_group, self.regions,
            self.percentiles, self.kde_group, self.title_group,
        )

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_curve(self) -> None:
        cfg = self.cfg
        col = _with_opacity(cfg.curve_color, cfg.curve_opacity)

        stroke = VMobject()
        stroke.set_points_as_corners(self._smoothed_pts)
        stroke.set_stroke(color=col, width=cfg.curve_stroke_width)
        stroke.set_fill(opacity=0)
        self.stroke = stroke

        if cfg.glow_width > 0 and cfg.glow_opacity > 0:
            glow = VMobject()
            glow.set_points_as_corners(self._smoothed_pts)
            glow.set_stroke(
                color=_with_opacity(cfg.curve_color, cfg.glow_opacity),
                width=cfg.glow_width,
            )
            glow.set_fill(opacity=0)
            self.glow = glow
            self.curve_group.add(glow)

        self.curve_group.add(stroke)

    def _build_fill(self) -> None:
        cfg = self.cfg
        fill_col = cfg.fill_color if cfg.fill_color is not None else cfg.curve_color
        fill_grp = _build_fill_polygon(
            self._smoothed_pts,
            floor_z=cfg.floor_z,
            y_pos=cfg.y_pos,
            n_gradient_strips=cfg.n_gradient_strips,
            color=fill_col,
            opacity=cfg.fill_opacity,
            gradient=cfg.fill_gradient,
        )
        self.fill_group.add(fill_grp)

    def _build_moments(self, scene: Optional[ThreeDScene]) -> None:
        cfg = self.cfg
        mu = self.dist.mean
        sig = self.dist.std
        y = cfg.y_pos
        floor_z = cfg.floor_z

        if mu is None:
            return

        x_mu = mu + cfg.x_pos
        z_mu_pdf = float(self.dist.pdf(mu)) / self._y_max_raw * cfg.z_scale

        # Mean line — dashed vertical from floor to PDF height
        mean_col = _with_opacity(cfg.mean_color, 0.85)
        mean_line = DashedLine(
            np.array([x_mu, y, floor_z]),
            np.array([x_mu, y, z_mu_pdf * 1.05]),
            dash_length=0.07, dashed_ratio=0.45,
            color=mean_col, stroke_width=cfg.moment_stroke_width,
        )
        mean_dot = Dot3D(
            point=np.array([x_mu, y, floor_z]),
            radius=0.06, color=mean_col,
        )

        mu_lbl = Text(f"μ = {mu:.3g}", font_size=17, color=cfg.mean_color)
        mu_lbl.move_to(np.array([x_mu + 0.12, y, z_mu_pdf + 0.28]))
        if scene is not None:
            scene.add_fixed_orientation_mobjects(mu_lbl)

        self.mean_line = mean_line
        self.mean_dot = mean_dot
        self.mean_label = VGroup(mu_lbl)
        self.moments_group.add(mean_line, mean_dot, mu_lbl)

        # Sigma brackets (±1σ and ±2σ)
        if cfg.show_sigma_brackets and sig is not None:
            sig_col = _with_opacity(cfg.sigma_color, 0.70)
            for n_sig, label_text in [(1, "±1σ"), (2, "±2σ")]:
                lo_x = (mu - n_sig * sig) + cfg.x_pos
                hi_x = (mu + n_sig * sig) + cfg.x_pos
                z_bracket = floor_z - 0.12 * n_sig

                bracket = VGroup(
                    Line(np.array([lo_x, y, z_bracket]),
                         np.array([hi_x, y, z_bracket]),
                         color=sig_col,
                         stroke_width=cfg.moment_stroke_width * 0.8),
                    Line(np.array([lo_x, y, z_bracket]),
                         np.array([lo_x, y, z_bracket + 0.08]),
                         color=sig_col,
                         stroke_width=cfg.moment_stroke_width * 0.8),
                    Line(np.array([hi_x, y, z_bracket]),
                         np.array([hi_x, y, z_bracket + 0.08]),
                         color=sig_col,
                         stroke_width=cfg.moment_stroke_width * 0.8),
                )
                lbl = Text(label_text, font_size=14, color=cfg.sigma_color)
                lbl.move_to(np.array([(lo_x + hi_x) / 2, y, z_bracket - 0.22]))
                if scene is not None:
                    scene.add_fixed_orientation_mobjects(lbl)
                self.moments_group.add(bracket, lbl)

    def _build_title(self, scene: Optional[ThreeDScene]) -> None:
        cfg = self.cfg
        x_cen = (self._x_min + self._x_max) / 2 + cfg.x_pos
        z_top = cfg.z_scale * 1.12

        title_lbl = Text(
            self.dist.name,
            font_size=cfg.title_font,
            color=_with_opacity(cfg.curve_color, 0.88),
        )
        title_lbl.move_to(np.array([x_cen, cfg.y_pos, z_top]))
        self.title_group.add(title_lbl)

        if cfg.show_param_label:
            param_lbl = Text(
                self.dist.param_string,
                font_size=cfg.param_label_font,
                color=_with_opacity(cfg.curve_color, 0.60),
            )
            param_lbl.move_to(np.array([x_cen, cfg.y_pos, z_top - 0.38]))
            self.title_group.add(param_lbl)

        for mob in self.title_group:
            if scene is not None:
                scene.add_fixed_orientation_mobjects(mob)

    # ------------------------------------------------------------------
    # Public layer API
    # ------------------------------------------------------------------

    def shade_region(
        self,
        x_lo: float,
        x_hi: float,
        color: Optional[ManimColor] = None,
        label: bool = True,
        label_fmt: str = "P = {p:.{d}f}",
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Shade the PDF region between *x_lo* and *x_hi*.

        Computes P(x_lo ≤ X ≤ x_hi) by numerical integration and adds
        an optional probability label above the shaded region.

        Parameters
        ----------
        x_lo, x_hi : float
            Region boundaries in data coordinates.
        color : ManimColor or None
            Uses next colour from ``cfg.region_colors`` if None.
        label : bool
            Whether to add a probability annotation.
        label_fmt : str
            Format string with ``{p}`` (probability) and ``{d}`` (decimals).
        scene : ThreeDScene or None

        Returns
        -------
        VGroup — the region fill + label (added to ``self.regions``).
        """
        cfg = self.cfg
        col = (color if color is not None
               else cfg.region_colors[self._region_count % len(cfg.region_colors)])
        self._region_count += 1

        # Clip to visible x range
        lo = max(x_lo, self._x_min)
        hi = min(x_hi, self._x_max)

        # Find curve points within [lo, hi]
        mask = (self._xs >= lo) & (self._xs <= hi)
        xs_r = self._xs[mask] + cfg.x_pos
        ys_r = self._ys[mask]

        # Add boundary interpolations
        y_lo = float(self.dist.pdf(lo)) / self._y_max_raw * cfg.z_scale
        y_hi = float(self.dist.pdf(hi)) / self._y_max_raw * cfg.z_scale
        xs_full = np.concatenate([[lo + cfg.x_pos], xs_r, [hi + cfg.x_pos]])
        ys_full = np.concatenate([[y_lo], ys_r, [y_hi]])

        region_curve_pts = np.column_stack([
            xs_full, np.full(len(xs_full), cfg.y_pos), ys_full
        ])

        fill = _build_fill_polygon(
            region_curve_pts,
            floor_z=cfg.floor_z,
            y_pos=cfg.y_pos,
            n_gradient_strips=1,
            color=col,
            opacity=cfg.region_opacity,
            gradient=False,
        )

        grp = VGroup(fill)

        # Probability label
        if label:
            p = self.dist.cdf(hi) - self.dist.cdf(lo)
            lbl_text = label_fmt.format(p=p, d=cfg.region_label_decimals)
            lbl = Text(lbl_text, font_size=cfg.region_label_font, color=col)
            x_mid = (lo + hi) / 2 + cfg.x_pos
            z_mid = float(self.dist.pdf((lo + hi) / 2)) / self._y_max_raw * cfg.z_scale
            lbl.move_to(np.array([x_mid, cfg.y_pos, z_mid + 0.38]))
            grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        self.regions.add(grp)
        return grp

    def shade_tail_left(
        self,
        x_crit: float,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Shade P(X ≤ x_crit) — the left tail."""
        return self.shade_region(
            self._x_min, x_crit,
            color=color, label=label, scene=scene,
        )

    def shade_tail_right(
        self,
        x_crit: float,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Shade P(X ≥ x_crit) — the right tail."""
        return self.shade_region(
            x_crit, self._x_max,
            color=color, label=label, scene=scene,
        )

    def shade_two_tails(
        self,
        alpha: float = 0.05,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Shade both tails for a two-sided test at level *alpha*.

        Returns a VGroup containing both tail fills.
        """
        x_lo = self.dist.ppf(alpha / 2)
        x_hi = self.dist.ppf(1 - alpha / 2)
        g_left = self.shade_tail_left(x_lo, label=label, color=color, scene=scene)
        g_right = self.shade_tail_right(x_hi, label=label, color=color, scene=scene)
        return VGroup(g_left, g_right)

    def shade_between_sigmas(
        self,
        n_sigma: float = 1.0,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Shade the region μ ± n_sigma·σ.

        Useful for showing the 68–95–99.7 rule.
        """
        mu = self.dist.mean
        sig = self.dist.std
        if mu is None or sig is None:
            return VGroup()
        return self.shade_region(
            mu - n_sigma * sig, mu + n_sigma * sig,
            color=color, label=label,
            label_fmt=f"P(μ ± {n_sigma:.0f}σ) = {{p:.{self.cfg.region_label_decimals}f}}",
            scene=scene,
        )

    def add_percentile_marker(
        self,
        p: float,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Add a vertical marker line at the *p*-th percentile.

        Parameters
        ----------
        p : float
            Probability in (0, 1).
        label : bool
            If True, show a "{100p}th percentile" label.

        Returns
        -------
        VGroup — the marker line + label (added to ``self.percentiles``).
        """
        cfg = self.cfg
        col = color if color is not None else cfg.percentile_color
        x_p = self.dist.ppf(p) + cfg.x_pos
        z_top = float(self.dist.pdf(x_p - cfg.x_pos)) / self._y_max_raw * cfg.z_scale

        marker = DashedLine(
            np.array([x_p, cfg.y_pos, cfg.floor_z]),
            np.array([x_p, cfg.y_pos, z_top * 1.05]),
            dash_length=0.07, dashed_ratio=0.5,
            color=_with_opacity(col, 0.80),
            stroke_width=1.8,
        )
        dot = Dot3D(
            point=np.array([x_p, cfg.y_pos, cfg.floor_z]),
            radius=0.06, color=_with_opacity(col, 0.90),
        )
        grp = VGroup(marker, dot)

        if label:
            pct = int(round(p * 100))
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(pct % 10 if pct % 100 not in (11, 12, 13) else 0, "th")
            lbl = Text(
                f"{pct}{suffix}",
                font_size=cfg.percentile_font,
                color=col,
            )
            lbl.move_to(np.array([x_p, cfg.y_pos, -0.28]))
            grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        self.percentiles.add(grp)
        self._percentile_count += 1
        return grp

    def add_percentile_markers(
        self,
        ps: List[float],
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Add multiple percentile markers at once.

        Returns a VGroup of all added marker VGroups.
        """
        all_grps = VGroup()
        for p in ps:
            all_grps.add(self.add_percentile_marker(p, scene=scene))
        return all_grps

    def add_kde_overlay(
        self,
        data: np.ndarray,
        bandwidth: Optional[float] = None,
        color: Optional[ManimColor] = None,
        stroke_width: float = 2.2,
        scene: Optional[ThreeDScene] = None,
    ) -> VMobject:
        """Draw an empirical KDE curve over the theoretical PDF.

        Uses Silverman's rule-of-thumb bandwidth if *bandwidth* is None.
        The KDE is scaled to match the theoretical PDF's peak height.

        Parameters
        ----------
        data : np.ndarray
            Observed data values.
        bandwidth : float or None
        color : ManimColor or None
            Defaults to a lighter version of the curve colour.

        Returns
        -------
        VMobject — the KDE stroke (added to ``self.kde_group``).
        """
        cfg = self.cfg
        n = len(data)
        std = float(np.std(data))
        bw = bandwidth if bandwidth is not None else max(1.06 * std * n**(-0.2), 1e-3)
        col = color if color is not None else _lighten(cfg.curve_color, 1.45)

        xs = np.linspace(self._x_min, self._x_max, 300)
        zs = np.array([
            float(np.mean(
                np.exp(-0.5 * ((x - data) / bw)**2) / (bw * np.sqrt(TAU))
            ))
            for x in xs
        ])
        z_max_kde = zs.max() if zs.max() > 0 else 1.0
        zs_scaled = zs / z_max_kde * cfg.z_scale

        pts = np.column_stack([xs + cfg.x_pos, np.full(len(xs), cfg.y_pos), zs_scaled])
        smoothed = _catmull_rom(pts, resolution=12)

        kde_stroke = VMobject()
        kde_stroke.set_points_as_corners(smoothed)
        kde_stroke.set_stroke(color=_with_opacity(col, 0.80), width=stroke_width)
        kde_stroke.set_fill(opacity=0)

        # KDE glow
        kde_glow = VMobject()
        kde_glow.set_points_as_corners(smoothed)
        kde_glow.set_stroke(color=_with_opacity(col, 0.12), width=stroke_width * 2.5)
        kde_glow.set_fill(opacity=0)

        self.kde_group.add(kde_glow, kde_stroke)
        return kde_stroke

    def add_critical_value(
        self,
        x_crit: float,
        label: str = "",
        color: ManimColor = ManimColor("#E8593C"),
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Add a critical-value annotation line at *x_crit*.

        Combines a dashed vertical line from the floor to above the curve
        with an optional text label.  Returns the VGroup for animation.
        """
        cfg = self.cfg
        sc = x_crit + cfg.x_pos
        z_pdf = float(self.dist.pdf(x_crit)) / self._y_max_raw * cfg.z_scale
        col = _with_opacity(color, 0.82)

        ln = DashedLine(
            np.array([sc, cfg.y_pos, cfg.floor_z]),
            np.array([sc, cfg.y_pos, cfg.z_scale * 1.08]),
            dash_length=0.08, dashed_ratio=0.45,
            color=col, stroke_width=2.0,
        )
        dot = Dot3D(
            point=np.array([sc, cfg.y_pos, z_pdf]),
            radius=0.08, color=color,
        )
        grp = VGroup(ln, dot)

        if label:
            lbl = Text(label, font_size=19, color=color)
            lbl.move_to(np.array([sc, cfg.y_pos, cfg.z_scale * 1.18]))
            grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        self.add(grp)
        return grp

    # ------------------------------------------------------------------
    # Parameter morphing
    # ------------------------------------------------------------------

    def morph_to_distribution(
        self,
        new_dist: PDFDistribution,
        run_time: float = 1.5,
        update_title: bool = True,
        scene: Optional[ThreeDScene] = None,
    ) -> AnimationGroup:
        """Return an animation that morphs this visualiser to *new_dist*.

        Updates the stroke, glow, and fill to the new distribution's shape.
        Rebuilds the title annotation if *update_title* is True.

        Parameters
        ----------
        new_dist : PDFDistribution
            Target distribution.
        run_time : float
        update_title : bool
            If True, also transform the title and parameter labels.

        Returns
        -------
        AnimationGroup — play with ``scene.play(viz.morph_to_distribution(...))``.
        """
        cfg = self.cfg

        # Compute new curve points
        xs = np.arange(self._x_min, self._x_max + self._x_step * 0.5, self._x_step)
        ys_raw = np.array([max(float(new_dist.pdf(x)), 0.0) for x in xs])
        y_max_new = ys_raw.max() if ys_raw.max() > 0 else 1.0
        ys_new = ys_raw / y_max_new * cfg.z_scale

        new_curve_pts = np.column_stack([
            xs + cfg.x_pos,
            np.full(len(xs), cfg.y_pos),
            ys_new,
        ])
        new_smoothed = _catmull_rom(new_curve_pts, resolution=cfg.smooth_resolution)

        old_stroke_pts = np.array(self.stroke.get_all_points())
        if len(old_stroke_pts) == 0:
            old_stroke_pts = self._smoothed_pts

        # Resample both to the same length for clean interpolation
        n_interp = max(len(new_smoothed), len(old_stroke_pts))
        from_pts = _resample_equal(old_stroke_pts, n_interp)
        to_pts = _resample_equal(new_smoothed, n_interp)

        def stroke_updater(mob: VMobject, alpha: float) -> None:
            t = rate_functions.ease_in_out_cubic(alpha)
            mob.set_points_as_corners(from_pts + (to_pts - from_pts) * t)

        anims: List = [UpdateFromAlphaFunc(self.stroke, stroke_updater, run_time=run_time)]

        if hasattr(self, "glow"):
            glow_from = _resample_equal(old_stroke_pts, n_interp)

            def glow_updater(mob: VMobject, alpha: float) -> None:
                t = rate_functions.ease_in_out_cubic(alpha)
                mob.set_points_as_corners(glow_from + (to_pts - glow_from) * t)

            anims.append(UpdateFromAlphaFunc(self.glow, glow_updater, run_time=run_time))

        # Update fill (rebuild and cross-fade)
        new_fill_grp = _build_fill_polygon(
            new_smoothed,
            floor_z=cfg.floor_z,
            y_pos=cfg.y_pos,
            n_gradient_strips=cfg.n_gradient_strips,
            color=cfg.fill_color or cfg.curve_color,
            opacity=cfg.fill_opacity,
            gradient=cfg.fill_gradient,
        )
        anims.append(FadeOut(self.fill_group, run_time=run_time * 0.4))
        anims.append(FadeIn(new_fill_grp, run_time=run_time * 0.6))

        # Update stored state
        self._smoothed_pts = new_smoothed
        self.dist = new_dist
        self._y_max_raw = y_max_new
        self._ys = ys_new
        self._ys_raw = ys_raw

        return AnimationGroup(*anims)

    # ------------------------------------------------------------------
    # Animation helpers (layer-by-layer)
    # ------------------------------------------------------------------

    def animate_curve(self, run_time: float = 1.5) -> AnimationGroup:
        """Trace the PDF curve left-to-right."""
        anims = [Create(self.stroke, run_time=run_time)]
        if hasattr(self, "glow"):
            anims.append(FadeIn(self.glow, run_time=run_time * 0.3))
        return AnimationGroup(*anims, lag_ratio=0.0)

    def animate_fill(self, run_time: float = 1.0) -> UpdateFromAlphaFunc:
        """Grow the area fill upward from the floor."""
        cfg = self.cfg
        full_fill = self.fill_group
        smoothed = self._smoothed_pts
        floor_z = cfg.floor_z
        y_pos = cfg.y_pos
        fill_col = cfg.fill_color or cfg.curve_color

        def updater(mob: VGroup, alpha: float) -> None:
            t = smooth(alpha)
            interp_pts = np.column_stack([
                smoothed[:, 0],
                smoothed[:, 1],
                floor_z + (smoothed[:, 2] - floor_z) * t,
            ])
            mob.become(
                _build_fill_polygon(
                    interp_pts, floor_z=floor_z, y_pos=y_pos,
                    n_gradient_strips=cfg.n_gradient_strips,
                    color=fill_col, opacity=cfg.fill_opacity,
                    gradient=cfg.fill_gradient,
                )
            )

        return UpdateFromAlphaFunc(full_fill, updater, run_time=run_time)

    def animate_moments(self, run_time: float = 0.8) -> AnimationGroup:
        """Fade in all moment annotations."""
        return FadeIn(self.moments_group, run_time=run_time)

    def animate_title(self, run_time: float = 0.6) -> FadeIn:
        return FadeIn(self.title_group, run_time=run_time)

    def animate_percentiles(
        self,
        lag: float = 0.12,
        run_time_per: float = 0.4,
    ) -> LaggedStart:
        """Reveal percentile markers one by one."""
        return LaggedStart(
            *[Create(m, run_time=run_time_per) for m in self.percentiles],
            lag_ratio=lag,
        )

    def animate_regions(
        self,
        lag: float = 0.10,
        run_time_per: float = 0.45,
    ) -> LaggedStart:
        """Fade in probability regions one by one."""
        return LaggedStart(
            *[FadeIn(r, run_time=run_time_per) for r in self.regions],
            lag_ratio=lag,
        )

    def full_reveal(
        self,
        scene: ThreeDScene,
        curve_rt: float = 1.5,
        fill_rt: float = 1.0,
        moments_rt: float = 0.8,
        title_rt: float = 0.6,
    ) -> None:
        """Play the complete layer-by-layer reveal directly on *scene*."""
        scene.play(self.animate_curve(run_time=curve_rt))
        scene.play(self.animate_fill(run_time=fill_rt))
        if len(self.moments_group) > 0:
            scene.play(self.animate_moments(run_time=moments_rt))
        if len(self.title_group) > 0:
            scene.play(self.animate_title(run_time=title_rt))


def _resample_equal(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample *pts* to *n* points via arc-length parameterisation."""
    if len(pts) == n:
        return pts.copy()
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) > 0 else np.zeros(3), (n, 1))
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    total = cum[-1]
    if total < 1e-9:
        return np.tile(pts[0], (n, 1))
    ts = np.linspace(0.0, total, n)
    out = np.zeros((n, 3))
    for i, t in enumerate(ts):
        idx = int(np.searchsorted(cum, t, side="right")) - 1
        idx = max(0, min(idx, len(pts) - 2))
        seg = deltas[idx] if deltas[idx] > 1e-12 else 1e-12
        frac = (t - cum[idx]) / seg
        out[i] = pts[idx] + frac * (pts[idx + 1] - pts[idx])
    return out


# ===========================================================================
# MultiplePDFComparison3D
# ===========================================================================

class MultiplePDFComparison3D(VGroup):
    """Overlay multiple PDFVisualizer3D curves on the same axes.

    Manages a list of ``PDFVisualizer3D`` objects with compatible x-ranges,
    staggering their fills and rendering them at different y-depths
    (``depth_stagger > 0``) or overlaid at the same depth.

    Parameters
    ----------
    distributions : list of PDFDistribution
        Distributions to compare.
    x_range : (float, float, float) or None
        Shared x-range.  If None each distribution uses its own default.
    colors : list of ManimColor or None
    configs : list of PDFConfig or None
        Per-distribution configs.  If None, ``COMPARISON_PDF`` is used
        with the corresponding colour.
    depth_stagger : float
        Y-offset between successive curves (0 = all overlaid).
    show_divergence : bool
        If True and there are exactly two distributions, shade the
        region between the two curves.
    show_legend : bool
        Whether to add a floating legend.
    scene : ThreeDScene or None

    Attributes
    ----------
    visualizers : list of PDFVisualizer3D
    divergence_group : VGroup
        Shaded region between two overlaid PDFs (if ``show_divergence``).
    legend : VGroup
    """

    _DEFAULT_COLORS: List[ManimColor] = [
        ManimColor("#4A90D9"),
        ManimColor("#E8593C"),
        ManimColor("#2DAA6E"),
        ManimColor("#E0AA40"),
        ManimColor("#9B59B6"),
        ManimColor("#1ABC9C"),
    ]

    def __init__(
        self,
        distributions: List[PDFDistribution],
        x_range: Optional[Tuple[float, float, float]] = None,
        colors: Optional[List[ManimColor]] = None,
        configs: Optional[List[PDFConfig]] = None,
        depth_stagger: float = 0.0,
        show_divergence: bool = False,
        show_legend: bool = True,
        x_center: float = 0.0,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        n = len(distributions)
        self._colors = colors or [self._DEFAULT_COLORS[i % len(self._DEFAULT_COLORS)] for i in range(n)]

        # Resolve shared x_range
        if x_range is None:
            all_lo = [d.default_x_range[0] for d in distributions]
            all_hi = [d.default_x_range[1] for d in distributions]
            lo = min(all_lo)
            hi = max(all_hi)
            step = (hi - lo) / 200
            x_range = (lo, hi, step)

        self.visualizers: List[PDFVisualizer3D] = []

        for i, (dist, col) in enumerate(zip(distributions, self._colors)):
            if configs is not None and i < len(configs):
                cfg = configs[i]
            else:
                cfg = PDFConfig(**COMPARISON_PDF.__dict__)
                cfg.curve_color = col
                cfg.fill_color = col

            cfg.y_pos = i * depth_stagger
            cfg.x_pos = x_center

            viz = PDFVisualizer3D(
                dist,
                x_range=x_range,
                config=cfg,
                scene=scene,
            )
            self.visualizers.append(viz)
            self.add(viz)

        # Divergence shading (two curves only)
        self.divergence_group = VGroup()
        if show_divergence and len(distributions) == 2:
            self._build_divergence(x_range)
        self.add(self.divergence_group)

        # Legend
        self.legend = VGroup()
        if show_legend:
            self._build_legend(distributions, scene)
        self.add(self.legend)

    # ------------------------------------------------------------------

    def _build_divergence(self, x_range: Tuple[float, float, float]) -> None:
        """Shade the area between two PDFs (KL-divergence visualization)."""
        v0, v1 = self.visualizers[0], self.visualizers[1]
        cfg0, cfg1 = v0.cfg, v1.cfg
        xs = np.linspace(x_range[0], x_range[1], 400)

        z0 = np.array([max(float(v0.dist.pdf(x)), 0.0) / v0._y_max_raw * cfg0.z_scale for x in xs])
        z1 = np.array([max(float(v1.dist.pdf(x)), 0.0) / v1._y_max_raw * cfg1.z_scale for x in xs])

        # Upper envelope minus lower: two regions (dist0 > dist1, dist0 < dist1)
        for (upper_z, lower_z, col) in [
            (np.maximum(z0, z1), np.minimum(z0, z1), ManimColor("#AAAAAA")),
        ]:
            # Build alternating strips based on which curve is higher
            for i in range(len(xs) - 1):
                z_u = (upper_z[i] + upper_z[i+1]) / 2
                z_l = (lower_z[i] + lower_z[i+1]) / 2
                if abs(z_u - z_l) < 0.01:
                    continue
                x_l, x_r = xs[i] + cfg0.x_pos, xs[i+1] + cfg0.x_pos
                y = (cfg0.y_pos + cfg1.y_pos) / 2
                strip_col = self._colors[0] if z0[i] > z1[i] else self._colors[1]
                strip = Polygon(
                    np.array([x_l, y, z_l]),
                    np.array([x_r, y, z_l]),
                    np.array([x_r, y, z_u]),
                    np.array([x_l, y, z_u]),
                    fill_color=_with_opacity(strip_col, 0.14),
                    fill_opacity=1.0, stroke_width=0,
                )
                self.divergence_group.add(strip)

    def _build_legend(
        self,
        distributions: List[PDFDistribution],
        scene: Optional[ThreeDScene],
    ) -> None:
        """Build a floating legend listing distribution names and colours."""
        for i, (dist, col) in enumerate(zip(distributions, self._colors)):
            swatch = Line(
                ORIGIN, RIGHT * 0.38,
                color=_with_opacity(col, 0.90),
                stroke_width=3.0,
            )
            lbl = Text(
                f"{dist.name}  ({dist.param_string})",
                font_size=16,
                color=col,
            )
            row = VGroup(swatch, lbl)
            row.arrange(RIGHT, buff=0.12)
            row.move_to(np.array([5.5, 0, 3.5 - i * 0.35]))
            self.legend.add(row)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(row)

    # ------------------------------------------------------------------

    def animate_reveal(
        self,
        stagger: float = 0.3,
        run_time_per: float = 1.2,
    ) -> LaggedStart:
        """Reveal all curves with a staggered draw animation."""
        return LaggedStart(
            *[viz.animate_curve(run_time=run_time_per) for viz in self.visualizers],
            lag_ratio=stagger / run_time_per,
        )

    def animate_fills(
        self,
        stagger: float = 0.15,
        run_time_per: float = 0.9,
    ) -> LaggedStart:
        """Grow all fills with a stagger."""
        return LaggedStart(
            *[viz.animate_fill(run_time=run_time_per) for viz in self.visualizers],
            lag_ratio=stagger / run_time_per,
        )

    def highlight_distribution(
        self,
        index: int,
        dim_others: bool = True,
        dim_opacity: float = 0.20,
    ) -> AnimationGroup:
        """Emphasise one distribution, dimming the others."""
        anims = []
        for i, viz in enumerate(self.visualizers):
            if i == index:
                anims.append(viz.stroke.animate.set_stroke(
                    width=viz.cfg.curve_stroke_width * 1.8
                ))
            elif dim_others:
                anims.append(viz.animate.set_opacity(dim_opacity))
        return AnimationGroup(*anims, run_time=0.5)

    def morph_parameter(
        self,
        dist_index: int,
        new_distribution: PDFDistribution,
        run_time: float = 1.5,
        scene: Optional[ThreeDScene] = None,
    ) -> AnimationGroup:
        """Morph one distribution in the comparison to *new_distribution*."""
        return self.visualizers[dist_index].morph_to_distribution(
            new_distribution, run_time=run_time, scene=scene
        )


# ===========================================================================
# Ready-to-render ThreeDScene subclasses
# ===========================================================================

class NormalPDFScene(ThreeDScene):
    """Normal PDF with σ-brackets, 68–95–99.7 rule, and z-score annotations.

    Render:  manim -pql pdf_viz.py NormalPDFScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        dist = NormalDist(mu=0.0, sigma=1.0)
        viz = PDFVisualizer3D(
            dist,
            x_range=(-4.0, 4.0, 0.04),
            config=TEACHING_PDF,
            scene=self,
        )
        self.add(viz)

        viz.full_reveal(self)
        self.wait(0.5)

        # Shade 68-95-99.7
        for n_sig, col in [
            (1, ManimColor("#2DAA6E")),
            (2, ManimColor("#E0AA40")),
            (3, ManimColor("#E8593C")),
        ]:
            region = viz.shade_between_sigmas(n_sig, color=col, scene=self)
            self.play(FadeIn(region, run_time=0.55))
            self.wait(0.4)

        self.wait(2)


class TDistScene(ThreeDScene):
    """t-distribution for df = 1, 3, 10, 30 compared to Normal.

    Render:  manim -pql pdf_viz.py TDistScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.02)

        normal = NormalDist(0, 1)
        dists = [StudentTDist(df=df) for df in [1, 3, 10, 30]]
        colors = [
            ManimColor("#E8593C"),
            ManimColor("#E0AA40"),
            ManimColor("#2DAA6E"),
            ManimColor("#9B59B6"),
        ]

        comp = MultiplePDFComparison3D(
            distributions=[normal] + dists,
            x_range=(-5.0, 5.0, 0.05),
            colors=[ManimColor("#4A90D9")] + colors,
            depth_stagger=0.0,
            show_divergence=False,
            show_legend=True,
            scene=self,
        )
        self.add(comp)
        self.play(comp.animate_reveal(stagger=0.35, run_time_per=1.2))
        self.play(comp.animate_fills(stagger=0.12, run_time_per=0.8))
        self.wait(3)


class GammaPDFScene(ThreeDScene):
    """Gamma distribution sweeping shape k from 1 → 5, rate λ = 1.

    Render:  manim -pql pdf_viz.py GammaPDFScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        dist = GammaDist(shape=1.0, rate=1.0)
        cfg = PDFConfig(**TEACHING_PDF.__dict__)
        cfg.curve_color = ManimColor("#E8593C")
        cfg.fill_color = ManimColor("#E8593C")
        cfg.show_sigma_brackets = False

        viz = PDFVisualizer3D(dist, x_range=(0.0, 12.0, 0.06), config=cfg, scene=self)
        self.add(viz)
        viz.full_reveal(self, curve_rt=1.2, fill_rt=0.9, moments_rt=0.5, title_rt=0.4)
        self.wait(0.5)

        for k in [2.0, 3.0, 4.0, 5.0]:
            new_dist = GammaDist(shape=k, rate=1.0)
            self.play(
                viz.morph_to_distribution(new_dist, run_time=1.0, scene=self),
                run_time=1.0,
            )
            self.wait(0.4)

        self.wait(2)


class PDFComparisonScene(ThreeDScene):
    """Side-by-side PDFs: Normal, t(5), Laplace, Cauchy.

    All zero-centred, similar scale, depth-staggered for depth cues.

    Render:  manim -pql pdf_viz.py PDFComparisonScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        dists = [
            NormalDist(0, 1),
            StudentTDist(df=5),
            LaplaceDist(mu=0, b=0.707),   # same variance as N(0,1)
            CauchyDist(x0=0, gamma=0.5),
        ]
        comp = MultiplePDFComparison3D(
            distributions=dists,
            x_range=(-5.0, 5.0, 0.05),
            depth_stagger=0.6,
            show_legend=True,
            scene=self,
        )
        self.add(comp)
        self.play(comp.animate_reveal(stagger=0.30, run_time_per=1.2))
        self.play(comp.animate_fills(stagger=0.12, run_time_per=0.8))
        self.wait(0.5)

        # Highlight heavy-tailed distributions
        self.play(comp.highlight_distribution(3, dim_others=True))
        self.wait(2)
        self.play(AnimationGroup(*[
            v.animate.set_opacity(1.0) for v in comp.visualizers
        ]))
        self.wait(2)


class PDFParameterSweepScene(ThreeDScene):
    """Beta(α, β) PDF: sweep α from 0.5 → 5 while β = 2 stays fixed.

    Render:  manim -pql pdf_viz.py PDFParameterSweepScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.03)

        dist = BetaDist(alpha=0.5, beta=2.0)
        cfg = PDFConfig(**TEACHING_PDF.__dict__)
        cfg.curve_color = ManimColor("#9B59B6")
        cfg.fill_color = ManimColor("#9B59B6")
        cfg.show_sigma_brackets = False

        viz = PDFVisualizer3D(dist, config=cfg, scene=self)
        self.add(viz)
        viz.full_reveal(self, curve_rt=1.0, fill_rt=0.8, moments_rt=0.4, title_rt=0.3)
        self.wait(0.5)

        alpha_values = np.arange(1.0, 5.5, 0.5)
        for alpha in alpha_values:
            new_dist = BetaDist(alpha=alpha, beta=2.0)
            self.play(
                viz.morph_to_distribution(new_dist, run_time=0.65, scene=self),
            )
            self.wait(0.2)

        self.wait(1)

        # Shade the mode region
        mode = (dist.alpha - 1) / (dist.alpha + dist.beta - 2) if dist.alpha > 1 else 0.0
        viz.add_critical_value(mode, label="mode", scene=self)
        self.play(FadeIn(viz[-1], run_time=0.5))
        self.wait(2)