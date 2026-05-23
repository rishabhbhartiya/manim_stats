"""
manim_stats/distributions/pmf_viz.py
======================================
Production-quality probability mass function (PMF) visualizer.

Design philosophy
-----------------
Discrete distributions require a fundamentally different visual treatment
from continuous ones.  A PMF is a collection of *point masses*, not an
area under a curve, so the primary geometry is **3D prism bars** — one
bar per support value k — rather than a smooth stroke.

``PMFVisualizer3D`` is built from independent layers, each a public
VGroup attribute that can be animated separately:

    bars_group      – one 3-faced prism per mass point, height ∝ P(X=k).
    spine           – thin baseline connecting all bar feet.
    mean_marker     – vertical marker + dot at the expected value.
    mode_marker     – marker at the mode (tallest bar).
    regions         – highlighted bars (different colour) for P(a≤X≤b).
    cdf_overlay     – step-function CDF drawn above the PMF bars.
    title_group     – distribution name + parameter label.
    k_labels        – integer tick labels below each bar.
    prob_labels     – P(X=k) floating above each bar (optional).

Key differences from ``PDFVisualizer3D``
-----------------------------------------
- No smoothing.  Each bar is independent; parameter morphing reshapes
  individual bars rather than warping a continuous curve.
- Mode ≠ mean in general (Geometric, Poisson at boundary, etc.).
  Both are annotated separately.
- ``shade_region`` recolours whole bars rather than filling a polygon.
- CDF is a step function, not a smooth curve.
- Parameter morphing uses per-bar ``UpdateFromAlphaFunc`` so bar heights
  interpolate independently with no visual coupling.

Distribution support
--------------------
All PMFs are implemented from closed-form combinatorial expressions using
the Lanczos log-gamma approximation.  No scipy dependency.

    BernoulliDist(p)                    support {0, 1}
    BinomialDist(n, p)                  support {0, …, n}
    GeometricDist(p)                    support {1, 2, …} (trials-until-success)
    NegBinomialDist(r, p)               support {r, r+1, …}
    PoissonDist(lam)                    support {0, 1, 2, …}
    HypergeometricDist(N, K, n)         support {max(0,n+K-N), …, min(n,K)}
    DiscreteUniformDist(a, b)           support {a, …, b}
    MultinomialDist(n, probs)           marginal bars for each category

Classes
-------
PMFConfig
PMFVisualizer3D
MultiplePMFComparison3D
PMFDistribution            (abstract base)
BernoulliDist
BinomialDist
GeometricDist
NegBinomialDist
PoissonDist
HypergeometricDist
DiscreteUniformDist

Ready-to-render scenes
----------------------
BinomialPMFScene
PoissonPMFScene
GeometricPMFScene
PMFComparisonScene
BinomialNSweepScene

Usage
-----
    from manim import *
    from manim_stats.distributions.pmf_viz import PMFVisualizer3D, BinomialDist

    class MyScene(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65*DEGREES, theta=-50*DEGREES)
            viz = PMFVisualizer3D(BinomialDist(n=10, p=0.4))
            self.play(viz.animate_bars())
            self.play(viz.animate_mean_mode())
            viz.shade_tail_right(k_crit=8)
            self.play(FadeIn(viz.regions))
            self.wait()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Union, Dict, Sequence
from abc import ABC, abstractmethod
import math
import numpy as np

from manim import (
    VGroup, VMobject, Polygon, Line, DashedLine, Dot3D,
    Text, MathTex,
    ThreeDScene,
    AnimationGroup, LaggedStart, Succession,
    Create, FadeIn, FadeOut, Transform,
    UpdateFromAlphaFunc, Flash, GrowFromEdge,
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
# Special function — log-gamma (self-contained, matches pdf_viz.py)
# ---------------------------------------------------------------------------

def _lgamma(x: float) -> float:
    """Natural log of the gamma function via Lanczos approximation."""
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if x < 0.5:
        return math.log(PI / math.sin(PI * x)) - _lgamma(1 - x)
    x -= 1
    a = c[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += c[i] / (x + i)
    return 0.5 * math.log(TAU) + (x + 0.5) * math.log(t) - t + math.log(a)


def _log_comb(n: int, k: int) -> float:
    """log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)."""
    if k < 0 or k > n:
        return float("-inf")
    return _lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1)


# ===========================================================================
# Abstract base
# ===========================================================================

class PMFDistribution(ABC):
    """Abstract base class for all discrete distributions.

    Subclasses implement ``pmf(k)``, ``cdf(k)``, ``support``, ``mean``,
    ``variance``, ``name``, and ``param_string``.

    The ``support`` property returns the finite or truncated integer range
    that will be displayed.  For infinite-support distributions (Geometric,
    Poisson, NegBinomial) a practical truncation at ``P(X > k) < 1e-4`` is
    used.
    """

    @abstractmethod
    def pmf(self, k: int) -> float:
        """P(X = k)."""

    def cdf(self, k: int) -> float:
        """P(X ≤ k) — default: sum pmf from support[0] to k."""
        lo = self.support[0]
        return sum(self.pmf(j) for j in range(lo, k + 1))

    @property
    @abstractmethod
    def support(self) -> List[int]:
        """Ordered list of integer support values to display."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def param_string(self) -> str: ...

    @property
    @abstractmethod
    def mean(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def variance(self) -> Optional[float]: ...

    @property
    def std(self) -> Optional[float]:
        v = self.variance
        return math.sqrt(v) if v is not None and v >= 0 else None

    def mode(self) -> int:
        """Return the k in support that maximises pmf(k)."""
        ks = self.support
        return max(ks, key=lambda k: self.pmf(k))

    def pmf_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ks, ps) arrays over the full support."""
        ks = np.array(self.support, dtype=int)
        ps = np.array([self.pmf(int(k)) for k in ks], dtype=float)
        return ks, ps


# ===========================================================================
# Discrete distributions
# ===========================================================================

class BernoulliDist(PMFDistribution):
    """Bernoulli(p).  Single trial: success (1) or failure (0).

    Parameters
    ----------
    p : float
        P(X = 1) ∈ (0, 1).
    """

    def __init__(self, p: float = 0.5):
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        self.p = float(p)

    def pmf(self, k: int) -> float:
        if k == 0:
            return 1.0 - self.p
        if k == 1:
            return self.p
        return 0.0

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0
        if k < 1:
            return 1.0 - self.p
        return 1.0

    @property
    def support(self) -> List[int]:
        return [0, 1]

    @property
    def name(self) -> str:
        return "Bernoulli"

    @property
    def param_string(self) -> str:
        return f"p = {self.p:.3g}"

    @property
    def mean(self) -> float:
        return self.p

    @property
    def variance(self) -> float:
        return self.p * (1 - self.p)


class BinomialDist(PMFDistribution):
    """Binomial(n, p).  Counts of successes in n independent trials.

    Parameters
    ----------
    n : int
        Number of trials.
    p : float
        P(success) per trial.
    """

    def __init__(self, n: int = 10, p: float = 0.5):
        if n < 1:
            raise ValueError("n must be ≥ 1")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.n = int(n)
        self.p = float(p)

    def pmf(self, k: int) -> float:
        if k < 0 or k > self.n:
            return 0.0
        if self.p == 0:
            return 1.0 if k == 0 else 0.0
        if self.p == 1:
            return 1.0 if k == self.n else 0.0
        log_p = (_log_comb(self.n, k)
                 + k * math.log(self.p)
                 + (self.n - k) * math.log(1 - self.p))
        return math.exp(log_p)

    def cdf(self, k: int) -> float:
        return sum(self.pmf(j) for j in range(max(0, k + 1)))

    @property
    def support(self) -> List[int]:
        return list(range(self.n + 1))

    @property
    def name(self) -> str:
        return "Binomial"

    @property
    def param_string(self) -> str:
        return f"n = {self.n},  p = {self.p:.3g}"

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)


class GeometricDist(PMFDistribution):
    """Geometric(p).  Trials until first success (inclusive).

    P(X = k) = (1-p)^(k-1) · p,  k = 1, 2, 3, …

    Parameters
    ----------
    p : float
        P(success) per trial.
    max_k : int
        Truncation of the support (default: until P(X > max_k) < 1e-5).
    """

    def __init__(self, p: float = 0.3, max_k: Optional[int] = None):
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        self.p = float(p)
        if max_k is not None:
            self._max_k = int(max_k)
        else:
            # Truncate at P(X > k) = (1-p)^k < 1e-5
            if p >= 1.0:
                self._max_k = 1
            else:
                self._max_k = int(math.ceil(math.log(1e-5) / math.log(1 - p)))
            self._max_k = min(max(self._max_k, 5), 50)

    def pmf(self, k: int) -> float:
        if k < 1:
            return 0.0
        return self.p * (1 - self.p) ** (k - 1)

    def cdf(self, k: int) -> float:
        if k < 1:
            return 0.0
        return 1.0 - (1 - self.p) ** k

    @property
    def support(self) -> List[int]:
        return list(range(1, self._max_k + 1))

    @property
    def name(self) -> str:
        return "Geometric"

    @property
    def param_string(self) -> str:
        return f"p = {self.p:.3g}"

    @property
    def mean(self) -> float:
        return 1.0 / self.p

    @property
    def variance(self) -> float:
        return (1 - self.p) / self.p**2


class NegBinomialDist(PMFDistribution):
    """Negative Binomial(r, p).  Trials until r-th success.

    P(X = k) = C(k-1, r-1) · p^r · (1-p)^(k-r),  k = r, r+1, …

    Parameters
    ----------
    r : int
        Number of required successes.
    p : float
        P(success) per trial.
    max_k : int or None
        Truncation of support.
    """

    def __init__(self, r: int = 3, p: float = 0.4, max_k: Optional[int] = None):
        if r < 1:
            raise ValueError("r must be ≥ 1")
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        self.r = int(r)
        self.p = float(p)
        if max_k is not None:
            self._max_k = int(max_k)
        else:
            mean_k = r / p
            self._max_k = min(int(mean_k + 6 * math.sqrt(r * (1 - p)) / p) + r, 80)

    def pmf(self, k: int) -> float:
        if k < self.r:
            return 0.0
        log_p = (_log_comb(k - 1, self.r - 1)
                 + self.r * math.log(self.p)
                 + (k - self.r) * math.log(1 - self.p))
        return math.exp(log_p)

    @property
    def support(self) -> List[int]:
        return list(range(self.r, self._max_k + 1))

    @property
    def name(self) -> str:
        return "Negative Binomial"

    @property
    def param_string(self) -> str:
        return f"r = {self.r},  p = {self.p:.3g}"

    @property
    def mean(self) -> float:
        return self.r / self.p

    @property
    def variance(self) -> float:
        return self.r * (1 - self.p) / self.p**2


class PoissonDist(PMFDistribution):
    """Poisson(λ).  Count of rare events in a fixed interval.

    P(X = k) = e^(-λ) · λ^k / k!,  k = 0, 1, 2, …

    Parameters
    ----------
    lam : float
        Rate parameter λ > 0.
    max_k : int or None
        Truncation of support.
    """

    def __init__(self, lam: float = 4.0, max_k: Optional[int] = None):
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.lam = float(lam)
        if max_k is not None:
            self._max_k = int(max_k)
        else:
            # P(X > max_k) < 1e-5 via Chebyshev-ish heuristic
            self._max_k = min(int(lam + 6 * math.sqrt(lam)) + 5, 100)

    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        log_p = k * math.log(self.lam) - self.lam - _lgamma(k + 1)
        return math.exp(log_p)

    def cdf(self, k: int) -> float:
        return sum(self.pmf(j) for j in range(max(0, k + 1)))

    @property
    def support(self) -> List[int]:
        return list(range(0, self._max_k + 1))

    @property
    def name(self) -> str:
        return "Poisson"

    @property
    def param_string(self) -> str:
        return f"λ = {self.lam:.3g}"

    @property
    def mean(self) -> float:
        return self.lam

    @property
    def variance(self) -> float:
        return self.lam


class HypergeometricDist(PMFDistribution):
    """Hypergeometric(N, K, n).  Draws without replacement.

    P(X = k) = C(K,k) · C(N-K, n-k) / C(N, n)

    Parameters
    ----------
    N : int
        Population size.
    K : int
        Number of success states in population.
    n : int
        Number of draws.
    """

    def __init__(self, N: int = 20, K: int = 8, n: int = 5):
        if not (0 < n <= N and 0 <= K <= N):
            raise ValueError("Must have 0 < n ≤ N and 0 ≤ K ≤ N")
        self.N = int(N)
        self.K = int(K)
        self.n = int(n)
        self._lo = max(0, n + K - N)
        self._hi = min(n, K)

    def pmf(self, k: int) -> float:
        if k < self._lo or k > self._hi:
            return 0.0
        log_p = (_log_comb(self.K, k)
                 + _log_comb(self.N - self.K, self.n - k)
                 - _log_comb(self.N, self.n))
        return math.exp(log_p)

    @property
    def support(self) -> List[int]:
        return list(range(self._lo, self._hi + 1))

    @property
    def name(self) -> str:
        return "Hypergeometric"

    @property
    def param_string(self) -> str:
        return f"N={self.N},  K={self.K},  n={self.n}"

    @property
    def mean(self) -> float:
        return self.n * self.K / self.N

    @property
    def variance(self) -> float:
        N, K, n = self.N, self.K, self.n
        return n * (K / N) * (1 - K / N) * (N - n) / (N - 1)


class DiscreteUniformDist(PMFDistribution):
    """Discrete Uniform(a, b).  Equal probability on {a, a+1, …, b}.

    Parameters
    ----------
    a, b : int
        Support endpoints (inclusive).
    """

    def __init__(self, a: int = 1, b: int = 6):
        if b < a:
            raise ValueError("b must be ≥ a")
        self.a = int(a)
        self.b = int(b)
        self._p = 1.0 / (b - a + 1)

    def pmf(self, k: int) -> float:
        return self._p if self.a <= k <= self.b else 0.0

    def cdf(self, k: int) -> float:
        if k < self.a:
            return 0.0
        if k > self.b:
            return 1.0
        return (k - self.a + 1) * self._p

    @property
    def support(self) -> List[int]:
        return list(range(self.a, self.b + 1))

    @property
    def name(self) -> str:
        return "Discrete Uniform"

    @property
    def param_string(self) -> str:
        return f"a = {self.a},  b = {self.b}"

    @property
    def mean(self) -> float:
        return (self.a + self.b) / 2

    @property
    def variance(self) -> float:
        return ((self.b - self.a + 1) ** 2 - 1) / 12


# ===========================================================================
# PMFConfig
# ===========================================================================

@dataclass
class PMFConfig:
    """Complete visual configuration for ``PMFVisualizer3D``.

    Bar geometry
    ~~~~~~~~~~~~
    ``bar_width``          : width of each bar along x.
    ``bar_depth``          : depth of each bar along y.
    ``bar_spacing``        : centre-to-centre x distance between bars.
                             If 0, auto-computed from support size.
    ``z_scale``            : scene z-units for P(X=k) = 1 (the maximum
                             possible probability).  Actual heights scale
                             proportionally so the modal bar reaches
                             z_scale × modal_prob.

    Bar shading
    ~~~~~~~~~~~
    ``bar_color``          : base colour.
    ``side_shade_factor``  : darkness multiplier for the right face.
    ``top_shade_factor``   : brightness multiplier for the top face.
    ``bar_opacity``        : overall bar opacity.
    ``edge_color``         : edge stroke colour.
    ``edge_stroke_width``  : edge stroke width.
    ``edge_opacity``       : edge stroke opacity.
    ``gloss_opacity``      : gloss strip opacity on front face (0 = off).
    ``gloss_height_frac``  : height fraction occupied by gloss strip.

    Region highlighting
    ~~~~~~~~~~~~~~~~~~~
    ``region_colors``      : colour cycle for highlighted regions.
    ``region_opacity``     : opacity of highlighted bars.
    ``dimmed_opacity``     : opacity of bars *outside* a highlighted region.

    Mean / mode markers
    ~~~~~~~~~~~~~~~~~~~
    ``mean_color``         : colour of the mean marker and annotation.
    ``mode_color``         : colour of the mode marker and annotation.
    ``marker_stroke_width``: stroke width of marker lines.

    CDF overlay
    ~~~~~~~~~~~
    ``cdf_color``          : colour of the CDF step line.
    ``cdf_stroke_width``   : stroke width of CDF line.
    ``cdf_opacity``        : CDF line opacity.
    ``cdf_glow_opacity``   : glow halo opacity behind CDF line.

    Probability labels
    ~~~~~~~~~~~~~~~~~~
    ``show_prob_labels``   : show P(X=k) text above each bar.
    ``prob_label_font``    : font size.
    ``prob_label_decimals``: decimal places.
    ``prob_label_threshold``: only label bars where P(X=k) ≥ threshold.

    Axis labels
    ~~~~~~~~~~~
    ``show_k_labels``      : show integer k below each bar.
    ``k_label_font``       : font size.
    ``k_label_step``       : label every k-th value (1 = all, 2 = even, …).

    Title
    ~~~~~
    ``show_title``         : show distribution name.
    ``title_font``         : font size.
    ``show_param_label``   : show parameter string.
    ``param_label_font``   : font size.

    Layout
    ~~~~~~
    ``x_start``            : x position of the first (leftmost) bar.
    ``y_pos``              : y (depth) position.
    ``floor_z``            : z-coordinate of bar bases.
    """

    # Bar geometry
    bar_width: float = 0.50
    bar_depth: float = 0.35
    bar_spacing: float = 0.0      # 0 = auto
    z_scale: float = 4.0

    # Bar shading
    bar_color: ManimColor = ManimColor("#4A90D9")
    side_shade_factor: float = 0.62
    top_shade_factor: float = 1.32
    bar_opacity: float = 0.90
    edge_color: ManimColor = WHITE
    edge_stroke_width: float = 0.6
    edge_opacity: float = 0.25
    gloss_opacity: float = 0.14
    gloss_height_frac: float = 0.28

    # Region highlighting
    region_colors: List[ManimColor] = field(default_factory=lambda: [
        ManimColor("#E8593C"),
        ManimColor("#2DAA6E"),
        ManimColor("#9B59B6"),
        ManimColor("#1ABC9C"),
    ])
    region_opacity: float = 0.92
    dimmed_opacity: float = 0.18

    # Mean / mode markers
    mean_color: ManimColor = ManimColor("#E0AA40")
    mode_color: ManimColor = ManimColor("#E8593C")
    marker_stroke_width: float = 1.8

    # CDF overlay
    cdf_color: ManimColor = ManimColor("#9B59B6")
    cdf_stroke_width: float = 2.2
    cdf_opacity: float = 0.85
    cdf_glow_opacity: float = 0.12

    # Probability labels
    show_prob_labels: bool = False
    prob_label_font: int = 14
    prob_label_decimals: int = 3
    prob_label_threshold: float = 0.01

    # Axis labels
    show_k_labels: bool = True
    k_label_font: int = 16
    k_label_step: int = 1

    # Title
    show_title: bool = True
    title_font: int = 22
    show_param_label: bool = True
    param_label_font: int = 18

    # Layout
    x_start: float = 0.0
    y_pos: float = 0.0
    floor_z: float = 0.0


# ── Presets ─────────────────────────────────────────────────────────────────

MINIMAL_PMF = PMFConfig(
    gloss_opacity=0.0,
    edge_opacity=0.15,
    show_prob_labels=False,
    show_k_labels=True,
    show_title=False,
    show_param_label=False,
)

TEACHING_PMF = PMFConfig(
    gloss_opacity=0.16,
    edge_opacity=0.30,
    show_prob_labels=True,
    prob_label_font=15,
    show_k_labels=True,
    show_title=True,
    show_param_label=True,
)

COMPARISON_PMF = PMFConfig(
    bar_opacity=0.80,
    gloss_opacity=0.08,
    edge_opacity=0.18,
    show_prob_labels=False,
    show_k_labels=True,
    show_title=False,
    show_param_label=False,
)


# ===========================================================================
# _PMFBar3D  — internal
# ===========================================================================

class _PMFBar3D(VGroup):
    """A single PMF bar: 3-faced prism (front, right, top) with gloss.

    Matches the geometry of ``Bar3D`` from ``bar_chart3d.py``.
    Stored as three Polygon objects so faces can be independently
    updated during parameter morphing.

    Parameters
    ----------
    x, y_pos, floor_z : float
        Bar centre and floor coordinates.
    width, depth : float
        Bar dimensions.
    height : float
        Bar height (z direction).
    color, side_color, top_color : ManimColor
    bar_opacity : float
    edge_color : ManimColor
    edge_stroke_width, edge_opacity : float
    gloss_opacity, gloss_height_frac : float
    k : int
        Associated k-value (stored for selection logic).
    prob : float
        Associated probability (stored for label logic).
    """

    def __init__(
        self,
        x: float,
        y_pos: float,
        floor_z: float,
        width: float,
        depth: float,
        height: float,
        color: ManimColor,
        side_color: ManimColor,
        top_color: ManimColor,
        bar_opacity: float,
        edge_color: ManimColor,
        edge_stroke_width: float,
        edge_opacity: float,
        gloss_opacity: float,
        gloss_height_frac: float,
        k: int = 0,
        prob: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.prob = prob
        self._x = x
        self._y = y_pos
        self._floor = floor_z
        self._width = width
        self._depth = depth
        self._height = height
        self._base_color = color
        self._side_color = side_color
        self._top_color = top_color
        self._opacity = bar_opacity

        hw, hd = width / 2, depth / 2
        z0, z1 = floor_z, floor_z + height

        # Front face
        self.face_front = Polygon(
            np.array([x-hw, y_pos-hd, z0]),
            np.array([x+hw, y_pos-hd, z0]),
            np.array([x+hw, y_pos-hd, z1]),
            np.array([x-hw, y_pos-hd, z1]),
            fill_color=_with_opacity(color, bar_opacity),
            fill_opacity=1.0, stroke_width=0,
        )
        # Right face
        self.face_right = Polygon(
            np.array([x+hw, y_pos-hd, z0]),
            np.array([x+hw, y_pos+hd, z0]),
            np.array([x+hw, y_pos+hd, z1]),
            np.array([x+hw, y_pos-hd, z1]),
            fill_color=_with_opacity(side_color, bar_opacity * 0.95),
            fill_opacity=1.0, stroke_width=0,
        )
        # Top face
        self.face_top = Polygon(
            np.array([x-hw, y_pos-hd, z1]),
            np.array([x+hw, y_pos-hd, z1]),
            np.array([x+hw, y_pos+hd, z1]),
            np.array([x-hw, y_pos+hd, z1]),
            fill_color=_with_opacity(top_color, bar_opacity * 0.88),
            fill_opacity=1.0, stroke_width=0,
        )

        # Silhouette edges
        ecol = _with_opacity(edge_color, edge_opacity)
        self.edges = VGroup(*[
            Line(a, b, color=ecol, stroke_width=edge_stroke_width)
            for a, b in [
                (np.array([x-hw,y_pos-hd,z0]), np.array([x+hw,y_pos-hd,z0])),
                (np.array([x+hw,y_pos-hd,z0]), np.array([x+hw,y_pos-hd,z1])),
                (np.array([x+hw,y_pos-hd,z1]), np.array([x-hw,y_pos-hd,z1])),
                (np.array([x-hw,y_pos-hd,z1]), np.array([x-hw,y_pos-hd,z0])),
                (np.array([x+hw,y_pos-hd,z0]), np.array([x+hw,y_pos+hd,z0])),
                (np.array([x+hw,y_pos+hd,z0]), np.array([x+hw,y_pos+hd,z1])),
                (np.array([x+hw,y_pos+hd,z1]), np.array([x+hw,y_pos-hd,z1])),
                (np.array([x-hw,y_pos-hd,z1]), np.array([x+hw,y_pos-hd,z1])),
                (np.array([x+hw,y_pos-hd,z1]), np.array([x+hw,y_pos+hd,z1])),
                (np.array([x-hw,y_pos-hd,z1]), np.array([x-hw,y_pos+hd,z1])),
                (np.array([x-hw,y_pos+hd,z1]), np.array([x+hw,y_pos+hd,z1])),
            ]
        ])

        # Gloss strip on front face
        self.gloss = VGroup()
        if gloss_opacity > 0 and height > 0.02:
            gh = height * gloss_height_frac
            zb = z1 - gh
            gloss_poly = Polygon(
                np.array([x-hw, y_pos-hd, zb]),
                np.array([x+hw, y_pos-hd, zb]),
                np.array([x+hw, y_pos-hd, z1]),
                np.array([x-hw, y_pos-hd, z1]),
                fill_color=_with_opacity(WHITE, gloss_opacity),
                fill_opacity=1.0, stroke_width=0,
            )
            self.gloss.add(gloss_poly)

        self.add(self.face_front, self.face_right, self.face_top,
                 self.edges, self.gloss)

    # ------------------------------------------------------------------

    def set_height_instant(self, new_height: float) -> None:
        """Reshape bar to *new_height* immediately (no animation)."""
        self._height = new_height
        x, y = self._x, self._y
        hw, hd = self._width / 2, self._depth / 2
        z0, z1 = self._floor, self._floor + max(new_height, 1e-4)

        for face, pts in [
            (self.face_front, [
                [x-hw,y-hd,z0],[x+hw,y-hd,z0],
                [x+hw,y-hd,z1],[x-hw,y-hd,z1],[x-hw,y-hd,z0]]),
            (self.face_right, [
                [x+hw,y-hd,z0],[x+hw,y+hd,z0],
                [x+hw,y+hd,z1],[x+hw,y-hd,z1],[x+hw,y-hd,z0]]),
            (self.face_top, [
                [x-hw,y-hd,z1],[x+hw,y-hd,z1],
                [x+hw,y+hd,z1],[x-hw,y+hd,z1],[x-hw,y-hd,z1]]),
        ]:
            face.set_points_as_corners([np.array(p) for p in pts])

    def animate_grow(self, run_time: float = 0.55) -> UpdateFromAlphaFunc:
        """Grow bar from floor to full height."""
        target = self._height
        x, y = self._x, self._y
        hw, hd = self._width / 2, self._depth / 2
        z0 = self._floor
        cfg_gloss_h = self._height  # capture for closure

        def updater(mob: _PMFBar3D, alpha: float) -> None:
            h = max(smooth(alpha) * target, 1e-4)
            z1 = z0 + h
            mob.face_front.set_points_as_corners([
                np.array([x-hw,y-hd,z0]), np.array([x+hw,y-hd,z0]),
                np.array([x+hw,y-hd,z1]), np.array([x-hw,y-hd,z1]),
                np.array([x-hw,y-hd,z0]),
            ])
            mob.face_right.set_points_as_corners([
                np.array([x+hw,y-hd,z0]), np.array([x+hw,y+hd,z0]),
                np.array([x+hw,y+hd,z1]), np.array([x+hw,y-hd,z1]),
                np.array([x+hw,y-hd,z0]),
            ])
            mob.face_top.set_points_as_corners([
                np.array([x-hw,y-hd,z1]), np.array([x+hw,y-hd,z1]),
                np.array([x+hw,y+hd,z1]), np.array([x-hw,y+hd,z1]),
                np.array([x-hw,y-hd,z1]),
            ])

        return UpdateFromAlphaFunc(self, updater, run_time=run_time)

    def highlight(
        self,
        color: ManimColor,
        opacity: float = 0.92,
    ) -> "None":
        """Recolour all faces to a highlight colour."""
        self.face_front.set_fill(_with_opacity(color, opacity))
        self.face_right.set_fill(_with_opacity(_darken(color, 0.62), opacity * 0.95))
        self.face_top.set_fill(_with_opacity(_lighten(color, 1.32), opacity * 0.88))

    def restore_color(self) -> None:
        """Restore original colours."""
        op = self._opacity
        self.face_front.set_fill(_with_opacity(self._base_color, op))
        self.face_right.set_fill(_with_opacity(self._side_color, op * 0.95))
        self.face_top.set_fill(_with_opacity(self._top_color, op * 0.88))


# ===========================================================================
# PMFVisualizer3D
# ===========================================================================

class PMFVisualizer3D(VGroup):
    """Full-featured 3D PMF visualizer for a single discrete distribution.

    Visual layers (all public VGroup attributes, independently animatable):

    - ``bars_group``    : one ``_PMFBar3D`` per support value.
    - ``spine``         : thin baseline connecting bar centres.
    - ``mean_marker``   : dashed vertical + dot at E[X].
    - ``mode_marker``   : dashed vertical + dot at mode(X).
    - ``regions``       : re-coloured subsets of bars for P(a≤X≤b).
    - ``cdf_overlay``   : step-function CDF line above bars.
    - ``title_group``   : name + parameter labels.
    - ``k_labels``      : integer labels below each bar.
    - ``prob_labels``   : P(X=k) labels above each bar.

    Parameters
    ----------
    distribution : PMFDistribution
    config : PMFConfig
    scene : ThreeDScene or None
    """

    def __init__(
        self,
        distribution: PMFDistribution,
        config: Optional[PMFConfig] = None,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dist = distribution
        self.cfg = config if config is not None else PMFConfig()
        self._scene = scene

        # Compute bar heights in scene units
        ks_arr, ps_arr = distribution.pmf_array()
        self._ks: np.ndarray = ks_arr
        self._ps: np.ndarray = ps_arr
        self._heights: np.ndarray = ps_arr * self.cfg.z_scale

        # Resolve bar spacing
        cfg = self.cfg
        n_bars = len(ks_arr)
        if cfg.bar_spacing > 0:
            spacing = cfg.bar_spacing
        else:
            # Auto: fit comfortably; wider support → narrower spacing
            spacing = max(0.50, min(1.00, 6.0 / max(n_bars, 1)))
        self._spacing = spacing

        # X positions for each bar (centred around x_start)
        total_width = (n_bars - 1) * spacing
        self._bar_xs: np.ndarray = (
            cfg.x_start
            - total_width / 2
            + np.arange(n_bars) * spacing
        )

        # Override bar width to fit spacing
        bar_w = min(cfg.bar_width, spacing * 0.82)

        # Derived face colours
        col_base  = cfg.bar_color
        col_right = _darken(col_base, cfg.side_shade_factor)
        col_top   = _lighten(col_base, cfg.top_shade_factor)

        # Public layer groups
        self.bars_group   = VGroup()
        self.spine        = VGroup()
        self.mean_marker  = VGroup()
        self.mode_marker  = VGroup()
        self.regions      = VGroup()
        self.cdf_overlay  = VGroup()
        self.title_group  = VGroup()
        self.k_labels     = VGroup()
        self.prob_labels  = VGroup()

        # Internal bar references for quick access
        self._bars: List[_PMFBar3D] = []
        self._k_to_idx: Dict[int, int] = {}

        # Build layers
        self._build_bars(bar_w, col_base, col_right, col_top)
        self._build_spine(bar_w)
        self._build_mean_mode_markers(scene)
        if cfg.show_k_labels:
            self._build_k_labels(bar_w, scene)
        if cfg.show_prob_labels:
            self._build_prob_labels(scene)
        if cfg.show_title:
            self._build_title(scene)

        self.add(
            self.spine, self.bars_group,
            self.mean_marker, self.mode_marker,
            self.regions, self.cdf_overlay,
            self.k_labels, self.prob_labels,
            self.title_group,
        )

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_bars(
        self,
        bar_w: float,
        col_base: ManimColor,
        col_right: ManimColor,
        col_top: ManimColor,
    ) -> None:
        cfg = self.cfg
        for i, (k, p, h, x) in enumerate(
            zip(self._ks, self._ps, self._heights, self._bar_xs)
        ):
            bar = _PMFBar3D(
                x=float(x),
                y_pos=cfg.y_pos,
                floor_z=cfg.floor_z,
                width=bar_w,
                depth=cfg.bar_depth,
                height=max(float(h), 1e-4),
                color=col_base,
                side_color=col_right,
                top_color=col_top,
                bar_opacity=cfg.bar_opacity,
                edge_color=cfg.edge_color,
                edge_stroke_width=cfg.edge_stroke_width,
                edge_opacity=cfg.edge_opacity,
                gloss_opacity=cfg.gloss_opacity,
                gloss_height_frac=cfg.gloss_height_frac,
                k=int(k),
                prob=float(p),
            )
            self._bars.append(bar)
            self._k_to_idx[int(k)] = i
            self.bars_group.add(bar)

    def _build_spine(self, bar_w: float) -> None:
        """Thin baseline connecting leftmost to rightmost bar foot."""
        cfg = self.cfg
        if len(self._bar_xs) < 2:
            return
        x0 = float(self._bar_xs[0]) - bar_w / 2
        x1 = float(self._bar_xs[-1]) + bar_w / 2
        col = _with_opacity(cfg.bar_color, 0.35)
        self.spine.add(Line(
            np.array([x0, cfg.y_pos - cfg.bar_depth / 2, cfg.floor_z]),
            np.array([x1, cfg.y_pos - cfg.bar_depth / 2, cfg.floor_z]),
            color=col, stroke_width=1.0,
        ))

    def _build_mean_mode_markers(self, scene: Optional[ThreeDScene]) -> None:
        cfg = self.cfg
        mu = self.dist.mean
        mode_k = self.dist.mode()

        if mu is not None:
            self._add_float_marker(
                value=mu,
                color=cfg.mean_color,
                label=f"μ = {mu:.3g}",
                z_extra=0.32,
                scene=scene,
                group=self.mean_marker,
            )

        mode_idx = self._k_to_idx.get(mode_k)
        if mode_idx is not None:
            mode_x = float(self._bar_xs[mode_idx])
            mode_h = float(self._heights[mode_idx])
            self._add_vertical_marker(
                x=mode_x,
                z_top=mode_h + cfg.floor_z + 0.35,
                color=cfg.mode_color,
                label=f"mode = {mode_k}",
                scene=scene,
                group=self.mode_marker,
            )

    def _add_float_marker(
        self,
        value: float,
        color: ManimColor,
        label: str,
        z_extra: float,
        scene: Optional[ThreeDScene],
        group: VGroup,
    ) -> None:
        """Add a floating marker at a non-integer x position."""
        cfg = self.cfg
        # Interpolate scene x from data value
        if len(self._ks) < 2:
            return
        k_min, k_max = float(self._ks[0]), float(self._ks[-1])
        t = (value - k_min) / (k_max - k_min) if k_max > k_min else 0.5
        x_scene = float(self._bar_xs[0]) + t * (float(self._bar_xs[-1]) - float(self._bar_xs[0]))
        z_pdf = float(self.dist.pmf(int(round(value)))) * cfg.z_scale
        col = _with_opacity(color, 0.82)

        ln = DashedLine(
            np.array([x_scene, cfg.y_pos - cfg.bar_depth / 2, cfg.floor_z]),
            np.array([x_scene, cfg.y_pos - cfg.bar_depth / 2, z_pdf + z_extra]),
            dash_length=0.07, dashed_ratio=0.45,
            color=col, stroke_width=cfg.marker_stroke_width,
        )
        dot = DashedLine(
            np.array([x_scene, cfg.y_pos - cfg.bar_depth / 2, cfg.floor_z - 0.07]),
            np.array([x_scene, cfg.y_pos - cfg.bar_depth / 2, cfg.floor_z + 0.07]),
            dash_length=1000, dashed_ratio=1.0,
            color=col, stroke_width=cfg.marker_stroke_width * 2,
        )
        lbl = Text(label, font_size=17, color=color)
        lbl.move_to(np.array([x_scene, cfg.y_pos - cfg.bar_depth / 2, z_pdf + z_extra + 0.28]))
        group.add(ln, dot, lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)

    def _add_vertical_marker(
        self,
        x: float,
        z_top: float,
        color: ManimColor,
        label: str,
        scene: Optional[ThreeDScene],
        group: VGroup,
    ) -> None:
        cfg = self.cfg
        col = _with_opacity(color, 0.78)
        ln = Line(
            np.array([x, cfg.y_pos - cfg.bar_depth / 2, cfg.floor_z]),
            np.array([x, cfg.y_pos - cfg.bar_depth / 2, z_top]),
            color=col, stroke_width=cfg.marker_stroke_width,
        )
        lbl = Text(label, font_size=17, color=color)
        lbl.move_to(np.array([x, cfg.y_pos - cfg.bar_depth / 2, z_top + 0.25]))
        group.add(ln, lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)

    def _build_k_labels(self, bar_w: float, scene: Optional[ThreeDScene]) -> None:
        cfg = self.cfg
        for i, (k, x) in enumerate(zip(self._ks, self._bar_xs)):
            if i % cfg.k_label_step != 0:
                continue
            lbl = Text(str(int(k)), font_size=cfg.k_label_font,
                       color=_with_opacity(WHITE, 0.55))
            lbl.move_to(np.array([
                float(x),
                cfg.y_pos - cfg.bar_depth / 2,
                cfg.floor_z - 0.25,
            ]))
            self.k_labels.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

    def _build_prob_labels(self, scene: Optional[ThreeDScene]) -> None:
        cfg = self.cfg
        for i, (k, p, h, x) in enumerate(
            zip(self._ks, self._ps, self._heights, self._bar_xs)
        ):
            if p < cfg.prob_label_threshold:
                continue
            fmt = f"{p:.{cfg.prob_label_decimals}f}"
            lbl = Text(fmt, font_size=cfg.prob_label_font,
                       color=_with_opacity(WHITE, 0.72))
            lbl.move_to(np.array([
                float(x),
                cfg.y_pos - cfg.bar_depth / 2,
                cfg.floor_z + float(h) + 0.20,
            ]))
            self.prob_labels.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

    def _build_title(self, scene: Optional[ThreeDScene]) -> None:
        cfg = self.cfg
        x_cen = float(np.mean(self._bar_xs)) if len(self._bar_xs) > 0 else cfg.x_start
        z_top = cfg.z_scale * 1.12

        title_lbl = Text(
            self.dist.name, font_size=cfg.title_font,
            color=_with_opacity(cfg.bar_color, 0.88),
        )
        title_lbl.move_to(np.array([x_cen, cfg.y_pos - cfg.bar_depth / 2, z_top]))
        self.title_group.add(title_lbl)

        if cfg.show_param_label:
            param_lbl = Text(
                self.dist.param_string, font_size=cfg.param_label_font,
                color=_with_opacity(cfg.bar_color, 0.60),
            )
            param_lbl.move_to(np.array([x_cen, cfg.y_pos - cfg.bar_depth / 2, z_top - 0.38]))
            self.title_group.add(param_lbl)

        for mob in self.title_group:
            if scene is not None:
                scene.add_fixed_orientation_mobjects(mob)

    # ------------------------------------------------------------------
    # Public region API
    # ------------------------------------------------------------------

    def shade_region(
        self,
        k_lo: int,
        k_hi: int,
        color: Optional[ManimColor] = None,
        dim_outside: bool = False,
        label: bool = True,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Highlight bars for k ∈ [k_lo, k_hi] by recolouring them.

        Parameters
        ----------
        k_lo, k_hi : int
            Inclusive range of k values to highlight.
        color : ManimColor or None
            Uses next colour from ``cfg.region_colors`` if None.
        dim_outside : bool
            If True, dim all bars outside [k_lo, k_hi].
        label : bool
            If True, add a P(k_lo ≤ X ≤ k_hi) annotation above the region.
        scene : ThreeDScene or None

        Returns
        -------
        VGroup — the recoloured bars + label, added to ``self.regions``.
        """
        cfg = self.cfg
        col = (color if color is not None
               else cfg.region_colors[len(self.regions) % len(cfg.region_colors)])

        region_grp = VGroup()

        # Compute probability
        p_region = sum(
            self.dist.pmf(k) for k in self._ks
            if k_lo <= k <= k_hi
        )

        for i, (k, bar) in enumerate(zip(self._ks, self._bars)):
            if k_lo <= k <= k_hi:
                # Clone face colours to highlight colour
                bar.face_front.set_fill(_with_opacity(col, cfg.region_opacity))
                bar.face_right.set_fill(_with_opacity(_darken(col, 0.62), cfg.region_opacity * 0.95))
                bar.face_top.set_fill(_with_opacity(_lighten(col, 1.32), cfg.region_opacity * 0.88))
                region_grp.add(bar)
            elif dim_outside:
                bar.set_opacity(cfg.dimmed_opacity)

        if label and len(region_grp) > 0:
            # Position label above the tallest bar in range
            region_bars = [b for k, b in zip(self._ks, self._bars)
                           if k_lo <= k <= k_hi]
            max_h = max(b._height for b in region_bars) if region_bars else 0.5
            x_mid = float(np.mean([b._x for b in region_bars]))
            lbl = Text(
                f"P = {p_region:.{cfg.prob_label_decimals}f}",
                font_size=19,
                color=col,
            )
            lbl.move_to(np.array([
                x_mid, cfg.y_pos - cfg.bar_depth / 2,
                cfg.floor_z + max_h + 0.40,
            ]))
            region_grp.add(lbl)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(lbl)

        self.regions.add(region_grp)
        return region_grp

    def shade_tail_left(
        self,
        k_crit: int,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Highlight P(X ≤ k_crit) — left tail."""
        return self.shade_region(int(self._ks[0]), k_crit,
                                 color=color, label=label, scene=scene)

    def shade_tail_right(
        self,
        k_crit: int,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Highlight P(X ≥ k_crit) — right tail."""
        return self.shade_region(k_crit, int(self._ks[-1]),
                                 color=color, label=label, scene=scene)

    def shade_two_tails(
        self,
        alpha: float = 0.05,
        label: bool = True,
        color: Optional[ManimColor] = None,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Shade the rejection region for a two-sided test at level *alpha*."""
        ks = self._ks
        ps = self._ps
        # Find critical values from the tails
        cum_lo = 0.0
        k_lo = int(ks[0])
        for k, p in zip(ks, ps):
            cum_lo += p
            if cum_lo >= alpha / 2:
                k_lo = int(k)
                break
        cum_hi = 0.0
        k_hi = int(ks[-1])
        for k, p in zip(reversed(ks), reversed(ps)):
            cum_hi += p
            if cum_hi >= alpha / 2:
                k_hi = int(k)
                break
        g_left  = self.shade_tail_left(k_lo, label=label, color=color, scene=scene)
        g_right = self.shade_tail_right(k_hi, label=label, color=color, scene=scene)
        return VGroup(g_left, g_right)

    def restore_colors(self) -> None:
        """Remove all region highlights and restore original bar colours."""
        for bar in self._bars:
            bar.restore_color()
            bar.set_opacity(1.0)

    # ------------------------------------------------------------------
    # CDF overlay
    # ------------------------------------------------------------------

    def build_cdf_overlay(
        self,
        scene: Optional[ThreeDScene] = None,
    ) -> VGroup:
        """Build a step-function CDF line above the PMF bars.

        The CDF steps up at each k.  Returns the VGroup so the caller
        can animate it with ``Create`` or ``FadeIn``.
        """
        cfg = self.cfg
        col = _with_opacity(cfg.cdf_color, cfg.cdf_opacity)
        y = cfg.y_pos - cfg.bar_depth / 2
        floor = cfg.floor_z

        pts: List[np.ndarray] = []
        cum = 0.0
        for k, p, x in zip(self._ks, self._ps, self._bar_xs):
            z_before = cum * cfg.z_scale
            cum += p
            z_after = cum * cfg.z_scale
            # Horizontal segment at z_before, then vertical jump to z_after
            pts.append(np.array([float(x) - self._spacing / 2, y, z_before]))
            pts.append(np.array([float(x) + self._spacing / 2, y, z_before]))
            pts.append(np.array([float(x) + self._spacing / 2, y, z_after]))

        # Final horizontal to the right
        if pts:
            pts.append(np.array([pts[-1][0] + self._spacing * 0.5, y, pts[-1][2]]))

        stroke = VMobject()
        stroke.set_points_as_corners(pts)
        stroke.set_stroke(color=col, width=cfg.cdf_stroke_width)
        stroke.set_fill(opacity=0)

        self.cdf_overlay.add(stroke)

        if cfg.cdf_glow_opacity > 0:
            glow = VMobject()
            glow.set_points_as_corners(pts)
            glow.set_stroke(
                color=_with_opacity(cfg.cdf_color, cfg.cdf_glow_opacity),
                width=cfg.cdf_stroke_width * 2.8,
            )
            glow.set_fill(opacity=0)
            self.cdf_overlay.add(glow)

        # CDF label
        lbl = Text("CDF", font_size=16, color=cfg.cdf_color)
        if pts:
            lbl.move_to(pts[-1] + np.array([0.3, 0, 0]))
        self.cdf_overlay.add(lbl)
        if scene is not None:
            scene.add_fixed_orientation_mobjects(lbl)

        return self.cdf_overlay

    # ------------------------------------------------------------------
    # Parameter morphing
    # ------------------------------------------------------------------

    def morph_to_distribution(
        self,
        new_dist: PMFDistribution,
        run_time: float = 1.2,
        update_title: bool = True,
        scene: Optional[ThreeDScene] = None,
    ) -> AnimationGroup:
        """Morph bars to a new distribution over the same support.

        Each bar interpolates independently from its current height to
        the new PMF value at that k.  If the new distribution has a
        different support, bars outside the new support shrink to zero;
        new k values not previously visible are not added.

        Parameters
        ----------
        new_dist : PMFDistribution
        run_time : float
        update_title : bool
            Fade-in new title/param labels.

        Returns
        -------
        AnimationGroup
        """
        cfg = self.cfg
        anims: List = []
        new_params: Dict[int, float] = {}

        new_ks, new_ps = new_dist.pmf_array()
        for k, p in zip(new_ks, new_ps):
            new_params[int(k)] = float(p)

        for i, (k, bar) in enumerate(zip(self._ks, self._bars)):
            k_int = int(k)
            old_h = bar._height
            new_p  = new_params.get(k_int, 0.0)
            new_h  = new_p * cfg.z_scale
            x, y   = bar._x, bar._y
            hw, hd = bar._width / 2, bar._depth / 2
            z0     = bar._floor

            def make_bar_updater(b, oh, nh, bx, by, bhw, bhd, bz0):
                def updater(mob, alpha: float) -> None:
                    t = rate_functions.ease_in_out_cubic(alpha)
                    h = max(oh + (nh - oh) * t, 1e-4)
                    z1 = bz0 + h
                    mob.face_front.set_points_as_corners([
                        np.array([bx-bhw,by-bhd,bz0]),
                        np.array([bx+bhw,by-bhd,bz0]),
                        np.array([bx+bhw,by-bhd,z1]),
                        np.array([bx-bhw,by-bhd,z1]),
                        np.array([bx-bhw,by-bhd,bz0]),
                    ])
                    mob.face_right.set_points_as_corners([
                        np.array([bx+bhw,by-bhd,bz0]),
                        np.array([bx+bhw,by+bhd,bz0]),
                        np.array([bx+bhw,by+bhd,z1]),
                        np.array([bx+bhw,by-bhd,z1]),
                        np.array([bx+bhw,by-bhd,bz0]),
                    ])
                    mob.face_top.set_points_as_corners([
                        np.array([bx-bhw,by-bhd,z1]),
                        np.array([bx+bhw,by-bhd,z1]),
                        np.array([bx+bhw,by+bhd,z1]),
                        np.array([bx-bhw,by+bhd,z1]),
                        np.array([bx-bhw,by-bhd,z1]),
                    ])
                return updater

            anims.append(UpdateFromAlphaFunc(
                bar,
                make_bar_updater(bar, old_h, new_h, x, y, hw, hd, z0),
                run_time=run_time,
            ))
            bar._height = new_h
            bar.prob = new_params.get(k_int, 0.0)

        # Update prob labels (fade out old, rebuild)
        if len(self.prob_labels) > 0 and update_title:
            anims.append(FadeOut(self.prob_labels, run_time=run_time * 0.4))

        # Update title labels
        if update_title and cfg.show_title:
            old_title = VGroup(*[m for m in self.title_group])
            self.dist = new_dist
            self.title_group.remove(*list(self.title_group))
            self._build_title(scene)
            anims.append(
                AnimationGroup(
                    FadeOut(old_title, run_time=run_time * 0.3),
                    FadeIn(self.title_group, run_time=run_time * 0.5),
                )
            )
        else:
            self.dist = new_dist

        # Update stored arrays
        self._ps = np.array([new_params.get(int(k), 0.0) for k in self._ks])
        self._heights = self._ps * cfg.z_scale

        return AnimationGroup(*anims)

    # ------------------------------------------------------------------
    # Animation helpers
    # ------------------------------------------------------------------

    def animate_bars(
        self,
        lag: float = 0.04,
        run_time_per: float = 0.55,
        left_to_right: bool = True,
    ) -> LaggedStart:
        """Grow all bars from the floor with a stagger.

        Parameters
        ----------
        left_to_right : bool
            If True, bars grow left-to-right.  False = right-to-left.
        """
        bars = self._bars if left_to_right else list(reversed(self._bars))
        return LaggedStart(
            *[b.animate_grow(run_time=run_time_per) for b in bars],
            lag_ratio=lag,
        )

    def animate_mean_mode(self, run_time: float = 0.7) -> AnimationGroup:
        """Fade in mean and mode markers together."""
        anims = []
        if len(self.mean_marker) > 0:
            anims.append(Create(self.mean_marker, run_time=run_time))
        if len(self.mode_marker) > 0:
            anims.append(Create(self.mode_marker, run_time=run_time))
        return AnimationGroup(*anims)

    def animate_cdf(self, run_time: float = 1.2) -> Create:
        """Draw the CDF step function left-to-right."""
        if len(self.cdf_overlay) == 0:
            self.build_cdf_overlay()
        return Create(self.cdf_overlay, run_time=run_time)

    def animate_title(self, run_time: float = 0.5) -> FadeIn:
        return FadeIn(self.title_group, run_time=run_time)

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
        bar_rt: float = 1.0,
        mean_mode_rt: float = 0.6,
        title_rt: float = 0.5,
    ) -> None:
        """Play the complete layer-by-layer reveal directly on *scene*."""
        scene.play(self.animate_bars(run_time_per=bar_rt / max(len(self._bars), 1) * 0.8))
        if cfg := self.cfg:
            if len(self.k_labels) > 0:
                scene.play(FadeIn(self.k_labels, run_time=0.35))
        scene.play(self.animate_mean_mode(run_time=mean_mode_rt))
        if len(self.title_group) > 0:
            scene.play(self.animate_title(run_time=title_rt))


# ===========================================================================
# MultiplePMFComparison3D
# ===========================================================================

class MultiplePMFComparison3D(VGroup):
    """Display two PMF distributions on the same k-axis.

    Two display modes:
    - ``"grouped"``       – bars for each distribution placed side by side
                            for each k value.
    - ``"superimposed"``  – bars share x position; second distribution
                            rendered semi-transparent behind the first.

    Parameters
    ----------
    dist_a, dist_b : PMFDistribution
        The two distributions to compare.  Must share at least part of
        their support for a meaningful comparison.
    mode : str
        ``"grouped"`` or ``"superimposed"``.
    colors : (ManimColor, ManimColor) or None
    config_a, config_b : PMFConfig or None
    x_start : float
    scene : ThreeDScene or None

    Attributes
    ----------
    viz_a, viz_b : PMFVisualizer3D
    """

    def __init__(
        self,
        dist_a: PMFDistribution,
        dist_b: PMFDistribution,
        mode: str = "grouped",
        colors: Optional[Tuple[ManimColor, ManimColor]] = None,
        config_a: Optional[PMFConfig] = None,
        config_b: Optional[PMFConfig] = None,
        x_start: float = 0.0,
        scene: Optional[ThreeDScene] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mode = mode
        col_a = (colors[0] if colors else ManimColor("#4A90D9"))
        col_b = (colors[1] if colors else ManimColor("#E8593C"))

        # Union of supports
        support_a = set(dist_a.support)
        support_b = set(dist_b.support)
        all_ks = sorted(support_a | support_b)
        n = len(all_ks)

        spacing = max(0.55, min(1.05, 7.0 / max(n, 1)))

        if mode == "grouped":
            bar_w = spacing * 0.40
            # dist_a at x - bar_w*0.6, dist_b at x + bar_w*0.6
            offset = bar_w * 0.62

            cfg_a = PMFConfig(**COMPARISON_PMF.__dict__)
            cfg_a.bar_color = col_a
            cfg_a.bar_width = bar_w
            cfg_a.bar_spacing = spacing
            cfg_a.x_start = x_start - offset
            cfg_a.show_k_labels = True

            cfg_b = PMFConfig(**COMPARISON_PMF.__dict__)
            cfg_b.bar_color = col_b
            cfg_b.bar_width = bar_w
            cfg_b.bar_spacing = spacing
            cfg_b.x_start = x_start + offset
            cfg_b.show_k_labels = False
            cfg_b.show_title = False

        else:  # superimposed
            bar_w = spacing * 0.72

            cfg_a = PMFConfig(**COMPARISON_PMF.__dict__)
            cfg_a.bar_color = col_a
            cfg_a.bar_width = bar_w
            cfg_a.bar_spacing = spacing
            cfg_a.x_start = x_start
            cfg_a.bar_opacity = 0.90

            cfg_b = PMFConfig(**COMPARISON_PMF.__dict__)
            cfg_b.bar_color = col_b
            cfg_b.bar_width = bar_w * 0.82
            cfg_b.bar_spacing = spacing
            cfg_b.x_start = x_start
            cfg_b.bar_opacity = 0.50
            cfg_b.show_k_labels = False
            cfg_b.show_title = False

        if config_a is not None:
            cfg_a = config_a
        if config_b is not None:
            cfg_b = config_b

        self.viz_a = PMFVisualizer3D(dist_a, config=cfg_a, scene=scene)
        self.viz_b = PMFVisualizer3D(dist_b, config=cfg_b, scene=scene)
        self.add(self.viz_a, self.viz_b)

        # Legend
        self._build_legend(dist_a, dist_b, col_a, col_b, scene)

    def _build_legend(
        self,
        dist_a: PMFDistribution,
        dist_b: PMFDistribution,
        col_a: ManimColor,
        col_b: ManimColor,
        scene: Optional[ThreeDScene],
    ) -> None:
        self.legend = VGroup()
        entries = [(dist_a, col_a), (dist_b, col_b)]
        for i, (dist, col) in enumerate(entries):
            swatch = Line(ORIGIN, RIGHT * 0.38,
                          color=_with_opacity(col, 0.90), stroke_width=3.0)
            lbl = Text(
                f"{dist.name}  ({dist.param_string})",
                font_size=16, color=col,
            )
            row = VGroup(swatch, lbl)
            row.arrange(RIGHT, buff=0.12)
            row.move_to(np.array([5.2, 0, 3.8 - i * 0.38]))
            self.legend.add(row)
            if scene is not None:
                scene.add_fixed_orientation_mobjects(row)
        self.add(self.legend)

    def animate_reveal(
        self,
        stagger: float = 0.4,
        run_time_per: float = 1.0,
    ) -> AnimationGroup:
        """Reveal both distributions with a stagger."""
        return AnimationGroup(
            self.viz_a.animate_bars(run_time_per=run_time_per / max(len(self.viz_a._bars), 1) * 0.8),
            LaggedStart(
                FadeIn(VGroup(), run_time=stagger),
                self.viz_b.animate_bars(run_time_per=run_time_per / max(len(self.viz_b._bars), 1) * 0.8),
            ),
            lag_ratio=0.0,
        )


# ===========================================================================
# Ready-to-render ThreeDScene subclasses
# ===========================================================================

class BinomialPMFScene(ThreeDScene):
    """Binomial(10, 0.4) PMF with mean, mode, and right-tail shading.

    Render:  manim -pql pmf_viz.py BinomialPMFScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        dist = BinomialDist(n=10, p=0.4)
        viz = PMFVisualizer3D(dist, config=TEACHING_PMF, scene=self)
        self.add(viz)

        viz.full_reveal(self)
        self.wait(0.5)

        # Shade P(X ≥ 7) — right tail rejection region
        region = viz.shade_tail_right(7, label=True, scene=self)
        self.play(FadeIn(viz.regions, run_time=0.55))
        self.wait(0.5)

        # Add CDF overlay
        viz.build_cdf_overlay(scene=self)
        self.play(viz.animate_cdf(run_time=1.2))
        self.wait(2)


class PoissonPMFScene(ThreeDScene):
    """Poisson PMF for λ = 1, 2, 4, 8 — showing mean = variance = λ.

    Render:  manim -pql pmf_viz.py PoissonPMFScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        dist = PoissonDist(lam=1.0)
        cfg = PMFConfig(**TEACHING_PMF.__dict__)
        cfg.bar_color = ManimColor("#E8593C")
        cfg.mean_color = ManimColor("#4A90D9")
        cfg.mode_color = ManimColor("#2DAA6E")

        viz = PMFVisualizer3D(dist, config=cfg, scene=self)
        self.add(viz)
        viz.full_reveal(self)
        self.wait(0.5)

        for lam in [2.0, 4.0, 8.0]:
            new_dist = PoissonDist(lam=lam)
            self.play(
                viz.morph_to_distribution(new_dist, run_time=1.0, scene=self),
            )
            self.wait(0.6)

        self.wait(2)


class GeometricPMFScene(ThreeDScene):
    """Geometric(p) PMF for p = 0.5, 0.3, 0.1 — showing memorylessness.

    Render:  manim -pql pmf_viz.py GeometricPMFScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        dist = GeometricDist(p=0.5)
        cfg = PMFConfig(**TEACHING_PMF.__dict__)
        cfg.bar_color = ManimColor("#2DAA6E")

        viz = PMFVisualizer3D(dist, config=cfg, scene=self)
        self.add(viz)
        viz.full_reveal(self)
        self.wait(0.5)

        for p in [0.3, 0.1]:
            new_dist = GeometricDist(p=p)
            self.play(
                viz.morph_to_distribution(new_dist, run_time=1.2, scene=self),
            )
            self.wait(0.8)

        self.wait(2)


class PMFComparisonScene(ThreeDScene):
    """Binomial(20, 0.3) vs Poisson(6) — same mean, different variance.

    Render:  manim -pql pmf_viz.py PMFComparisonScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.020)

        binom = BinomialDist(n=20, p=0.3)  # mean = 6
        poisson = PoissonDist(lam=6.0)

        comp = MultiplePMFComparison3D(
            dist_a=binom,
            dist_b=poisson,
            mode="grouped",
            scene=self,
        )
        self.add(comp)
        self.play(comp.animate_reveal(stagger=0.35, run_time_per=0.9))
        self.play(FadeIn(comp.legend, run_time=0.5))
        self.wait(2)


class BinomialNSweepScene(ThreeDScene):
    """Binomial(n, 0.3) PMF for n = 5 → 50, showing CLT convergence.

    Render:  manim -pql pmf_viz.py BinomialNSweepScene
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-52 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.025)

        dist = BinomialDist(n=5, p=0.3)
        cfg = PMFConfig(**TEACHING_PMF.__dict__)
        cfg.bar_color = ManimColor("#9B59B6")
        cfg.show_prob_labels = False

        viz = PMFVisualizer3D(dist, config=cfg, scene=self)
        self.add(viz)
        viz.full_reveal(self, bar_rt=0.8)
        self.wait(0.4)

        for n in [10, 20, 30, 50]:
            new_dist = BinomialDist(n=n, p=0.3)
            self.play(
                viz.morph_to_distribution(new_dist, run_time=0.90, scene=self),
            )
            self.wait(0.5)

        self.wait(2)