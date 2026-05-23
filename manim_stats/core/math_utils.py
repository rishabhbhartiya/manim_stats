"""
manim_stats/core/math_utils.py
================================
Pure-math utility layer for the Manim Statistics Extension.

Every numerical computation needed by any asset in this library lives here.
No Manim imports — this module is scene-independent and fully testable.

Sections
--------
 1.  Distribution Registry     — unified PDF/CDF/PPF/SF/logPDF interface
 2.  Descriptive Statistics     — central tendency, dispersion, shape
 3.  Kernel Density Estimation  — KDE with multiple kernels & bandwidths
 4.  Regression & Residuals     — OLS, WLS, diagnostics, prediction bands
 5.  Correlation                — Pearson, Spearman, Kendall, partial, distance
 6.  Hypothesis Testing Math    — statistics, p-values, critical values, effect sizes
 7.  Probability Geometry       — trees, grids, Bayes tables, combinatorics
 8.  Sampling Utilities         — bootstrap, permutation, stratified, systematic
 9.  Surface / Mesh Generation  — 3D point arrays for geometry construction
10.  Interpolation & Smoothing  — spline, LOWESS, Bézier, moving average
11.  Information Theory         — entropy, KL, MI, cross-entropy
12.  Matrix Utilities           — covariance, eigen, Cholesky, PCA ellipse
13.  Numerical Integration      — adaptive quadrature, Monte Carlo, trapezoid
14.  Axis & Tick Helpers        — nice numbers, auto-range, tick generation
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Any, Callable, Dict, Generator, List,
    Literal, Optional, Sequence, Tuple, Union,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, interpolate as sci_interpolate, stats, special
from scipy.linalg import cholesky, eigh


# convenient type aliases
FloatArray = NDArray[np.float64]
IntArray   = NDArray[np.int64]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DISTRIBUTION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistributionResult:
    """
    Container returned by every DistributionFunction evaluation.

    Carries the primary array plus optional derived arrays so callers
    never have to recompute redundant quantities.
    """
    x:        FloatArray                   # input points
    pdf:      Optional[FloatArray] = None
    cdf:      Optional[FloatArray] = None
    sf:       Optional[FloatArray] = None  # survival = 1 - CDF
    log_pdf:  Optional[FloatArray] = None
    log_cdf:  Optional[FloatArray] = None
    ppf:      Optional[FloatArray] = None  # quantile / inverse CDF
    # Distribution metadata
    mean:     Optional[float]      = None
    variance: Optional[float]      = None
    skewness: Optional[float]      = None
    kurtosis: Optional[float]      = None  # excess kurtosis
    support:  Optional[Tuple[float, float]] = None   # (lo, hi)
    name:     str                  = ""
    params:   Dict[str, float]     = field(default_factory=dict)


class DistributionFunction:
    """
    Unified interface around ``scipy.stats`` continuous and discrete
    distributions.  Every distribution in the library is accessed
    through this class so asset code never imports scipy directly.

    Usage
    -----
    >>> df = DistributionFunction.normal(mu=0, sigma=1)
    >>> r  = df.evaluate(x=np.linspace(-4, 4, 200))
    >>> r.pdf    # array of shape (200,)
    >>> r.mean   # 0.0
    """

    def __init__(
        self,
        rv:      Any,                          # scipy rv_continuous / rv_discrete
        name:    str,
        params:  Dict[str, float],
        support: Tuple[float, float],
        discrete: bool = False,
    ) -> None:
        self._rv       = rv
        self._name     = name
        self._params   = params
        self._support  = support
        self._discrete = discrete

    # ── evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        x:          Optional[ArrayLike] = None,
        n_points:   int                 = 300,
        x_padding:  float               = 0.05,   # fraction of range to pad
    ) -> DistributionResult:
        """
        Evaluate PDF/CDF/SF/logPDF/logCDF over *x*.

        If *x* is None, a sensible range is chosen automatically based
        on the 0.001–0.999 quantile interval of the distribution.
        """
        if x is None:
            lo = float(self._rv.ppf(0.001))
            hi = float(self._rv.ppf(0.999))
            pad = (hi - lo) * x_padding
            x = np.linspace(lo - pad, hi + pad, n_points)
        x = np.asarray(x, dtype=float)

        if self._discrete:
            x_int = np.round(x).astype(int)
            pdf   = self._rv.pmf(x_int).astype(float)
            cdf   = self._rv.cdf(x_int).astype(float)
            sf    = 1.0 - cdf
            logpdf = np.where(pdf > 0, np.log(pdf), -np.inf)
            logcdf = np.where(cdf > 0, np.log(cdf), -np.inf)
        else:
            pdf    = self._rv.pdf(x)
            cdf    = self._rv.cdf(x)
            sf     = self._rv.sf(x)
            logpdf = self._rv.logpdf(x)
            logcdf = self._rv.logcdf(x)

        # Moments (may be nan for heavy-tailed)
        try:    mean = float(self._rv.mean())
        except: mean = None
        try:    var  = float(self._rv.var())
        except: var  = None
        try:    skew = float(self._rv.stats(moments="s"))
        except: skew = None
        try:    kurt = float(self._rv.stats(moments="k"))
        except: kurt = None

        # PPF grid (quantile function) — useful for QQ-plots
        p   = np.linspace(0.001, 0.999, len(x))
        ppf = self._rv.ppf(p)

        return DistributionResult(
            x=x, pdf=pdf, cdf=cdf, sf=sf,
            log_pdf=logpdf, log_cdf=logcdf, ppf=ppf,
            mean=mean, variance=var,
            skewness=skew, kurtosis=kurt,
            support=self._support,
            name=self._name, params=self._params,
        )

    def pdf(self,  x: ArrayLike) -> FloatArray:
        if self._discrete:
            return self._rv.pmf(np.round(np.asarray(x)).astype(int))
        return self._rv.pdf(np.asarray(x, dtype=float))

    def cdf(self,  x: ArrayLike) -> FloatArray:
        return self._rv.cdf(np.asarray(x, dtype=float))

    def ppf(self,  p: ArrayLike) -> FloatArray:
        return self._rv.ppf(np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9))

    def sf(self,   x: ArrayLike) -> FloatArray:
        return self._rv.sf(np.asarray(x, dtype=float))

    def sample(self, n: int, seed: Optional[int] = None) -> FloatArray:
        return self._rv.rvs(size=n, random_state=seed)

    # ── factories: continuous ─────────────────────────────────────────────

    @classmethod
    def normal(cls, mu: float = 0.0, sigma: float = 1.0) -> "DistributionFunction":
        return cls(stats.norm(loc=mu, scale=sigma),
                   "Normal", {"mu": mu, "sigma": sigma},
                   (-np.inf, np.inf))

    @classmethod
    def student_t(cls, df: float, mu: float = 0.0, sigma: float = 1.0) -> "DistributionFunction":
        return cls(stats.t(df=df, loc=mu, scale=sigma),
                   "Student-t", {"df": df, "mu": mu, "sigma": sigma},
                   (-np.inf, np.inf))

    @classmethod
    def chi_squared(cls, df: float) -> "DistributionFunction":
        return cls(stats.chi2(df=df),
                   "Chi-Squared", {"df": df},
                   (0.0, np.inf))

    @classmethod
    def f_distribution(cls, dfn: float, dfd: float) -> "DistributionFunction":
        return cls(stats.f(dfn=dfn, dfd=dfd),
                   "F", {"dfn": dfn, "dfd": dfd},
                   (0.0, np.inf))

    @classmethod
    def exponential(cls, lam: float = 1.0) -> "DistributionFunction":
        # scipy uses scale = 1/lambda
        return cls(stats.expon(scale=1.0 / lam),
                   "Exponential", {"lambda": lam},
                   (0.0, np.inf))

    @classmethod
    def gamma(cls, alpha: float, beta: float = 1.0) -> "DistributionFunction":
        return cls(stats.gamma(a=alpha, scale=1.0 / beta),
                   "Gamma", {"alpha": alpha, "beta": beta},
                   (0.0, np.inf))

    @classmethod
    def beta(cls, a: float, b: float) -> "DistributionFunction":
        return cls(stats.beta(a=a, b=b),
                   "Beta", {"a": a, "b": b},
                   (0.0, 1.0))

    @classmethod
    def uniform_continuous(cls, lo: float = 0.0, hi: float = 1.0) -> "DistributionFunction":
        return cls(stats.uniform(loc=lo, scale=hi - lo),
                   "Uniform", {"lo": lo, "hi": hi},
                   (lo, hi))

    @classmethod
    def log_normal(cls, mu: float = 0.0, sigma: float = 1.0) -> "DistributionFunction":
        return cls(stats.lognorm(s=sigma, scale=math.exp(mu)),
                   "Log-Normal", {"mu": mu, "sigma": sigma},
                   (0.0, np.inf))

    @classmethod
    def weibull(cls, c: float, scale: float = 1.0) -> "DistributionFunction":
        return cls(stats.weibull_min(c=c, scale=scale),
                   "Weibull", {"c": c, "scale": scale},
                   (0.0, np.inf))

    @classmethod
    def cauchy(cls, x0: float = 0.0, gamma: float = 1.0) -> "DistributionFunction":
        return cls(stats.cauchy(loc=x0, scale=gamma),
                   "Cauchy", {"x0": x0, "gamma": gamma},
                   (-np.inf, np.inf))

    @classmethod
    def pareto(cls, alpha: float, xm: float = 1.0) -> "DistributionFunction":
        return cls(stats.pareto(b=alpha, scale=xm),
                   "Pareto", {"alpha": alpha, "xm": xm},
                   (xm, np.inf))

    @classmethod
    def laplace(cls, mu: float = 0.0, b: float = 1.0) -> "DistributionFunction":
        return cls(stats.laplace(loc=mu, scale=b),
                   "Laplace", {"mu": mu, "b": b},
                   (-np.inf, np.inf))

    @classmethod
    def logistic(cls, mu: float = 0.0, s: float = 1.0) -> "DistributionFunction":
        return cls(stats.logistic(loc=mu, scale=s),
                   "Logistic", {"mu": mu, "s": s},
                   (-np.inf, np.inf))

    # ── factories: discrete ───────────────────────────────────────────────

    @classmethod
    def bernoulli(cls, p: float) -> "DistributionFunction":
        return cls(stats.bernoulli(p=p),
                   "Bernoulli", {"p": p},
                   (0, 1), discrete=True)

    @classmethod
    def binomial(cls, n: int, p: float) -> "DistributionFunction":
        return cls(stats.binom(n=n, p=p),
                   "Binomial", {"n": n, "p": p},
                   (0, n), discrete=True)

    @classmethod
    def poisson(cls, lam: float) -> "DistributionFunction":
        return cls(stats.poisson(mu=lam),
                   "Poisson", {"lambda": lam},
                   (0, np.inf), discrete=True)

    @classmethod
    def geometric(cls, p: float) -> "DistributionFunction":
        return cls(stats.geom(p=p),
                   "Geometric", {"p": p},
                   (1, np.inf), discrete=True)

    @classmethod
    def negative_binomial(cls, r: int, p: float) -> "DistributionFunction":
        return cls(stats.nbinom(n=r, p=p),
                   "Negative Binomial", {"r": r, "p": p},
                   (0, np.inf), discrete=True)

    @classmethod
    def hypergeometric(cls, M: int, n: int, N: int) -> "DistributionFunction":
        """
        M = population size, n = number of success states in population,
        N = number of draws.
        """
        return cls(stats.hypergeom(M=M, n=n, N=N),
                   "Hypergeometric", {"M": M, "n": n, "N": N},
                   (max(0, N - (M - n)), min(n, N)), discrete=True)

    @classmethod
    def uniform_discrete(cls, lo: int = 0, hi: int = 5) -> "DistributionFunction":
        return cls(stats.randint(low=lo, high=hi + 1),
                   "Discrete Uniform", {"lo": lo, "hi": hi},
                   (lo, hi), discrete=True)

    # ── mixture & custom ──────────────────────────────────────────────────

    @classmethod
    def gaussian_mixture(
        cls,
        means:  Sequence[float],
        sigmas: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> "DistributionFunction":
        """
        Gaussian mixture model — returns a custom DistributionFunction
        backed by a synthetic scipy-like object.
        """
        means   = np.asarray(means,  dtype=float)
        sigmas  = np.asarray(sigmas, dtype=float)
        weights = np.asarray(weights, dtype=float) if weights is not None \
                  else np.ones(len(means)) / len(means)
        weights = weights / weights.sum()

        class _GMMRv:
            def pdf(self, x):
                x = np.asarray(x, float)
                return sum(w * stats.norm(loc=m, scale=s).pdf(x)
                           for w, m, s in zip(weights, means, sigmas))
            def cdf(self, x):
                x = np.asarray(x, float)
                return sum(w * stats.norm(loc=m, scale=s).cdf(x)
                           for w, m, s in zip(weights, means, sigmas))
            def logpdf(self, x): return np.log(np.clip(self.pdf(x), 1e-300, None))
            def logcdf(self, x): return np.log(np.clip(self.cdf(x), 1e-300, None))
            def sf(self, x):     return 1.0 - self.cdf(x)
            def ppf(self, p):
                from scipy.optimize import brentq
                p = np.asarray(p, float)
                lo = means.min() - 6 * sigmas.max()
                hi = means.max() + 6 * sigmas.max()
                return np.array([brentq(lambda x: self.cdf(x) - pi, lo, hi)
                                 for pi in p.flat])
            def rvs(self, size=1, random_state=None):
                rng = np.random.default_rng(random_state)
                idx = rng.choice(len(means), size=size, p=weights)
                return np.array([rng.normal(means[i], sigmas[i]) for i in idx])
            def mean(self):   return float((weights * means).sum())
            def var(self):
                e_x2 = (weights * (sigmas**2 + means**2)).sum()
                return float(e_x2 - self.mean()**2)
            def stats(self, moments="mvsk"): return None  # simplified

        rv = _GMMRv()
        lo = float(means.min() - 6 * sigmas.max())
        hi = float(means.max() + 6 * sigmas.max())
        return cls(rv, "Gaussian Mixture",
                   {"components": int(len(means))},
                   (lo, hi))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DescriptiveStats:
    """
    All descriptive statistics for a 1-D dataset in one object.
    Computed lazily on first access via ``compute()``.
    """
    n:                int
    mean:             float
    trimmed_mean:     float   # 10% trimmed by default
    weighted_mean:    Optional[float]
    median:           float
    mode:             float   # first mode for continuous data
    variance_pop:     float
    variance_sample:  float
    std_pop:          float
    std_sample:       float
    mad:              float   # median absolute deviation
    iqr:              float
    range:            float
    minimum:          float
    maximum:          float
    skewness:         float   # Fisher's moment coefficient
    kurtosis:         float   # excess kurtosis (normal = 0)
    cv:               float   # coefficient of variation (%)
    gini:             float   # Gini coefficient
    # Five-number summary
    q0:               float   # min
    q1:               float   # 25th percentile
    q2:               float   # median
    q3:               float   # 75th percentile
    q4:               float   # max
    # Extended percentiles — computed on demand
    percentiles:      FloatArray  = field(default_factory=lambda: np.array([]))


def compute_descriptive(
    data:    ArrayLike,
    weights: Optional[ArrayLike] = None,
    trim:    float                = 0.10,
    pcts:    Sequence[float]      = (1, 5, 10, 25, 50, 75, 90, 95, 99),
) -> DescriptiveStats:
    """
    Compute the full ``DescriptiveStats`` for *data*.

    Parameters
    ----------
    data    : 1-D array-like
    weights : optional frequency / probability weights (same length as data)
    trim    : fraction to trim from each tail for trimmed mean
    pcts    : percentile values to include in .percentiles
    """
    x  = np.asarray(data, dtype=float).ravel()
    x  = x[np.isfinite(x)]
    n  = len(x)
    if n == 0:
        raise ValueError("Empty data array after removing non-finite values.")

    # Weighted mean
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        wmean = float(np.dot(x, w))
    else:
        wmean = None

    # Mode — for continuous data we bin and pick the highest-frequency bin
    hist, edges = np.histogram(x, bins=min(50, n // 2 + 1))
    mode_val = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))

    # Gini coefficient
    xs   = np.sort(x - x.min() + 1e-9)   # shift to positive
    idx  = np.arange(1, n + 1)
    gini = float((2 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)

    q0, q1, q2, q3, q4 = np.percentile(x, [0, 25, 50, 75, 100])

    return DescriptiveStats(
        n             = n,
        mean          = float(np.mean(x)),
        trimmed_mean  = float(stats.trim_mean(x, trim)),
        weighted_mean = wmean,
        median        = float(np.median(x)),
        mode          = mode_val,
        variance_pop  = float(np.var(x, ddof=0)),
        variance_sample = float(np.var(x, ddof=1)) if n > 1 else 0.0,
        std_pop       = float(np.std(x, ddof=0)),
        std_sample    = float(np.std(x, ddof=1)) if n > 1 else 0.0,
        mad           = float(np.median(np.abs(x - np.median(x)))),
        iqr           = float(q3 - q1),
        range         = float(x.max() - x.min()),
        minimum       = float(x.min()),
        maximum       = float(x.max()),
        skewness      = float(stats.skew(x)),
        kurtosis      = float(stats.kurtosis(x)),
        cv            = float(np.std(x, ddof=1) / np.mean(x) * 100)
                        if np.mean(x) != 0 else 0.0,
        gini          = gini,
        q0=float(q0), q1=float(q1), q2=float(q2), q3=float(q3), q4=float(q4),
        percentiles   = np.percentile(x, list(pcts)),
    )


def compute_running_stats(data: ArrayLike) -> Dict[str, FloatArray]:
    """
    Compute running (cumulative) mean, variance, and standard deviation.
    Useful for animating convergence of the sample mean.
    Uses Welford's online algorithm for numerical stability.
    """
    x   = np.asarray(data, dtype=float).ravel()
    n   = len(x)
    means = np.zeros(n)
    M2s   = np.zeros(n)
    count = 0
    mean  = 0.0
    M2    = 0.0
    for i, xi in enumerate(x):
        count += 1
        delta  = xi - mean
        mean  += delta / count
        delta2 = xi - mean
        M2    += delta * delta2
        means[i] = mean
        M2s[i]   = M2
    variances = np.where(np.arange(1, n + 1) > 1,
                         M2s / (np.arange(1, n + 1) - 1), 0.0)
    return {
        "means":     means,
        "variances": variances,
        "stds":      np.sqrt(variances),
        "counts":    np.arange(1, n + 1, dtype=float),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KERNEL DENSITY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

KernelName = Literal["gaussian", "epanechnikov", "tophat", "biweight", "triangular"]


def _kernel_func(name: KernelName) -> Callable[[FloatArray], FloatArray]:
    """Return a callable K(u) for the named kernel (vectorized, unit bandwidth)."""
    if name == "gaussian":
        return lambda u: np.exp(-0.5 * u**2) / math.sqrt(2 * math.pi)
    elif name == "epanechnikov":
        return lambda u: np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    elif name == "tophat":
        return lambda u: np.where(np.abs(u) <= 1, 0.5, 0.0)
    elif name == "biweight":
        return lambda u: np.where(np.abs(u) <= 1, (15/16) * (1 - u**2)**2, 0.0)
    elif name == "triangular":
        return lambda u: np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
    else:
        raise ValueError(f"Unknown kernel: {name!r}")


def bandwidth_silverman(x: FloatArray) -> float:
    """Silverman's rule-of-thumb bandwidth."""
    n   = len(x)
    std = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    s   = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * s * n**(-0.2)


def bandwidth_scott(x: FloatArray) -> float:
    """Scott's rule bandwidth."""
    return float(np.std(x, ddof=1)) * len(x)**(-1/5)


def bandwidth_cv(
    x:          FloatArray,
    bandwidths: Optional[FloatArray] = None,
    cv_folds:   int                  = 5,
) -> float:
    """
    Cross-validation bandwidth selection (leave-one-out log-likelihood).
    Searches over a log-spaced grid of candidate bandwidths.
    """
    if bandwidths is None:
        h_min = bandwidth_silverman(x) * 0.1
        h_max = bandwidth_silverman(x) * 3.0
        bandwidths = np.exp(np.linspace(math.log(h_min), math.log(h_max), 30))

    best_h  = bandwidths[0]
    best_ll = -np.inf
    kernel  = _kernel_func("gaussian")
    n       = len(x)

    for h in bandwidths:
        # Leave-one-out log-likelihood
        ll = 0.0
        for i in range(n):
            others = np.delete(x, i)
            u      = (x[i] - others) / h
            dens   = kernel(u).mean() / h
            if dens > 0:
                ll += math.log(dens)
        if ll > best_ll:
            best_ll = ll
            best_h  = h

    return float(best_h)


@dataclass
class KDEResult:
    x:          FloatArray
    density:    FloatArray
    bandwidth:  float
    kernel:     str
    n_samples:  int


def compute_kde(
    data:           ArrayLike,
    x_eval:         Optional[ArrayLike]  = None,
    kernel:         KernelName           = "gaussian",
    bandwidth:      Union[float, Literal["silverman", "scott", "cv"]] = "silverman",
    n_points:       int                  = 256,
    x_padding:      float                = 0.15,
) -> KDEResult:
    """
    Compute kernel density estimate of *data* at *x_eval* points.

    Parameters
    ----------
    data      : 1-D sample
    x_eval    : evaluation grid (auto-generated if None)
    kernel    : kernel function name
    bandwidth : float or selection rule string
    n_points  : resolution of auto-generated x grid
    x_padding : fractional padding beyond data range
    """
    x   = np.asarray(data, dtype=float).ravel()
    n   = len(x)

    if isinstance(bandwidth, str):
        if bandwidth == "silverman":
            h = bandwidth_silverman(x)
        elif bandwidth == "scott":
            h = bandwidth_scott(x)
        elif bandwidth == "cv":
            h = bandwidth_cv(x)
        else:
            raise ValueError(f"Unknown bandwidth rule: {bandwidth!r}")
    else:
        h = float(bandwidth)

    if x_eval is None:
        pad = (x.max() - x.min()) * x_padding
        x_eval = np.linspace(x.min() - pad, x.max() + pad, n_points)
    else:
        x_eval = np.asarray(x_eval, dtype=float)

    K      = _kernel_func(kernel)
    density = np.array([K((xi - x) / h).mean() / h for xi in x_eval])

    return KDEResult(x=x_eval, density=density,
                     bandwidth=h, kernel=kernel, n_samples=n)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REGRESSION & RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OLSResult:
    """Full OLS regression result with diagnostics."""
    # Coefficients
    intercept:          float
    slopes:             FloatArray         # shape (p,) for p predictors
    coef_se:            FloatArray         # standard errors shape (p+1,)
    coef_t_stats:       FloatArray
    coef_p_values:      FloatArray
    # Fit quality
    r_squared:          float
    adj_r_squared:      float
    rmse:               float
    aic:                float
    bic:                float
    # Residuals
    fitted:             FloatArray
    residuals:          FloatArray
    standardized_resid: FloatArray
    studentized_resid:  FloatArray
    # Influence
    leverage:           FloatArray         # hat matrix diagonal
    cooks_distance:     FloatArray
    # Prediction
    x_grid:             Optional[FloatArray] = None   # for plotting
    y_pred:             Optional[FloatArray] = None
    ci_lower:           Optional[FloatArray] = None   # confidence band
    ci_upper:           Optional[FloatArray] = None
    pi_lower:           Optional[FloatArray] = None   # prediction band
    pi_upper:           Optional[FloatArray] = None
    # Model info
    n:                  int   = 0
    p:                  int   = 0          # number of predictors (excl. intercept)
    f_statistic:        float = 0.0
    f_p_value:          float = 1.0


def compute_ols(
    x:           ArrayLike,
    y:           ArrayLike,
    add_intercept: bool   = True,
    weights:     Optional[ArrayLike] = None,
    x_grid_n:    int      = 200,
    alpha:       float    = 0.05,
) -> OLSResult:
    """
    Ordinary (or weighted) least squares regression.

    Parameters
    ----------
    x             : predictor(s) — shape (n,) or (n, p)
    y             : response — shape (n,)
    add_intercept : prepend a column of ones to X
    weights       : optional observation weights (WLS)
    x_grid_n      : points for prediction / confidence bands
    alpha         : significance level for bands
    """
    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape

    if add_intercept:
        X = np.hstack([np.ones((n, 1)), X])

    W = np.diag(np.asarray(weights, float)) if weights is not None else np.eye(n)

    # β = (XᵀWX)⁻¹ XᵀWy
    XtW   = X.T @ W
    XtWX  = XtW @ X
    XtWy  = XtW @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

    fitted    = X @ beta
    residuals = y - fitted
    dof       = n - len(beta)
    mse       = float((residuals**2).sum() / dof) if dof > 0 else 0.0
    rmse      = math.sqrt(mse)

    # Hat matrix diagonal (leverage)
    try:
        H_diag = np.einsum("ij,jk,ki->i", X, np.linalg.inv(XtWX), X.T)
    except np.linalg.LinAlgError:
        H_diag = np.full(n, len(beta) / n)

    # Standardised & studentized residuals
    h         = np.clip(H_diag, 0, 1 - 1e-9)
    std_resid = residuals / (rmse * np.sqrt(1 - h + 1e-12))
    # Studentized (external) — refit leaving each point out is expensive;
    # use the DFFITS approximation instead
    stu_resid = std_resid * np.sqrt((n - p - 2) /
                                    np.clip(n - p - 1 - std_resid**2, 1e-6, None))

    # Cook's distance
    cooks = (std_resid**2 * h) / (len(beta) * (1 - h + 1e-12))

    # R² / adjusted R²
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / max(dof, 1)

    # F-statistic
    k         = len(beta) - 1  # number of slopes
    ss_reg    = ss_tot - ss_res
    f_stat    = (ss_reg / max(k, 1)) / mse if mse > 0 else 0.0
    f_p_value = float(stats.f.sf(f_stat, k, dof)) if k > 0 else 1.0

    # Coefficient SE & t-tests
    coef_var = mse * np.linalg.pinv(XtWX)
    coef_se  = np.sqrt(np.diag(np.abs(coef_var)))
    t_stats  = beta / np.where(coef_se > 1e-15, coef_se, 1e-15)
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=dof)

    # AIC / BIC
    log_lik = -0.5 * n * (math.log(2 * math.pi * mse) + 1)
    k_aic   = len(beta) + 1
    aic     = 2 * k_aic - 2 * log_lik
    bic     = k_aic * math.log(n) - 2 * log_lik

    # Prediction grid for 1-D case
    x_grid = y_pred = ci_lo = ci_hi = pi_lo = pi_hi = None
    if p == 1:
        xi      = X[:, 1] if add_intercept else X[:, 0]
        x_grid  = np.linspace(xi.min(), xi.max(), x_grid_n)
        X_grid  = np.column_stack([np.ones(x_grid_n), x_grid]) if add_intercept \
                  else x_grid[:, None]
        y_pred  = X_grid @ beta
        t_crit  = stats.t.ppf(1 - alpha / 2, df=dof)
        # Confidence band (mean response)
        se_mean = np.sqrt(mse * np.einsum("ij,jk,ik->i",
                                          X_grid, np.linalg.pinv(XtWX), X_grid))
        ci_lo   = y_pred - t_crit * se_mean
        ci_hi   = y_pred + t_crit * se_mean
        # Prediction band (individual response)
        se_pred = np.sqrt(mse * (1 + np.einsum("ij,jk,ik->i",
                                                X_grid, np.linalg.pinv(XtWX), X_grid)))
        pi_lo   = y_pred - t_crit * se_pred
        pi_hi   = y_pred + t_crit * se_pred

    return OLSResult(
        intercept          = float(beta[0]) if add_intercept else 0.0,
        slopes             = beta[1:] if add_intercept else beta,
        coef_se            = coef_se,
        coef_t_stats       = t_stats,
        coef_p_values      = p_values,
        r_squared          = r2,
        adj_r_squared      = adj_r2,
        rmse               = rmse,
        aic                = aic,
        bic                = bic,
        fitted             = fitted,
        residuals          = residuals,
        standardized_resid = std_resid,
        studentized_resid  = stu_resid,
        leverage           = H_diag,
        cooks_distance     = cooks,
        x_grid             = x_grid,
        y_pred             = y_pred,
        ci_lower           = ci_lo,
        ci_upper           = ci_hi,
        pi_lower           = pi_lo,
        pi_upper           = pi_hi,
        n                  = n,
        p                  = p,
        f_statistic        = f_stat,
        f_p_value          = f_p_value,
    )


def compute_vif(X: ArrayLike) -> FloatArray:
    """
    Variance Inflation Factors for each column of the design matrix *X*.
    VIF_j = 1 / (1 - R²_j) where R²_j is from regressing column j on all others.
    """
    X  = np.asarray(X, dtype=float)
    n, p = X.shape
    vif = np.zeros(p)
    for j in range(p):
        y_j  = X[:, j]
        X_j  = np.delete(X, j, axis=1)
        res  = compute_ols(X_j, y_j, add_intercept=True)
        vif[j] = 1.0 / max(1 - res.r_squared, 1e-9)
    return vif


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrelationResult:
    method:      str
    r:           float          # correlation coefficient
    p_value:     float
    ci_lower:    float          # Fisher-z 95% CI lower
    ci_upper:    float
    n:           int


def pearson_correlation(x: ArrayLike, y: ArrayLike,
                        alpha: float = 0.05) -> CorrelationResult:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n    = len(x)
    r, p = stats.pearsonr(x, y)
    # Fisher z-transform CI
    z    = math.atanh(np.clip(r, -0.9999, 0.9999))
    se   = 1.0 / math.sqrt(max(n - 3, 1))
    z_c  = stats.norm.ppf(1 - alpha / 2)
    return CorrelationResult(
        method="pearson", r=r, p_value=p, n=n,
        ci_lower=math.tanh(z - z_c * se),
        ci_upper=math.tanh(z + z_c * se),
    )


def spearman_correlation(x: ArrayLike, y: ArrayLike) -> CorrelationResult:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = stats.spearmanr(x, y)
    return CorrelationResult(method="spearman", r=float(r), p_value=float(p),
                             ci_lower=float("nan"), ci_upper=float("nan"),
                             n=len(x))


def kendall_tau(x: ArrayLike, y: ArrayLike) -> CorrelationResult:
    x, y = np.asarray(x, float), np.asarray(y, float)
    tau, p = stats.kendalltau(x, y)
    return CorrelationResult(method="kendall_tau", r=float(tau), p_value=float(p),
                             ci_lower=float("nan"), ci_upper=float("nan"),
                             n=len(x))


def correlation_matrix(
    data:   ArrayLike,
    method: Literal["pearson", "spearman"] = "pearson",
) -> Tuple[FloatArray, FloatArray]:
    """
    Return (r_matrix, p_matrix) for a (n × p) data matrix.
    """
    X = np.asarray(data, dtype=float)
    _, p = X.shape
    r_mat = np.eye(p)
    p_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            if method == "pearson":
                r, pv = stats.pearsonr(X[:, i], X[:, j])
            else:
                r, pv = stats.spearmanr(X[:, i], X[:, j])
            r_mat[i, j] = r_mat[j, i] = float(r)
            p_mat[i, j] = p_mat[j, i] = float(pv)
    return r_mat, p_mat


def partial_correlation(
    x:       ArrayLike,
    y:       ArrayLike,
    controls: ArrayLike,
) -> CorrelationResult:
    """
    Partial correlation of x and y controlling for *controls*.
    Uses residual-on-residual approach.
    """
    x       = np.asarray(x, float)
    y       = np.asarray(y, float)
    Z       = np.asarray(controls, float)
    if Z.ndim == 1: Z = Z[:, None]
    res_x   = compute_ols(Z, x).residuals
    res_y   = compute_ols(Z, y).residuals
    return pearson_correlation(res_x, res_y)


def distance_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """
    Székely's distance correlation dCor(X, Y) ∈ [0, 1].
    Detects non-linear dependence unlike Pearson.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n    = len(x)

    def _dcov2(A: FloatArray, B: FloatArray) -> float:
        return float((A * B).mean())

    def _center(D: FloatArray) -> FloatArray:
        row = D.mean(axis=1, keepdims=True)
        col = D.mean(axis=0, keepdims=True)
        grd = D.mean()
        return D - row - col + grd

    A = _center(np.abs(x[:, None] - x[None, :]))
    B = _center(np.abs(y[:, None] - y[None, :]))
    dcov2_xy = _dcov2(A, B)
    dcov2_xx = _dcov2(A, A)
    dcov2_yy = _dcov2(B, B)
    denom    = math.sqrt(dcov2_xx * dcov2_yy)
    return math.sqrt(max(dcov2_xy / denom, 0)) if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HYPOTHESIS TESTING MATH
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_name:      str
    statistic:      float
    p_value:        float
    df:             Optional[float]
    critical_value: float            # at alpha
    reject_h0:      bool
    alpha:          float
    tail:           Literal["two", "left", "right"]
    effect_size:    Optional[float]  # Cohen's d / η² / r depending on test
    effect_label:   str              # "Cohen's d", "eta_sq", etc.
    ci:             Optional[Tuple[float, float]] = None


def z_test_one_sample(
    x:        ArrayLike,
    mu0:      float,
    sigma:    float,
    alpha:    float = 0.05,
    tail:     Literal["two", "left", "right"] = "two",
) -> TestResult:
    x    = np.asarray(x, float)
    n    = len(x)
    xbar = float(x.mean())
    z    = (xbar - mu0) / (sigma / math.sqrt(n))
    if tail == "two":
        p = 2 * float(stats.norm.sf(abs(z)))
        crit = float(stats.norm.ppf(1 - alpha / 2))
    elif tail == "right":
        p = float(stats.norm.sf(z))
        crit = float(stats.norm.ppf(1 - alpha))
    else:
        p = float(stats.norm.cdf(z))
        crit = float(stats.norm.ppf(alpha))
    d = abs(xbar - mu0) / sigma
    se = sigma / math.sqrt(n)
    z_c = stats.norm.ppf(1 - alpha / 2)
    return TestResult("One-Sample Z-Test", z, p, None, crit,
                      p < alpha, alpha, tail, d, "Cohen's d",
                      ci=(xbar - z_c * se, xbar + z_c * se))


def t_test_one_sample(
    x:      ArrayLike,
    mu0:    float,
    alpha:  float = 0.05,
    tail:   Literal["two", "left", "right"] = "two",
) -> TestResult:
    x     = np.asarray(x, float)
    n     = len(x)
    xbar  = float(x.mean())
    s     = float(x.std(ddof=1))
    se    = s / math.sqrt(n)
    t     = (xbar - mu0) / se
    df    = n - 1
    if tail == "two":
        p    = 2 * float(stats.t.sf(abs(t), df))
        crit = float(stats.t.ppf(1 - alpha / 2, df))
    elif tail == "right":
        p    = float(stats.t.sf(t, df))
        crit = float(stats.t.ppf(1 - alpha, df))
    else:
        p    = float(stats.t.cdf(t, df))
        crit = float(stats.t.ppf(alpha, df))
    d   = abs(xbar - mu0) / s
    t_c = stats.t.ppf(1 - alpha / 2, df)
    return TestResult("One-Sample t-Test", t, p, df, crit,
                      p < alpha, alpha, tail, d, "Cohen's d",
                      ci=(xbar - t_c * se, xbar + t_c * se))


def t_test_two_sample(
    x1:      ArrayLike,
    x2:      ArrayLike,
    equal_var: bool   = True,
    alpha:   float    = 0.05,
    tail:    Literal["two", "left", "right"] = "two",
) -> TestResult:
    t, p  = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=tail)
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    n1, n2  = len(x1), len(x2)
    s1, s2  = x1.std(ddof=1), x2.std(ddof=1)
    # Pooled or Welch df
    if equal_var:
        df   = n1 + n2 - 2
        sp   = math.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / df)
        d    = abs(float(x1.mean() - x2.mean())) / sp
    else:
        num  = (s1**2/n1 + s2**2/n2)**2
        den  = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        df   = num / den
        d    = abs(float(x1.mean() - x2.mean())) / math.sqrt((s1**2 + s2**2)/2)
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return TestResult("Two-Sample t-Test", float(t), float(p), df, crit,
                      float(p) < alpha, alpha, tail, d, "Cohen's d")


def chi_square_gof(
    observed:  ArrayLike,
    expected:  Optional[ArrayLike] = None,
    alpha:     float = 0.05,
) -> TestResult:
    """Chi-square goodness-of-fit test."""
    obs = np.asarray(observed, float)
    exp = np.asarray(expected, float) if expected is not None \
          else np.full_like(obs, obs.mean())
    chi2, p = stats.chisquare(obs, f_exp=exp)
    df   = len(obs) - 1
    crit = float(stats.chi2.ppf(1 - alpha, df))
    w    = math.sqrt(float(chi2) / obs.sum())  # Cohen's w
    return TestResult("Chi-Square GoF", float(chi2), float(p), df, crit,
                      float(p) < alpha, alpha, "right", w, "Cohen's w")


def chi_square_independence(
    contingency: ArrayLike,
    alpha:       float = 0.05,
) -> TestResult:
    """Chi-square test of independence on a contingency table."""
    ct   = np.asarray(contingency, float)
    chi2, p, df, _ = stats.chi2_contingency(ct)
    crit = float(stats.chi2.ppf(1 - alpha, df))
    n    = ct.sum()
    k    = min(ct.shape) - 1
    v    = math.sqrt(float(chi2) / (n * k))   # Cramér's V
    return TestResult("Chi-Square Independence", float(chi2), float(p), df, crit,
                      float(p) < alpha, alpha, "right", v, "Cramér's V")


def f_test_anova(
    groups: Sequence[ArrayLike],
    alpha:  float = 0.05,
) -> TestResult:
    """One-way ANOVA F-test."""
    arrs  = [np.asarray(g, float) for g in groups]
    f, p  = stats.f_oneway(*arrs)
    k     = len(arrs)
    n     = sum(len(a) for a in arrs)
    df1, df2 = k - 1, n - k
    crit  = float(stats.f.ppf(1 - alpha, df1, df2))
    grand = np.concatenate(arrs).mean()
    ss_b  = sum(len(a) * (a.mean() - grand)**2 for a in arrs)
    ss_t  = sum(((a - grand)**2).sum() for a in arrs)
    eta2  = float(ss_b / ss_t) if ss_t > 0 else 0.0
    return TestResult("One-Way ANOVA", float(f), float(p), (df1, df2), crit,
                      float(p) < alpha, alpha, "right", eta2, "eta_sq")


def compute_power(
    effect_size: float,
    n:           int,
    alpha:       float = 0.05,
    test:        Literal["z", "t", "chi2"] = "t",
    df:          Optional[int] = None,
) -> float:
    """
    Compute statistical power for a given effect size and sample size.
    Uses the non-centrality parameter of the respective distribution.
    """
    if test == "z":
        ncp  = effect_size * math.sqrt(n)
        crit = stats.norm.ppf(1 - alpha / 2)
        power = stats.norm.sf(crit - ncp) + stats.norm.cdf(-crit - ncp)
    elif test == "t":
        ncp  = effect_size * math.sqrt(n)
        df_  = df if df is not None else n - 1
        crit = stats.t.ppf(1 - alpha / 2, df_)
        power = stats.nct.sf(crit, df_, ncp) + stats.nct.cdf(-crit, df_, ncp)
    elif test == "chi2":
        ncp  = effect_size**2 * n
        df_  = df if df is not None else 1
        crit = stats.chi2.ppf(1 - alpha, df_)
        power = stats.ncx2.sf(crit, df_, ncp)
    else:
        raise ValueError(f"Unknown test: {test!r}")
    return float(np.clip(power, 0, 1))


def sample_size_for_power(
    effect_size: float,
    power:       float = 0.80,
    alpha:       float = 0.05,
    test:        Literal["z", "t"] = "t",
    max_n:       int   = 10_000,
) -> int:
    """
    Find the minimum sample size to achieve *power* for *effect_size*.
    Binary-search over [2, max_n].
    """
    lo, hi = 2, max_n
    while lo < hi:
        mid = (lo + hi) // 2
        p   = compute_power(effect_size, mid, alpha, test)
        if p >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PROBABILITY GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbTreeNode:
    label:       str
    probability: float
    cumulative:  float         # product along the path from root
    depth:       int
    parent_idx:  Optional[int]
    x:           float = 0.0   # layout x in scene units
    y:           float = 0.0   # layout y in scene units


def build_probability_tree(
    branch_probs: Sequence[Sequence[float]],
    branch_labels: Optional[Sequence[Sequence[str]]] = None,
    h_spacing:    float = 2.5,
    v_spacing:    float = 1.2,
) -> List[ProbTreeNode]:
    """
    Build a fully labelled probability tree.

    Parameters
    ----------
    branch_probs  : list of lists.  Level 0 is the root split;
                    each inner list gives branch probabilities at that level.
                    E.g. [[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]] for a 3-level tree
                    where every node splits into 2.
    branch_labels : matching labels (auto-generated if None)
    h_spacing     : horizontal gap between levels in scene units
    v_spacing     : vertical gap between sibling branches

    Returns
    -------
    Flat list of ProbTreeNode in BFS order — use x, y for Manim positioning.
    """
    n_levels = len(branch_probs)
    nodes: List[ProbTreeNode] = []

    # Root node
    nodes.append(ProbTreeNode("root", 1.0, 1.0, 0, None, 0.0, 0.0))

    for level, probs in enumerate(branch_probs):
        # Collect parent nodes at this level
        parents = [nd for nd in nodes if nd.depth == level]
        n_branches = len(probs)
        labels = (branch_labels[level]
                  if branch_labels and len(branch_labels) > level
                  else [str(i) for i in range(n_branches)])

        # Layout: spread children vertically around each parent
        for parent in parents:
            y_offsets = np.linspace(
                -(n_branches - 1) * v_spacing / 2,
                (n_branches - 1) * v_spacing / 2,
                n_branches,
            )
            for k, (p, lbl, dy) in enumerate(zip(probs, labels, y_offsets)):
                cum = parent.cumulative * p
                nodes.append(ProbTreeNode(
                    label=lbl, probability=float(p),
                    cumulative=cum, depth=level + 1,
                    parent_idx=nodes.index(parent),
                    x=parent.x + h_spacing,
                    y=parent.y + dy,
                ))
    return nodes


def bayes_update(
    prior:      ArrayLike,
    likelihood: ArrayLike,
) -> FloatArray:
    """
    Discrete Bayes update: posterior ∝ likelihood × prior.
    Both arrays must be the same length (number of hypotheses).
    Returns normalised posterior.
    """
    prior  = np.asarray(prior, float)
    lik    = np.asarray(likelihood, float)
    post   = prior * lik
    return post / post.sum()


def generate_sample_space_grid(
    outcomes: Sequence[Any],
    events:   Optional[Dict[str, Callable[[Any], bool]]] = None,
) -> Dict[str, Any]:
    """
    Build a sample space grid for visualising events and probabilities.

    Parameters
    ----------
    outcomes : list of outcome labels (or (row, col) tuples for 2-D spaces)
    events   : named predicates — e.g. {"A": lambda x: x > 3}

    Returns
    -------
    dict with keys: outcomes, event_memberships, probabilities
    """
    n   = len(outcomes)
    out = {
        "outcomes":          list(outcomes),
        "probabilities":     [1.0 / n] * n,
        "event_memberships": {},
    }
    if events:
        for name, pred in events.items():
            out["event_memberships"][name] = [bool(pred(o)) for o in outcomes]
    return out


def combinatorics(
    n: int,
    r: int,
    with_replacement: bool = False,
    ordered: bool = False,
) -> int:
    """
    Count arrangements:

        ordered=True,  with_replacement=False  → permutations P(n,r)
        ordered=False, with_replacement=False  → combinations C(n,r)
        ordered=True,  with_replacement=True   → n^r
        ordered=False, with_replacement=True   → C(n+r-1, r)
    """
    if ordered and not with_replacement:
        return math.perm(n, r)
    elif not ordered and not with_replacement:
        return math.comb(n, r)
    elif ordered and with_replacement:
        return n ** r
    else:
        return math.comb(n + r - 1, r)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAMPLING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_sample(
    data:       ArrayLike,
    n_boot:     int   = 1000,
    statistic:  Callable[[FloatArray], float] = np.mean,
    seed:       Optional[int] = None,
    alpha:      float = 0.05,
) -> Dict[str, Any]:
    """
    Non-parametric bootstrap with percentile confidence interval.

    Returns
    -------
    dict with 'bootstrap_stats', 'ci_lower', 'ci_upper', 'se', 'bias'
    """
    rng  = np.random.default_rng(seed)
    x    = np.asarray(data, dtype=float)
    n    = len(x)
    obs  = statistic(x)
    boot = np.array([statistic(rng.choice(x, size=n, replace=True))
                     for _ in range(n_boot)])
    pct  = np.percentile(boot, [alpha/2*100, (1-alpha/2)*100])
    return {
        "observed":         obs,
        "bootstrap_stats":  boot,
        "ci_lower":         float(pct[0]),
        "ci_upper":         float(pct[1]),
        "se":               float(boot.std(ddof=1)),
        "bias":             float(boot.mean() - obs),
        "n_boot":           n_boot,
    }


def permutation_test(
    x:          ArrayLike,
    y:          ArrayLike,
    statistic:  Callable[[FloatArray, FloatArray], float] = lambda a, b: a.mean() - b.mean(),
    n_perm:     int           = 5000,
    tail:       Literal["two", "left", "right"] = "two",
    seed:       Optional[int] = None,
) -> Dict[str, Any]:
    """
    Permutation test for difference in *statistic* between x and y.
    """
    rng  = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    combined = np.concatenate([x, y])
    nx       = len(x)
    obs      = statistic(x, y)
    null_dist = np.array([
        statistic(combined[perm := rng.permutation(len(combined))][:nx],
                  combined[perm][nx:])
        for _ in range(n_perm)
    ])
    if tail == "two":
        p = float((np.abs(null_dist) >= abs(obs)).mean())
    elif tail == "right":
        p = float((null_dist >= obs).mean())
    else:
        p = float((null_dist <= obs).mean())
    return {"observed": obs, "null_distribution": null_dist, "p_value": p,
            "n_perm": n_perm}


def stratified_sample_indices(
    strata:     ArrayLike,
    n_per_stratum: int    = 10,
    seed:       Optional[int] = None,
) -> Dict[Any, IntArray]:
    """
    Return dict mapping each stratum label to *n_per_stratum* random indices.
    """
    rng   = np.random.default_rng(seed)
    arr   = np.asarray(strata)
    cats  = np.unique(arr)
    return {
        c: rng.choice(np.where(arr == c)[0],
                      size=min(n_per_stratum, (arr == c).sum()),
                      replace=False)
        for c in cats
    }


def systematic_sample_indices(n: int, k: int) -> IntArray:
    """
    Systematic sample: every k-th element starting from a random start in [0, k).
    Returns array of selected indices.
    """
    start = np.random.randint(0, k)
    return np.arange(start, n, k)


def cluster_sample_indices(
    cluster_ids: ArrayLike,
    n_clusters:  int,
    seed:        Optional[int] = None,
) -> IntArray:
    """
    Select *n_clusters* clusters at random, return all indices within them.
    """
    rng  = np.random.default_rng(seed)
    ids  = np.asarray(cluster_ids)
    unique = np.unique(ids)
    chosen = rng.choice(unique, size=min(n_clusters, len(unique)), replace=False)
    return np.where(np.isin(ids, chosen))[0]


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SURFACE / MESH GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def bivariate_normal_mesh(
    mu:         ArrayLike          = (0.0, 0.0),
    sigma:      ArrayLike          = ((1.0, 0.0), (0.0, 1.0)),
    x_range:    Tuple[float, float] = (-3.5, 3.5),
    y_range:    Tuple[float, float] = (-3.5, 3.5),
    resolution: int                = 80,
    z_scale:    float              = 1.0,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Generate (X, Y, Z) mesh arrays for a bivariate normal PDF surface.
    Suitable for Manim Surface() construction via a lambda.

    Returns
    -------
    X, Y : 2-D meshgrids of shape (resolution, resolution)
    Z    : PDF values scaled by z_scale
    """
    mu    = np.asarray(mu, float)
    Sigma = np.asarray(sigma, float)
    xs    = np.linspace(*x_range, resolution)
    ys    = np.linspace(*y_range, resolution)
    X, Y  = np.meshgrid(xs, ys)
    pos   = np.dstack([X, Y])
    rv    = stats.multivariate_normal(mean=mu, cov=Sigma)
    Z     = rv.pdf(pos) * z_scale
    return X, Y, Z


def make_surface_func(
    X: FloatArray, Y: FloatArray, Z: FloatArray
) -> Callable[[float, float], np.ndarray]:
    """
    Return a Manim-compatible parametric surface function
    f(u, v) → [x, y, z] by bilinear interpolation over (X, Y, Z).

    u maps to X column index (0–1), v maps to Y row index (0–1).
    """
    from scipy.interpolate import RegularGridInterpolator
    xs = X[0, :]
    ys = Y[:, 0]
    interp = RegularGridInterpolator((ys, xs), Z, method="linear",
                                      bounds_error=False, fill_value=0.0)

    def surface_fn(u: float, v: float) -> np.ndarray:
        x = xs[0] + u * (xs[-1] - xs[0])
        y = ys[0] + v * (ys[-1] - ys[0])
        z = float(interp([[y, x]]))
        return np.array([x, y, z])

    return surface_fn


def contour_levels(Z: FloatArray, n_levels: int = 8) -> FloatArray:
    """
    Return *n_levels* contour thresholds spaced to capture equal probability
    mass (quantile-based, not linearly spaced).
    """
    flat = Z.ravel()
    flat = flat[flat > flat.max() * 1e-4]
    return np.percentile(flat, np.linspace(5, 95, n_levels))


def histogram_mesh(
    data:       ArrayLike,
    bins:       Union[int, ArrayLike] = 20,
    normalize:  bool = True,
    z_depth:    float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Compute 3-D bar geometry data for a histogram.
    Each element describes one bar:
        x_left, x_right, height, center_x, width, z_depth
    """
    x     = np.asarray(data, float)
    counts, edges = np.histogram(x, bins=bins,
                                  density=normalize)
    bars = []
    for i, (cnt, lo, hi) in enumerate(zip(counts, edges[:-1], edges[1:])):
        bars.append({
            "x_left":   float(lo),
            "x_right":  float(hi),
            "center_x": float((lo + hi) / 2),
            "width":    float(hi - lo),
            "height":   float(cnt),
            "z_depth":  z_depth,
            "bin_index": i,
        })
    return bars


def scatter_point_cloud(
    x:          ArrayLike,
    y:          ArrayLike,
    z:          Optional[ArrayLike] = None,
    color_by:   Optional[ArrayLike] = None,
    jitter:     float               = 0.0,
    seed:       Optional[int]       = None,
) -> Dict[str, FloatArray]:
    """
    Prepare a scatter point cloud for 3-D Dot3D placement.
    If z is None, points are placed on the z=0 plane.

    Returns
    -------
    dict with keys: x, y, z, color_values (normalised to [0,1])
    """
    rng  = np.random.default_rng(seed)
    xv   = np.asarray(x, float) + rng.uniform(-jitter, jitter, len(x))
    yv   = np.asarray(y, float) + rng.uniform(-jitter, jitter, len(y))
    zv   = np.asarray(z, float) if z is not None else np.zeros(len(xv))
    if color_by is not None:
        cv = np.asarray(color_by, float)
        r  = cv.max() - cv.min()
        cv = (cv - cv.min()) / r if r > 0 else np.zeros_like(cv)
    else:
        cv = np.zeros(len(xv))
    return {"x": xv, "y": yv, "z": zv, "color_values": cv}


# ─────────────────────────────────────────────────────────────────────────────
# 10.  INTERPOLATION & SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def smooth_curve(
    x:       ArrayLike,
    y:       ArrayLike,
    n_out:   int   = 300,
    kind:    Literal["cubic", "quadratic", "linear", "akima"] = "cubic",
) -> Tuple[FloatArray, FloatArray]:
    """
    Interpolate / smooth a set of (x, y) points to *n_out* densely spaced pts.
    Returns (x_smooth, y_smooth).
    """
    xv = np.asarray(x, float)
    yv = np.asarray(y, float)
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    # Remove duplicate x values
    _, unique_idx = np.unique(xv, return_index=True)
    xv, yv = xv[unique_idx], yv[unique_idx]

    xi = np.linspace(xv.min(), xv.max(), n_out)
    if kind == "akima":
        f  = sci_interpolate.Akima1DInterpolator(xv, yv)
    else:
        f  = sci_interpolate.interp1d(xv, yv, kind=kind,
                                       bounds_error=False,
                                       fill_value="extrapolate")
    return xi, f(xi)


def lowess_smooth(
    x:         ArrayLike,
    y:         ArrayLike,
    frac:      float = 0.3,
    n_out:     int   = 300,
) -> Tuple[FloatArray, FloatArray]:
    """
    Locally-weighted scatterplot smoothing (LOWESS).
    Returns (x_smooth, y_smooth) evaluated on a uniform grid.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    xv = np.asarray(x, float)
    yv = np.asarray(y, float)
    smoothed = lowess(yv, xv, frac=frac, return_sorted=True)
    xi = np.linspace(xv.min(), xv.max(), n_out)
    f  = sci_interpolate.interp1d(smoothed[:, 0], smoothed[:, 1],
                                   kind="linear",
                                   bounds_error=False,
                                   fill_value=(smoothed[0, 1], smoothed[-1, 1]))
    return xi, f(xi)


def bezier_points(
    control_pts: ArrayLike,
    n:           int = 100,
) -> FloatArray:
    """
    Evaluate a degree-(k-1) Bézier curve at *n* uniformly spaced parameter values.
    *control_pts* shape: (k, d) where d is the dimension (2 or 3).
    Returns array of shape (n, d).
    """
    P  = np.asarray(control_pts, float)
    k  = len(P)
    ts = np.linspace(0, 1, n)
    curve = np.zeros((n, P.shape[1]))
    for i, t in enumerate(ts):
        b = np.zeros(P.shape[1])
        for j, pt in enumerate(P):
            b += math.comb(k - 1, j) * (1 - t)**(k - 1 - j) * t**j * pt
        curve[i] = b
    return curve


def moving_average(
    x:       ArrayLike,
    window:  int,
    mode:    Literal["simple", "exponential", "weighted"] = "simple",
    alpha:   float = 0.3,   # for exponential
) -> FloatArray:
    """
    Apply a moving average to *x*.
    Returns array of the same length (edges padded with NaN for simple/weighted).
    """
    xv  = np.asarray(x, float)
    n   = len(xv)
    out = np.full(n, np.nan)
    if mode == "simple":
        for i in range(window - 1, n):
            out[i] = xv[i - window + 1:i + 1].mean()
    elif mode == "exponential":
        out[0] = xv[0]
        for i in range(1, n):
            out[i] = alpha * xv[i] + (1 - alpha) * out[i - 1]
    elif mode == "weighted":
        w = np.arange(1, window + 1, dtype=float)
        w /= w.sum()
        for i in range(window - 1, n):
            out[i] = (w * xv[i - window + 1:i + 1]).sum()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 11.  INFORMATION THEORY
# ─────────────────────────────────────────────────────────────────────────────

def entropy(p: ArrayLike, base: float = math.e) -> float:
    """
    Shannon entropy H(P) = -Σ p_i log(p_i).
    *p* need not be normalised — it will be normalised internally.
    """
    pv   = np.asarray(p, float)
    pv   = pv[pv > 0]
    pv  /= pv.sum()
    return float(-np.sum(pv * np.log(pv) / math.log(base)))


def kl_divergence(p: ArrayLike, q: ArrayLike, base: float = math.e) -> float:
    """
    KL divergence D_KL(P || Q) = Σ p_i log(p_i / q_i).
    Undefined if q_i = 0 where p_i > 0 — returns inf in that case.
    """
    pv = np.asarray(p, float)
    qv = np.asarray(q, float)
    pv /= pv.sum(); qv /= qv.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio  = np.where((pv > 0) & (qv > 0), pv * np.log(pv / qv) / math.log(base), 0.0)
        inf_mask = (pv > 0) & (qv == 0)
    if inf_mask.any():
        return math.inf
    return float(ratio.sum())


def js_divergence(p: ArrayLike, q: ArrayLike) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded [0,1] in nats/bits).
    """
    pv = np.asarray(p, float); pv /= pv.sum()
    qv = np.asarray(q, float); qv /= qv.sum()
    m  = 0.5 * (pv + qv)
    return 0.5 * kl_divergence(pv, m) + 0.5 * kl_divergence(qv, m)


def mutual_information(
    joint: ArrayLike,
    base:  float = math.e,
) -> float:
    """
    Mutual information I(X; Y) from a (m × n) joint probability table.
    """
    P  = np.asarray(joint, float)
    P /= P.sum()
    px = P.sum(axis=1, keepdims=True)
    py = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(P > 0,
                             np.log(P / (px * py + 1e-300)) / math.log(base),
                             0.0)
    return float((P * log_ratio).sum())


def cross_entropy(p: ArrayLike, q: ArrayLike, base: float = math.e) -> float:
    """H(P, Q) = H(P) + D_KL(P || Q)."""
    return entropy(p, base) + kl_divergence(p, q, base)


def differential_entropy(
    dist: "DistributionFunction",
    n_points: int = 1000,
) -> float:
    """
    Numerical differential entropy h(X) = -∫ f(x) log f(x) dx
    for a continuous distribution.
    """
    res = dist.evaluate(n_points=n_points)
    f   = np.clip(res.pdf, 1e-300, None)
    dx  = res.x[1] - res.x[0]
    return float(-np.sum(f * np.log(f) * dx))


# ─────────────────────────────────────────────────────────────────────────────
# 12.  MATRIX UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def covariance_matrix(
    data:    ArrayLike,
    ddof:    int = 1,
    weights: Optional[ArrayLike] = None,
) -> FloatArray:
    """
    Compute (p × p) sample covariance matrix from (n × p) data.
    Optionally weighted.
    """
    X = np.asarray(data, float)
    if weights is None:
        return np.cov(X.T, ddof=ddof)
    w   = np.asarray(weights, float)
    w  /= w.sum()
    mu  = (w[:, None] * X).sum(axis=0)
    Xc  = X - mu
    return (w[:, None] * Xc).T @ Xc / (1 - (w**2).sum())


def pca(
    data:       ArrayLike,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Principal Component Analysis via eigendecomposition of the covariance matrix.

    Returns
    -------
    dict with:
        components      — (k × p) matrix of eigenvectors
        explained_var   — variance explained by each PC
        explained_ratio — fraction of total variance
        scores          — (n × k) projected data
        loadings        — (p × k) = components.T * sqrt(explained_var)
    """
    X    = np.asarray(data, float)
    n, p = X.shape
    k    = n_components or p
    Xc   = X - X.mean(axis=0)
    C    = np.cov(Xc.T, ddof=1)
    vals, vecs = eigh(C)       # ascending order
    idx  = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    vals = np.maximum(vals, 0)[:k]
    vecs = vecs[:, :k]
    scores   = Xc @ vecs
    total    = vals.sum() + 1e-15
    loadings = vecs * np.sqrt(vals)
    return {
        "components":      vecs.T,
        "explained_var":   vals,
        "explained_ratio": vals / total,
        "scores":          scores,
        "loadings":        loadings,
        "cumulative_ratio": np.cumsum(vals / total),
    }


def confidence_ellipse_params(
    x:      ArrayLike,
    y:      ArrayLike,
    n_std:  float = 2.0,
) -> Dict[str, float]:
    """
    Parameters of the 2-D confidence ellipse for scatter data.

    Returns
    -------
    dict: center_x, center_y, width, height, angle_deg
    Suitable for Manim Ellipse() with rotation.
    """
    xv = np.asarray(x, float)
    yv = np.asarray(y, float)
    C  = np.cov(xv, yv)
    vals, vecs = eigh(C)
    vals  = np.maximum(vals, 0)
    order = np.argsort(vals)[::-1]
    vals  = vals[order]
    vecs  = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    w     = 2 * n_std * math.sqrt(vals[0])
    h     = 2 * n_std * math.sqrt(vals[1])
    return {
        "center_x":  float(xv.mean()),
        "center_y":  float(yv.mean()),
        "width":     w,
        "height":    h,
        "angle_deg": angle,
    }


def cholesky_sample(
    mu:     ArrayLike,
    sigma:  ArrayLike,
    n:      int,
    seed:   Optional[int] = None,
) -> FloatArray:
    """
    Draw *n* samples from N(mu, Sigma) using Cholesky decomposition.
    Numerically more stable than np.random.multivariate_normal for
    near-singular covariances.

    Returns
    -------
    FloatArray of shape (n, len(mu))
    """
    rng = np.random.default_rng(seed)
    mu  = np.asarray(mu, float)
    Sig = np.asarray(sigma, float)
    # Add tiny jitter for numerical stability
    Sig = Sig + np.eye(len(mu)) * 1e-10
    L   = cholesky(Sig, lower=True)
    z   = rng.standard_normal((n, len(mu)))
    return (L @ z.T).T + mu


def make_positive_semidefinite(M: ArrayLike) -> FloatArray:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix
    (Higham 1988 nearest correlation matrix algorithm — simplified).
    Used when a user-supplied covariance matrix is ill-conditioned.
    """
    A    = np.asarray(M, float)
    A    = (A + A.T) / 2
    vals, vecs = eigh(A)
    vals = np.maximum(vals, 0)
    return (vecs * vals) @ vecs.T


# ─────────────────────────────────────────────────────────────────────────────
# 13.  NUMERICAL INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def area_under_curve(
    f:      Callable[[FloatArray], FloatArray],
    a:      float,
    b:      float,
    method: Literal["quad", "trapz", "simpson", "montecarlo"] = "quad",
    n:      int   = 1000,
    seed:   Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute ∫_a^b f(x) dx.

    Returns
    -------
    (estimate, error_estimate)
    """
    if method == "quad":
        val, err = integrate.quad(f, a, b, limit=200)
        return float(val), float(err)
    elif method == "trapz":
        x   = np.linspace(a, b, n)
        val = float(np.trapezoid(f(x), x))
        return val, 0.0
    elif method == "simpson":
        x   = np.linspace(a, b, n | 1)   # ensure odd
        val = float(integrate.simpson(f(x), x=x))
        return val, 0.0
    elif method == "montecarlo":
        rng  = np.random.default_rng(seed)
        xs   = rng.uniform(a, b, n)
        val  = float((b - a) * f(xs).mean())
        err  = float((b - a) * f(xs).std() / math.sqrt(n))
        return val, err
    else:
        raise ValueError(f"Unknown integration method: {method!r}")


def cdf_from_pdf(
    pdf_vals: FloatArray,
    x:        FloatArray,
) -> FloatArray:
    """
    Numerically compute CDF from PDF values using cumulative trapezoid integration.
    Normalises so CDF(x[-1]) = 1.
    """
    cdf = integrate.cumulative_trapezoid(pdf_vals, x, initial=0)
    total = cdf[-1]
    return cdf / total if total > 0 else cdf


def pdf_from_samples(
    samples:  ArrayLike,
    x_eval:   ArrayLike,
    bandwidth: Union[float, str] = "silverman",
) -> FloatArray:
    """
    Estimate PDF at *x_eval* from *samples* via KDE.
    Thin wrapper around compute_kde for convenience.
    """
    result = compute_kde(samples, x_eval=x_eval, bandwidth=bandwidth)
    return result.density


# ─────────────────────────────────────────────────────────────────────────────
# 14.  AXIS & TICK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def nice_number(x: float, round_: bool = False) -> float:
    """
    Round *x* to a 'nice' number (1, 2, 5, 10, …) for axis ticks.
    Wilkinson's algorithm.
    """
    exp   = math.floor(math.log10(abs(x))) if x != 0 else 0
    frac  = abs(x) / 10**exp
    if round_:
        if   frac < 1.5: nf = 1
        elif frac < 3.0: nf = 2
        elif frac < 7.0: nf = 5
        else:             nf = 10
    else:
        if   frac <= 1: nf = 1
        elif frac <= 2: nf = 2
        elif frac <= 5: nf = 5
        else:           nf = 10
    return math.copysign(nf * 10**exp, x)


def auto_range(
    data:       ArrayLike,
    padding:    float = 0.05,
    n_ticks:    int   = 5,
) -> Tuple[float, float, float]:
    """
    Compute (min, max, step) for axis range based on data.
    Returns a 'nice' step suitable for Manim's ThreeDAxes range tuple.
    """
    x      = np.asarray(data, float)
    lo, hi = float(x.min()), float(x.max())
    span   = hi - lo if hi != lo else 1.0
    pad    = span * padding
    lo     -= pad
    hi     += pad
    raw_step = (hi - lo) / n_ticks
    step     = nice_number(raw_step, round_=True)
    lo       = math.floor(lo / step) * step
    hi       = math.ceil(hi  / step) * step
    return lo, hi, step


def generate_ticks(
    lo:    float,
    hi:    float,
    step:  float,
    minor_per_major: int = 4,
) -> Dict[str, FloatArray]:
    """
    Generate major and minor tick positions between *lo* and *hi*.

    Returns
    -------
    dict: 'major' array, 'minor' array, 'labels' list of formatted strings
    """
    major = np.arange(math.ceil(lo / step) * step,
                      hi + step * 1e-6, step)
    minor_step = step / minor_per_major
    minor = np.arange(math.ceil(lo / minor_step) * minor_step,
                      hi + minor_step * 1e-6, minor_step)
    # Remove minor ticks that coincide with major ticks
    minor = minor[~np.isin(np.round(minor, 10),
                            np.round(major, 10))]
    labels = []
    for m in major:
        if step >= 1:
            labels.append(f"{m:.0f}")
        elif step >= 0.1:
            labels.append(f"{m:.1f}")
        elif step >= 0.01:
            labels.append(f"{m:.2f}")
        else:
            labels.append(f"{m:.3g}")
    return {"major": major, "minor": minor, "labels": labels}


def format_stat_value(
    value: float,
    decimals: int = 4,
    sci_threshold: float = 1e-3,
) -> str:
    """
    Format a numeric stat value for display:
      - Very small p-values → scientific notation
      - Otherwise → fixed decimal
    """
    if abs(value) < sci_threshold and value != 0:
        return f"{value:.{decimals}e}"
    return f"{value:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Distribution
    "DistributionResult",
    "DistributionFunction",
    # Descriptive
    "DescriptiveStats",
    "compute_descriptive",
    "compute_running_stats",
    # KDE
    "KernelName",
    "KDEResult",
    "compute_kde",
    "bandwidth_silverman",
    "bandwidth_scott",
    "bandwidth_cv",
    # Regression
    "OLSResult",
    "compute_ols",
    "compute_vif",
    # Correlation
    "CorrelationResult",
    "pearson_correlation",
    "spearman_correlation",
    "kendall_tau",
    "correlation_matrix",
    "partial_correlation",
    "distance_correlation",
    # Hypothesis testing
    "TestResult",
    "z_test_one_sample",
    "t_test_one_sample",
    "t_test_two_sample",
    "chi_square_gof",
    "chi_square_independence",
    "f_test_anova",
    "compute_power",
    "sample_size_for_power",
    # Probability geometry
    "ProbTreeNode",
    "build_probability_tree",
    "bayes_update",
    "generate_sample_space_grid",
    "combinatorics",
    # Sampling
    "bootstrap_sample",
    "permutation_test",
    "stratified_sample_indices",
    "systematic_sample_indices",
    "cluster_sample_indices",
    # Mesh / surface
    "bivariate_normal_mesh",
    "make_surface_func",
    "contour_levels",
    "histogram_mesh",
    "scatter_point_cloud",
    # Interpolation
    "smooth_curve",
    "lowess_smooth",
    "bezier_points",
    "moving_average",
    # Information theory
    "entropy",
    "kl_divergence",
    "js_divergence",
    "mutual_information",
    "cross_entropy",
    "differential_entropy",
    # Matrix
    "covariance_matrix",
    "pca",
    "confidence_ellipse_params",
    "cholesky_sample",
    "make_positive_semidefinite",
    # Integration
    "area_under_curve",
    "cdf_from_pdf",
    "pdf_from_samples",
    # Axis helpers
    "nice_number",
    "auto_range",
    "generate_ticks",
    "format_stat_value",
]