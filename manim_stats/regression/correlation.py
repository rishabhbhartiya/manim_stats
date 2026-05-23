"""
manim_stats/regression/correlation.py
======================================
Correlation analysis, OLS regression fitting, diagnostic measures,
and Manim visualisation mobjects for the regression / correlation topic area.

Architecture
------------
  Layer A  Pure-math computation      — pearson, spearman, kendall, partial_r,
                                        point_biserial, phi, cramers_v
  Layer B  Regression fitting          — ols_fit, ridge_fit, robust_fit (IRLS)
  Layer C  Structured result classes   — CorrelationResult, RegressionResult,
                                        PartialCorrelationResult, InfluenceMeasures
  Layer D  Manim mobjects              — CorrelationEllipse3D, CorrelationMatrix3D,
                                        RegressionLine3D, ResidualArrows3D,
                                        InfluenceMap3D, CIBand3D
  Layer E  Scene-level animations      — build_scatter_to_line, morph_r_value,
                                        reveal_residuals, animate_influence_removal
  Layer F  Formula registry bridge     — CORR_FORMULAS, formula builders that
                                        extend tex_utils.FORMULAS

Design notes
------------
* Layers A–C are pure Python + NumPy/SciPy — usable in notebooks, scripts, or
  tests without Manim installed.
* Every result dataclass has a ``.to_formula()`` method returning a TexFormula
  (from core.tex_utils) so scenes can overlay the live equation alongside the
  visualisation.
* Manim mobjects follow the ``VGroup`` sub-class pattern established in
  Card3D: __init__ builds sub-mobjects, public methods return Animation objects.
* All colour constants are pulled from core.colors so every mobject respects
  the active StatsTheme.
* SciPy is an optional dependency for p-value computation; if absent the
  module falls back to NumPy-only implementations with a user warning.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
# Graceful Manim import
# ---------------------------------------------------------------------------
try:
    import manim as mn
    from manim import (
        VGroup, VMobject,
        Axes, NumberPlane,
        Line, Arrow, DashedLine,
        Dot, Circle, Ellipse, Square, Rectangle, RoundedRectangle, Polygon,
        Text, MathTex, Tex,
        ManimColor, WHITE, BLACK, GRAY, DARK_GRAY,
        RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE,
        UP, DOWN, LEFT, RIGHT, ORIGIN,
        TAU, PI,
        Write, Create, FadeIn, FadeOut, FadeToColor,
        Transform, ReplacementTransform, TransformMatchingShapes,
        Rotate, Flash, Indicate, Circumscribe,
        AnimationGroup, Succession, LaggedStart,
        rate_functions,
        interpolate_color,
        SurroundingRectangle,
    )
    _MANIM_AVAILABLE = True
except ImportError:
    _MANIM_AVAILABLE = False
    VGroup = VMobject = object  # type: ignore

# ---------------------------------------------------------------------------
# Project imports (graceful — avoids hard circular dependency)
# ---------------------------------------------------------------------------
try:
    from manim_stats.core.colors import (
        REGRESSION_FAMILY, NORMAL_FAMILY, INFERENCE_FAMILY,
        DISCRETE_FAMILY, NEUTRAL_FAMILY,
        DARK_THEME, LIGHT_THEME,
        StatColor, ColorFamily, StatsTheme,
        PURPLE_600, PURPLE_200, PURPLE_800,
        CORAL_600, CORAL_200,
        TEAL_600, TEAL_200,
        BLUE_600, BLUE_200,
        AMBER_600, AMBER_200,
        GRAY_400, GRAY_600, GRAY_800,
        WHITE as _WHITE, BLACK as _BLACK,
        diverging_map, gradient_ramp,
    )
    _COLORS_AVAILABLE = True
except ImportError:
    _COLORS_AVAILABLE = False

try:
    from manim_stats.core.tex_utils import (
        TexFormula, TexDerivationStep,
        _frac, _sqrt, _sum, _exp, _expected, _var, _cov,
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


def _require_scipy(name: str) -> None:
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            f"{name} requires SciPy for p-value computation.  "
            "Install with: pip install scipy  "
            "or pass compute_pvalue=False to suppress this error."
        )


# ===========================================================================
# LAYER A — Pure-math correlation functions
# Each returns a raw numeric result (or CorrelationResult) with no Manim deps.
# ===========================================================================

class CorrelationMethod(Enum):
    """Supported correlation coefficient types."""
    PEARSON        = "pearson"
    SPEARMAN       = "spearman"
    KENDALL        = "kendall"
    POINT_BISERIAL = "point_biserial"
    PHI            = "phi"
    CRAMERS_V      = "cramers_v"
    PARTIAL        = "partial"


@dataclass
class CorrelationResult:
    """
    Structured output of a correlation computation.

    Attributes
    ----------
    r : float
        Correlation coefficient.  For Cramers V this is V (0–1 unsigned).
    method : CorrelationMethod
        Which coefficient was computed.
    n : int
        Sample size.
    p_value : float or None
        Two-tailed p-value.  None when SciPy is unavailable and
        ``compute_pvalue=False`` was passed.
    ci_low, ci_high : float or None
        95% confidence interval via Fisher z-transformation (Pearson only).
        None for non-Pearson methods or n < 4.
    df : int or None
        Degrees of freedom used for the t/z test.
    z_stat : float or None
        Test statistic (t for Pearson/Spearman/Point-biserial, z for Kendall).
    controlled_for : list[str] or None
        Variable names partialled out (PartialCorrelationResult only).

    Derived properties
    ------------------
    .stars              — significance stars: "", "*", "**", "***"
    .effect_size_label  — Cohen's (1988) label: "negligible"/"small"/"medium"/"large"
    .significant        — bool: p_value < 0.05 (None if p_value is None)
    .abs_r              — |r|
    """

    r:              float
    method:         CorrelationMethod
    n:              int
    p_value:        Optional[float]  = None
    ci_low:         Optional[float]  = None
    ci_high:        Optional[float]  = None
    df:             Optional[int]    = None
    z_stat:         Optional[float]  = None
    controlled_for: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def abs_r(self) -> float:
        return abs(self.r)

    @property
    def stars(self) -> str:
        """Significance stars based on p_value."""
        if self.p_value is None:
            return ""
        if self.p_value < 0.001:
            return "***"
        if self.p_value < 0.01:
            return "**"
        if self.p_value < 0.05:
            return "*"
        return ""

    @property
    def significant(self) -> Optional[bool]:
        if self.p_value is None:
            return None
        return self.p_value < 0.05

    @property
    def effect_size_label(self) -> str:
        """
        Cohen (1988) benchmarks for |r|:
        < 0.10 negligible, < 0.30 small, < 0.50 medium, >= 0.50 large.
        """
        a = self.abs_r
        if a < 0.10:
            return "negligible"
        if a < 0.30:
            return "small"
        if a < 0.50:
            return "medium"
        return "large"

    @property
    def ci_str(self) -> str:
        """Formatted 95% CI string, e.g. '[0.574, 0.887]'."""
        if self.ci_low is None or self.ci_high is None:
            return "N/A"
        return f"[{self.ci_low:.3f}, {self.ci_high:.3f}]"

    def summary(self) -> str:
        """One-line human-readable summary."""
        p_str = f"p={self.p_value:.4g}" if self.p_value is not None else "p=N/A"
        return (
            f"{self.method.value}  r={self.r:.4f}{self.stars}  "
            f"n={self.n}  {p_str}  95%CI={self.ci_str}  "
            f"effect={self.effect_size_label}"
        )

    # ------------------------------------------------------------------
    # tex_utils bridge
    # ------------------------------------------------------------------

    def to_formula(self) -> "TexFormula":
        """
        Return a ``TexFormula`` showing the result as a typeset equation, e.g.::

            r = 0.774^{**}\quad (n=30,\; p=5.3 \times 10^{-7})

        Requires tex_utils to be importable (it is a project-internal dep).
        """
        if not _TEX_AVAILABLE:
            raise ImportError(
                "to_formula() requires manim_stats.core.tex_utils."
            )
        sym = {
            CorrelationMethod.PEARSON:        "r",
            CorrelationMethod.SPEARMAN:       r"\rho",
            CorrelationMethod.KENDALL:        r"\tau",
            CorrelationMethod.POINT_BISERIAL: r"r_{pb}",
            CorrelationMethod.PHI:            r"\phi",
            CorrelationMethod.CRAMERS_V:      "V",
            CorrelationMethod.PARTIAL:        r"r_{partial}",
        }[self.method]

        stars_tex  = r"^{\!" + self.stars + r"}" if self.stars else ""
        p_tex      = (
            rf"\;p={self.p_value:.3g}"
            if self.p_value is not None else ""
        )
        n_tex      = rf"\;n={self.n}"
        raw        = rf"{sym} = {self.r:.4f}{stars_tex}\quad({n_tex}{p_tex})"

        return TexFormula(
            name        = f"corr_result_{self.method.value}",
            raw         = raw,
            description = self.summary(),
            parts       = {
                "coefficient": sym,
                "value":       f"{self.r:.4f}",
                "stars":       self.stars,
            },
            tags        = ["correlation", "result", self.method.value],
        )

    def __repr__(self) -> str:
        return (
            f"CorrelationResult(r={self.r:.4f}, "
            f"method={self.method.value}, n={self.n}, "
            f"p={self.p_value:.4g if self.p_value is not None else 'None'})"
        )


# ---------------------------------------------------------------------------
# A.1  Pearson r
# ---------------------------------------------------------------------------

def pearson(
    x:               np.ndarray,
    y:               np.ndarray,
    compute_pvalue:  bool = True,
    alpha:           float = 0.05,
) -> CorrelationResult:
    """
    Compute Pearson's r with Fisher z confidence interval.

    Parameters
    ----------
    x, y : array-like of shape (n,)
        Paired observations.  NaNs are listwise-deleted.
    compute_pvalue : bool
        If False, skip the SciPy t-test (faster; p_value will be None).
    alpha : float
        Significance level for the confidence interval (default 0.05 → 95% CI).

    Returns
    -------
    CorrelationResult
        .r, .p_value, .ci_low, .ci_high (Fisher z), .df = n-2, .z_stat = t

    Notes
    -----
    CI uses Fisher's z-transformation:
        z = arctanh(r)
        SE(z) = 1 / sqrt(n - 3)
        CI = tanh(z ± z_crit * SE)
    """
    x, y = _clean_paired(x, y)
    n    = len(x)
    if n < 3:
        raise ValueError(f"pearson() requires n >= 3, got n={n}.")

    r = float(np.corrcoef(x, y)[0, 1])
    r = np.clip(r, -1 + 1e-12, 1 - 1e-12)   # guard arctanh domain

    p_value = z_stat = None
    if compute_pvalue:
        _require_scipy("pearson(compute_pvalue=True)")
        r_sc, p_value = _sp.pearsonr(x, y)
        z_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r**2)

    # Fisher z CI
    ci_low = ci_high = None
    if n >= 4:
        z       = math.atanh(r)
        se_z    = 1.0 / math.sqrt(n - 3)
        if _SCIPY_AVAILABLE:
            z_crit  = float(_sp.norm.ppf(1 - alpha / 2))
        else:
            z_crit  = 1.959963985   # 97.5th percentile of N(0,1)
        ci_low  = float(math.tanh(z - z_crit * se_z))
        ci_high = float(math.tanh(z + z_crit * se_z))

    return CorrelationResult(
        r       = r,
        method  = CorrelationMethod.PEARSON,
        n       = n,
        p_value = p_value,
        ci_low  = ci_low,
        ci_high = ci_high,
        df      = n - 2,
        z_stat  = z_stat,
    )


# ---------------------------------------------------------------------------
# A.2  Spearman rho
# ---------------------------------------------------------------------------

def spearman(
    x:              np.ndarray,
    y:              np.ndarray,
    compute_pvalue: bool = True,
) -> CorrelationResult:
    """
    Compute Spearman's rank correlation.

    Uses the standard formula on rank-transformed data:
        rho = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))

    where d_i = rank(x_i) - rank(y_i).  Ties are broken with average ranks.
    For n >= 10 the t approximation t = rho * sqrt((n-2)/(1-rho^2)) is used
    for the p-value.

    Returns
    -------
    CorrelationResult
        .r = rho, .df = n-2, .z_stat = t approximation (for n >= 10)
    """
    x, y = _clean_paired(x, y)
    n    = len(x)
    if n < 3:
        raise ValueError(f"spearman() requires n >= 3, got n={n}.")

    rx   = _rankdata(x)
    ry   = _rankdata(y)
    d_sq = np.sum((rx - ry) ** 2)
    rho  = float(1.0 - 6.0 * d_sq / (n * (n**2 - 1)))
    rho  = np.clip(rho, -1.0, 1.0)

    p_value = z_stat = None
    if compute_pvalue:
        _require_scipy("spearman(compute_pvalue=True)")
        _, p_value = _sp.spearmanr(x, y)
        if n >= 10:
            t_val   = rho * math.sqrt((n - 2) / (1.0 - rho**2 + 1e-14))
            z_stat  = t_val

    return CorrelationResult(
        r       = rho,
        method  = CorrelationMethod.SPEARMAN,
        n       = n,
        p_value = p_value,
        df      = n - 2,
        z_stat  = z_stat,
    )


# ---------------------------------------------------------------------------
# A.3  Kendall's tau-b
# ---------------------------------------------------------------------------

def kendall(
    x:              np.ndarray,
    y:              np.ndarray,
    compute_pvalue: bool = True,
) -> CorrelationResult:
    """
    Compute Kendall's tau-b.

    tau-b adjusts for ties in both x and y, making it the most appropriate
    Kendall variant for discrete or tied data.

    Returns
    -------
    CorrelationResult
        .r = tau_b, .z_stat = z approximation
    """
    x, y = _clean_paired(x, y)
    n    = len(x)
    if n < 3:
        raise ValueError(f"kendall() requires n >= 3, got n={n}.")

    p_value = z_stat = None
    if compute_pvalue:
        _require_scipy("kendall(compute_pvalue=True)")
        tau, p_value = _sp.kendalltau(x, y)
    else:
        tau = _kendall_tau_b_numpy(x, y)

    if not compute_pvalue:
        tau = _kendall_tau_b_numpy(x, y)

    # z approximation (valid for n >= 10)
    if n >= 10:
        v    = 2.0 * (2 * n + 5) / (9 * n * (n - 1))
        z_stat = float(tau) / math.sqrt(v) if v > 0 else None

    return CorrelationResult(
        r       = float(tau),
        method  = CorrelationMethod.KENDALL,
        n       = n,
        p_value = p_value,
        z_stat  = z_stat,
    )


def _kendall_tau_b_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Pure NumPy O(n log n) Kendall tau-b via merge-sort concordance count."""
    n = len(x)
    order  = np.argsort(x, stable=True)
    y_sort = y[order]

    # Count concordant (C) and discordant (D) pairs by merge-sort on y_sort
    C, D, tx, ty = 0, 0, 0, 0

    def _merge_count(arr):
        nonlocal D
        if len(arr) <= 1:
            return arr
        mid   = len(arr) // 2
        left  = _merge_count(arr[:mid])
        right = _merge_count(arr[mid:])
        merged, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                D += len(right) - j
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    _merge_count(list(y_sort))

    P        = n * (n - 1) // 2
    C        = P - D
    # Tie corrections
    tx_vals  = np.unique(x, return_counts=True)[1]
    ty_vals  = np.unique(y, return_counts=True)[1]
    tx       = np.sum(tv * (tv - 1) // 2 for tv in tx_vals)
    ty       = np.sum(tv * (tv - 1) // 2 for tv in ty_vals)
    denom    = math.sqrt((P - tx) * (P - ty))
    return float((C - D) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# A.4  Point-biserial r
# ---------------------------------------------------------------------------

def point_biserial(
    continuous:     np.ndarray,
    binary:         np.ndarray,
    compute_pvalue: bool = True,
) -> CorrelationResult:
    """
    Compute the point-biserial correlation between a continuous and a
    binary (0/1) variable.

    Mathematically equivalent to Pearson r between the two variables but
    has a simpler closed-form:

        r_pb = (M1 - M0) / s_n  *  sqrt(n1 * n0 / n^2)

    where M1, M0 are group means, s_n the pooled (population) SD, n1 n0 group sizes.

    Parameters
    ----------
    continuous : array-like
        Continuous variable.
    binary : array-like
        Binary variable with values in {0, 1}.  Will be coerced to int.

    Returns
    -------
    CorrelationResult
    """
    c, b = _clean_paired(continuous, np.asarray(binary, dtype=float))
    n    = len(c)

    unique_b = np.unique(b)
    if not np.array_equal(np.sort(unique_b), [0.0, 1.0]):
        # Recode to 0/1 by rank of unique values
        b = (b == np.max(unique_b)).astype(float)

    n1 = int(np.sum(b == 1))
    n0 = int(np.sum(b == 0))
    if n1 == 0 or n0 == 0:
        raise ValueError("point_biserial() requires both groups (0 and 1) to be non-empty.")

    M1   = float(np.mean(c[b == 1]))
    M0   = float(np.mean(c[b == 0]))
    s_n  = float(np.std(c, ddof=0))    # population SD
    r_pb = (M1 - M0) / s_n * math.sqrt(n1 * n0 / n**2)

    p_value = z_stat = None
    if compute_pvalue:
        _require_scipy("point_biserial(compute_pvalue=True)")
        _, p_value = _sp.pointbiserialr(b, c)
        z_stat = r_pb * math.sqrt(n - 2) / math.sqrt(1 - r_pb**2 + 1e-14)

    return CorrelationResult(
        r       = r_pb,
        method  = CorrelationMethod.POINT_BISERIAL,
        n       = n,
        p_value = p_value,
        df      = n - 2,
        z_stat  = z_stat,
    )


# ---------------------------------------------------------------------------
# A.5  Phi coefficient (2×2 contingency tables)
# ---------------------------------------------------------------------------

def phi_coefficient(
    table:          np.ndarray,
    compute_pvalue: bool = True,
) -> CorrelationResult:
    """
    Compute the phi coefficient for a 2x2 contingency table.

    phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))

    Parameters
    ----------
    table : ndarray of shape (2, 2)
        Contingency table [[a, b], [c, d]].

    Returns
    -------
    CorrelationResult
        .r = phi, .n = total cell count
    """
    table = np.asarray(table, dtype=float)
    if table.shape != (2, 2):
        raise ValueError(
            f"phi_coefficient() requires a (2,2) table, got {table.shape}."
        )
    a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
    n    = int(np.sum(table))
    denom = math.sqrt((a+b) * (c+d) * (a+c) * (b+d))
    phi  = float((a * d - b * c) / denom) if denom > 0 else 0.0

    p_value = None
    if compute_pvalue:
        _require_scipy("phi_coefficient(compute_pvalue=True)")
        chi2, p_value, _, _ = _sp.chi2_contingency(table, correction=False)

    return CorrelationResult(
        r       = phi,
        method  = CorrelationMethod.PHI,
        n       = n,
        p_value = p_value,
    )


# ---------------------------------------------------------------------------
# A.6  Cramér's V (r×c contingency tables)
# ---------------------------------------------------------------------------

def cramers_v(
    table:          np.ndarray,
    compute_pvalue: bool = True,
    bias_corrected: bool = True,
) -> CorrelationResult:
    """
    Compute Cramér's V for an r×c contingency table.

    Standard V = sqrt(chi^2 / (n * (k-1)))   where k = min(r, c).

    Bias-corrected version (Bergsma 2013):
        phi_c^2 = max(0, chi^2/n - (k-1)/(n-1))
        V_c     = sqrt(phi_c^2 / (k_tilde - 1))
        k_tilde = k - (k-1)^2 / (n-1)

    Parameters
    ----------
    table : array-like
        r × c contingency table of observed counts.
    bias_corrected : bool
        Apply the Bergsma–Wicher bias correction (default True).

    Returns
    -------
    CorrelationResult
        .r = V (0–1 unsigned), .n = total count
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2:
        raise ValueError("cramers_v() requires a 2-D contingency table.")
    n     = float(np.sum(table))
    r, c  = table.shape
    k     = min(r, c)

    _require_scipy("cramers_v")
    chi2, p_value, _, _ = _sp.chi2_contingency(table)

    if bias_corrected:
        phi2      = max(0.0, chi2 / n - (k - 1) / (n - 1))
        k_tilde   = k - (k - 1)**2 / (n - 1)
        r_tilde   = r - (r - 1)**2 / (n - 1)
        V         = math.sqrt(phi2 / (min(k_tilde, r_tilde) - 1)) if min(k_tilde, r_tilde) > 1 else 0.0
    else:
        V = math.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0.0

    return CorrelationResult(
        r       = V,
        method  = CorrelationMethod.CRAMERS_V,
        n       = int(n),
        p_value = p_value if compute_pvalue else None,
    )


# ---------------------------------------------------------------------------
# A.7  Partial correlation
# ---------------------------------------------------------------------------

@dataclass
class PartialCorrelationResult(CorrelationResult):
    """
    Extension of CorrelationResult for partial / semi-partial correlations.

    Additional attributes
    ---------------------
    control_vars : list[str]
        Names of the variables that were partialled out.
    x_name, y_name : str
        Names of the two focal variables.
    semi_partial : bool
        True if this is a semi-partial (part) correlation rather than partial.
    r_squared_full : float or None
        R^2 of the full model (for computing the unique variance contribution).
    r_squared_without_x : float or None
        R^2 of the model without the focal X variable.
    unique_variance : float or None
        r_squared_full - r_squared_without_x  (unique contribution of X).
    """
    control_vars:         List[str]      = field(default_factory=list)
    x_name:               str            = "X"
    y_name:               str            = "Y"
    semi_partial:         bool           = False
    r_squared_full:       Optional[float] = None
    r_squared_without_x:  Optional[float] = None

    @property
    def unique_variance(self) -> Optional[float]:
        if self.r_squared_full is None or self.r_squared_without_x is None:
            return None
        return self.r_squared_full - self.r_squared_without_x

    def summary(self) -> str:
        kind     = "semi-partial" if self.semi_partial else "partial"
        ctrl_str = ", ".join(self.control_vars) or "none"
        base     = super().summary()
        return f"{base}  [{kind}, controlling for: {ctrl_str}]"


def partial_correlation(
    x:               np.ndarray,
    y:               np.ndarray,
    controls:        np.ndarray,
    x_name:          str  = "X",
    y_name:          str  = "Y",
    control_names:   Optional[List[str]] = None,
    compute_pvalue:  bool = True,
    semi_partial:    bool = False,
) -> PartialCorrelationResult:
    """
    Compute the partial (or semi-partial) correlation between X and Y,
    controlling for one or more covariates Z.

    Method
    ------
    1. Regress X on Z → residuals e_X
    2. Regress Y on Z → residuals e_Y
    3. Partial r = Pearson(e_X, e_Y)

    For semi-partial (part) correlation, only X is residualised:
    3. Semi-partial r = Pearson(e_X, Y)

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Focal variables.
    controls : array-like, shape (n,) or (n, k)
        Control variable(s).  Automatically prepended with an intercept column.
    x_name, y_name : str
        Labels used in the result summary and formula.
    control_names : list[str], optional
        Labels for control variables.
    compute_pvalue : bool
    semi_partial : bool
        If True, compute the semi-partial (part) correlation.

    Returns
    -------
    PartialCorrelationResult
    """
    x  = np.asarray(x, dtype=float).ravel()
    y  = np.asarray(y, dtype=float).ravel()
    Z  = np.asarray(controls, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n  = len(x)

    # Add intercept
    Zc = np.column_stack([np.ones(n), Z])

    def _residualise(v: np.ndarray) -> np.ndarray:
        beta, _, _, _ = np.linalg.lstsq(Zc, v, rcond=None)
        return v - Zc @ beta

    e_x = _residualise(x)
    e_y = _residualise(y) if not semi_partial else y

    result = pearson(e_x, e_y, compute_pvalue=compute_pvalue)

    ctrl_names = control_names or [f"Z{i+1}" for i in range(Z.shape[1])]

    return PartialCorrelationResult(
        r             = result.r,
        method        = CorrelationMethod.PARTIAL,
        n             = n,
        p_value       = result.p_value,
        ci_low        = result.ci_low,
        ci_high       = result.ci_high,
        df            = n - 2 - Z.shape[1],
        z_stat        = result.z_stat,
        control_vars  = ctrl_names,
        x_name        = x_name,
        y_name        = y_name,
        semi_partial  = semi_partial,
    )


# ---------------------------------------------------------------------------
# A.8  Correlation matrix
# ---------------------------------------------------------------------------

def correlation_matrix(
    data:           np.ndarray,
    method:         CorrelationMethod = CorrelationMethod.PEARSON,
    compute_pvalue: bool = True,
    var_names:      Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute a full p×p correlation matrix.

    Parameters
    ----------
    data : array-like, shape (n, p)
        Each column is a variable.
    method : CorrelationMethod
        Only PEARSON and SPEARMAN are supported for full matrices.
    compute_pvalue : bool
    var_names : list[str], optional
        Column labels.  Defaults to ["X1", "X2", …].

    Returns
    -------
    (R, P, names)
        R : ndarray (p, p)  — correlation coefficients, 1 on diagonal
        P : ndarray (p, p)  — p-values; NaN on diagonal; None-matrix if not computed
        names : list[str]   — variable names
    """
    data  = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, p  = data.shape
    names = var_names or [f"X{i+1}" for i in range(p)]

    R = np.ones((p, p))
    P = np.full((p, p), np.nan)

    _corr_fn = pearson if method == CorrelationMethod.PEARSON else spearman

    for i in range(p):
        for j in range(i + 1, p):
            res = _corr_fn(
                data[:, i], data[:, j],
                compute_pvalue=compute_pvalue,
            )
            R[i, j] = R[j, i] = res.r
            if res.p_value is not None:
                P[i, j] = P[j, i] = res.p_value

    if not compute_pvalue:
        P = None  # type: ignore

    return R, P, names


# ===========================================================================
# LAYER B — Regression fitting and diagnostics
# ===========================================================================

@dataclass
class RegressionResult:
    """
    Structured output of an OLS regression fit.

    Attributes
    ----------
    beta : ndarray, shape (k,)
        Estimated coefficients.  beta[0] is the intercept if ``fit_intercept=True``.
    se : ndarray, shape (k,)
        Standard errors of each coefficient.
    t_stat : ndarray, shape (k,)
        t-statistics: beta / se.
    p_values : ndarray, shape (k,)
        Two-tailed p-values for each coefficient.
    r_squared : float
        Coefficient of determination R^2.
    adj_r_squared : float
        Adjusted R^2 = 1 - (1-R^2) * (n-1) / (n-k-1).
    f_stat : float
        F-statistic for overall model fit.
    f_p_value : float
        p-value of the F-test.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    fitted : ndarray, shape (n,)
        Predicted values y_hat.
    residuals : ndarray, shape (n,)
        Ordinary residuals y - y_hat.
    x : ndarray, shape (n, k)
        Design matrix (with intercept column if fit_intercept=True).
    y : ndarray, shape (n,)
        Response variable.
    feature_names : list[str]
        Names for each column of X (excluding intercept).
    fit_intercept : bool
    n : int
        Sample size.
    k : int
        Number of predictors (excluding intercept).
    """

    beta:          np.ndarray
    se:            np.ndarray
    t_stat:        np.ndarray
    p_values:      np.ndarray
    r_squared:     float
    adj_r_squared: float
    f_stat:        float
    f_p_value:     float
    aic:           float
    bic:           float
    fitted:        np.ndarray
    residuals:     np.ndarray
    x:             np.ndarray
    y:             np.ndarray
    feature_names: List[str]
    fit_intercept: bool
    n:             int
    k:             int

    # ------------------------------------------------------------------
    # Derived diagnostic quantities
    # ------------------------------------------------------------------

    @property
    def rss(self) -> float:
        """Residual sum of squares."""
        return float(np.sum(self.residuals**2))

    @property
    def tss(self) -> float:
        """Total sum of squares."""
        return float(np.sum((self.y - self.y.mean())**2))

    @property
    def mse(self) -> float:
        """Mean squared error of residuals."""
        df = self.n - self.k - (1 if self.fit_intercept else 0)
        return self.rss / df if df > 0 else float("nan")

    @property
    def rmse(self) -> float:
        """Root mean squared error."""
        return math.sqrt(self.mse)

    @property
    def sigma_hat(self) -> float:
        """Estimated residual standard deviation (same as rmse)."""
        return self.rmse

    @property
    def hat_matrix_diag(self) -> np.ndarray:
        """
        Diagonal of the hat matrix H = X(X'X)^-1 X'.
        Leverage values for each observation.
        """
        try:
            XtX_inv = np.linalg.inv(self.x.T @ self.x)
            return np.einsum("ij,jk,ki->i", self.x, XtX_inv, self.x.T)
        except np.linalg.LinAlgError:
            return np.full(self.n, float("nan"))

    def coef_table(self, digits: int = 4) -> str:
        """
        Return a formatted coefficient table string, e.g.::

            Coeff         Estimate    SE        t       p
            (Intercept)   0.0581    0.0893    0.651   0.5203
            X1            0.6268    0.0916    6.843   0.0000 ***

        Parameters
        ----------
        digits : int
            Decimal places for each numeric column.

        Returns
        -------
        str
        """
        names = (["(Intercept)"] if self.fit_intercept else []) + list(self.feature_names)
        rows  = [f"{'Coeff':<18} {'Estimate':>10} {'SE':>10} {'t':>10} {'p':>12}"]
        rows.append("-" * 64)
        for name, b, se, t, p in zip(names, self.beta, self.se, self.t_stat, self.p_values):
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            rows.append(
                f"{name:<18} {b:>10.{digits}f} {se:>10.{digits}f} "
                f"{t:>10.{digits}f} {p:>12.4g} {stars}"
            )
        rows.append("-" * 64)
        rows.append(f"R^2={self.r_squared:.4f}  Adj R^2={self.adj_r_squared:.4f}  "
                    f"F({self.k},{self.n-self.k-1})={self.f_stat:.4f}  p={self.f_p_value:.4g}  "
                    f"AIC={self.aic:.2f}  BIC={self.bic:.2f}")
        return "\n".join(rows)

    def to_formula(self) -> "TexFormula":
        """
        Return a TexFormula for the fitted equation, e.g.::

            hat{Y} = 0.058 + 0.627 X_1
        """
        if not _TEX_AVAILABLE:
            raise ImportError("to_formula() requires manim_stats.core.tex_utils.")
        terms = []
        b_arr = list(self.beta)
        if self.fit_intercept:
            intercept = b_arr.pop(0)
            terms.append(f"{intercept:.4f}")
        for i, (b, name) in enumerate(zip(b_arr, self.feature_names)):
            sign = "+" if b >= 0 else "-"
            terms.append(f"{sign} {abs(b):.4f}\\,{name}")
        raw = r"\hat{Y} = " + " ".join(terms)
        return TexFormula(
            name        = "regression_fit",
            raw         = raw,
            description = f"Fitted OLS: R^2={self.r_squared:.4f}",
            parts       = {"lhs": r"\hat{Y}"},
            tags        = ["regression", "ols", "fitted"],
        )

    def __repr__(self) -> str:
        return (
            f"RegressionResult("
            f"n={self.n}, k={self.k}, "
            f"R^2={self.r_squared:.4f}, "
            f"RMSE={self.rmse:.4f})"
        )


@dataclass
class InfluenceMeasures:
    """
    Diagnostic influence measures for each observation in an OLS fit.

    All arrays have length n (one value per observation).

    Attributes
    ----------
    leverage : ndarray
        Hat-matrix diagonal h_ii.  High leverage: h_ii > 2k/n.
    studentized_residuals : ndarray
        Internally studentized (standardised) residuals.
    externally_studentized : ndarray
        Externally studentized residuals (leave-one-out SD).
    cooks_d : ndarray
        Cook's distance: measures influence of each observation on all fitted values.
    dffits : ndarray
        DFFITS: scaled change in fitted value when observation is deleted.
    dfbetas : ndarray, shape (n, k)
        Change in each coefficient when observation is deleted (scaled by SE).
    """

    leverage:                np.ndarray
    studentized_residuals:   np.ndarray
    externally_studentized:  np.ndarray
    cooks_d:                 np.ndarray
    dffits:                  np.ndarray
    dfbetas:                 np.ndarray   # shape (n, k)

    # ------------------------------------------------------------------
    # Outlier / high-influence masks
    # ------------------------------------------------------------------

    def high_leverage_mask(self, threshold: Optional[float] = None, k: int = 1, n: int = 1) -> np.ndarray:
        """
        Boolean mask of high-leverage observations.

        Default threshold: 2 * (k+1) / n  (Huber's rule).
        """
        thresh = threshold if threshold is not None else 2 * (k + 1) / max(n, 1)
        return self.leverage > thresh

    def outlier_mask(self, threshold: float = 2.5) -> np.ndarray:
        """Boolean mask: |externally studentized residual| > threshold."""
        return np.abs(self.externally_studentized) > threshold

    def influential_mask(self, cooks_threshold: float = 1.0) -> np.ndarray:
        """Boolean mask: Cook's D > threshold (default 1.0, conservative 4/n also common)."""
        return self.cooks_d > cooks_threshold

    def concerning_obs(
        self,
        leverage_mult: float  = 2.0,
        outlier_thresh: float = 2.5,
        cooks_thresh:   float = 1.0,
        k:              int   = 1,
        n:              int   = 1,
    ) -> np.ndarray:
        """
        Return indices of observations that are high-leverage AND/OR outliers
        AND/OR influential (Cook's D).
        """
        lev   = self.high_leverage_mask(threshold=leverage_mult*(k+1)/max(n,1), k=k, n=n)
        out   = self.outlier_mask(outlier_thresh)
        inf   = self.influential_mask(cooks_thresh)
        return np.where(lev | out | inf)[0]

    def __repr__(self) -> str:
        n_inf  = int(np.sum(self.influential_mask()))
        n_lev  = int(np.sum(self.leverage > 0.5))
        n_out  = int(np.sum(self.outlier_mask()))
        return (
            f"InfluenceMeasures("
            f"n_influential={n_inf}, "
            f"n_high_leverage={n_lev}, "
            f"n_outliers={n_out})"
        )


# ---------------------------------------------------------------------------
# B.1  OLS fitting
# ---------------------------------------------------------------------------

def ols_fit(
    x:               np.ndarray,
    y:               np.ndarray,
    fit_intercept:   bool = True,
    feature_names:   Optional[List[str]] = None,
    compute_pvalue:  bool = True,
) -> RegressionResult:
    """
    Ordinary Least Squares regression via the normal equations.

    Solves beta = (X'X)^{-1} X'y using ``np.linalg.lstsq`` for numerical
    stability.  When X is near-singular a PseudoInverse fallback is used
    with a UserWarning.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, k)
        Predictor matrix.  A 1-D array is treated as a single predictor.
    y : array-like, shape (n,)
        Response variable.
    fit_intercept : bool
        If True (default), prepend a column of ones to X.
    feature_names : list[str], optional
        Names for each column of X (before intercept is added).
    compute_pvalue : bool
        If True (default), compute t-test p-values via SciPy.

    Returns
    -------
    RegressionResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    k = x.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(k)]

    # Build design matrix
    X = np.column_stack([np.ones(n), x]) if fit_intercept else x

    # Solve
    try:
        XtX     = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        beta    = XtX_inv @ X.T @ y
    except np.linalg.LinAlgError:
        warnings.warn(
            "X'X is singular; falling back to pseudoinverse (lstsq).",
            UserWarning, stacklevel=2,
        )
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        XtX_inv       = np.linalg.pinv(X.T @ X)

    y_hat   = X @ beta
    resid   = y - y_hat
    rss     = float(np.sum(resid**2))
    tss     = float(np.sum((y - y.mean())**2))
    r2      = 1.0 - rss / tss if tss > 0 else 0.0

    # Degrees of freedom
    df_res  = n - k - (1 if fit_intercept else 0)
    df_reg  = k
    s2      = rss / df_res if df_res > 0 else float("nan")
    var_beta = np.diag(XtX_inv) * s2
    se       = np.sqrt(np.abs(var_beta))
    t_stat   = beta / np.where(se > 0, se, float("nan"))

    # Adjusted R^2
    adj_r2   = 1.0 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")

    # F-statistic
    msr      = (tss - rss) / df_reg if df_reg > 0 else float("nan")
    f_stat   = msr / s2 if s2 > 0 else float("nan")

    # Information criteria (using log-likelihood of normal errors)
    ll       = -0.5 * n * (1 + math.log(2 * math.pi) + math.log(s2)) if s2 > 0 else float("nan")
    n_params = len(beta)
    aic      = -2 * ll + 2 * n_params
    bic      = -2 * ll + math.log(n) * n_params

    # p-values
    if compute_pvalue and _SCIPY_AVAILABLE:
        p_vals  = 2 * (1 - _sp.t.cdf(np.abs(t_stat), df=df_res))
        f_pval  = float(1 - _sp.f.cdf(f_stat, df_res, df_res)) if not math.isnan(f_stat) else float("nan")
    else:
        p_vals  = np.full_like(t_stat, float("nan"))
        f_pval  = float("nan")

    return RegressionResult(
        beta          = beta,
        se            = se,
        t_stat        = t_stat,
        p_values      = p_vals,
        r_squared     = r2,
        adj_r_squared = adj_r2,
        f_stat        = float(f_stat),
        f_p_value     = f_pval,
        aic           = float(aic),
        bic           = float(bic),
        fitted        = y_hat,
        residuals     = resid,
        x             = X,
        y             = y,
        feature_names = names,
        fit_intercept = fit_intercept,
        n             = n,
        k             = k,
    )


# ---------------------------------------------------------------------------
# B.2  Ridge regression
# ---------------------------------------------------------------------------

def ridge_fit(
    x:             np.ndarray,
    y:             np.ndarray,
    alpha:         float = 1.0,
    fit_intercept: bool  = True,
    feature_names: Optional[List[str]] = None,
) -> RegressionResult:
    """
    Ridge regression (L2-penalised OLS).

    Solves:  beta = (X'X + alpha * I)^{-1} X'y

    The intercept is never penalised: alpha is applied only to the slope
    coefficients regardless of ``fit_intercept``.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, k)
    y : array-like, shape (n,)
    alpha : float
        Regularisation strength (0 → OLS).
    fit_intercept : bool
    feature_names : list[str], optional

    Returns
    -------
    RegressionResult
        Note: .p_values and .t_stat are NaN (ridge has no exact distribution).
    """
    x  = np.asarray(x, dtype=float)
    y  = np.asarray(y, dtype=float).ravel()
    n  = len(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    k     = x.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(k)]

    # Center X and y when fitting intercept
    if fit_intercept:
        x_mean  = x.mean(axis=0)
        y_mean  = y.mean()
        xc      = x - x_mean
        yc      = y - y_mean
    else:
        xc, yc  = x, y
        x_mean  = np.zeros(k)
        y_mean  = 0.0

    # Ridge normal equations (applied to centred data, no penalty on intercept)
    A      = xc.T @ xc + alpha * np.eye(k)
    slopes = np.linalg.solve(A, xc.T @ yc)

    if fit_intercept:
        intercept = y_mean - x_mean @ slopes
        beta      = np.concatenate([[intercept], slopes])
        X_design  = np.column_stack([np.ones(n), x])
    else:
        beta      = slopes
        X_design  = x

    y_hat   = X_design @ beta
    resid   = y - y_hat
    rss     = float(np.sum(resid**2))
    tss     = float(np.sum((y - y.mean())**2))
    r2      = 1.0 - rss / tss if tss > 0 else 0.0
    df_res  = n - k - (1 if fit_intercept else 0)
    s2      = rss / df_res if df_res > 0 else float("nan")
    adj_r2  = 1.0 - (1 - r2) * (n - 1) / max(1, n - k - 1)
    ll      = -0.5 * n * (1 + math.log(2 * math.pi) + math.log(s2)) if s2 > 0 else float("nan")
    n_p     = len(beta)
    aic     = -2 * ll + 2 * n_p
    bic     = -2 * ll + math.log(n) * n_p

    nan_arr = np.full(len(beta), float("nan"))
    return RegressionResult(
        beta          = beta,
        se            = nan_arr.copy(),
        t_stat        = nan_arr.copy(),
        p_values      = nan_arr.copy(),
        r_squared     = r2,
        adj_r_squared = adj_r2,
        f_stat        = float("nan"),
        f_p_value     = float("nan"),
        aic           = float(aic),
        bic           = float(bic),
        fitted        = y_hat,
        residuals     = resid,
        x             = X_design,
        y             = y,
        feature_names = names,
        fit_intercept = fit_intercept,
        n             = n,
        k             = k,
    )


# ---------------------------------------------------------------------------
# B.3  Influence measures
# ---------------------------------------------------------------------------

def influence_measures(result: RegressionResult) -> InfluenceMeasures:
    """
    Compute full set of regression influence diagnostics for a fitted OLS model.

    Computed quantities
    -------------------
    leverage  : hat-matrix diagonal h_ii
    studentized_residuals : e_i / (s * sqrt(1 - h_ii))
    externally_studentized : e_i / (s_{-i} * sqrt(1 - h_ii))
        where s_{-i} is leave-one-out residual SD — computed via the
        Sherman-Morrison formula (no need to refit n times).
    cooks_d   : h_ii * student_i^2 / ((k+1) * (1-h_ii))
    dffits    : student_i * sqrt(h_ii / (1-h_ii))
    dfbetas   : change in standardised beta_j when obs i is deleted

    Parameters
    ----------
    result : RegressionResult
        Must have been fitted with ``ols_fit`` (ridge not supported).

    Returns
    -------
    InfluenceMeasures
    """
    X     = result.x
    y     = result.y
    resid = result.residuals
    n, p  = X.shape
    df    = n - p

    # Leverage
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    h = np.einsum("ij,jk,ki->i", X, XtX_inv, X.T)
    h = np.clip(h, 0.0, 1.0 - 1e-10)

    # Studentized residuals
    s2          = float(np.sum(resid**2) / df) if df > 0 else float("nan")
    student     = resid / (math.sqrt(s2) * np.sqrt(1 - h))

    # Externally studentized (leave-one-out) via Sherman–Morrison
    # s_{-i}^2 = (df * s^2 - resid_i^2/(1-h_ii)) / (df-1)
    s2_loo      = (df * s2 - resid**2 / (1 - h)) / (df - 1)
    s2_loo      = np.clip(s2_loo, 1e-14, None)
    ext_student = resid / (np.sqrt(s2_loo) * np.sqrt(1 - h))

    # Cook's D
    cooks = (student**2 / p) * (h / (1 - h))

    # DFFITS
    dffits = ext_student * np.sqrt(h / (1 - h))

    # DFBETAS: (X'X)^{-1} x_i e_i / (s_{-i} * sqrt((X'X)^{-1}_{jj}))
    # Shape: (n, p)
    C           = XtX_inv @ X.T                      # (p, n)
    beta_change = C.T * resid[:, None] / (1 - h)[:, None]   # (n, p)
    se_beta     = np.sqrt(np.diag(XtX_inv) * s2)
    dfbetas     = beta_change / (np.sqrt(s2_loo)[:, None] * se_beta[None, :])

    return InfluenceMeasures(
        leverage               = h,
        studentized_residuals  = student,
        externally_studentized = ext_student,
        cooks_d                = cooks,
        dffits                 = dffits,
        dfbetas                = dfbetas,
    )


# ===========================================================================
# LAYER D — Manim mobjects
# ===========================================================================

class CorrelationEllipse3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    A confidence / concentration ellipse for a bivariate (X, Y) cloud.

    The ellipse represents the contour of the bivariate normal distribution
    implied by the observed means, SDs, and correlation.  At ``level=0.95``
    it encloses ~95% of the probability mass.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Data points.
    level : float
        Probability content of the ellipse.  Default 0.95.
    n_points : int
        Number of points on the ellipse boundary.  Default 100.
    stroke_color : ManimColor, optional
    fill_color : ManimColor, optional
    fill_opacity : float
    stroke_width : float

    Key sub-mobjects
    ----------------
    .ellipse   : VMobject — the ellipse boundary
    .axes_lines: VGroup   — the two principal axes (optional, toggleable)
    .center_dot: Dot      — centre of the ellipse

    Animations
    ----------
    .animate_grow(run_time)     — scale from zero
    .morph_to(r, run_time)      — morph the shape as r changes continuously
    """

    def __init__(
        self,
        x:             np.ndarray,
        y:             np.ndarray,
        level:         float = 0.95,
        n_points:      int   = 120,
        stroke_color   = None,
        fill_color     = None,
        fill_opacity:  float = 0.18,
        stroke_width:  float = 2.5,
        show_axes:     bool  = False,
        **kwargs,
    ) -> None:
        _require_manim("CorrelationEllipse3D")
        super().__init__(**kwargs)

        x    = np.asarray(x, dtype=float)
        y    = np.asarray(y, dtype=float)
        self._x = x
        self._y = y
        self._level    = level
        self._n_points = n_points

        # Pick colors from theme if not given
        if _COLORS_AVAILABLE:
            stroke_color = stroke_color or ManimColor(REGRESSION_FAMILY.base.hex)
            fill_color   = fill_color   or ManimColor(REGRESSION_FAMILY.light.hex)
        else:
            stroke_color = stroke_color or PURPLE
            fill_color   = fill_color   or PURPLE

        self._stroke_color  = stroke_color
        self._fill_color    = fill_color
        self._fill_opacity  = fill_opacity
        self._stroke_width  = stroke_width

        self._build(x, y, level, n_points, show_axes)

    # ------------------------------------------------------------------
    # Internal geometry builders
    # ------------------------------------------------------------------

    def _ellipse_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        level: float,
        n: int,
    ) -> np.ndarray:
        """
        Return (n, 3) array of 3-D points on the confidence ellipse.

        The ellipse is derived from the eigendecomposition of the 2×2
        covariance matrix:
            Sigma = [[Var(X), Cov(X,Y)], [Cov(X,Y), Var(Y)]]
        scaled by chi2(2) quantile to achieve the desired probability level.
        """
        mu_x, mu_y = float(np.mean(x)), float(np.mean(y))
        cov        = np.cov(x, y)    # 2x2
        if cov.shape != (2, 2) or np.any(np.isnan(cov)):
            # Degenerate: return a tiny circle
            t = np.linspace(0, 2 * math.pi, n, endpoint=False)
            return np.column_stack([0.01 * np.cos(t), 0.01 * np.sin(t), np.zeros(n)])

        # Chi-squared quantile (2 DOF)
        if _SCIPY_AVAILABLE:
            chi2_q = float(_sp.chi2.ppf(level, df=2))
        else:
            # Closed form: chi2(2) quantile = -2 ln(1 - p)
            chi2_q = -2.0 * math.log(1.0 - level)

        # Eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            t = np.linspace(0, 2 * math.pi, n, endpoint=False)
            return np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])

        eigvals = np.maximum(eigvals, 1e-12)
        # Semi-axes
        a = math.sqrt(chi2_q * eigvals[1])   # larger eigenvalue
        b = math.sqrt(chi2_q * eigvals[0])   # smaller eigenvalue

        t      = np.linspace(0, 2 * math.pi, n, endpoint=False)
        circle = np.column_stack([a * np.cos(t), b * np.sin(t)])
        # Rotate by eigenvector angle
        R      = eigvecs[:, ::-1]   # columns are eigvecs, largest first
        pts    = (R @ circle.T).T
        pts[:, 0] += mu_x
        pts[:, 1] += mu_y

        return np.column_stack([pts, np.zeros(n)])

    def _build(
        self,
        x: np.ndarray, y: np.ndarray,
        level: float, n_points: int,
        show_axes: bool,
    ) -> None:
        pts_3d  = self._ellipse_points(x, y, level, n_points)

        # Build as a closed VMobject polygon
        from manim import VMobject
        self.ellipse = VMobject(
            fill_color    = self._fill_color,
            fill_opacity  = self._fill_opacity,
            stroke_color  = self._stroke_color,
            stroke_width  = self._stroke_width,
        )
        self.ellipse.set_points_as_corners(
            [*pts_3d, pts_3d[0]]   # close the loop
        )
        self.add(self.ellipse)

        # Centre dot
        cx, cy = float(np.mean(x)), float(np.mean(y))
        self.center_dot = Dot(
            point  = [cx, cy, 0.0],
            color  = self._stroke_color,
            radius = 0.05,
        )
        self.add(self.center_dot)

        # Principal axes (optional)
        if show_axes:
            self._add_principal_axes(x, y, level)

    def _add_principal_axes(
        self, x: np.ndarray, y: np.ndarray, level: float
    ) -> None:
        cov     = np.cov(x, y)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if _SCIPY_AVAILABLE:
            chi2_q = float(_sp.chi2.ppf(level, df=2))
        else:
            chi2_q = -2.0 * math.log(1.0 - level)
        cx, cy  = float(np.mean(x)), float(np.mean(y))
        center  = np.array([cx, cy, 0.0])
        self.axes_lines = VGroup()
        for i in range(2):
            scale   = math.sqrt(chi2_q * max(eigvals[i], 1e-12))
            end_vec = eigvecs[:, i] * scale
            line    = DashedLine(
                start       = center - np.array([*end_vec, 0.0]),
                end         = center + np.array([*end_vec, 0.0]),
                color       = self._stroke_color,
                stroke_width= 1.2,
                dash_length = 0.06,
            )
            self.axes_lines.add(line)
        self.add(self.axes_lines)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_grow(self, run_time: float = 1.0) -> "mn.Animation":
        """Scale the ellipse from zero at its centre."""
        _require_manim("animate_grow")
        self.scale(0.001, about_point=self.center_dot.get_center())
        return self.animate(run_time=run_time, rate_func=rate_functions.ease_out_back).scale(
            1000, about_point=self.center_dot.get_center()
        )

    def morph_to(
        self,
        r_target:     float,
        std_x:        float  = 1.0,
        std_y:        float  = 1.0,
        run_time:     float  = 1.2,
    ) -> "mn.Animation":
        """
        Animate the ellipse morphing to represent a new correlation coefficient.

        Parameters
        ----------
        r_target : float
            Target Pearson r (-1 to 1).
        std_x, std_y : float
            Fixed standard deviations to use (keeps size stable while r changes).
        run_time : float

        Returns
        -------
        manim.Transform
        """
        _require_manim("morph_to")
        n = len(self._x)
        # Synthesise new x, y with the target correlation
        new_x   = np.random.default_rng(0).standard_normal(n) * std_x
        noise   = np.random.default_rng(1).standard_normal(n)
        new_y   = r_target * (new_x / std_x) * std_y + math.sqrt(max(0, 1 - r_target**2)) * noise * std_y
        new_y  *= std_y / np.std(new_y) if np.std(new_y) > 0 else 1.0

        target_ellipse = CorrelationEllipse3D(
            x             = new_x,
            y             = new_y,
            level         = self._level,
            n_points      = self._n_points,
            stroke_color  = self._stroke_color,
            fill_color    = self._fill_color,
            fill_opacity  = self._fill_opacity,
            stroke_width  = self._stroke_width,
        )
        target_ellipse.move_to(self.get_center())
        return Transform(self, target_ellipse, run_time=run_time,
                         rate_func=rate_functions.ease_in_out_sine)


# ---------------------------------------------------------------------------
# CorrelationMatrix3D
# ---------------------------------------------------------------------------

class CorrelationMatrix3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    A fully labeled heatmap-style correlation matrix for p variables.

    Each cell shows the correlation coefficient and optional significance
    stars.  The diagonal shows the variable name.  Colors use a diverging
    purple → white → coral colormap (REGRESSION_FAMILY → INFERENCE_FAMILY).

    Parameters
    ----------
    R : ndarray, shape (p, p)
        Correlation matrix.
    P : ndarray or None, shape (p, p)
        p-value matrix.  Stars are shown when P is provided.
    var_names : list[str]
        Variable labels.
    cell_size : float
        Side length of each cell square.
    show_values : bool
        Overlay numeric r values on each cell.
    show_diagonal : bool
        If True, diagonal cells show variable name; otherwise blank.
    cmap_low : ManimColor
        Color for r = -1.  Defaults to REGRESSION_FAMILY.dark.
    cmap_high : ManimColor
        Color for r = +1.  Defaults to INFERENCE_FAMILY.dark.
    cmap_mid : ManimColor
        Color for r = 0.  Defaults to off-white.

    Key sub-mobjects
    ----------------
    .cells          — VGroup of Rectangle mobjects (p*p total)
    .value_labels   — VGroup of Text/MathTex labels
    .var_name_labels— VGroup of diagonal name labels
    .row_labels     — VGroup of left row labels
    .col_labels     — VGroup of top column labels

    Animations
    ----------
    .animate_build(run_time, stagger)  — reveal cells row by row
    .highlight_cell(i, j, color)       — flash cell (i,j)
    .morph_data(R_new, P_new, run_time)— transition to new data
    """

    def __init__(
        self,
        R:              np.ndarray,
        P:              Optional[np.ndarray]  = None,
        var_names:      Optional[List[str]]   = None,
        cell_size:      float                  = 0.65,
        show_values:    bool                   = True,
        show_diagonal:  bool                   = True,
        font_size:      int                    = 16,
        cmap_low        = None,
        cmap_high       = None,
        cmap_mid        = None,
        **kwargs,
    ) -> None:
        _require_manim("CorrelationMatrix3D")
        super().__init__(**kwargs)

        R         = np.asarray(R, dtype=float)
        p         = R.shape[0]
        self._R   = R
        self._P   = P
        self._p   = p
        self._cell_size   = cell_size
        self._show_values = show_values
        self._font_size   = font_size

        self._var_names = var_names or [f"X{i+1}" for i in range(p)]

        # Resolve colormap endpoints
        if _COLORS_AVAILABLE:
            self._cmap_low  = cmap_low  or ManimColor(REGRESSION_FAMILY.dark.hex)
            self._cmap_high = cmap_high or ManimColor(INFERENCE_FAMILY.dark.hex)
            self._cmap_mid  = cmap_mid  or WHITE
        else:
            self._cmap_low  = cmap_low  or BLUE
            self._cmap_high = cmap_high or RED
            self._cmap_mid  = cmap_mid  or WHITE

        self._build(R, P, show_diagonal)

    # ------------------------------------------------------------------
    # Color mapping
    # ------------------------------------------------------------------

    def _r_to_color(self, r: float):
        """Map r in [-1, 1] to a Manim color via diverging map."""
        r = float(np.clip(r, -1.0, 1.0))
        if r < 0:
            t = (r + 1.0)        # 0 (r=-1) → 1 (r=0)
            return interpolate_color(self._cmap_low, self._cmap_mid, t)
        else:
            t = r                # 0 (r=0) → 1 (r=+1)
            return interpolate_color(self._cmap_mid, self._cmap_high, t)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(
        self,
        R: np.ndarray,
        P: Optional[np.ndarray],
        show_diagonal: bool,
    ) -> None:
        p           = self._p
        cs          = self._cell_size
        total_w     = p * cs
        total_h     = p * cs
        origin      = np.array([-total_w / 2 + cs / 2, total_h / 2 - cs / 2, 0.0])

        self.cells          = VGroup()
        self.value_labels   = VGroup()
        self.var_name_labels= VGroup()
        self.row_labels     = VGroup()
        self.col_labels     = VGroup()

        for i in range(p):
            for j in range(p):
                pos = origin + np.array([j * cs, -i * cs, 0.0])

                if i == j:
                    # Diagonal cell
                    cell_color = WHITE if _MANIM_AVAILABLE else None
                    cell = Rectangle(
                        width        = cs * 0.95,
                        height       = cs * 0.95,
                        fill_color   = cell_color,
                        fill_opacity = 0.12,
                        stroke_color = GRAY,
                        stroke_width = 0.8,
                    ).move_to(pos)
                    self.cells.add(cell)
                    if show_diagonal:
                        lbl = Text(
                            self._var_names[i],
                            font_size = self._font_size,
                            color     = GRAY,
                        ).move_to(pos)
                        self.var_name_labels.add(lbl)
                else:
                    r_val  = float(R[i, j])
                    color  = self._r_to_color(r_val)
                    # Opacity proportional to |r|
                    opacity = 0.15 + 0.80 * abs(r_val)
                    cell   = Rectangle(
                        width        = cs * 0.95,
                        height       = cs * 0.95,
                        fill_color   = color,
                        fill_opacity = opacity,
                        stroke_color = DARK_GRAY,
                        stroke_width = 0.5,
                    ).move_to(pos)
                    self.cells.add(cell)

                    if self._show_values:
                        # Significance stars
                        stars = ""
                        if P is not None and not np.isnan(P[i, j]):
                            pv = float(P[i, j])
                            if pv < 0.001:   stars = "***"
                            elif pv < 0.01:  stars = "**"
                            elif pv < 0.05:  stars = "*"

                        val_str  = f"{r_val:.2f}{stars}"
                        # Choose label color for readability
                        lbl_color = BLACK if abs(r_val) < 0.5 else WHITE
                        lbl = Text(
                            val_str,
                            font_size = self._font_size * 0.85,
                            color     = lbl_color,
                        ).move_to(pos)
                        self.value_labels.add(lbl)

        # Row labels (left side)
        for i in range(p):
            y_pos = origin[1] - i * cs
            lbl   = Text(
                self._var_names[i],
                font_size = self._font_size,
                color     = GRAY,
            ).move_to(np.array([origin[0] - cs * 1.1, y_pos, 0.0]))
            self.row_labels.add(lbl)

        # Column labels (top)
        for j in range(p):
            x_pos = origin[0] + j * cs
            lbl   = Text(
                self._var_names[j],
                font_size = self._font_size,
                color     = GRAY,
            ).move_to(np.array([x_pos, origin[1] + cs * 1.1, 0.0]))
            self.col_labels.add(lbl)

        self.add(self.cells, self.value_labels, self.var_name_labels,
                 self.row_labels, self.col_labels)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_build(
        self,
        run_time: float = 2.0,
        stagger:  float = 0.08,
    ) -> "mn.Animation":
        """Reveal cells row by row with a staggered FadeIn."""
        _require_manim("animate_build")
        all_mobs = list(self.cells) + list(self.value_labels) + list(self.var_name_labels)
        return LaggedStart(
            *[FadeIn(m, run_time=run_time * 0.4) for m in all_mobs],
            lag_ratio = stagger,
        )

    def highlight_cell(
        self,
        i: int,
        j: int,
        color = None,
        run_time: float = 0.6,
    ) -> "mn.Animation":
        """
        Flash cell (i, j) with an Indicate animation.

        Parameters
        ----------
        i, j : int
            Row and column (0-indexed).
        color : ManimColor, optional
        run_time : float
        """
        _require_manim("highlight_cell")
        if color is None:
            color = YELLOW if _MANIM_AVAILABLE else None
        idx  = i * self._p + j
        if idx >= len(self.cells):
            return AnimationGroup()
        cell = self.cells[idx]
        return Indicate(cell, color=color, scale_factor=1.15, run_time=run_time)

    def morph_data(
        self,
        R_new:    np.ndarray,
        P_new:    Optional[np.ndarray] = None,
        run_time: float = 1.5,
    ) -> "mn.Animation":
        """Crossfade to a new correlation matrix."""
        _require_manim("morph_data")
        target = CorrelationMatrix3D(
            R          = R_new,
            P          = P_new,
            var_names  = self._var_names,
            cell_size  = self._cell_size,
            show_values= self._show_values,
            font_size  = self._font_size,
            cmap_low   = self._cmap_low,
            cmap_high  = self._cmap_high,
            cmap_mid   = self._cmap_mid,
        ).move_to(self.get_center())
        return Transform(self, target, run_time=run_time,
                         rate_func=rate_functions.ease_in_out_sine)


# ---------------------------------------------------------------------------
# RegressionLine3D
# ---------------------------------------------------------------------------

class RegressionLine3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    The fitted regression line (simple linear regression) overlaid on a scatter.

    Draws the line y = beta0 + beta1 * x across the visible x-range,
    with an optional shaded 95% confidence band.

    Parameters
    ----------
    result : RegressionResult
        Must be a simple (k=1) OLS fit.
    x_range : tuple (x_min, x_max), optional
        Extent of the line.  Defaults to observed data range.
    show_ci_band : bool
        Draw a shaded 95% CI band around the fitted line.
    line_color : ManimColor, optional
    band_color : ManimColor, optional
    stroke_width : float
    ci_level : float
        Confidence level for the band.  Default 0.95.

    Key sub-mobjects
    ----------------
    .line          : Line
    .ci_band       : VMobject (shaded polygon, optional)
    .equation_label: MathTex  (shows fitted equation, optional)

    Animations
    ----------
    .animate_draw(run_time)        — draw line left-to-right
    .morph_to_fit(new_result, run_time) — rotate/translate to a new fit
    .reveal_ci_band(run_time)      — fade in the CI band
    """

    def __init__(
        self,
        result:          RegressionResult,
        x_range:         Optional[Tuple[float, float]] = None,
        show_ci_band:    bool = True,
        show_equation:   bool = True,
        line_color       = None,
        band_color       = None,
        stroke_width:    float = 2.8,
        ci_level:        float = 0.95,
        **kwargs,
    ) -> None:
        _require_manim("RegressionLine3D")
        if result.k != 1:
            raise ValueError(
                f"RegressionLine3D expects k=1 predictor, got k={result.k}."
            )
        super().__init__(**kwargs)

        self._result      = result
        self._ci_level    = ci_level

        # Colors
        if _COLORS_AVAILABLE:
            line_color = line_color or ManimColor(REGRESSION_FAMILY.base.hex)
            band_color = band_color or ManimColor(REGRESSION_FAMILY.light.hex)
        else:
            line_color = line_color or PURPLE
            band_color = band_color or PURPLE

        # Extract x from design matrix (intercept is col 0)
        x_raw  = result.x[:, 1] if result.fit_intercept else result.x[:, 0]
        b0     = result.beta[0] if result.fit_intercept else 0.0
        b1     = result.beta[1] if result.fit_intercept else result.beta[0]

        x_min, x_max = x_range if x_range else (float(x_raw.min()), float(x_raw.max()))

        # Fitted line
        y_at_min = b0 + b1 * x_min
        y_at_max = b0 + b1 * x_max
        self.line = Line(
            start        = [x_min, y_at_min, 0.0],
            end          = [x_max, y_at_max, 0.0],
            color        = line_color,
            stroke_width = stroke_width,
        )
        self.add(self.line)

        # CI band
        if show_ci_band:
            self._build_ci_band(
                result, x_raw, b0, b1,
                x_min, x_max, band_color, ci_level,
            )

        # Equation label
        if show_equation:
            self._build_equation(b0, b1, result.r_squared, line_color)

    def _build_ci_band(
        self,
        result: RegressionResult,
        x_raw: np.ndarray,
        b0: float, b1: float,
        x_min: float, x_max: float,
        band_color, ci_level: float,
    ) -> None:
        """
        Build a shaded CI band polygon.

        95% pointwise CI for E[Y|x]:
            y_hat(x) ± t_{n-2,alpha/2} * s * sqrt(1/n + (x-xbar)^2 / Sxx)
        """
        n    = result.n
        xbar = float(np.mean(x_raw))
        Sxx  = float(np.sum((x_raw - xbar)**2))
        s    = result.sigma_hat

        if _SCIPY_AVAILABLE:
            t_crit = float(_sp.t.ppf((1 + ci_level) / 2, df=n - 2))
        else:
            t_crit = 1.959963985   # z-approx

        n_pts  = 60
        xs     = np.linspace(x_min, x_max, n_pts)
        y_hat  = b0 + b1 * xs
        margin = t_crit * s * np.sqrt(1 / n + (xs - xbar)**2 / max(Sxx, 1e-12))

        upper  = y_hat + margin
        lower  = y_hat - margin

        # Build polygon: upper curve forward + lower curve backward
        upper_pts = [[xs[i], upper[i], 0.001] for i in range(n_pts)]
        lower_pts = [[xs[i], lower[i], 0.001] for i in range(n_pts - 1, -1, -1)]
        all_pts   = upper_pts + lower_pts

        from manim import VMobject
        self.ci_band = VMobject(
            fill_color   = band_color,
            fill_opacity = 0.25,
            stroke_width = 0,
        )
        self.ci_band.set_points_as_corners(
            [np.array(p) for p in all_pts] + [np.array(all_pts[0])]
        )
        self.add(self.ci_band)

    def _build_equation(
        self,
        b0: float,
        b1: float,
        r2: float,
        color,
    ) -> None:
        sign   = "+" if b1 >= 0 else "-"
        eq_str = (
            rf"\hat{{y}} = {b0:.3f} {sign} {abs(b1):.3f}x"
            rf"\quad R^2 = {r2:.3f}"
        )
        self.equation_label = MathTex(
            eq_str,
            font_size = 24,
            color     = color,
        )
        self.add(self.equation_label)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_draw(self, run_time: float = 1.2) -> "mn.Animation":
        """Draw the regression line from left to right."""
        _require_manim("animate_draw")
        return Create(self.line, run_time=run_time,
                      rate_func=rate_functions.ease_in_out_sine)

    def reveal_ci_band(self, run_time: float = 0.8) -> "mn.Animation":
        """Fade in the CI band after the line is drawn."""
        _require_manim("reveal_ci_band")
        if not hasattr(self, "ci_band"):
            return AnimationGroup()
        return FadeIn(self.ci_band, run_time=run_time)

    def morph_to_fit(
        self,
        new_result: RegressionResult,
        run_time:   float = 1.2,
    ) -> "mn.Animation":
        """Animate the line (and band) transforming to a new regression fit."""
        _require_manim("morph_to_fit")
        x_raw  = new_result.x[:, 1] if new_result.fit_intercept else new_result.x[:, 0]
        target = RegressionLine3D(
            result       = new_result,
            x_range      = (float(x_raw.min()), float(x_raw.max())),
            show_ci_band = hasattr(self, "ci_band"),
            show_equation= hasattr(self, "equation_label"),
        ).move_to(self.get_center())
        return Transform(self, target, run_time=run_time,
                         rate_func=rate_functions.ease_in_out_sine)


# ---------------------------------------------------------------------------
# ResidualArrows3D
# ---------------------------------------------------------------------------

class ResidualArrows3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Vertical arrows from each fitted value to the observed value,
    visualising the OLS residuals.

    Positive residuals (y > y_hat) are shown in one color, negative in another.
    Arrow width scales with |residual| so large residuals are visually prominent.

    Parameters
    ----------
    result : RegressionResult
        Simple OLS fit (k=1 recommended for clarity).
    pos_color : ManimColor, optional
        Color for positive residuals.  Default TEAL_600.
    neg_color : ManimColor, optional
        Color for negative residuals.  Default CORAL_600.
    max_stroke_width : float
        Maximum arrow stroke width (for the largest residual).
    show_labels : bool
        Label the largest residual with its value.
    outlier_threshold : float
        |studentized residual| above which an observation is flagged.

    Key sub-mobjects
    ----------------
    .positive_arrows : VGroup
    .negative_arrows : VGroup
    .outlier_markers : VGroup  (circles around flagged observations)

    Animations
    ----------
    .animate_appear(run_time, stagger)    — grow arrows from the regression line
    .flash_outliers(run_time)             — Indicate outlier_markers
    .update_fit(new_result, run_time)     — morph residuals to a new model
    """

    def __init__(
        self,
        result:              RegressionResult,
        pos_color            = None,
        neg_color            = None,
        max_stroke_width:    float = 4.0,
        min_stroke_width:    float = 0.8,
        show_labels:         bool  = False,
        outlier_threshold:   float = 2.0,
        **kwargs,
    ) -> None:
        _require_manim("ResidualArrows3D")
        super().__init__(**kwargs)

        if _COLORS_AVAILABLE:
            pos_color = pos_color or ManimColor(TEAL_600.hex)
            neg_color = neg_color or ManimColor(CORAL_600.hex)
        else:
            pos_color = pos_color or GREEN
            neg_color = neg_color or RED

        self._result            = result
        self._pos_color         = pos_color
        self._neg_color         = neg_color
        self._outlier_threshold = outlier_threshold

        self.positive_arrows = VGroup()
        self.negative_arrows = VGroup()
        self.outlier_markers = VGroup()

        self._build(result, pos_color, neg_color,
                    max_stroke_width, min_stroke_width,
                    show_labels, outlier_threshold)

    def _build(
        self,
        result: RegressionResult,
        pos_color, neg_color,
        max_sw: float, min_sw: float,
        show_labels: bool, outlier_thresh: float,
    ) -> None:
        x_col  = result.x[:, 1] if result.fit_intercept else result.x[:, 0]
        y_hat  = result.fitted
        y      = result.y
        resid  = result.residuals
        n      = result.n

        # Influence measures for outlier detection
        try:
            infl = influence_measures(result)
            ext_stud = infl.externally_studentized
        except Exception:
            ext_stud = np.zeros(n)

        max_abs_resid = float(np.max(np.abs(resid))) or 1.0

        for i in range(n):
            xi    = float(x_col[i])
            yi    = float(y[i])
            yhati = float(y_hat[i])
            ri    = float(resid[i])

            if abs(ri) < 1e-10:
                continue

            stroke_w = min_sw + (max_sw - min_sw) * abs(ri) / max_abs_resid
            color    = pos_color if ri > 0 else neg_color

            arrow = Arrow(
                start        = [xi, yhati, 0.001],
                end          = [xi, yi,    0.002],
                buff         = 0,
                color        = color,
                stroke_width = stroke_w,
                tip_length   = min(0.12, abs(ri) * 0.4),
                max_tip_length_to_length_ratio = 0.4,
            )
            if ri > 0:
                self.positive_arrows.add(arrow)
            else:
                self.negative_arrows.add(arrow)

            # Outlier marker
            if abs(float(ext_stud[i])) > outlier_thresh:
                marker = Circle(
                    radius       = 0.15,
                    color        = YELLOW if _MANIM_AVAILABLE else None,
                    stroke_width = 1.8,
                    fill_opacity = 0,
                ).move_to([xi, yi, 0.003])
                self.outlier_markers.add(marker)

        self.add(self.positive_arrows, self.negative_arrows, self.outlier_markers)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_appear(
        self, run_time: float = 1.4, stagger: float = 0.04
    ) -> "mn.Animation":
        """Grow all residual arrows from zero length."""
        _require_manim("animate_appear")
        all_arrows = list(self.positive_arrows) + list(self.negative_arrows)
        return LaggedStart(
            *[Create(a, run_time=run_time * 0.5) for a in all_arrows],
            lag_ratio = stagger,
        )

    def flash_outliers(self, run_time: float = 1.0) -> "mn.Animation":
        """Flash the outlier marker circles."""
        _require_manim("flash_outliers")
        if not self.outlier_markers:
            return AnimationGroup()
        return LaggedStart(
            *[Flash(m, color=YELLOW, run_time=run_time, flash_radius=0.25)
              for m in self.outlier_markers],
            lag_ratio = 0.15,
        )

    def update_fit(
        self,
        new_result: RegressionResult,
        run_time:   float = 1.0,
    ) -> "mn.Animation":
        """Morph residual arrows to reflect a new regression fit."""
        _require_manim("update_fit")
        target = ResidualArrows3D(
            result            = new_result,
            pos_color         = self._pos_color,
            neg_color         = self._neg_color,
            outlier_threshold = self._outlier_threshold,
        ).move_to(self.get_center())
        return Transform(self, target, run_time=run_time,
                         rate_func=rate_functions.ease_in_out_quad)


# ===========================================================================
# LAYER E — Scene-level animation factories
# ===========================================================================

def build_scatter_to_line(
    axes:       "mn.Axes",
    result:     RegressionResult,
    dot_color   = None,
    run_time:   float = 3.0,
) -> "mn.Animation":
    """
    Canonical animation sequence: scatter cloud appears, then the regression
    line is drawn through it.

    Steps
    -----
    1. LaggedStart: all scatter dots appear
    2. Pause (implicit)
    3. Regression line draws left-to-right
    4. CI band fades in
    5. Equation label writes in

    Parameters
    ----------
    axes : manim.Axes
        Already-added Axes mobject on which to draw.
    result : RegressionResult
        Simple OLS fit (k=1).
    dot_color : ManimColor, optional
    run_time : float
        Total duration.

    Returns
    -------
    manim.Succession
    """
    _require_manim("build_scatter_to_line")

    if _COLORS_AVAILABLE:
        dot_color = dot_color or ManimColor(NORMAL_FAMILY.base.hex)
    else:
        dot_color = dot_color or BLUE

    x_col = result.x[:, 1] if result.fit_intercept else result.x[:, 0]

    # Scatter dots
    dots = VGroup(*[
        Dot(
            point  = axes.c2p(float(x_col[i]), float(result.y[i])),
            color  = dot_color,
            radius = 0.055,
        )
        for i in range(result.n)
    ])

    reg_line = RegressionLine3D(result=result, show_ci_band=True, show_equation=True)

    t_dots = run_time * 0.35
    t_line = run_time * 0.30
    t_band = run_time * 0.18
    t_eq   = run_time * 0.17

    anim = Succession(
        LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                    lag_ratio=0.03, run_time=t_dots),
        reg_line.animate_draw(run_time=t_line),
        reg_line.reveal_ci_band(run_time=t_band),
        Write(reg_line.equation_label, run_time=t_eq)
        if hasattr(reg_line, "equation_label") else AnimationGroup(),
    )
    return anim, dots, reg_line


def morph_r_value(
    ellipse:   CorrelationEllipse3D,
    r_values:  Sequence[float],
    run_time_each: float = 1.2,
) -> "mn.Animation":
    """
    Animate a CorrelationEllipse3D morphing through a sequence of r values.

    Parameters
    ----------
    ellipse : CorrelationEllipse3D
    r_values : sequence of float
        Each value triggers one morph step.
    run_time_each : float
        Duration per morph step.

    Returns
    -------
    manim.Succession
    """
    _require_manim("morph_r_value")
    steps = [
        ellipse.morph_to(r, run_time=run_time_each)
        for r in r_values
    ]
    return Succession(*steps)


def reveal_residuals(
    scene,
    result:   RegressionResult,
    reg_line: RegressionLine3D,
    run_time: float = 2.0,
) -> None:
    """
    Play the full residual-reveal sequence on *scene*:
    1. Residual arrows appear staggered
    2. Outlier markers flash
    3. Brief pause

    Parameters
    ----------
    scene : manim.Scene
    result : RegressionResult
    reg_line : RegressionLine3D
        The already-displayed regression line.
    run_time : float
    """
    _require_manim("reveal_residuals")
    arrows = ResidualArrows3D(result=result)
    scene.add(arrows)
    scene.play(arrows.animate_appear(run_time=run_time * 0.7))
    if arrows.outlier_markers:
        scene.play(arrows.flash_outliers(run_time=run_time * 0.3))
    scene.wait(0.4)


def animate_influence_removal(
    scene,
    result:       RegressionResult,
    obs_index:    int,
    axes:         "mn.Axes",
    dot_mob,
    run_time:     float = 2.0,
) -> "RegressionResult":
    """
    Remove observation *obs_index* from the data, refit OLS, and animate
    the regression line updating.

    Sequence
    --------
    1. Flash the removed observation's dot in red
    2. FadeOut the dot
    3. Morph the regression line to the new fit
    4. Return the re-fitted RegressionResult

    Parameters
    ----------
    scene : manim.Scene
    result : RegressionResult
        Original fit.
    obs_index : int
        Index (0-based) to remove.
    axes : manim.Axes
    dot_mob : Dot
        The Manim dot mobject for the removed point.
    run_time : float

    Returns
    -------
    RegressionResult — the re-fitted model without observation obs_index.
    """
    _require_manim("animate_influence_removal")
    # Flash
    scene.play(
        Flash(dot_mob, color=RED, run_time=run_time * 0.25, flash_radius=0.25),
        Indicate(dot_mob, color=RED, scale_factor=1.4, run_time=run_time * 0.25),
    )
    scene.play(FadeOut(dot_mob, run_time=run_time * 0.15))

    # Refit without the observation
    mask  = np.ones(result.n, dtype=bool)
    mask[obs_index] = False
    x_col = result.x[mask, 1] if result.fit_intercept else result.x[mask, 0]
    y_new = result.y[mask]
    new_result = ols_fit(x_col, y_new)

    return new_result


# ===========================================================================
# LAYER F — Formula registry bridge
# ===========================================================================

def _build_pearson_formula() -> "TexFormula":
    if not _TEX_AVAILABLE:
        return None  # type: ignore
    num = _sum(r"(x_i - \bar{x})(y_i - \bar{y})", "i=1", "n")
    den = (
        _sqrt(_sum(r"(x_i - \bar{x})^2", "i=1", "n"))
        + r"\,"
        + _sqrt(_sum(r"(y_i - \bar{y})^2", "i=1", "n"))
    )
    raw = rf"r = {_frac(num, den)}"
    return TexFormula(
        name        = "pearson_r_detailed",
        raw         = raw,
        description = "Pearson r — sample cross-product ratio form",
        parts       = {
            "numerator":   num,
            "denominator": den,
            "cross_product": r"(x_i - \bar{x})(y_i - \bar{y})",
            "ss_x":        _sum(r"(x_i - \bar{x})^2", "i=1", "n"),
            "ss_y":        _sum(r"(y_i - \bar{y})^2", "i=1", "n"),
        },
        steps       = [
            TexDerivationStep(
                lhs = "r",
                rhs = _frac(
                    r"\mathrm{Cov}(X,Y)",
                    r"\mathrm{SD}(X)\,\mathrm{SD}(Y)"
                ),
                annotation = "Definition: standardised covariance",
            ),
            TexDerivationStep(
                lhs = "",
                rhs = _frac(
                    _frac(_sum(r"(x_i-\bar{x})(y_i-\bar{y})", "i",""),  "n-1"),
                    _frac(
                        _sqrt(_sum(r"(x_i-\bar{x})^2","i","")) + r"\," +
                        _sqrt(_sum(r"(y_i-\bar{y})^2","i","")),
                        "n-1"
                    ),
                ),
                annotation = "Expand sample Cov and SD",
            ),
            TexDerivationStep(
                lhs = "",
                rhs = raw.split("=",1)[1].strip(),
                annotation = "Cancel (n-1) factors",
            ),
        ],
        tags        = ["correlation", "pearson", "regression"],
    )


def _build_spearman_formula() -> "TexFormula":
    if not _TEX_AVAILABLE:
        return None  # type: ignore
    raw = (
        r"\rho = 1 - "
        + _frac(
            r"6\," + _sum("d_i^2", "i=1", "n"),
            r"n(n^2 - 1)"
        )
    )
    return TexFormula(
        name        = "spearman_rho_formula",
        raw         = raw,
        description = "Spearman rank correlation (no-ties closed form)",
        parts       = {
            "rank_diff_sq":  _sum("d_i^2", "i=1", "n"),
            "denominator":   r"n(n^2 - 1)",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"d_i",
                rhs = r"\mathrm{rank}(x_i) - \mathrm{rank}(y_i)",
                annotation = "Rank difference for each pair",
            ),
            TexDerivationStep(
                lhs = r"\rho",
                rhs = raw.split("=",1)[1].strip(),
                annotation = "Closed form (no ties)",
            ),
        ],
        tags        = ["correlation", "spearman", "nonparametric"],
    )


def _build_ols_derivation_formula() -> "TexFormula":
    if not _TEX_AVAILABLE:
        return None  # type: ignore
    raw = (
        r"\hat{\boldsymbol{\beta}} = "
        r"\left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}"
    )
    return TexFormula(
        name        = "ols_normal_equations",
        raw         = raw,
        description = "OLS normal equations (matrix form)",
        parts       = {
            "beta_hat":    r"\hat{\boldsymbol{\beta}}",
            "gram_matrix": r"\mathbf{X}^\top \mathbf{X}",
            "projection":  r"\mathbf{X}^\top \mathbf{y}",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"\mathrm{RSS}(\boldsymbol{\beta})",
                rhs = r"(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})^\top(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})",
                annotation = "Residual sum of squares",
            ),
            TexDerivationStep(
                lhs = r"\frac{\partial\,\mathrm{RSS}}{\partial\boldsymbol{\beta}}",
                rhs = r"-2\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}) = \mathbf{0}",
                annotation = "Set gradient to zero",
            ),
            TexDerivationStep(
                lhs = r"\mathbf{X}^\top\mathbf{X}\,\hat{\boldsymbol{\beta}}",
                rhs = r"\mathbf{X}^\top\mathbf{y}",
                annotation = "Normal equations",
            ),
            TexDerivationStep(
                lhs = r"\hat{\boldsymbol{\beta}}",
                rhs = raw.split("=",1)[1].strip(),
                annotation = "Solve (assuming X'X invertible)",
            ),
        ],
        tags        = ["regression", "ols", "matrix", "derivation"],
    )


def _build_partial_r_formula() -> "TexFormula":
    if not _TEX_AVAILABLE:
        return None  # type: ignore
    raw = (
        r"r_{XY \cdot Z} = "
        + _frac(
            r"r_{XY} - r_{XZ}\,r_{YZ}",
            _sqrt(r"(1-r_{XZ}^2)(1-r_{YZ}^2)")
        )
    )
    return TexFormula(
        name        = "partial_r_formula",
        raw         = raw,
        description = "Partial correlation (single control Z)",
        parts       = {
            "r_xy":  r"r_{XY}",
            "r_xz":  r"r_{XZ}",
            "r_yz":  r"r_{YZ}",
            "numerator":   r"r_{XY} - r_{XZ}\,r_{YZ}",
            "denominator": _sqrt(r"(1-r_{XZ}^2)(1-r_{YZ}^2)"),
        },
        steps       = [
            TexDerivationStep(
                lhs = r"e_{X|Z}",
                rhs = r"X - \hat{X}(Z)",
                annotation = "Residualise X on Z",
            ),
            TexDerivationStep(
                lhs = r"e_{Y|Z}",
                rhs = r"Y - \hat{Y}(Z)",
                annotation = "Residualise Y on Z",
            ),
            TexDerivationStep(
                lhs = r"r_{XY\cdot Z}",
                rhs = r"r(e_{X|Z},\, e_{Y|Z})",
                annotation = "Pearson r of residuals",
            ),
        ],
        tags        = ["correlation", "partial", "regression"],
    )


def _build_cooks_d_formula() -> "TexFormula":
    if not _TEX_AVAILABLE:
        return None  # type: ignore
    raw = (
        r"D_i = "
        + _frac(
            r"e_i^2",
            r"p\,s^2"
        )
        + r"\cdot"
        + _frac(r"h_{ii}", r"(1-h_{ii})^2")
    )
    return TexFormula(
        name        = "cooks_d_formula",
        raw         = raw,
        description = "Cook's distance — influence of observation i",
        parts       = {
            "residual_sq": r"e_i^2",
            "leverage":    r"h_{ii}",
            "leverage_term": _frac(r"h_{ii}", r"(1-h_{ii})^2"),
        },
        tags        = ["regression", "influence", "diagnostics"],
    )


# Build CORR_FORMULAS registry
CORR_FORMULAS: Dict[str, "TexFormula"] = {}

if _TEX_AVAILABLE:
    _formulas_to_add = [
        _build_pearson_formula(),
        _build_spearman_formula(),
        _build_ols_derivation_formula(),
        _build_partial_r_formula(),
        _build_cooks_d_formula(),
    ]
    for _f in _formulas_to_add:
        if _f is not None:
            CORR_FORMULAS[_f.name] = _f
            # Also register in the global tex_utils catalog
            try:
                register_formula(_f)
            except (ValueError, KeyError):
                pass   # already registered — that's fine


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def _clean_paired(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coerce x, y to float arrays and apply listwise deletion of NaN rows.

    Returns
    -------
    (x_clean, y_clean) — both 1-D float64 arrays of equal length.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length; got {len(x)} and {len(y)}."
        )
    valid = ~(np.isnan(x) | np.isnan(y))
    n_dropped = int(np.sum(~valid))
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} NaN pair(s) removed by listwise deletion.",
            UserWarning, stacklevel=3,
        )
    return x[valid], y[valid]


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank transform (handles ties), pure NumPy."""
    n      = len(x)
    order  = np.argsort(x, stable=True)
    ranks  = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    # Average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    # For each unique value, replace ranks with their average
    for ui, cnt in enumerate(counts):
        if cnt > 1:
            mask       = inv == ui
            avg_rank   = float(np.mean(ranks[mask]))
            ranks[mask] = avg_rank
    return ranks


# ===========================================================================
# __all__
# ===========================================================================

__all__ = [
    # Layer A enums
    "CorrelationMethod",

    # Layer A functions
    "pearson",
    "spearman",
    "kendall",
    "point_biserial",
    "phi_coefficient",
    "cramers_v",
    "partial_correlation",
    "correlation_matrix",

    # Layer B functions
    "ols_fit",
    "ridge_fit",
    "influence_measures",

    # Layer C dataclasses
    "CorrelationResult",
    "PartialCorrelationResult",
    "RegressionResult",
    "InfluenceMeasures",

    # Layer D Manim mobjects
    "CorrelationEllipse3D",
    "CorrelationMatrix3D",
    "RegressionLine3D",
    "ResidualArrows3D",

    # Layer E scene animations
    "build_scatter_to_line",
    "morph_r_value",
    "reveal_residuals",
    "animate_influence_removal",

    # Layer F formula registry
    "CORR_FORMULAS",

    # Helpers
    "_clean_paired",
    "_rankdata",
]