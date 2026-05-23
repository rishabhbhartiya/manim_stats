"""
manim_stats/regression/residuals.py
=====================================
Residual analysis, diagnostic test statistics, and Manim diagnostic-plot
mobjects for OLS regression assumption checking.

Scope
-----
Everything a statistics educator needs to **understand, test, and visualise**
OLS regression residuals:

  Layer A  Pure-math residual transforms — ordinary, standardised, internally
           and externally studentized, PRESS residuals, jackknife residuals,
           ACF, PACF (pure NumPy/SciPy — no statsmodels dependency).

  Layer B  Diagnostic test statistics   — Breusch-Pagan, White, Durbin-Watson,
           Ljung-Box, Shapiro-Wilk, Jarque-Bera, Anderson-Darling, RESET,
           Rainbow linearity test.  Each returns a typed result dataclass.

  Layer C  Structured result dataclasses — NormalityResult,
           HomoscedasticityResult, AutocorrResult, LinearityResult,
           ResidualDiagnostics (bundle of all four).

  Layer D  Manim mobjects               — ResidualVsFittedPlot, QQPlot3D,
           ScaleLocationPlot, InfluencePlot3D, ACFPlot3D, DiagnosticPanel.

  Layer E  Scene-level animations       — build_diagnostic_scene,
           animate_assumption_violation, animate_fix_heteroscedasticity.

  Layer F  Formula registry bridge      — RESIDUAL_FORMULAS extending
           tex_utils.FORMULAS.

Relationship to existing files
-------------------------------
* ``correlation.py``      — provides ``RegressionResult``, ``InfluenceMeasures``,
                             ``influence_measures()``, ``ols_fit()``.
                             This file imports all four and adds a pure-residuals
                             perspective on top.
* ``regression_plane.py`` — 3-D plane residuals (``PlaneResiduals3D``) are
                             distinct: those are 3-D vertical lines in a scene.
                             This file handles 2-D diagnostic *plots*.

Design notes
------------
* No statsmodels dependency — ACF, PACF, Ljung-Box, and all tests are
  implemented using pure NumPy/SciPy.  A ``UserWarning`` is emitted when a
  calculation needs more observations than available.
* Every test result dataclass has ``.passes(alpha)`` for programmatic checks
  and ``.summary_line()`` for human-readable output.
* ``ResidualDiagnostics`` is the single entry-point: ``diagnose(result)``
  computes everything and returns the bundle.
* Manim mobjects follow the VGroup sub-class pattern used throughout the
  project.  All colours reference ``core.colors`` families.
* LOWESS smoothing for the Residuals vs Fitted plot is implemented in pure
  NumPy (Gaussian kernel local regression) so it works without scikit-learn.
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
        Line, DashedLine, Arrow,
        Dot, Circle,
        Text, MathTex,
        ManimColor, WHITE, BLACK, GRAY, DARK_GRAY, LIGHT_GRAY,
        RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE,
        UP, DOWN, LEFT, RIGHT, ORIGIN,
        TAU, PI,
        Write, Create, FadeIn, FadeOut,
        Transform, ReplacementTransform,
        Rotate, Flash, Indicate,
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
# Project imports
# ---------------------------------------------------------------------------
try:
    from manim_stats.regression.correlation import (
        RegressionResult,
        InfluenceMeasures,
        influence_measures,
        ols_fit,
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
        AMBER_600, AMBER_200, AMBER_800,
        BLUE_600, BLUE_200,
        GRAY_400, GRAY_200, GRAY_600,
        StatColor,
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
        raise ImportError(f"{name} requires Manim.  Install with: pip install manim")


def _require_scipy(name: str) -> None:
    if not _SCIPY_AVAILABLE:
        raise ImportError(f"{name} requires SciPy.  Install with: pip install scipy")


def _require_corr(name: str) -> None:
    if not _CORR_AVAILABLE:
        raise ImportError(f"{name} requires manim_stats.regression.correlation.")


# ===========================================================================
# LAYER A — Pure-math residual transforms
# All functions are pure NumPy/SciPy. No Manim or project-level deps.
# ===========================================================================

def ordinary_residuals(result: "RegressionResult") -> np.ndarray:
    """
    Return the ordinary (raw) OLS residuals  e = y - X*beta.

    These are the same as ``result.residuals`` but exposed here for
    symmetry with the other transform functions.
    """
    return result.residuals.copy()


def standardised_residuals(result: "RegressionResult") -> np.ndarray:
    """
    Internally studentized (standardised) residuals:

        r_i = e_i / (s * sqrt(1 - h_ii))

    where ``s`` is the residual standard deviation and ``h_ii`` is the
    leverage (hat-matrix diagonal).  Under the classical assumptions,
    r_i ~ t_{n-p} but the r_i are not independent.

    Equivalent to what R calls ``rstandard()``.

    Returns
    -------
    ndarray, shape (n,)
    """
    _require_corr("standardised_residuals")
    e  = result.residuals
    h  = _hat_diag(result)
    s  = result.sigma_hat
    denom = s * np.sqrt(np.clip(1.0 - h, 1e-14, None))
    return e / np.where(denom > 0, denom, float("nan"))


def externally_studentized_residuals(result: "RegressionResult") -> np.ndarray:
    """
    Externally studentized (jackknife / deleted) residuals:

        t_i = e_i / (s_{-i} * sqrt(1 - h_ii))

    where ``s_{-i}`` is the leave-one-out residual standard deviation,
    computed efficiently via the Sherman-Morrison formula:

        s_{-i}^2 = (df * s^2 - e_i^2 / (1 - h_ii)) / (df - 1)

    Under the classical assumptions, t_i ~ t_{n-p-1}.
    Equivalent to what R calls ``rstudent()``.

    Returns
    -------
    ndarray, shape (n,)
    """
    _require_corr("externally_studentized_residuals")
    e  = result.residuals
    h  = _hat_diag(result)
    n, df = result.n, result.n - result.k - (1 if result.fit_intercept else 0)
    s2 = float(np.sum(e**2) / max(df, 1))
    h  = np.clip(h, 0.0, 1.0 - 1e-10)
    s2_loo = (df * s2 - e**2 / (1.0 - h)) / max(df - 1, 1)
    s2_loo = np.clip(s2_loo, 1e-14, None)
    return e / (np.sqrt(s2_loo) * np.sqrt(1.0 - h))


def press_residuals(result: "RegressionResult") -> np.ndarray:
    """
    PRESS (Predicted Residual Error Sum of Squares) residuals:

        e^{PRESS}_i = e_i / (1 - h_ii)

    These are the residuals obtained by refitting the model without
    observation i and predicting y_i — computed cheaply via the
    Sherman-Morrison identity.

    The PRESS statistic is ``sum(press_residuals**2)``.

    Returns
    -------
    ndarray, shape (n,)
    """
    _require_corr("press_residuals")
    e = result.residuals
    h = np.clip(_hat_diag(result), 0.0, 1.0 - 1e-10)
    return e / (1.0 - h)


def press_statistic(result: "RegressionResult") -> float:
    """
    PRESS = sum of squared PRESS residuals.

    A good predictive model has PRESS close to RSS.  The ratio
    PRESS / RSS measures optimism of the in-sample fit.
    """
    return float(np.sum(press_residuals(result)**2))


def jackknife_betas(result: "RegressionResult") -> np.ndarray:
    """
    Leave-one-out coefficient vectors, computed via the Sherman-Morrison
    rank-1 downdate formula — O(n * k^2) rather than O(n^2 * k).

    beta_{-i} = beta - (X'X)^{-1} x_i e_i / (1 - h_ii)

    Returns
    -------
    ndarray, shape (n, p)
        Row i is the coefficient vector fitted without observation i.
        p = number of columns in the design matrix.
    """
    _require_corr("jackknife_betas")
    X  = result.x
    e  = result.residuals
    h  = np.clip(_hat_diag(result), 0.0, 1.0 - 1e-10)
    try:
        XtXi = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtXi = np.linalg.pinv(X.T @ X)

    beta   = result.beta
    # C = (X'X)^{-1} X'  shape (p, n)
    C      = XtXi @ X.T
    # delta_beta[i] = C[:, i] * e[i] / (1 - h[i])   shape (p,)
    deltas = C * (e / (1.0 - h))[np.newaxis, :]     # (p, n)
    return (beta[:, np.newaxis] - deltas).T          # (n, p)


def acf_values(
    e:        np.ndarray,
    max_lag:  int = 20,
) -> np.ndarray:
    """
    Sample autocorrelation function of residuals up to ``max_lag``.

    Uses the standard biased estimator:
        rho_k = (1/n) sum_{i=1}^{n-k} e_c_i * e_c_{i+k}  /  Var(e_c)

    Returns
    -------
    ndarray, shape (max_lag,)
        rho_1, rho_2, ..., rho_{max_lag}
    """
    e  = np.asarray(e, dtype=float).ravel()
    n  = len(e)
    ec = e - e.mean()
    v  = float(np.sum(ec**2)) / n
    if v < 1e-12:
        return np.zeros(max_lag)
    return np.array([
        float(np.sum(ec[:n-k] * ec[k:])) / (n * v)
        for k in range(1, max_lag + 1)
    ])


def pacf_values(
    e:       np.ndarray,
    max_lag: int = 20,
) -> np.ndarray:
    """
    Partial autocorrelation function via successive OLS projections
    (Yule-Walker / Durbin-Levinson equivalent).

    For lag k, regress ``e[k:]`` on ``[e[k-1:n-1], e[k-2:n-2], ..., e[0:n-k]]``
    and return the coefficient on the last column.

    Returns
    -------
    ndarray, shape (max_lag,)
        phi_1, phi_2, ..., phi_{max_lag}
    """
    e   = np.asarray(e, dtype=float).ravel()
    n   = len(e)
    out = []
    for k in range(1, max_lag + 1):
        if n - k < k + 2:
            out.append(float("nan"))
            continue
        Z = np.column_stack([e[k - j:n - j] for j in range(1, k + 1)])
        b, _, _, _ = np.linalg.lstsq(Z, e[k:], rcond=None)
        out.append(float(b[-1]))
    return np.array(out)


def lowess_smooth(
    x:     np.ndarray,
    y:     np.ndarray,
    frac:  float = 0.40,
    n_eval: int  = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-kernel local linear regression (LOWESS approximation),
    implemented in pure NumPy — no scikit-learn or statsmodels needed.

    Parameters
    ----------
    x, y : array-like
        Data to smooth.
    frac : float
        Bandwidth as a fraction of x range.  Larger = smoother.
    n_eval : int
        Number of evaluation points.

    Returns
    -------
    (x_smooth, y_smooth) — each ndarray of length n_eval.
    """
    x   = np.asarray(x, dtype=float).ravel()
    y   = np.asarray(y, dtype=float).ravel()
    lo, hi = x.min(), x.max()
    span   = max(hi - lo, 1e-6)
    bw     = span * frac
    xs     = np.linspace(lo, hi, n_eval)
    ys     = np.empty(n_eval)

    for i, xi in enumerate(xs):
        w = np.exp(-0.5 * ((x - xi) / bw)**2)
        w_sum = w.sum()
        if w_sum < 1e-14:
            ys[i] = float(np.mean(y))
            continue
        A = np.column_stack([np.ones(len(x)), x])
        W = np.diag(w)
        try:
            b, _, _, _ = np.linalg.lstsq(A.T @ W @ A, A.T @ (W @ y), rcond=None)
            ys[i] = float(b[0] + b[1] * xi)
        except np.linalg.LinAlgError:
            ys[i] = float(np.average(y, weights=w))

    return xs, ys


def qq_coordinates(
    residuals: np.ndarray,
    distribution: str = "norm",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical vs sample quantile pairs for a Q-Q plot.

    Uses the Blom plotting position formula:
        p_i = (i - 3/8) / (n + 1/4)

    Parameters
    ----------
    residuals : array-like
        Residuals to plot (externally studentized recommended).
    distribution : str
        SciPy distribution name for theoretical quantiles.
        Default ``'norm'``.

    Returns
    -------
    (theoretical_q, sample_q) — both sorted ndarrays of length n.
    """
    _require_scipy("qq_coordinates")
    r   = np.asarray(residuals, dtype=float)
    n   = len(r)
    ps  = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    dist = getattr(_sp, distribution)
    theoretical = dist.ppf(ps)
    sample_q    = np.sort(r)
    return theoretical, sample_q


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hat_diag(result: "RegressionResult") -> np.ndarray:
    """Compute hat-matrix diagonal h_ii from a RegressionResult."""
    X = result.x
    try:
        XtXi = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtXi = np.linalg.pinv(X.T @ X)
    return np.clip(
        np.einsum("ij,jk,ki->i", X, XtXi, X.T),
        0.0, 1.0,
    )


# ===========================================================================
# LAYER B — Diagnostic test statistics
# Each function returns a typed dataclass (Layer C).
# ===========================================================================

# ---------------------------------------------------------------------------
# B.1  Normality tests
# ---------------------------------------------------------------------------

@dataclass
class NormalityResult:
    """
    Bundle of residual normality test results.

    Attributes
    ----------
    sw_stat, sw_p : float or None
        Shapiro-Wilk W statistic and p-value.  None when n > 5000 (test unreliable).
    jb_stat, jb_p : float
        Jarque-Bera chi-squared statistic and p-value.
    ad_stat : float
        Anderson-Darling statistic.
    ad_critical_5pct : float
        Anderson-Darling 5% critical value (from SciPy table).
    skewness : float
        Sample skewness of the residuals.
    excess_kurtosis : float
        Sample excess kurtosis of the residuals.
    n : int
        Number of residuals.
    """
    sw_stat:          Optional[float]
    sw_p:             Optional[float]
    jb_stat:          float
    jb_p:             float
    ad_stat:          float
    ad_critical_5pct: float
    skewness:         float
    excess_kurtosis:  float
    n:                int

    def passes(self, alpha: float = 0.05) -> bool:
        """
        Return True if all available tests fail to reject normality at *alpha*.
        (i.e., normality assumption is not rejected.)
        """
        ok = True
        if self.sw_p is not None:
            ok = ok and (self.sw_p > alpha)
        ok = ok and (self.jb_p > alpha)
        ok = ok and (self.ad_stat < self.ad_critical_5pct)
        return ok

    def summary_line(self) -> str:
        sw_str = (f"SW W={self.sw_stat:.4f} p={self.sw_p:.4g}"
                  if self.sw_p is not None else "SW n/a")
        return (
            f"Normality: {sw_str} | "
            f"JB={self.jb_stat:.4f} p={self.jb_p:.4g} | "
            f"AD={self.ad_stat:.4f} (crit5%={self.ad_critical_5pct:.4f}) | "
            f"skew={self.skewness:.4f} kurt={self.excess_kurtosis:.4f}"
        )

    def verdict(self, alpha: float = 0.05) -> str:
        return "PASS" if self.passes(alpha) else "FAIL"


def test_normality(
    result:          "RegressionResult",
    use_studentized: bool = True,
) -> NormalityResult:
    """
    Test whether the OLS residuals are consistent with a normal distribution.

    Runs Shapiro-Wilk (n ≤ 5000), Jarque-Bera, and Anderson-Darling tests.

    Parameters
    ----------
    result : RegressionResult
    use_studentized : bool
        If True (default), tests the externally studentized residuals;
        otherwise uses raw residuals.  Studentized residuals are preferable
        because they are approximately i.i.d. t_{n-p-1}.

    Returns
    -------
    NormalityResult
    """
    _require_scipy("test_normality")
    _require_corr("test_normality")

    r = (externally_studentized_residuals(result)
         if use_studentized else result.residuals)
    n = len(r)

    # Shapiro-Wilk (reliable for n ≤ 5000)
    sw_stat = sw_p = None
    if n <= 5000:
        sw_stat, sw_p = _sp.shapiro(r)
        sw_stat, sw_p = float(sw_stat), float(sw_p)

    # Jarque-Bera
    jb_stat, jb_p = _sp.jarque_bera(r)

    # Anderson-Darling
    ad_res = _sp.anderson(r, dist="norm")
    # Critical value at 5% is index 2 in the standard table
    ad_crit = float(ad_res.critical_values[2])

    sk = float(_sp.skew(r))
    ku = float(_sp.kurtosis(r, fisher=True))

    return NormalityResult(
        sw_stat          = sw_stat,
        sw_p             = sw_p,
        jb_stat          = float(jb_stat),
        jb_p             = float(jb_p),
        ad_stat          = float(ad_res.statistic),
        ad_critical_5pct = ad_crit,
        skewness         = sk,
        excess_kurtosis  = ku,
        n                = n,
    )


# ---------------------------------------------------------------------------
# B.2  Homoscedasticity tests
# ---------------------------------------------------------------------------

@dataclass
class HomoscedasticityResult:
    """
    Bundle of heteroscedasticity test results.

    Attributes
    ----------
    bp_stat, bp_p : float
        Breusch-Pagan LM statistic and p-value.
    white_stat, white_p : float
        White's test LM statistic and p-value.
    fitted_resid_r : float
        Pearson r between |residual| and fitted value — a simple
        heteroscedasticity trend indicator.
    n : int
    """
    bp_stat:          float
    bp_p:             float
    white_stat:       float
    white_p:          float
    fitted_resid_r:   float
    n:                int

    def passes(self, alpha: float = 0.05) -> bool:
        """Return True if neither test rejects homoscedasticity at *alpha*."""
        return self.bp_p > alpha and self.white_p > alpha

    @property
    def fan_direction(self) -> str:
        """
        Rough description of the heteroscedasticity pattern based on
        sign of the fitted-|residual| correlation.
        """
        r = self.fitted_resid_r
        if abs(r) < 0.10:
            return "none detected"
        return "fan-out (positive)" if r > 0 else "fan-in (negative)"

    def summary_line(self) -> str:
        return (
            f"Homoscedasticity: "
            f"BP={self.bp_stat:.4f} p={self.bp_p:.4g} | "
            f"White={self.white_stat:.4f} p={self.white_p:.4g} | "
            f"trend r={self.fitted_resid_r:.4f} ({self.fan_direction})"
        )

    def verdict(self, alpha: float = 0.05) -> str:
        return "PASS" if self.passes(alpha) else "FAIL"


def test_homoscedasticity(
    result: "RegressionResult",
) -> HomoscedasticityResult:
    """
    Test the constant-variance (homoscedasticity) assumption.

    Runs:
    * Breusch-Pagan: regress squared residuals on X, LM = n * R^2 ~ chi^2(k).
    * White: regress squared residuals on X, X^2, cross-products, LM ~ chi^2(p-1).

    Parameters
    ----------
    result : RegressionResult

    Returns
    -------
    HomoscedasticityResult
    """
    _require_scipy("test_homoscedasticity")
    _require_corr("test_homoscedasticity")

    X   = result.x
    e   = result.residuals
    e2  = e**2
    n   = result.n
    k   = result.k

    # Breusch-Pagan: LM = n * R^2 from regressing e^2 on X
    b_bp, _, _, _ = np.linalg.lstsq(X, e2, rcond=None)
    e2h_bp = X @ b_bp
    tss2   = np.sum((e2 - e2.mean())**2)
    r2_bp  = (1.0 - np.sum((e2 - e2h_bp)**2) / tss2) if tss2 > 0 else 0.0
    bp_s   = float(n * r2_bp)
    bp_p   = float(1.0 - _sp.chi2.cdf(bp_s, df=k))

    # White test: augment with squared columns and cross-products of X
    x_cols = [X[:, j] for j in range(1, X.shape[1])]  # skip intercept
    white_cols = [np.ones(n)]
    for j, xj in enumerate(x_cols):
        white_cols.append(xj)
        white_cols.append(xj**2)
        for xk in x_cols[j+1:]:
            white_cols.append(xj * xk)
    Xw = np.column_stack(white_cols)
    bw, _, _, _ = np.linalg.lstsq(Xw, e2, rcond=None)
    e2h_w  = Xw @ bw
    r2_w   = (1.0 - np.sum((e2 - e2h_w)**2) / tss2) if tss2 > 0 else 0.0
    white_s = float(n * r2_w)
    white_p = float(1.0 - _sp.chi2.cdf(white_s, df=Xw.shape[1] - 1))

    # Simple trend indicator: r(|e|, y_hat)
    yh  = result.fitted
    abs_e = np.abs(e)
    trend_r = float(np.corrcoef(abs_e, yh)[0, 1]) if np.std(abs_e) > 0 else 0.0

    return HomoscedasticityResult(
        bp_stat        = bp_s,
        bp_p           = bp_p,
        white_stat     = white_s,
        white_p        = white_p,
        fitted_resid_r = trend_r,
        n              = n,
    )


# ---------------------------------------------------------------------------
# B.3  Autocorrelation tests
# ---------------------------------------------------------------------------

@dataclass
class AutocorrResult:
    """
    Bundle of residual autocorrelation test results.

    Attributes
    ----------
    dw_stat : float
        Durbin-Watson statistic.  ~2 = no autocorr, <2 = positive, >2 = negative.
    ljung_box_stat, ljung_box_p : float
        Ljung-Box Q statistic at ``max_lag`` and its p-value (chi-squared df=max_lag).
    acf : ndarray
        Sample ACF values rho_1 … rho_{max_lag}.
    pacf : ndarray
        Sample PACF values phi_1 … phi_{max_lag}.
    max_lag : int
    n : int
    """
    dw_stat:         float
    ljung_box_stat:  float
    ljung_box_p:     float
    acf:             np.ndarray
    pacf:            np.ndarray
    max_lag:         int
    n:               int

    @property
    def dw_conclusion(self) -> str:
        """Informal interpretation of the Durbin-Watson statistic."""
        d = self.dw_stat
        if d < 1.5:
            return "positive autocorrelation"
        if d > 2.5:
            return "negative autocorrelation"
        return "no autocorrelation"

    @property
    def significant_lags(self) -> List[int]:
        """
        Lags where |ACF| exceeds the approximate 95 % CI = ±1.96/sqrt(n).
        """
        ci = 1.96 / math.sqrt(max(self.n, 2))
        return [k + 1 for k, a in enumerate(self.acf) if abs(a) > ci]

    def passes(self, alpha: float = 0.05) -> bool:
        """Return True if Ljung-Box fails to reject no-autocorrelation at alpha."""
        return self.ljung_box_p > alpha

    def summary_line(self) -> str:
        return (
            f"Autocorrelation: "
            f"DW={self.dw_stat:.4f} ({self.dw_conclusion}) | "
            f"Ljung-Box Q({self.max_lag})={self.ljung_box_stat:.4f} "
            f"p={self.ljung_box_p:.4g} | "
            f"sig lags={self.significant_lags}"
        )

    def verdict(self, alpha: float = 0.05) -> str:
        return "PASS" if self.passes(alpha) else "FAIL"


def test_autocorrelation(
    result:  "RegressionResult",
    max_lag: int = 10,
) -> AutocorrResult:
    """
    Test for autocorrelation in the OLS residuals.

    Computes Durbin-Watson, Ljung-Box Q, and sample ACF/PACF.

    Parameters
    ----------
    result : RegressionResult
    max_lag : int
        Number of lags for Ljung-Box and ACF/PACF.

    Returns
    -------
    AutocorrResult
    """
    _require_scipy("test_autocorrelation")
    _require_corr("test_autocorrelation")

    e   = result.residuals
    n   = result.n
    lag = min(max_lag, n // 4)

    # Durbin-Watson
    dw = float(np.sum(np.diff(e)**2) / max(np.sum(e**2), 1e-14))

    # ACF and PACF
    acf_arr  = acf_values(e, max_lag=lag)
    pacf_arr = pacf_values(e, max_lag=lag)

    # Ljung-Box
    acf_sq = acf_arr**2
    ns     = np.arange(1, lag + 1)
    lb_s   = float(n * (n + 2) * np.sum(acf_sq / np.maximum(n - ns, 1)))
    lb_p   = float(1.0 - _sp.chi2.cdf(lb_s, df=lag))

    return AutocorrResult(
        dw_stat        = dw,
        ljung_box_stat = lb_s,
        ljung_box_p    = lb_p,
        acf            = acf_arr,
        pacf           = pacf_arr,
        max_lag        = lag,
        n              = n,
    )


# ---------------------------------------------------------------------------
# B.4  Linearity tests
# ---------------------------------------------------------------------------

@dataclass
class LinearityResult:
    """
    Bundle of linearity-assumption test results.

    Attributes
    ----------
    reset_f, reset_p : float
        Ramsey RESET F-statistic and p-value.
    rainbow_f, rainbow_p : float
        Rainbow test F-statistic and p-value.
    n : int
    """
    reset_f:   float
    reset_p:   float
    rainbow_f: float
    rainbow_p: float
    n:         int

    def passes(self, alpha: float = 0.05) -> bool:
        return self.reset_p > alpha and self.rainbow_p > alpha

    def summary_line(self) -> str:
        return (
            f"Linearity: "
            f"RESET F={self.reset_f:.4f} p={self.reset_p:.4g} | "
            f"Rainbow F={self.rainbow_f:.4f} p={self.rainbow_p:.4g}"
        )

    def verdict(self, alpha: float = 0.05) -> str:
        return "PASS" if self.passes(alpha) else "FAIL"


def test_linearity(
    result:     "RegressionResult",
    reset_order: int = 2,
) -> LinearityResult:
    """
    Test the linearity assumption using the RESET and Rainbow tests.

    RESET (Ramsey 1969)
        Add y_hat^2 and y_hat^3 (``reset_order`` controls the highest power)
        to the model and F-test whether the new terms are jointly zero.

    Rainbow test
        Fit the model to the middle half of observations (sorted by y_hat)
        and test whether RSS_mid / RSS_full exceeds expectations under H0.

    Parameters
    ----------
    result : RegressionResult
    reset_order : int
        Highest power of y_hat added in the RESET test (default 2 → y_hat^2, y_hat^3).

    Returns
    -------
    LinearityResult
    """
    _require_scipy("test_linearity")
    _require_corr("test_linearity")

    X    = result.x
    y    = result.y
    yh   = result.fitted
    e    = result.residuals
    n    = result.n
    k    = result.k
    p    = X.shape[1]

    # RESET
    augments = np.column_stack([yh**(i+2) for i in range(reset_order)])
    Xr = np.column_stack([X, augments])
    br, _, _, _ = np.linalg.lstsq(Xr, y, rcond=None)
    rss_r  = float(np.sum((y - Xr @ br)**2))
    rss_o  = float(np.sum(e**2))
    q_rs   = reset_order
    df_res = n - p - q_rs
    if df_res > 0 and rss_r > 0:
        F_rs = ((rss_o - rss_r) / q_rs) / (rss_r / df_res)
    else:
        F_rs = float("nan")
    p_rs = float(1.0 - _sp.f.cdf(F_rs, q_rs, max(df_res, 1))) if not math.isnan(F_rs) else float("nan")

    # Rainbow — sort by y_hat, fit middle half
    order  = np.argsort(yh)
    lo_idx = n // 4
    hi_idx = 3 * n // 4
    mid    = order[lo_idx:hi_idx]
    nm     = len(mid)
    Xm, ym = X[mid], y[mid]
    bm, _, _, _  = np.linalg.lstsq(Xm, ym, rcond=None)
    rss_m  = float(np.sum((ym - Xm @ bm)**2))
    q_rb   = n - nm
    df_m   = nm - p
    if df_m > 0 and rss_m > 0:
        F_rb = ((rss_o - rss_m) / q_rb) / (rss_m / df_m)
    else:
        F_rb = float("nan")
    p_rb = float(1.0 - _sp.f.cdf(F_rb, q_rb, max(df_m, 1))) if not math.isnan(F_rb) else float("nan")

    return LinearityResult(
        reset_f   = float(F_rs),
        reset_p   = p_rs,
        rainbow_f = float(F_rb),
        rainbow_p = p_rb,
        n         = n,
    )


# ===========================================================================
# LAYER C — ResidualDiagnostics bundle
# ===========================================================================

@dataclass
class ResidualDiagnostics:
    """
    Complete bundle of OLS residual diagnostics for a single fitted model.

    Produced by :func:`diagnose` — the recommended entry-point.

    Attributes
    ----------
    result : RegressionResult
        The original fitted model.
    ordinary : ndarray
        Raw residuals e = y - y_hat.
    standardised : ndarray
        Internally studentized residuals.
    externally_stud : ndarray
        Externally studentized (jackknife) residuals.
    press_resid : ndarray
        PRESS residuals.
    press_stat : float
        PRESS statistic sum(press_resid^2).
    leverage : ndarray
        Hat-matrix diagonal h_ii.
    acf_vals : ndarray
        Sample ACF at lags 1..max_lag.
    pacf_vals : ndarray
        Sample PACF at lags 1..max_lag.
    qq_theoretical : ndarray
        Theoretical normal quantiles for Q-Q plot.
    qq_sample : ndarray
        Sorted externally studentized residuals for Q-Q plot.
    lowess_x, lowess_y : ndarray
        LOWESS smooth of (y_hat, e) for the Residuals vs Fitted plot.
    normality : NormalityResult
    homoscedasticity : HomoscedasticityResult
    autocorr : AutocorrResult
    linearity : LinearityResult
    influence : InfluenceMeasures or None
    """

    result:           "RegressionResult"
    ordinary:         np.ndarray
    standardised:     np.ndarray
    externally_stud:  np.ndarray
    press_resid:      np.ndarray
    press_stat:       float
    leverage:         np.ndarray
    acf_vals:         np.ndarray
    pacf_vals:        np.ndarray
    qq_theoretical:   np.ndarray
    qq_sample:        np.ndarray
    lowess_x:         np.ndarray
    lowess_y:         np.ndarray
    normality:        NormalityResult
    homoscedasticity: HomoscedasticityResult
    autocorr:         AutocorrResult
    linearity:        LinearityResult
    influence:        Optional["InfluenceMeasures"] = None

    # ------------------------------------------------------------------
    # Summary utilities
    # ------------------------------------------------------------------

    def all_pass(self, alpha: float = 0.05) -> bool:
        """Return True if every assumption test passes at *alpha*."""
        return (
            self.normality.passes(alpha)
            and self.homoscedasticity.passes(alpha)
            and self.autocorr.passes(alpha)
            and self.linearity.passes(alpha)
        )

    def failed_assumptions(self, alpha: float = 0.05) -> List[str]:
        """Return list of assumption names that fail at *alpha*."""
        failures = []
        if not self.normality.passes(alpha):
            failures.append("normality")
        if not self.homoscedasticity.passes(alpha):
            failures.append("homoscedasticity")
        if not self.autocorr.passes(alpha):
            failures.append("no_autocorrelation")
        if not self.linearity.passes(alpha):
            failures.append("linearity")
        return failures

    def summary_table(self, alpha: float = 0.05) -> str:
        """
        Return a formatted multi-line diagnostic summary table.

        Example output::

            ┌─────────────────────────────────────────────────────┐
            │  OLS Residual Diagnostics  n=60  k=1  R²=0.7893     │
            ├─────────────┬──────────┬──────────────────────────────┤
            │ Assumption  │ Verdict  │ Key statistics               │
            ├─────────────┼──────────┼──────────────────────────────┤
            │ Normality   │  PASS ✓  │ SW p=0.498  JB p=0.600      │
            │ Homosced.   │  PASS ✓  │ BP p=0.503  White p=0.799   │
            │ No Autocorr │  PASS ✓  │ DW=1.749  LB p=0.010        │
            │ Linearity   │  PASS ✓  │ RESET p=0.213               │
            └─────────────┴──────────┴──────────────────────────────┘
        """
        r   = self.result
        hdr = (f"OLS Residual Diagnostics  "
               f"n={r.n}  k={r.k}  R²={r.r_squared:.4f}")
        rows = [
            ("Normality",   self.normality.verdict(alpha),   self.normality.summary_line()),
            ("Homosced.",   self.homoscedasticity.verdict(alpha), self.homoscedasticity.summary_line()),
            ("No Autocorr", self.autocorr.verdict(alpha),    self.autocorr.summary_line()),
            ("Linearity",   self.linearity.verdict(alpha),   self.linearity.summary_line()),
        ]
        lines = [
            "=" * 72,
            f"  {hdr}",
            "=" * 72,
            f"  {'Assumption':<14}  {'Verdict':<8}  Key statistics",
            "-" * 72,
        ]
        for name, verdict, detail in rows:
            tick = "✓" if verdict == "PASS" else "✗"
            lines.append(f"  {name:<14}  {verdict} {tick}  {detail}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def __repr__(self) -> str:
        fail = self.failed_assumptions()
        status = "OK" if not fail else f"FAIL({','.join(fail)})"
        return (
            f"ResidualDiagnostics("
            f"n={self.result.n}, k={self.result.k}, "
            f"status={status})"
        )


def diagnose(
    result:          "RegressionResult",
    max_lag:         int   = 10,
    lowess_frac:     float = 0.40,
    compute_influence: bool = True,
) -> ResidualDiagnostics:
    """
    Run the full battery of OLS residual diagnostics and return a
    :class:`ResidualDiagnostics` bundle.

    This is the single recommended entry-point for residual analysis.

    Parameters
    ----------
    result : RegressionResult
        A fitted OLS model from ``correlation.ols_fit()``.
    max_lag : int
        Lag limit for ACF/PACF and Ljung-Box.
    lowess_frac : float
        Bandwidth fraction for the LOWESS smoother (0.3–0.6 typical).
    compute_influence : bool
        If True, compute the full InfluenceMeasures (hat matrix, Cook's D, etc.).
        Set False for very large n to save time.

    Returns
    -------
    ResidualDiagnostics
    """
    _require_corr("diagnose")

    e       = result.residuals
    yh      = result.fitted
    h       = _hat_diag(result)
    ord_r   = e.copy()
    std_r   = standardised_residuals(result)
    ext_r   = externally_studentized_residuals(result)
    pr      = press_residuals(result)
    ps      = float(np.sum(pr**2))

    # ACF / PACF
    lag      = min(max_lag, result.n // 4)
    acf_arr  = acf_values(e, max_lag=lag)
    pacf_arr = pacf_values(e, max_lag=lag)

    # Q-Q
    theo_q, samp_q = qq_coordinates(ext_r)

    # LOWESS
    lx, ly = lowess_smooth(yh, e, frac=lowess_frac)

    # Tests
    norm_r  = test_normality(result)
    homo_r  = test_homoscedasticity(result)
    auto_r  = test_autocorrelation(result, max_lag=lag)
    lin_r   = test_linearity(result)

    # Influence
    infl = None
    if compute_influence:
        try:
            infl = influence_measures(result)
        except Exception as ex:
            warnings.warn(f"influence_measures() failed: {ex}", UserWarning, stacklevel=2)

    return ResidualDiagnostics(
        result           = result,
        ordinary         = ord_r,
        standardised     = std_r,
        externally_stud  = ext_r,
        press_resid      = pr,
        press_stat       = ps,
        leverage         = h,
        acf_vals         = acf_arr,
        pacf_vals        = pacf_arr,
        qq_theoretical   = theo_q,
        qq_sample        = samp_q,
        lowess_x         = lx,
        lowess_y         = ly,
        normality        = norm_r,
        homoscedasticity = homo_r,
        autocorr         = auto_r,
        linearity        = lin_r,
        influence        = infl,
    )


# ===========================================================================
# LAYER D — Manim diagnostic-plot mobjects
# ===========================================================================

class ResidualVsFittedPlot(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Residuals vs Fitted values diagnostic plot.

    Contents
    --------
    * 2-D axes with fitted values on x and residuals on y.
    * Scatter dots coloured by sign (teal = positive, coral = negative).
    * LOWESS smooth curve (purple) showing the mean-residual trend.
    * Horizontal dashed zero-reference line.
    * Outlier index labels for |externally studentized residual| > threshold.

    Parameters
    ----------
    diag : ResidualDiagnostics
    width, height : float
        Plot bounding-box size in Manim world-units.
    residual_type : {'ordinary','standardised','externally_stud'}
        Which residual transform to display on the y-axis.
    dot_radius : float
    outlier_threshold : float
        Label observations whose |ext. studentized residual| exceeds this.
    show_lowess : bool
    show_outlier_labels : bool

    Key sub-mobjects
    ----------------
    .axes, .dots, .lowess_curve, .zero_line, .outlier_labels

    Animations
    ----------
    .animate_appear(run_time)
    .highlight_band(y_lo, y_hi, run_time)  — shade a horizontal band
    .morph_to(new_diag, run_time)          — transition to a different fit
    """

    def __init__(
        self,
        diag:                ResidualDiagnostics,
        width:               float = 4.0,
        height:              float = 3.2,
        residual_type:       str   = "externally_stud",
        dot_radius:          float = 0.055,
        outlier_threshold:   float = 2.5,
        show_lowess:         bool  = True,
        show_outlier_labels: bool  = True,
        title:               str   = "Residuals vs Fitted",
        **kwargs,
    ) -> None:
        _require_manim("ResidualVsFittedPlot")
        super().__init__(**kwargs)

        self._diag     = diag
        self._width    = width
        self._height   = height
        self._rtype    = residual_type
        self._dot_r    = dot_radius
        self._out_thr  = outlier_threshold
        self._show_low = show_lowess
        self._title    = title

        # Resolve residuals
        r_map = {
            "ordinary":         diag.ordinary,
            "standardised":     diag.standardised,
            "externally_stud":  diag.externally_stud,
        }
        resid   = r_map.get(residual_type, diag.externally_stud)
        yh      = diag.result.fitted

        # Colours
        if _COLORS_AVAILABLE:
            self._pos_color   = ManimColor(TEAL_600.hex)
            self._neg_color   = ManimColor(CORAL_600.hex)
            self._lowess_color= ManimColor(PURPLE_600.hex)
            self._axis_color  = ManimColor(GRAY_400.hex)
            self._label_color = ManimColor(AMBER_600.hex)
        else:
            self._pos_color    = GREEN
            self._neg_color    = RED
            self._lowess_color = PURPLE
            self._axis_color   = GRAY
            self._label_color  = YELLOW

        self._build(yh, resid, diag, show_lowess, show_outlier_labels,
                    dot_radius, outlier_threshold, width, height, title)

    # ------------------------------------------------------------------
    def _build(self, yh, resid, diag, show_lowess, show_labels,
               dot_r, out_thr, w, h, title) -> None:
        y_margin  = max(abs(resid)) * 0.18
        x_margin  = (yh.max() - yh.min()) * 0.12

        self.axes = Axes(
            x_range = [yh.min()  - x_margin, yh.max()  + x_margin,
                       (yh.max() - yh.min()) / 4],
            y_range = [resid.min() - y_margin, resid.max() + y_margin,
                       (resid.max() - resid.min()) / 4],
            x_length = w,
            y_length = h,
            axis_config = {
                "color": self._axis_color,
                "stroke_width": 1.4,
                "include_tip": True,
                "tip_width": 0.10,
                "tip_height": 0.10,
            },
        )

        # Axis labels
        x_lbl = MathTex(r"\hat{y}", font_size=20, color=self._axis_color)
        y_lbl = MathTex("e", font_size=20, color=self._axis_color)
        x_lbl.next_to(self.axes.x_axis, DOWN, buff=0.22)
        y_lbl.next_to(self.axes.y_axis, LEFT, buff=0.22)

        title_mob = Text(title, font_size=18, color=self._axis_color)
        title_mob.next_to(self.axes, UP, buff=0.12)

        # Zero reference line
        self.zero_line = DashedLine(
            start        = self.axes.c2p(yh.min() - x_margin, 0),
            end          = self.axes.c2p(yh.max() + x_margin, 0),
            color        = self._axis_color,
            stroke_width = 1.0,
            dash_length  = 0.10,
        )

        # Scatter dots
        self.dots = VGroup()
        for i in range(len(resid)):
            color = self._pos_color if resid[i] >= 0 else self._neg_color
            dot   = Dot(
                point  = self.axes.c2p(float(yh[i]), float(resid[i])),
                radius = dot_r,
                color  = color,
            )
            self.dots.add(dot)

        # LOWESS curve
        self.lowess_curve = VGroup()
        if show_lowess and len(diag.lowess_x) > 1:
            pts = [
                self.axes.c2p(float(diag.lowess_x[i]), float(diag.lowess_y[i]))
                for i in range(len(diag.lowess_x))
            ]
            for i in range(len(pts) - 1):
                seg = Line(
                    start        = pts[i],
                    end          = pts[i + 1],
                    color        = self._lowess_color,
                    stroke_width = 2.0,
                )
                self.lowess_curve.add(seg)

        # Outlier labels
        self.outlier_labels = VGroup()
        if show_labels:
            ext_r = diag.externally_stud
            for i, er in enumerate(ext_r):
                if abs(er) > out_thr:
                    lbl = Text(
                        str(i),
                        font_size = 14,
                        color     = self._label_color,
                    ).next_to(self.dots[i], UP if er > 0 else DOWN, buff=0.05)
                    self.outlier_labels.add(lbl)

        self.add(self.axes, x_lbl, y_lbl, title_mob,
                 self.zero_line, self.dots,
                 self.lowess_curve, self.outlier_labels)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def animate_appear(self, run_time: float = 1.8) -> "mn.Animation":
        """Create axes, then stagger dots in, then draw LOWESS."""
        _require_manim("animate_appear")
        t = run_time
        return Succession(
            Create(self.axes, run_time=t * 0.25),
            Create(self.zero_line, run_time=t * 0.10),
            LaggedStart(
                *[FadeIn(d, scale=0.5) for d in self.dots],
                lag_ratio=0.025, run_time=t * 0.40,
            ),
            Create(self.lowess_curve, run_time=t * 0.25) if self.lowess_curve else AnimationGroup(),
        )

    def highlight_band(
        self,
        y_lo:     float,
        y_hi:     float,
        color     = None,
        run_time: float = 0.6,
    ) -> "mn.Animation":
        """
        Shade a horizontal band [y_lo, y_hi] on the residual axis.

        Useful for highlighting the ±2 sigma band.
        """
        _require_manim("highlight_band")
        color = color or (self._pos_color if _COLORS_AVAILABLE else GREEN)
        x0 = self.axes.x_range[0]
        x1 = self.axes.x_range[1]
        corners = [
            self.axes.c2p(x0, y_lo),
            self.axes.c2p(x1, y_lo),
            self.axes.c2p(x1, y_hi),
            self.axes.c2p(x0, y_hi),
        ]
        band = VMobject(
            fill_color   = color,
            fill_opacity = 0.18,
            stroke_width = 0,
        )
        band.set_points_as_corners([*corners, corners[0]])
        self.add(band)
        return FadeIn(band, run_time=run_time)

    def morph_to(
        self,
        new_diag: ResidualDiagnostics,
        run_time: float = 1.2,
    ) -> "mn.Animation":
        """Cross-fade to a new diagnostic's residual plot."""
        _require_manim("morph_to")
        target = ResidualVsFittedPlot(
            diag          = new_diag,
            width         = self._width,
            height        = self._height,
            residual_type = self._rtype,
            dot_radius    = self._dot_r,
            show_lowess   = self._show_low,
            title         = self._title,
        ).move_to(self.get_center())
        return Transform(self, target, run_time=run_time,
                         rate_func=rate_functions.ease_in_out_sine)


# ---------------------------------------------------------------------------
# QQPlot3D
# ---------------------------------------------------------------------------

class QQPlot3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Normal Q-Q plot of residuals.

    Plots the sample quantiles of the (externally studentized) residuals
    against the theoretical quantiles of N(0,1).  Points on the 45° reference
    line indicate normality.  Colour encodes deviation from the line.

    Parameters
    ----------
    diag : ResidualDiagnostics
    width, height : float
    dot_radius : float
    show_ci_lines : bool
        Draw approximate 95 % CI envelopes around the reference line.

    Key sub-mobjects
    ----------------
    .axes, .dots, .reference_line, .ci_upper, .ci_lower

    Animations
    ----------
    .animate_build(run_time)
    .morph_distribution(dist_name, run_time)
    """

    def __init__(
        self,
        diag:           ResidualDiagnostics,
        width:          float = 3.8,
        height:         float = 3.2,
        dot_radius:     float = 0.055,
        show_ci_lines:  bool  = True,
        title:          str   = "Normal Q-Q",
        **kwargs,
    ) -> None:
        _require_manim("QQPlot3D")
        super().__init__(**kwargs)

        self._diag   = diag
        self._width  = width
        self._height = height
        self._dot_r  = dot_radius
        self._title  = title

        if _COLORS_AVAILABLE:
            self._ref_color  = ManimColor(REGRESSION_FAMILY.base.hex)
            self._pos_color  = ManimColor(CORAL_600.hex)
            self._neg_color  = ManimColor(TEAL_600.hex)
            self._on_color   = ManimColor(REGRESSION_FAMILY.light.hex)
            self._axis_color = ManimColor(GRAY_400.hex)
            self._ci_color   = ManimColor(GRAY_200.hex)
        else:
            self._ref_color = self._on_color = PURPLE
            self._pos_color = RED
            self._neg_color = GREEN
            self._axis_color = GRAY
            self._ci_color = LIGHT_GRAY

        self._build(diag, width, height, dot_radius, show_ci_lines, title)

    def _build(self, diag, w, h, dot_r, show_ci, title) -> None:
        tq = diag.qq_theoretical
        sq = diag.qq_sample

        margin = max(abs(tq).max(), abs(sq).max()) * 0.18

        self.axes = Axes(
            x_range = [tq.min() - margin, tq.max() + margin,
                       (tq.max() - tq.min()) / 4],
            y_range = [sq.min() - margin, sq.max() + margin,
                       (sq.max() - sq.min()) / 4],
            x_length = w,
            y_length = h,
            axis_config = {"color": self._axis_color, "stroke_width": 1.4,
                           "include_tip": True, "tip_width": 0.10, "tip_height": 0.10},
        )

        x_lbl = MathTex(r"\text{Theoretical quantiles}", font_size=16, color=self._axis_color)
        y_lbl = MathTex(r"\text{Sample quantiles}",     font_size=16, color=self._axis_color)
        x_lbl.next_to(self.axes.x_axis, DOWN, buff=0.22)
        y_lbl.next_to(self.axes.y_axis, LEFT, buff=0.22).rotate(PI / 2)
        title_mob = Text(title, font_size=18, color=self._axis_color)
        title_mob.next_to(self.axes, UP, buff=0.12)

        # Reference line y = x
        lo = min(tq.min(), sq.min()) - margin * 0.5
        hi = max(tq.max(), sq.max()) + margin * 0.5
        self.reference_line = Line(
            start        = self.axes.c2p(lo, lo),
            end          = self.axes.c2p(hi, hi),
            color        = self._ref_color,
            stroke_width = 1.5,
        )

        # Approximate 95% CI envelope: ±1.36/sqrt(n) * (1/pdf(tq))
        # Use the simple Kolmogorov-Smirnov band approximation
        self.ci_upper = VGroup()
        self.ci_lower = VGroup()
        if show_ci and _SCIPY_AVAILABLE:
            n = len(sq)
            ks_crit = 1.36 / math.sqrt(n)
            ci_pts_up   = []
            ci_pts_down = []
            for t in tq:
                phi_t = float(_sp.norm.pdf(t))
                if phi_t > 1e-4:
                    delta = ks_crit / phi_t
                    ci_pts_up.append(self.axes.c2p(float(t), float(t) + delta))
                    ci_pts_down.append(self.axes.c2p(float(t), float(t) - delta))
            if len(ci_pts_up) > 1:
                for i in range(len(ci_pts_up) - 1):
                    self.ci_upper.add(
                        Line(ci_pts_up[i], ci_pts_up[i+1],
                             color=self._ci_color, stroke_width=0.8)
                    )
                    self.ci_lower.add(
                        Line(ci_pts_down[i], ci_pts_down[i+1],
                             color=self._ci_color, stroke_width=0.8)
                    )

        # Scatter dots coloured by deviation from ref line
        self.dots = VGroup()
        for i in range(len(tq)):
            dev   = float(sq[i]) - float(tq[i])
            t_dev = min(abs(dev) / max(abs(sq - tq).max(), 1e-6), 1.0)
            color = (interpolate_color(self._on_color, self._pos_color, t_dev)
                     if dev > 0 else
                     interpolate_color(self._on_color, self._neg_color, t_dev))
            dot   = Dot(
                point  = self.axes.c2p(float(tq[i]), float(sq[i])),
                radius = dot_r,
                color  = color,
            )
            self.dots.add(dot)

        self.add(self.axes, x_lbl, y_lbl, title_mob,
                 self.ci_upper, self.ci_lower,
                 self.reference_line, self.dots)

    # ------------------------------------------------------------------
    def animate_build(self, run_time: float = 2.0) -> "mn.Animation":
        """Draw reference line, then stagger dots onto the plot."""
        _require_manim("animate_build")
        return Succession(
            Create(self.axes, run_time=run_time * 0.25),
            Create(self.reference_line, run_time=run_time * 0.15),
            FadeIn(self.ci_upper, run_time=run_time * 0.10) if self.ci_upper else AnimationGroup(),
            FadeIn(self.ci_lower, run_time=run_time * 0.10) if self.ci_lower else AnimationGroup(),
            LaggedStart(
                *[FadeIn(d, scale=0.5) for d in self.dots],
                lag_ratio=0.03, run_time=run_time * 0.40,
            ),
        )


# ---------------------------------------------------------------------------
# ScaleLocationPlot
# ---------------------------------------------------------------------------

class ScaleLocationPlot(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Scale-Location (Spread-Level) plot: sqrt(|standardised residuals|) vs fitted.

    A flat LOWESS through this plot confirms homoscedasticity.  An upward
    slope signals fan-out (variance increases with fitted values).

    Parameters
    ----------
    diag : ResidualDiagnostics
    width, height : float
    show_lowess : bool

    Key sub-mobjects
    ----------------
    .axes, .dots, .lowess_curve

    Animations
    ----------
    .animate_appear(run_time)
    """

    def __init__(
        self,
        diag:        ResidualDiagnostics,
        width:       float = 3.8,
        height:      float = 3.2,
        dot_radius:  float = 0.055,
        show_lowess: bool  = True,
        title:       str   = "Scale-Location",
        **kwargs,
    ) -> None:
        _require_manim("ScaleLocationPlot")
        super().__init__(**kwargs)

        self._diag   = diag
        self._width  = width
        self._height = height

        if _COLORS_AVAILABLE:
            dot_color      = ManimColor(NORMAL_FAMILY.base.hex)
            lowess_color   = ManimColor(REGRESSION_FAMILY.base.hex)
            self._ax_color = ManimColor(GRAY_400.hex)
        else:
            dot_color = lowess_color = BLUE
            self._ax_color = GRAY

        yh      = diag.result.fitted
        sqrt_r  = np.sqrt(np.abs(diag.standardised))
        lx, ly  = lowess_smooth(yh, sqrt_r)

        y_margin  = sqrt_r.max() * 0.18
        x_margin  = (yh.max() - yh.min()) * 0.12

        self.axes = Axes(
            x_range = [yh.min() - x_margin, yh.max() + x_margin,
                       (yh.max() - yh.min()) / 4],
            y_range = [0, sqrt_r.max() + y_margin,
                       (sqrt_r.max()) / 4],
            x_length = width,
            y_length = height,
            axis_config = {"color": self._ax_color, "stroke_width": 1.4,
                           "include_tip": True, "tip_width": 0.10, "tip_height": 0.10},
        )

        x_lbl = MathTex(r"\hat{y}", font_size=20, color=self._ax_color)
        y_lbl = MathTex(r"\sqrt{|r_i|}", font_size=20, color=self._ax_color)
        x_lbl.next_to(self.axes.x_axis, DOWN, buff=0.22)
        y_lbl.next_to(self.axes.y_axis, LEFT, buff=0.22)
        title_mob = Text(title, font_size=18, color=self._ax_color)
        title_mob.next_to(self.axes, UP, buff=0.12)

        self.dots = VGroup(*[
            Dot(self.axes.c2p(float(yh[i]), float(sqrt_r[i])),
                radius=dot_radius, color=dot_color)
            for i in range(len(yh))
        ])

        self.lowess_curve = VGroup()
        if show_lowess and len(lx) > 1:
            pts = [self.axes.c2p(float(lx[i]), float(ly[i])) for i in range(len(lx))]
            for i in range(len(pts) - 1):
                self.lowess_curve.add(
                    Line(pts[i], pts[i+1], color=lowess_color, stroke_width=2.0)
                )

        self.add(self.axes, x_lbl, y_lbl, title_mob, self.dots, self.lowess_curve)

    def animate_appear(self, run_time: float = 1.6) -> "mn.Animation":
        """Axes draw in, then dots stagger."""
        _require_manim("animate_appear")
        return Succession(
            Create(self.axes, run_time=run_time * 0.28),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in self.dots],
                        lag_ratio=0.025, run_time=run_time * 0.45),
            Create(self.lowess_curve, run_time=run_time * 0.27) if self.lowess_curve else AnimationGroup(),
        )


# ---------------------------------------------------------------------------
# InfluencePlot3D
# ---------------------------------------------------------------------------

class InfluencePlot3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    Influence plot (leverage vs externally studentized residual).

    Each observation is a bubble:
      * x position = leverage h_ii
      * y position = externally studentized residual t_i
      * bubble radius = sqrt(Cook's D_i) (scaled to fit axes)
      * color = magnitude of Cook's D (sequential coral)

    Horizontal dashed lines at ±outlier_threshold and vertical dashed
    line at the high-leverage threshold (2*(k+1)/n).  Cook's D = 0.5 and
    D = 1.0 contour curves drawn as guidelines.

    Parameters
    ----------
    diag : ResidualDiagnostics
    width, height : float
    outlier_threshold : float
    cook_contours : list[float]
        Cook's D values for which contour lines are drawn.

    Key sub-mobjects
    ----------------
    .axes, .bubbles, .threshold_lines, .cook_contours

    Animations
    ----------
    .animate_appear(run_time)
    .highlight_influential(cooks_thresh, run_time)
    """

    def __init__(
        self,
        diag:               ResidualDiagnostics,
        width:              float = 3.8,
        height:             float = 3.2,
        outlier_threshold:  float = 2.5,
        cook_contours:      Sequence[float] = (0.5, 1.0),
        title:              str   = "Influence Plot",
        **kwargs,
    ) -> None:
        _require_manim("InfluencePlot3D")
        super().__init__(**kwargs)

        self._diag  = diag
        self._width = width
        self._height = height
        self._out_t = outlier_threshold

        if _COLORS_AVAILABLE:
            self._lo_color  = ManimColor(CORAL_200.hex)
            self._hi_color  = ManimColor(CORAL_800.hex)
            self._ax_color  = ManimColor(GRAY_400.hex)
            self._thr_color = ManimColor(AMBER_600.hex)
        else:
            self._lo_color = ORANGE
            self._hi_color = RED
            self._ax_color = GRAY
            self._thr_color = YELLOW

        self._build(diag, width, height, outlier_threshold, cook_contours, title)

    def _build(self, diag, w, h, out_t, contours, title) -> None:
        h_arr  = diag.leverage
        t_arr  = diag.externally_stud
        cd_arr = diag.influence.cooks_d if diag.influence is not None else np.zeros(diag.result.n)
        n      = diag.result.n
        k      = diag.result.k

        h_max  = float(h_arr.max()) * 1.25 + 0.02
        t_lim  = max(abs(t_arr).max() * 1.25, float(out_t) * 1.4)

        self.axes = Axes(
            x_range = [0, h_max, h_max / 4],
            y_range = [-t_lim, t_lim, t_lim / 2],
            x_length = w,
            y_length = h,
            axis_config = {"color": self._ax_color, "stroke_width": 1.4,
                           "include_tip": True, "tip_width": 0.10, "tip_height": 0.10},
        )
        x_lbl = MathTex(r"h_{ii}", font_size=20, color=self._ax_color)
        y_lbl = MathTex(r"t_i",   font_size=20, color=self._ax_color)
        x_lbl.next_to(self.axes.x_axis, DOWN, buff=0.22)
        y_lbl.next_to(self.axes.y_axis, LEFT, buff=0.22)
        title_mob = Text(title, font_size=18, color=self._ax_color)
        title_mob.next_to(self.axes, UP, buff=0.12)

        # Threshold lines
        self.threshold_lines = VGroup()
        lev_thresh = 2.0 * (k + 1) / n
        for x_v in ([lev_thresh] if lev_thresh < h_max else []):
            self.threshold_lines.add(
                DashedLine(
                    self.axes.c2p(x_v, -t_lim),
                    self.axes.c2p(x_v, t_lim),
                    color=self._thr_color, stroke_width=1.0, dash_length=0.08,
                )
            )
        for y_v in (-out_t, out_t):
            self.threshold_lines.add(
                DashedLine(
                    self.axes.c2p(0, y_v),
                    self.axes.c2p(h_max, y_v),
                    color=self._thr_color, stroke_width=1.0, dash_length=0.08,
                )
            )

        # Cook's D contour curves: D = C means (t^2 / p) * (h/(1-h)^2) = C
        # => t = ±sqrt(C * p * (1-h)^2 / h)
        self.cook_contour_group = VGroup()
        p = k + 1
        for C in contours:
            for sign in (1, -1):
                pts = []
                for hv in np.linspace(max(lev_thresh * 0.5, 0.01), h_max * 0.95, 80):
                    val = C * p * (1 - hv)**2 / hv
                    if val >= 0:
                        tv = sign * math.sqrt(val)
                        if -t_lim <= tv <= t_lim:
                            pts.append(self.axes.c2p(hv, tv))
                if len(pts) > 1:
                    for i in range(len(pts) - 1):
                        self.cook_contour_group.add(
                            Line(pts[i], pts[i+1],
                                 color=self._thr_color,
                                 stroke_width=0.7)
                        )

        # Bubbles
        cd_max  = max(float(cd_arr.max()), 1e-6)
        max_r   = min(w, h) * 0.09
        min_r   = max_r * 0.15

        self.bubbles = VGroup()
        for i in range(n):
            hi   = float(h_arr[i])
            ti   = float(t_arr[i])
            cd   = float(cd_arr[i])
            t_cd = min(cd / cd_max, 1.0)
            color = interpolate_color(self._lo_color, self._hi_color, t_cd)
            radius = min_r + (max_r - min_r) * math.sqrt(t_cd)
            dot = Dot(
                point  = self.axes.c2p(hi, ti),
                radius = radius,
                color  = color,
            )
            dot.set_fill(opacity=0.7)
            dot.set_stroke(color=self._hi_color, width=0.5)
            self.bubbles.add(dot)

        self.add(self.axes, x_lbl, y_lbl, title_mob,
                 self.threshold_lines, self.cook_contour_group, self.bubbles)

    # ------------------------------------------------------------------
    def animate_appear(self, run_time: float = 1.8) -> "mn.Animation":
        _require_manim("animate_appear")
        return Succession(
            Create(self.axes, run_time=run_time * 0.25),
            FadeIn(self.threshold_lines, run_time=run_time * 0.10),
            FadeIn(self.cook_contour_group, run_time=run_time * 0.10),
            LaggedStart(*[FadeIn(b, scale=0.5) for b in self.bubbles],
                        lag_ratio=0.03, run_time=run_time * 0.55),
        )

    def highlight_influential(
        self,
        cooks_thresh: float = 0.5,
        run_time:     float = 1.0,
    ) -> "mn.Animation":
        """Flash all bubbles whose Cook's D exceeds cooks_thresh."""
        _require_manim("highlight_influential")
        cd_arr  = (self._diag.influence.cooks_d
                   if self._diag.influence is not None
                   else np.zeros(self._diag.result.n))
        targets = [self.bubbles[i] for i, cd in enumerate(cd_arr)
                   if cd > cooks_thresh]
        if not targets:
            return AnimationGroup()
        return LaggedStart(
            *[AnimationGroup(
                Flash(b, color=YELLOW, run_time=run_time, flash_radius=0.2),
                Indicate(b, color=YELLOW, scale_factor=1.8, run_time=run_time),
              ) for b in targets],
            lag_ratio=0.12,
        )


# ---------------------------------------------------------------------------
# ACFPlot3D
# ---------------------------------------------------------------------------

class ACFPlot3D(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    ACF or PACF bar-chart plot.

    Parameters
    ----------
    diag : ResidualDiagnostics
    kind : {'acf', 'pacf'}
    width, height : float
    show_ci : bool
        Draw ±1.96/sqrt(n) significance lines.

    Key sub-mobjects
    ----------------
    .axes, .bars, .ci_lines, .zero_line

    Animations
    ----------
    .animate_appear(run_time)
    .flash_significant(run_time)
    """

    def __init__(
        self,
        diag:     ResidualDiagnostics,
        kind:     str   = "acf",
        width:    float = 3.8,
        height:   float = 3.2,
        show_ci:  bool  = True,
        title:    Optional[str] = None,
        **kwargs,
    ) -> None:
        _require_manim("ACFPlot3D")
        super().__init__(**kwargs)

        self._diag  = diag
        self._kind  = kind
        self._width = width
        self._height = height

        if _COLORS_AVAILABLE:
            bar_color_pos  = ManimColor(TEAL_600.hex)
            bar_color_neg  = ManimColor(CORAL_600.hex)
            ci_color       = ManimColor(AMBER_600.hex)
            self._ax_color = ManimColor(GRAY_400.hex)
        else:
            bar_color_pos = GREEN
            bar_color_neg = RED
            ci_color      = YELLOW
            self._ax_color = GRAY

        values = diag.acf_vals if kind == "acf" else diag.pacf_vals
        n      = diag.result.n
        m      = len(values)
        ci     = 1.96 / math.sqrt(max(n, 2))

        title_str = title or ("ACF" if kind == "acf" else "PACF")
        y_max     = max(abs(values).max() * 1.25, ci * 1.6, 0.2)

        self.axes = Axes(
            x_range = [0, m + 1, max(m // 4, 1)],
            y_range = [-y_max, y_max, y_max / 2],
            x_length = width,
            y_length = height,
            axis_config = {"color": self._ax_color, "stroke_width": 1.4,
                           "include_tip": True, "tip_width": 0.10, "tip_height": 0.10},
        )
        x_lbl = Text("Lag", font_size=18, color=self._ax_color)
        y_lbl = MathTex(r"\hat{\rho}" if kind == "acf" else r"\hat{\phi}",
                        font_size=20, color=self._ax_color)
        x_lbl.next_to(self.axes.x_axis, DOWN, buff=0.22)
        y_lbl.next_to(self.axes.y_axis, LEFT, buff=0.22)
        title_mob = Text(title_str, font_size=18, color=self._ax_color)
        title_mob.next_to(self.axes, UP, buff=0.12)

        # Zero line
        self.zero_line = DashedLine(
            self.axes.c2p(0, 0), self.axes.c2p(m + 1, 0),
            color=self._ax_color, stroke_width=0.8, dash_length=0.10,
        )

        # Bars
        bar_width_frac = 0.55
        x_step = (self.axes.c2p(2, 0) - self.axes.c2p(1, 0))[0]
        self.bars = VGroup()
        for lag_idx, val in enumerate(values):
            x_pos = float(self.axes.c2p(lag_idx + 1, 0)[0])
            y0    = float(self.axes.c2p(0, 0)[1])
            y1    = float(self.axes.c2p(0, float(val))[1])
            color = bar_color_pos if val >= 0 else bar_color_neg
            bar   = Line(
                start        = [x_pos, y0, 0],
                end          = [x_pos, y1, 0],
                color        = color,
                stroke_width = abs(x_step) * bar_width_frac * 50,
            )
            self.bars.add(bar)

        # CI lines
        self.ci_lines = VGroup()
        if show_ci:
            for sign in (1, -1):
                self.ci_lines.add(
                    DashedLine(
                        self.axes.c2p(0, sign * ci),
                        self.axes.c2p(m + 1, sign * ci),
                        color=ci_color, stroke_width=1.2, dash_length=0.08,
                    )
                )

        self._values = values
        self._ci     = ci

        self.add(self.axes, x_lbl, y_lbl, title_mob,
                 self.zero_line, self.bars, self.ci_lines)

    def animate_appear(self, run_time: float = 1.4) -> "mn.Animation":
        """Axes draw in, then bars grow from zero."""
        _require_manim("animate_appear")
        return Succession(
            Create(self.axes, run_time=run_time * 0.30),
            Create(self.zero_line, run_time=run_time * 0.08),
            FadeIn(self.ci_lines, run_time=run_time * 0.12),
            LaggedStart(*[Create(b) for b in self.bars],
                        lag_ratio=0.06, run_time=run_time * 0.50),
        )

    def flash_significant(self, run_time: float = 0.8) -> "mn.Animation":
        """Flash bars that exceed the 95 % CI threshold."""
        _require_manim("flash_significant")
        sig = [self.bars[i] for i, v in enumerate(self._values)
               if abs(v) > self._ci]
        if not sig:
            return AnimationGroup()
        return LaggedStart(
            *[Flash(b, color=YELLOW, run_time=run_time, flash_radius=0.1)
              for b in sig],
            lag_ratio=0.1,
        )


# ---------------------------------------------------------------------------
# DiagnosticPanel — 2×2 grid of all four plots
# ---------------------------------------------------------------------------

class DiagnosticPanel(VGroup if _MANIM_AVAILABLE else object):  # type: ignore
    """
    A 2×2 grid containing the four canonical OLS diagnostic plots:

        ┌──────────────────┬──────────────────┐
        │  Residuals vs    │  Normal Q-Q       │
        │  Fitted          │                   │
        ├──────────────────┼──────────────────┤
        │  Scale-Location  │  Influence Plot   │
        └──────────────────┴──────────────────┘

    Parameters
    ----------
    diag : ResidualDiagnostics
    cell_width, cell_height : float
        Size of each sub-plot cell.
    gap : float
        Spacing between cells.

    Key sub-mobjects
    ----------------
    .rvf_plot   : ResidualVsFittedPlot
    .qq_plot    : QQPlot3D
    .sl_plot    : ScaleLocationPlot
    .infl_plot  : InfluencePlot3D

    Animations
    ----------
    .animate_build_sequence(run_time)  — reveal plots one at a time
    .highlight_assumption(name)        — Indicate the relevant cell
    .summary_text_panel(alpha)         — Write verdict text beside the grid
    """

    def __init__(
        self,
        diag:        ResidualDiagnostics,
        cell_width:  float = 3.5,
        cell_height: float = 2.8,
        gap:         float = 0.25,
        **kwargs,
    ) -> None:
        _require_manim("DiagnosticPanel")
        super().__init__(**kwargs)

        w, h, g = cell_width, cell_height, gap

        self.rvf_plot  = ResidualVsFittedPlot(diag, width=w, height=h,
                                               title="Residuals vs Fitted")
        self.qq_plot   = QQPlot3D(diag,           width=w, height=h,
                                   title="Normal Q-Q")
        self.sl_plot   = ScaleLocationPlot(diag,  width=w, height=h,
                                           title="Scale-Location")
        self.infl_plot = InfluencePlot3D(diag,    width=w, height=h,
                                         title="Influence Plot")

        # Position: top-left, top-right, bottom-left, bottom-right
        half_w = (w + g) / 2
        half_h = (h + g) / 2

        self.rvf_plot .move_to([-half_w,  half_h, 0])
        self.qq_plot  .move_to([ half_w,  half_h, 0])
        self.sl_plot  .move_to([-half_w, -half_h, 0])
        self.infl_plot.move_to([ half_w, -half_h, 0])

        self.add(self.rvf_plot, self.qq_plot,
                 self.sl_plot,  self.infl_plot)

    # ------------------------------------------------------------------
    def animate_build_sequence(self, run_time: float = 7.0) -> "mn.Animation":
        """Reveal all four plots in sequence."""
        _require_manim("animate_build_sequence")
        t = run_time / 4
        return Succession(
            self.rvf_plot .animate_appear(run_time=t),
            self.qq_plot  .animate_build(run_time=t),
            self.sl_plot  .animate_appear(run_time=t),
            self.infl_plot.animate_appear(run_time=t),
        )

    def highlight_assumption(
        self,
        name:     str,
        run_time: float = 0.8,
    ) -> "mn.Animation":
        """
        Flash a SurroundingRectangle around the plot most relevant to
        the named assumption.

        Parameters
        ----------
        name : {'normality', 'homoscedasticity', 'no_autocorrelation', 'linearity', 'influence'}
        """
        _require_manim("highlight_assumption")
        plot_map = {
            "normality":           self.qq_plot,
            "homoscedasticity":    self.sl_plot,
            "no_autocorrelation":  self.rvf_plot,
            "linearity":           self.rvf_plot,
            "influence":           self.infl_plot,
        }
        target = plot_map.get(name, self.rvf_plot)
        ring   = SurroundingRectangle(target, color=YELLOW, buff=0.06)
        return Succession(
            FadeIn(ring, run_time=run_time * 0.3),
            Indicate(target, color=YELLOW, scale_factor=1.02, run_time=run_time * 0.5),
            FadeOut(ring, run_time=run_time * 0.2),
        )

    def summary_text_panel(
        self,
        alpha:    float = 0.05,
        run_time: float = 1.2,
    ) -> "mn.Animation":
        """
        Write a compact verdicts panel (PASS / FAIL) beside the grid.

        Returns a Write animation for the text group.
        """
        _require_manim("summary_text_panel")
        diag = self.rvf_plot._diag

        if _COLORS_AVAILABLE:
            pass_c = ManimColor(TEAL_600.hex)
            fail_c = ManimColor(CORAL_600.hex)
            base_c = ManimColor(GRAY_400.hex)
        else:
            pass_c = GREEN
            fail_c = RED
            base_c = GRAY

        lines = [
            ("Normality",       diag.normality.verdict(alpha)),
            ("Homoscedasticity",diag.homoscedasticity.verdict(alpha)),
            ("No autocorr.",    diag.autocorr.verdict(alpha)),
            ("Linearity",       diag.linearity.verdict(alpha)),
        ]
        text_group = VGroup()
        for i, (lbl, verdict) in enumerate(lines):
            color = pass_c if verdict == "PASS" else fail_c
            tick  = "✓" if verdict == "PASS" else "✗"
            mob   = Text(f"{lbl}: {verdict} {tick}", font_size=16, color=color)
            mob.move_to(np.array([0, -i * 0.38, 0]))
            text_group.add(mob)
        text_group.next_to(self, RIGHT, buff=0.4)
        self.add(text_group)
        return Write(text_group, run_time=run_time)


# ===========================================================================
# LAYER E — Scene-level animation factories
# ===========================================================================

def build_diagnostic_scene(
    result:      "RegressionResult",
    scene,
    layout:      str   = "panel",
    run_time:    float = 8.0,
    alpha:       float = 0.05,
) -> "DiagnosticPanel":
    """
    Build and animate the complete OLS diagnostic scene on *scene*.

    Sequence
    --------
    1. Compute full ResidualDiagnostics.
    2. Build a DiagnosticPanel.
    3. Play animate_build_sequence.
    4. Print summary_text_panel.
    5. Highlight any failed assumptions.

    Parameters
    ----------
    result : RegressionResult
    scene : manim.Scene
    layout : {'panel'}
        Currently only the 2×2 panel layout is supported.
    run_time : float
        Total scene duration.
    alpha : float
        Significance level for pass/fail decisions.

    Returns
    -------
    DiagnosticPanel
    """
    _require_manim("build_diagnostic_scene")
    _require_corr("build_diagnostic_scene")

    diag  = diagnose(result)
    panel = DiagnosticPanel(diag)
    panel.center()

    scene.play(panel.animate_build_sequence(run_time=run_time * 0.70))
    scene.play(panel.summary_text_panel(alpha=alpha, run_time=run_time * 0.15))

    for assumption in diag.failed_assumptions(alpha):
        scene.play(panel.highlight_assumption(assumption, run_time=0.8))

    return panel


def animate_assumption_violation(
    scene,
    result_good:     "RegressionResult",
    result_bad:      "RegressionResult",
    assumption:      str,
    run_time:        float = 4.0,
) -> None:
    """
    Demonstrate an assumption violation by morphing from a well-specified
    model to a mis-specified one, then highlighting the failing diagnostic plot.

    Parameters
    ----------
    scene : manim.Scene
    result_good : RegressionResult
        A model where the assumption holds.
    result_bad : RegressionResult
        A model where the assumption is violated.
    assumption : str
        One of 'normality', 'homoscedasticity', 'linearity'.
    run_time : float
    """
    _require_manim("animate_assumption_violation")
    _require_corr("animate_assumption_violation")

    diag_good = diagnose(result_good)
    diag_bad  = diagnose(result_bad)

    # Show good model first
    rvf_good = ResidualVsFittedPlot(diag_good, title=f"{assumption}: OK")
    rvf_good.center()
    scene.play(rvf_good.animate_appear(run_time=run_time * 0.35))
    scene.wait(run_time * 0.10)

    # Morph to bad model
    scene.play(rvf_good.morph_to(diag_bad, run_time=run_time * 0.35))
    scene.wait(run_time * 0.10)

    # Flash the title to draw attention
    scene.play(Flash(rvf_good, color=RED, run_time=run_time * 0.10))


def animate_fix_heteroscedasticity(
    scene,
    result:      "RegressionResult",
    log_y:       bool  = True,
    run_time:    float = 5.0,
) -> "RegressionResult":
    """
    Animate the before / after effect of a log-transform on Y to fix
    heteroscedasticity.

    Steps
    -----
    1. Show Scale-Location plot of original model (fan-out pattern).
    2. Apply log(y) and refit.
    3. Morph the Scale-Location plot to the new (flatter) one.
    4. Return the re-fitted result.

    Parameters
    ----------
    scene : manim.Scene
    result : RegressionResult
        Must be k=1 for a clean demo.
    log_y : bool
        If True, apply log transform to y.  Otherwise apply sqrt.
    run_time : float

    Returns
    -------
    RegressionResult — the re-fitted model.
    """
    _require_manim("animate_fix_heteroscedasticity")
    _require_corr("animate_fix_heteroscedasticity")

    diag_orig = diagnose(result)
    sl_orig   = ScaleLocationPlot(diag_orig, title="Before: Scale-Location")
    sl_orig.center()
    scene.play(sl_orig.animate_appear(run_time=run_time * 0.30))
    scene.wait(run_time * 0.10)

    # Re-fit with transformed response
    y_new = np.log(np.clip(result.y, 1e-8, None)) if log_y else np.sqrt(np.clip(result.y, 0, None))
    x_raw = result.x[:, 1:] if result.fit_intercept else result.x
    result_new = ols_fit(x_raw, y_new,
                         fit_intercept=result.fit_intercept,
                         feature_names=result.feature_names)
    diag_new   = diagnose(result_new)
    sl_new     = ScaleLocationPlot(diag_new,
                                   title="After: Scale-Location (log y)")
    sl_new.move_to(sl_orig.get_center())
    scene.play(
        Transform(sl_orig, sl_new, run_time=run_time * 0.40,
                  rate_func=rate_functions.ease_in_out_sine)
    )
    scene.wait(run_time * 0.10)

    transform_label = Text(
        f"y → {'log(y)' if log_y else 'sqrt(y)'}",
        font_size=22,
        color=(ManimColor(AMBER_600.hex) if _COLORS_AVAILABLE else YELLOW),
    ).next_to(sl_orig, UP, buff=0.25)
    scene.play(Write(transform_label, run_time=run_time * 0.10))

    return result_new


# ===========================================================================
# LAYER F — Formula registry bridge
# ===========================================================================

def _build_gauss_markov_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"\hat{\boldsymbol{\beta}}_{OLS} = "
        r"\arg\min_{\boldsymbol{\beta}}\;"
        r"\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2"
        r"\quad \Rightarrow \quad "
        r"\mathrm{BLUE}"
    )
    return TexFormula(
        name        = "gauss_markov_ols",
        raw         = raw,
        description = "Gauss-Markov theorem: OLS is BLUE under classical assumptions",
        parts       = {
            "estimator": r"\hat{\boldsymbol{\beta}}_{OLS}",
            "rss":       r"\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2",
            "blue":      r"\mathrm{BLUE}",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"\text{Assumption A1}",
                rhs = r"\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},\; \mathbb{E}[\boldsymbol{\varepsilon}]=\mathbf{0}",
                annotation = "Linear model with zero-mean errors",
            ),
            TexDerivationStep(
                lhs = r"\text{Assumption A2}",
                rhs = r"\mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}",
                annotation = "Homoscedasticity + no autocorrelation",
            ),
            TexDerivationStep(
                lhs = r"\text{Assumption A3}",
                rhs = r"\mathrm{rank}(\mathbf{X}) = k+1",
                annotation = "No perfect multicollinearity",
            ),
            TexDerivationStep(
                lhs = r"\Rightarrow\; \hat{\boldsymbol{\beta}}_{OLS}",
                rhs = r"\text{is BLUE (Best Linear Unbiased Estimator)}",
                annotation = "Gauss-Markov conclusion",
            ),
        ],
        tags        = ["regression", "ols", "gauss_markov", "assumptions"],
    )


def _build_durbin_watson_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"d = "
        + _frac(
            _sum(r"(e_t - e_{t-1})^2", "t=2", "n"),
            _sum(r"e_t^2", "t=1", "n"),
        )
        + r"\quad \approx 2(1 - \hat{\rho}_1)"
    )
    return TexFormula(
        name        = "durbin_watson_formula",
        raw         = raw,
        description = "Durbin-Watson test statistic for first-order autocorrelation",
        parts       = {
            "numerator":   _sum(r"(e_t - e_{t-1})^2", "t=2", "n"),
            "denominator": _sum(r"e_t^2", "t=1", "n"),
            "approx":      r"2(1 - \hat{\rho}_1)",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"\hat{\rho}_1",
                rhs = _frac(
                    _sum(r"e_t e_{t-1}", "t=2", "n"),
                    _sum(r"e_t^2", "t=1", "n"),
                ),
                annotation = "Sample lag-1 autocorrelation",
            ),
            TexDerivationStep(
                lhs = "d",
                rhs = (
                    _frac(
                        _sum(r"(e_t^2 - 2e_t e_{t-1} + e_{t-1}^2)", "t=2", "n"),
                        _sum(r"e_t^2", "t=1", "n"),
                    )
                ),
                annotation = "Expand the squared difference",
            ),
            TexDerivationStep(
                lhs = "",
                rhs = r"\approx 2(1 - \hat{\rho}_1)",
                annotation = "Approximate when n is large",
            ),
        ],
        tags        = ["regression", "autocorrelation", "durbin_watson"],
    )


def _build_breusch_pagan_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"LM = n \cdot R^2_{e^2 \sim \mathbf{X}}"
        r"\;\overset{H_0}{\sim}\; \chi^2_{k}"
    )
    return TexFormula(
        name        = "breusch_pagan_formula",
        raw         = raw,
        description = "Breusch-Pagan LM test for heteroscedasticity",
        parts       = {
            "lm_stat":  r"LM",
            "r_sq":     r"R^2_{e^2 \sim \mathbf{X}}",
            "null_dist": r"\chi^2_{k}",
        },
        steps       = [
            TexDerivationStep(
                lhs = r"H_0",
                rhs = r"\mathrm{Var}(\varepsilon_i) = \sigma^2 \;\forall\, i",
                annotation = "Null: homoscedasticity",
            ),
            TexDerivationStep(
                lhs = "H_1",
                rhs = r"\mathrm{Var}(\varepsilon_i) = h(\mathbf{x}_i^\top \boldsymbol{\gamma})",
                annotation = "Alt: variance depends on X",
            ),
            TexDerivationStep(
                lhs = r"\text{Step 1}",
                rhs = r"\text{Regress } \hat{e}_i^2 \text{ on } \mathbf{X}",
                annotation = "Auxiliary regression",
            ),
            TexDerivationStep(
                lhs = r"LM",
                rhs = raw.split("\\sim")[0].strip(),
                annotation = "LM = n * R^2 of auxiliary regression",
            ),
        ],
        tags        = ["regression", "homoscedasticity", "breusch_pagan"],
    )


def _build_press_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"\mathrm{PRESS} = "
        + _sum(
            r"\left(\frac{e_i}{1-h_{ii}}\right)^2",
            "i=1", "n"
        )
        + r" = "
        + _sum(r"\left(y_i - \hat{y}_{(i)}\right)^2", "i=1", "n")
    )
    return TexFormula(
        name        = "press_statistic_formula",
        raw         = raw,
        description = "PRESS statistic — leave-one-out prediction error sum of squares",
        parts       = {
            "press":          r"\mathrm{PRESS}",
            "leverage_form":  _sum(r"\left(\frac{e_i}{1-h_{ii}}\right)^2", "i=1", "n"),
            "loo_form":       _sum(r"\left(y_i - \hat{y}_{(i)}\right)^2", "i=1", "n"),
        },
        steps       = [
            TexDerivationStep(
                lhs = r"\hat{y}_{(i)}",
                rhs = r"\mathbf{x}_i^\top \hat{\boldsymbol{\beta}}_{(i)}",
                annotation = "Fitted value without observation i",
            ),
            TexDerivationStep(
                lhs = r"y_i - \hat{y}_{(i)}",
                rhs = _frac("e_i", "1 - h_{ii}"),
                annotation = "Sherman-Morrison shortcut",
            ),
            TexDerivationStep(
                lhs = r"\mathrm{PRESS}",
                rhs = _sum(r"\left(\frac{e_i}{1-h_{ii}}\right)^2", "i=1", "n"),
                annotation = "Sum of squared LOO residuals",
            ),
        ],
        tags        = ["regression", "press", "cross_validation", "influence"],
    )


def _build_cooks_d_full_formula() -> Optional["TexFormula"]:
    if not _TEX_AVAILABLE:
        return None
    raw = (
        r"D_i = "
        + _frac(
            r"(\hat{\boldsymbol{\beta}} - \hat{\boldsymbol{\beta}}_{(i)})^\top "
            r"\mathbf{X}^\top\mathbf{X}\,"
            r"(\hat{\boldsymbol{\beta}} - \hat{\boldsymbol{\beta}}_{(i)})",
            r"p \hat{\sigma}^2",
        )
        + r" = "
        + _frac("e_i^2", r"p \hat{\sigma}^2")
        + r"\cdot"
        + _frac("h_{ii}", r"(1 - h_{ii})^2")
    )
    return TexFormula(
        name        = "cooks_d_full",
        raw         = raw,
        description = "Cook's distance — influence of observation i on all fitted values",
        parts       = {
            "beta_change":    (
                r"(\hat{\boldsymbol{\beta}} - \hat{\boldsymbol{\beta}}_{(i)})^\top"
                r"\mathbf{X}^\top\mathbf{X}\,"
                r"(\hat{\boldsymbol{\beta}} - \hat{\boldsymbol{\beta}}_{(i)})"
            ),
            "residual_sq":    "e_i^2",
            "leverage_term":  _frac("h_{ii}", r"(1-h_{ii})^2"),
        },
        steps       = [
            TexDerivationStep(
                lhs = r"D_i",
                rhs = _frac(
                    r"\|\hat{\mathbf{y}} - \hat{\mathbf{y}}_{(i)}\|^2",
                    r"p \hat{\sigma}^2",
                ),
                annotation = "Scaled shift in all fitted values",
            ),
            TexDerivationStep(
                lhs = "",
                rhs = (
                    _frac("e_i^2", r"p \hat{\sigma}^2")
                    + r"\cdot"
                    + _frac("h_{ii}", r"(1-h_{ii})^2")
                ),
                annotation = "Closed form via hat matrix",
            ),
        ],
        tags        = ["regression", "influence", "cooks_d", "diagnostics"],
    )


# Build RESIDUAL_FORMULAS registry
RESIDUAL_FORMULAS: Dict[str, "TexFormula"] = {}

if _TEX_AVAILABLE:
    _resid_formula_builders = [
        _build_gauss_markov_formula,
        _build_durbin_watson_formula,
        _build_breusch_pagan_formula,
        _build_press_formula,
        _build_cooks_d_full_formula,
    ]
    for _builder in _resid_formula_builders:
        _f = _builder()
        if _f is not None:
            RESIDUAL_FORMULAS[_f.name] = _f
            try:
                register_formula(_f)
            except (ValueError, KeyError):
                pass


# ===========================================================================
# __all__
# ===========================================================================

__all__ = [
    # Layer A — residual transforms
    "ordinary_residuals",
    "standardised_residuals",
    "externally_studentized_residuals",
    "press_residuals",
    "press_statistic",
    "jackknife_betas",
    "acf_values",
    "pacf_values",
    "lowess_smooth",
    "qq_coordinates",

    # Layer B — test statistics
    "test_normality",
    "test_homoscedasticity",
    "test_autocorrelation",
    "test_linearity",

    # Layer C — result dataclasses & entry point
    "NormalityResult",
    "HomoscedasticityResult",
    "AutocorrResult",
    "LinearityResult",
    "ResidualDiagnostics",
    "diagnose",

    # Layer D — Manim mobjects
    "ResidualVsFittedPlot",
    "QQPlot3D",
    "ScaleLocationPlot",
    "InfluencePlot3D",
    "ACFPlot3D",
    "DiagnosticPanel",

    # Layer E — scene animations
    "build_diagnostic_scene",
    "animate_assumption_violation",
    "animate_fix_heteroscedasticity",

    # Layer F — formula registry
    "RESIDUAL_FORMULAS",
]