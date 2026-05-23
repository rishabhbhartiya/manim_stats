"""
manim_stats/distributions/continuous_dists.py
===============================================
All 14 continuous distribution assets for the Manim Statistics Extension.

Each class inherits from ``ContinuousDistribution3D`` and adds:
    • Per-parameter ``ValueTracker`` registration
    • ``_rebuild_dist_fn()``      — reconstructs scipy backend from live params
    • ``_param_constraints()``    — bounds / validity predicates
    • Distribution-specific extra visualizations
    • Named class-method presets  (.standard(), .wide(), .skewed(), …)
    • ``show_formula()``          — MathTex PDF/PMF formula panel
    • Unique animations           — e.g. empirical_rule(), memoryless_demo(),
                                       hazard_rate_demo(), bayesian_update()

Distributions
-------------
 1.  NormalDist3D           — μ, σ
 2.  StudentTDist3D         — df, μ, σ
 3.  ChiSquaredDist3D       — df
 4.  FDist3D                — dfn, dfd
 5.  ExponentialDist3D      — λ (rate)
 6.  GammaDist3D            — α (shape), β (rate)
 7.  BetaDist3D             — a, b
 8.  UniformContDist3D      — lo, hi
 9.  LogNormalDist3D        — μ (log-mean), σ (log-std)
10.  WeibullDist3D          — c (shape), scale
11.  CauchyDist3D           — x₀ (location), γ (scale)
12.  ParetoDist3D           — α (shape), xₘ (scale)
13.  LaplaceDist3D          — μ, b (diversity)
14.  LogisticDist3D         — μ, s (scale)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
)

import numpy as np
from numpy.typing import ArrayLike

from manim import (
    VGroup, VMobject,
    Line3D, Arrow3D, Dot3D,
    DashedLine, Polygon,
    Text, MathTex,
    RoundedRectangle,
    Animation, AnimationGroup, Succession,
    Create, FadeIn, FadeOut, Write, Transform,
    GrowArrow, DrawBorderThenFill,
    UpdateFromAlphaFunc,
    ORIGIN, UP, DOWN, LEFT, RIGHT, IN, OUT,
    DEGREES, PI, TAU,
    interpolate_color, smooth, there_and_back,
    ValueTracker, always_redraw,
)

from ..core.base import (
    StatsTheme, StatsColorPalette,
    MaterialConfig, AnimationConfig,
    HighlightStyle, HighlightSystem,
    BuildStyle, ThemeMode,
)
from ..core.math_utils import (
    DistributionFunction, DistributionResult,
    area_under_curve, format_stat_value,
    FloatArray,
)
from ..axes.axes3d import StatsAxes3D, AxisID
from .base_dist import (
    ContinuousDistribution3D,
    BaseDistribution3D,
    RepresentationMode,
    ShadeFillStyle,
    ShadeRegionConfig,
    MomentMarkerConfig,
    MomentMarkerStyle,
    StatsAnnotationConfig,
    DistributionCurveConfig,
    BarConfig,
    PercentileProbe3D,
    ProbeConfig,
    FillRegionSystem,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _formula_panel(
    latex:      str,
    theme:      StatsColorPalette,
    position:   np.ndarray,
    font_size:  float = 28,
    title:      str   = "",
) -> VGroup:
    """
    Build a floating MathTex formula panel with an optional title.
    Returns a VGroup ready to add to any scene.
    """
    grp   = VGroup()
    color = theme.text_primary
    tc    = theme.accent

    if title:
        t = Text(title, font_size=font_size * 0.75, color=tc)
        t.move_to(position + UP * 0.55)
        grp.add(t)

    formula = MathTex(latex, font_size=font_size, color=color)
    formula.move_to(position)
    grp.add(formula)

    bg = RoundedRectangle(
        corner_radius=0.14,
        width=formula.width + 0.5,
        height=grp.height + 0.35,
    )
    bg.set_fill(theme.surface, opacity=0.85)
    bg.set_stroke(theme.border, width=1.2, opacity=0.6)
    bg.move_to(grp.get_center())
    grp.add_to_back(bg)
    return grp


def _annotation_arrow(
    axes:    StatsAxes3D,
    x_data:  float,
    y_data:  float,
    label:   str,
    color:   str,
    offset:  np.ndarray,
    font_size: float = 22,
    is_math: bool = True,
) -> VGroup:
    """Small arrow + label pointing to a specific point on the curve."""
    scene_pt  = axes.c2p(x_data, y_data)
    label_pt  = scene_pt + offset
    arrow     = Arrow3D(label_pt, scene_pt, color=color,
                        thickness=0.008, tip_length=0.12)
    mob_cls   = MathTex if is_math else Text
    lbl       = mob_cls(label, font_size=font_size, color=color)
    lbl.move_to(label_pt + UP * 0.2)
    return VGroup(arrow, lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NORMAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class NormalDist3D(ContinuousDistribution3D):
    r"""
    Normal (Gaussian) distribution  N(μ, σ²).

    Parameters
    ----------
    mu    : float  — mean μ
    sigma : float  — standard deviation σ > 0

    Unique features
    ---------------
    • ``empirical_rule()``        — shade 68 / 95 / 99.7% bands simultaneously
    • ``show_standard_normal()``  — overlay N(0,1) for comparison
    • ``animate_standardize()``   — animate (X-μ)/σ → Z transform
    • ``animate_sigma_sweep()``   — smoothly vary σ from narrow to wide
    • ``show_formula()``          — display PDF LaTeX

    Presets
    -------
    NormalDist3D.standard(axes)   — N(0, 1)
    NormalDist3D.wide(axes)       — N(0, 2)
    NormalDist3D.narrow(axes)     — N(0, 0.5)
    NormalDist3D.shifted(axes)    — N(2, 1)
    """

    # PDF LaTeX
    _FORMULA = (r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}"
                r"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}")

    def __init__(
        self,
        axes:   StatsAxes3D,
        mu:     float = 0.0,
        sigma:  float = 1.0,
        color:  Optional[str] = None,
        **kwargs,
    ) -> None:
        self._color_override = color
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.normal(mu, sigma),
            params  = {"mu": mu, "sigma": sigma},
            curve_cfg = DistributionCurveConfig(
                stroke_color = color,
                n_sample_points = 400,
                fill_opacity = 0.14,
            ),
            moment_cfg = MomentMarkerConfig(
                show_mean=True, show_sigma_bands=True, n_sigma_bands=2),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.normal(
            mu    = self.params.get("mu"),
            sigma = self.params.get("sigma"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"sigma": lambda v: v > 0}

    # ── unique visualizations ─────────────────────────────────────────────

    def empirical_rule(
        self,
        colors: Optional[Tuple[str, str, str]] = None,
    ) -> Tuple[VGroup, VGroup, VGroup]:
        """
        Shade the 68 / 95 / 99.7% bands (1σ / 2σ / 3σ) simultaneously.
        Returns (band_1, band_2, band_3) VGroups.
        """
        t = self._palette
        c1, c2, c3 = colors or (
            t.distribution_palette[0],
            t.distribution_palette[2],
            t.distribution_palette[3],
        )
        b1 = self.shade_by_sigma("emp_1s", n_sigma=1.0,
                                  color=c1, opacity=0.42)
        b2 = self.shade_by_sigma("emp_2s", n_sigma=2.0,
                                  color=c2, opacity=0.25)
        b3 = self.shade_by_sigma("emp_3s", n_sigma=3.0,
                                  color=c3, opacity=0.14)
        return b1, b2, b3

    def show_standard_normal(
        self,
        color: Optional[str] = None,
    ) -> "NormalDist3D":
        """
        Overlay N(0,1) on the same axes as a reference curve.
        Returns the overlay NormalDist3D (not added automatically —
        caller should ``scene.add()`` it).
        """
        c = color or self._palette.secondary
        return NormalDist3D(
            self._axes, mu=0.0, sigma=1.0, color=c,
            curve_cfg=DistributionCurveConfig(
                stroke_color=c, fill_opacity=0.0,
                show_fill=False, stroke_width=2.0,
            ),
            moment_cfg=MomentMarkerConfig(show_mean=False,
                                           show_sigma_bands=False),
        )

    def animate_sigma_sweep(
        self,
        target_sigma: float,
        run_time:     float = 3.0,
    ) -> Animation:
        """Smoothly vary σ, watching the curve narrow or widen."""
        return self.animate_param("sigma", target_sigma, run_time=run_time)

    def animate_mu_shift(
        self,
        target_mu: float,
        run_time:  float = 2.5,
    ) -> Animation:
        """Slide the bell curve left or right by changing μ."""
        return self.animate_param("mu", target_mu, run_time=run_time)

    def animate_standardize(
        self,
        run_time: float = 3.0,
    ) -> AnimationGroup:
        """Animate (X - μ)/σ → N(0,1) by simultaneously sending μ→0, σ→1."""
        return self.animate_params(
            {"mu": 0.0, "sigma": 1.0}, run_time=run_time)

    def show_formula(
        self,
        position: Optional[np.ndarray] = None,
    ) -> VGroup:
        """Display the PDF formula in a floating panel."""
        pos = position if position is not None \
              else np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Normal PDF")
        self.add(grp)
        return grp

    def z_score_annotation(
        self, x_val: float
    ) -> VGroup:
        """
        Add a labelled annotation showing the z-score at x_val.
        Returns the VGroup; caller adds to scene.
        """
        mu    = self.params.get("mu")
        sigma = self.params.get("sigma")
        z     = (x_val - mu) / sigma
        y_val = float(self._dist_fn.pdf([x_val])[0])
        t     = self._palette
        label = f"z = {z:.2f}"
        return _annotation_arrow(
            self._axes, x_val, y_val, label,
            color=t.accent,
            offset=RIGHT * 0.8 + UP * 0.4,
        )

    # ── presets ───────────────────────────────────────────────────────────

    @classmethod
    def standard(cls, axes: StatsAxes3D, **kwargs) -> "NormalDist3D":
        """N(0, 1) — the standard normal."""
        return cls(axes, mu=0.0, sigma=1.0, **kwargs)

    @classmethod
    def wide(cls, axes: StatsAxes3D, **kwargs) -> "NormalDist3D":
        return cls(axes, mu=0.0, sigma=2.0, **kwargs)

    @classmethod
    def narrow(cls, axes: StatsAxes3D, **kwargs) -> "NormalDist3D":
        return cls(axes, mu=0.0, sigma=0.5, **kwargs)

    @classmethod
    def shifted(cls, axes: StatsAxes3D, **kwargs) -> "NormalDist3D":
        return cls(axes, mu=2.0, sigma=1.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STUDENT'S t DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class StudentTDist3D(ContinuousDistribution3D):
    r"""
    Student's t-distribution  t(df, μ, σ).

    Parameters
    ----------
    df    : float  — degrees of freedom > 0
    mu    : float  — location (default 0)
    sigma : float  — scale (default 1)

    Unique features
    ---------------
    • ``animate_df_to_normal()``  — as df → ∞, t → N(0,1)
    • ``highlight_heavy_tails()`` — shade tail excess vs Normal
    • ``critical_values(alpha)``  — mark t_{α/2,df} critical values
    • ``show_formula()``

    Presets
    -------
    StudentTDist3D.df1(axes)    — Cauchy-like, df=1
    StudentTDist3D.df5(axes)
    StudentTDist3D.df30(axes)   — nearly normal
    """

    _FORMULA = (r"f(x) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}"
                r"{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)}"
                r"\!\left(1+\frac{x^2}{\nu}\right)^{\!-\frac{\nu+1}{2}}")

    def __init__(
        self,
        axes:   StatsAxes3D,
        df:     float = 5.0,
        mu:     float = 0.0,
        sigma:  float = 1.0,
        color:  Optional[str] = None,
        **kwargs,
    ) -> None:
        self._color_override = color
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.student_t(df, mu, sigma),
            params  = {"df": df, "mu": mu, "sigma": sigma},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.12),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.student_t(
            df    = self.params.get("df"),
            mu    = self.params.get("mu"),
            sigma = self.params.get("sigma"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"df": lambda v: v >= 1, "sigma": lambda v: v > 0}

    # ── unique visualizations ─────────────────────────────────────────────

    def animate_df_to_normal(
        self,
        run_time: float = 4.0,
    ) -> Animation:
        """Increase df from current to 120, watching t → N(0,1)."""
        return self.animate_param("df", 120.0, run_time=run_time)

    def highlight_heavy_tails(
        self,
        cutoff: float = 2.0,
        color:  Optional[str] = None,
    ) -> Tuple[VGroup, VGroup]:
        """
        Shade the tail excess regions |x| > cutoff on both sides.
        Returns (left_tail, right_tail).
        """
        c  = color or self._palette.negative
        lt = self.shade_tail_left("t_tail_left",  x=-cutoff, color=c, opacity=0.45)
        rt = self.shade_tail_right("t_tail_right", x= cutoff, color=c, opacity=0.45)
        return lt, rt

    def critical_values(
        self,
        alpha: float = 0.05,
    ) -> Tuple[PercentileProbe3D, PercentileProbe3D]:
        """Mark the two-tailed critical values t_{α/2, df}."""
        t_crit_lo = float(self._dist_fn.ppf(alpha / 2))
        t_crit_hi = float(self._dist_fn.ppf(1 - alpha / 2))
        p_lo = self.probe_at("t_crit_lo", t_crit_lo)
        p_hi = self.probe_at("t_crit_hi", t_crit_hi)
        return p_lo, p_hi

    def show_formula(
        self, position: Optional[np.ndarray] = None
    ) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Student's t PDF", font_size=22)
        self.add(grp)
        return grp

    @classmethod
    def df1(cls, axes: StatsAxes3D, **kwargs) -> "StudentTDist3D":
        return cls(axes, df=1.0, **kwargs)

    @classmethod
    def df5(cls, axes: StatsAxes3D, **kwargs) -> "StudentTDist3D":
        return cls(axes, df=5.0, **kwargs)

    @classmethod
    def df30(cls, axes: StatsAxes3D, **kwargs) -> "StudentTDist3D":
        return cls(axes, df=30.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CHI-SQUARED DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class ChiSquaredDist3D(ContinuousDistribution3D):
    r"""
    Chi-squared distribution  χ²(df).

    Parameters
    ----------
    df : float — degrees of freedom > 0

    Unique features
    ---------------
    • ``animate_df_convergence()`` — as df → ∞, χ²(df) → Normal
    • ``shade_critical_region()``  — right-tail critical region at α
    • ``show_normal_approx()``     — overlay Normal(df, 2*df) approximation
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\,\Gamma(k/2)},\;"
                r"x > 0")

    def __init__(
        self,
        axes:  StatsAxes3D,
        df:    float = 5.0,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.chi_squared(df),
            params  = {"df": df},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.13),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.chi_squared(
            self.params.get("df"))

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"df": lambda v: v >= 1}

    # ── unique visualizations ─────────────────────────────────────────────

    def animate_df_convergence(self, target_df: float = 40.0,
                                run_time: float = 4.0) -> Animation:
        """Watch χ²(df) become more bell-shaped as df grows."""
        return self.animate_param("df", target_df, run_time=run_time)

    def shade_critical_region(
        self,
        alpha: float = 0.05,
        color: Optional[str] = None,
    ) -> VGroup:
        """Shade the right-tail critical region for a χ² test at level α."""
        crit = float(self._dist_fn.ppf(1 - alpha))
        c    = color or self._palette.negative
        grp  = self.shade_tail_right("chi2_crit", x=crit,
                                      color=c, opacity=0.45)
        self.probe_at("chi2_crit_probe", crit)
        return grp

    def show_normal_approx(self) -> "NormalDist3D":
        """
        Overlay the Normal(df, 2*df) approximation.
        Returns the overlay (add to scene manually).
        """
        df  = self.params.get("df")
        mu  = df
        sig = math.sqrt(2 * df)
        return NormalDist3D(
            self._axes, mu=mu, sigma=sig,
            color=self._palette.secondary,
            curve_cfg=DistributionCurveConfig(
                stroke_color=self._palette.secondary,
                fill_opacity=0.0, show_fill=False,
                stroke_width=2.2,
            ),
            moment_cfg=MomentMarkerConfig(show_mean=False,
                                           show_sigma_bands=False),
        )

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.5, 2.0, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Chi-Squared PDF", font_size=22)
        self.add(grp)
        return grp

    @classmethod
    def df1(cls, axes: StatsAxes3D, **kwargs) -> "ChiSquaredDist3D":
        return cls(axes, df=1.0, **kwargs)

    @classmethod
    def df5(cls, axes: StatsAxes3D, **kwargs) -> "ChiSquaredDist3D":
        return cls(axes, df=5.0, **kwargs)

    @classmethod
    def df10(cls, axes: StatsAxes3D, **kwargs) -> "ChiSquaredDist3D":
        return cls(axes, df=10.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  F DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class FDist3D(ContinuousDistribution3D):
    r"""
    F-distribution  F(d₁, d₂).

    Parameters
    ----------
    dfn : float — numerator degrees of freedom > 0
    dfd : float — denominator degrees of freedom > 0

    Unique features
    ---------------
    • Two-parameter animation (dfn, dfd independently)
    • ``shade_critical_region(alpha)``
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}"
                r"{(d_1 x+d_2)^{d_1+d_2}}}}{x\,B(d_1/2,\,d_2/2)}")

    def __init__(
        self,
        axes:  StatsAxes3D,
        dfn:   float = 5.0,
        dfd:   float = 10.0,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.f_distribution(dfn, dfd),
            params  = {"dfn": dfn, "dfd": dfd},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.13),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.f_distribution(
            dfn=self.params.get("dfn"),
            dfd=self.params.get("dfd"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {
            "dfn": lambda v: v >= 1,
            "dfd": lambda v: v >= 1,
        }

    def shade_critical_region(
        self, alpha: float = 0.05, color: Optional[str] = None
    ) -> VGroup:
        crit = float(self._dist_fn.ppf(1 - alpha))
        c    = color or self._palette.negative
        grp  = self.shade_tail_right("f_crit", x=crit, color=c, opacity=0.42)
        self.probe_at("f_crit_probe", crit)
        return grp

    def animate_dfn_sweep(self, target: float = 30.0,
                           run_time: float = 3.5) -> Animation:
        return self.animate_param("dfn", target, run_time=run_time)

    def animate_dfd_sweep(self, target: float = 60.0,
                           run_time: float = 3.5) -> Animation:
        return self.animate_param("dfd", target, run_time=run_time)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.5, 2.0, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="F PDF", font_size=20)
        self.add(grp)
        return grp

    @classmethod
    def anova(cls, axes: StatsAxes3D, k: int = 3,
              n: int = 30, **kwargs) -> "FDist3D":
        """F for a k-group ANOVA with n total observations."""
        return cls(axes, dfn=float(k-1), dfd=float(n-k), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXPONENTIAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialDist3D(ContinuousDistribution3D):
    r"""
    Exponential distribution  Exp(λ).

    Parameters
    ----------
    lam : float — rate λ > 0  (mean = 1/λ)

    Unique features
    ---------------
    • ``memoryless_demo(s, t)``   — visualize P(X>s+t|X>s) = P(X>t)
    • ``show_half_life()``        — mark x = ln(2)/λ (median = half-life)
    • ``animate_rate_sweep()``    — vary λ
    • ``show_formula()``
    """

    _FORMULA = r"f(x) = \lambda e^{-\lambda x},\quad x \geq 0"

    def __init__(
        self,
        axes:  StatsAxes3D,
        lam:   float = 1.0,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.exponential(lam),
            params  = {"lam": lam},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.14),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.exponential(
            self.params.get("lam"))

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"lam": lambda v: v > 0}

    # ── unique visualizations ─────────────────────────────────────────────

    def memoryless_demo(
        self,
        s:     float = 1.0,
        t:     float = 1.0,
        color: Optional[str] = None,
    ) -> Tuple[VGroup, VGroup, VGroup]:
        """
        Visualize the memoryless property:
        P(X > s+t | X > s)  =  P(X > t).

        Shades:
            region_a = P(X > s)           — full right tail from s
            region_b = P(X > s+t)         — right tail from s+t
            region_c = P(X > t) on fresh  — a separate shade for P(X>t)

        Returns (region_a, region_b, label_group).
        """
        t_pal = self._palette
        c     = color or t_pal.accent
        c2    = t_pal.secondary

        lam   = self.params.get("lam")
        px_s  = math.exp(-lam * s)
        px_st = math.exp(-lam * (s + t))
        cond  = px_st / px_s   # = exp(-lam*t)

        r_a = self.shade_tail_right("mem_a", x=s,   color=c,  opacity=0.35)
        r_b = self.shade_tail_right("mem_b", x=s+t, color=c2, opacity=0.45)

        # Annotation
        axes  = self._axes
        y_mid = float(self._dist_fn.pdf([s + t / 2])[0]) * 0.5
        pos   = axes.c2p(s + t / 2, y_mid) + UP * 0.5
        lbl1  = MathTex(
            rf"P(X>{s+t:.1f}\mid X>{s:.1f}) = {cond:.4f}",
            font_size=20, color=c2)
        lbl1.move_to(pos)
        lbl2  = MathTex(
            rf"= P(X>{t:.1f}) = {cond:.4f}",
            font_size=20, color=c)
        lbl2.next_to(lbl1, DOWN, buff=0.1)
        lbl_grp = VGroup(lbl1, lbl2)
        self.add(lbl_grp)
        return r_a, r_b, lbl_grp

    def show_half_life(self) -> VGroup:
        """Mark the median (= half-life = ln(2)/λ) on the curve."""
        lam    = self.params.get("lam")
        x_half = math.log(2) / lam
        y_half = float(self._dist_fn.pdf([x_half])[0])
        t      = self._palette
        grp    = _annotation_arrow(
            self._axes, x_half, y_half,
            r"\text{Median} = \ln 2/\lambda",
            color=t.accent, offset=RIGHT * 1.0 + UP * 0.5,
        )
        self.add(grp)
        return grp

    def animate_rate_sweep(
        self, target_lam: float = 2.0, run_time: float = 3.0
    ) -> Animation:
        return self.animate_param("lam", target_lam, run_time=run_time)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.0, 2.0, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Exponential PDF")
        self.add(grp)
        return grp

    @classmethod
    def rate1(cls, axes: StatsAxes3D, **kwargs) -> "ExponentialDist3D":
        return cls(axes, lam=1.0, **kwargs)

    @classmethod
    def rate2(cls, axes: StatsAxes3D, **kwargs) -> "ExponentialDist3D":
        return cls(axes, lam=2.0, **kwargs)

    @classmethod
    def rate_half(cls, axes: StatsAxes3D, **kwargs) -> "ExponentialDist3D":
        return cls(axes, lam=0.5, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GAMMA DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class GammaDist3D(ContinuousDistribution3D):
    r"""
    Gamma distribution  Γ(α, β).

    Parameters
    ----------
    alpha : float — shape α > 0
    beta  : float — rate β > 0  (mean = α/β)

    Unique features
    ---------------
    • ``animate_shape_sweep()``     — vary α: Exp → skewed → bell
    • ``show_special_cases()``      — annotate α=1 (Exp) and α=df/2 (χ²)
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}"
                r"x^{\alpha-1}e^{-\beta x},\; x>0")

    def __init__(
        self,
        axes:   StatsAxes3D,
        alpha:  float = 2.0,
        beta:   float = 1.0,
        color:  Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.gamma(alpha, beta),
            params  = {"alpha": alpha, "beta": beta},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.13),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.gamma(
            alpha=self.params.get("alpha"),
            beta =self.params.get("beta"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"alpha": lambda v: v > 0, "beta": lambda v: v > 0}

    def animate_shape_sweep(
        self, target_alpha: float = 8.0, run_time: float = 4.0
    ) -> Animation:
        """Watch the shape morph from exponential-like to bell-like."""
        return self.animate_param("alpha", target_alpha, run_time=run_time)

    def show_special_cases(self) -> VGroup:
        """
        Annotate special cases:
            α=1   → Exponential(β)
            α=k/2 → χ²(k) when β = 1/2
        """
        t   = self._palette
        grp = VGroup()
        ann1 = Text("α=1: Exponential(β)",
                    font_size=20, color=t.accent)
        ann2 = Text("α=k/2, β=½: χ²(k)",
                    font_size=20, color=t.secondary)
        ann1.move_to(np.array([3.2, 1.8, 0.0]))
        ann2.next_to(ann1, DOWN, buff=0.15)
        grp.add(ann1, ann2)
        self.add(grp)
        return grp

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Gamma PDF", font_size=22)
        self.add(grp)
        return grp

    @classmethod
    def exponential_special(cls, axes: StatsAxes3D,
                             lam: float = 1.0, **kwargs) -> "GammaDist3D":
        """Gamma(1, λ) = Exponential(λ)."""
        return cls(axes, alpha=1.0, beta=lam, **kwargs)

    @classmethod
    def chi_squared_special(cls, axes: StatsAxes3D,
                             df: int = 5, **kwargs) -> "GammaDist3D":
        """Gamma(df/2, 1/2) = χ²(df)."""
        return cls(axes, alpha=df / 2, beta=0.5, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BETA DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class BetaDist3D(ContinuousDistribution3D):
    r"""
    Beta distribution  Beta(a, b)  supported on [0, 1].

    Parameters
    ----------
    a : float — shape parameter a > 0
    b : float — shape parameter b > 0

    Unique features
    ---------------
    • Rich shape variety:
        a=b=1   → Uniform(0,1)
        a=b<1   → U-shaped (bimodal extremes)
        a>1,b>1 → unimodal bell
        a<1,b>1 → J-shaped (left-heavy)
    • ``bayesian_update(n_successes, n_trials)`` — Beta posterior update
    • ``animate_a_sweep()`` / ``animate_b_sweep()``
    • ``show_formula()``
    • ``show_mode()``
    """

    _FORMULA = (r"f(x) = \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)},\;"
                r"x\in[0,1]")

    def __init__(
        self,
        axes:  StatsAxes3D,
        a:     float = 2.0,
        b:     float = 5.0,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.beta(a, b),
            params  = {"a": a, "b": b},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.15),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_mode=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.beta(
            a=self.params.get("a"),
            b=self.params.get("b"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"a": lambda v: v > 0, "b": lambda v: v > 0}

    # ── unique visualizations ─────────────────────────────────────────────

    def bayesian_update(
        self,
        n_successes: int,
        n_trials:    int,
        animate:     bool = False,
    ) -> "BetaDist3D":
        """
        Return a new BetaDist3D representing the Beta posterior
        after observing *n_successes* out of *n_trials*
        (conjugate prior update with Beta(a,b) prior).

        New params: a' = a + n_successes, b' = b + n_failures.
        """
        a_post = self.params.get("a") + n_successes
        b_post = self.params.get("b") + (n_trials - n_successes)
        posterior = BetaDist3D(
            self._axes, a=a_post, b=b_post,
            color=self._palette.secondary,
        )
        return posterior

    def animate_a_sweep(
        self, target_a: float = 8.0, run_time: float = 3.5
    ) -> Animation:
        return self.animate_param("a", target_a, run_time=run_time)

    def animate_b_sweep(
        self, target_b: float = 8.0, run_time: float = 3.5
    ) -> Animation:
        return self.animate_param("b", target_b, run_time=run_time)

    def show_mode_marker(self) -> Optional[VGroup]:
        """
        Annotate the mode = (a-1)/(a+b-2) (valid when a,b > 1).
        """
        a = self.params.get("a"); b = self.params.get("b")
        if a <= 1 or b <= 1:
            return None
        mode  = (a - 1) / (a + b - 2)
        y_mod = float(self._dist_fn.pdf([mode])[0])
        t     = self._palette
        grp   = _annotation_arrow(
            self._axes, mode, y_mod,
            r"\text{Mode}=\frac{a-1}{a+b-2}",
            color=t.positive,
            offset=RIGHT * 0.6 + UP * 0.5,
        )
        self.add(grp)
        return grp

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Beta PDF")
        self.add(grp)
        return grp

    @classmethod
    def uniform(cls, axes: StatsAxes3D, **kwargs) -> "BetaDist3D":
        """Beta(1,1) = Uniform(0,1)."""
        return cls(axes, a=1.0, b=1.0, **kwargs)

    @classmethod
    def symmetric(cls, axes: StatsAxes3D,
                  k: float = 3.0, **kwargs) -> "BetaDist3D":
        """Beta(k,k) — symmetric bell."""
        return cls(axes, a=k, b=k, **kwargs)

    @classmethod
    def u_shaped(cls, axes: StatsAxes3D, **kwargs) -> "BetaDist3D":
        """Beta(0.5, 0.5) — U-shaped (arcsine)."""
        return cls(axes, a=0.5, b=0.5, **kwargs)

    @classmethod
    def j_shaped(cls, axes: StatsAxes3D, **kwargs) -> "BetaDist3D":
        """Beta(0.5, 2.0) — J-shaped."""
        return cls(axes, a=0.5, b=2.0, **kwargs)

    @classmethod
    def informative_prior(cls, axes: StatsAxes3D,
                          **kwargs) -> "BetaDist3D":
        """Beta(10, 3) — right-skewed informative prior."""
        return cls(axes, a=10.0, b=3.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  UNIFORM (CONTINUOUS) DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class UniformContDist3D(ContinuousDistribution3D):
    r"""
    Continuous Uniform distribution  U(lo, hi).

    Parameters
    ----------
    lo : float — lower bound
    hi : float — upper bound > lo

    Unique features
    ---------------
    • Flat rectangular body with prominent width annotation
    • ``show_mean_variance()``  — annotate μ = (lo+hi)/2, σ² = (hi-lo)²/12
    • ``animate_expand()``      — animate hi → hi*2 (widening interval)
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{1}{b-a},\quad a \leq x \leq b")

    def __init__(
        self,
        axes:  StatsAxes3D,
        lo:    float = 0.0,
        hi:    float = 1.0,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.uniform_continuous(lo, hi),
            params  = {"lo": lo, "hi": hi},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color,
                fill_opacity=0.22,
                stroke_width=3.5,
            ),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        lo = self.params.get("lo")
        hi = self.params.get("hi")
        if hi > lo:
            self._dist_fn = DistributionFunction.uniform_continuous(lo, hi)

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {
            "lo": lambda v: v < self.params.get("hi"),
            "hi": lambda v: v > self.params.get("lo"),
        }

    def show_mean_variance(self) -> VGroup:
        lo  = self.params.get("lo")
        hi  = self.params.get("hi")
        mu  = (lo + hi) / 2
        var = (hi - lo) ** 2 / 12
        t   = self._palette
        grp = VGroup()
        lbl1 = MathTex(
            rf"\mu = \frac{{a+b}}{{2}} = {mu:.3g}",
            font_size=22, color=t.accent)
        lbl2 = MathTex(
            rf"\sigma^2 = \frac{{(b-a)^2}}{{12}} = {var:.3g}",
            font_size=22, color=t.secondary)
        lbl1.move_to(np.array([3.0, 2.0, 0.0]))
        lbl2.next_to(lbl1, DOWN, buff=0.15)
        grp.add(lbl1, lbl2)
        self.add(grp)
        return grp

    def animate_expand(
        self, target_hi: float = 3.0, run_time: float = 2.5
    ) -> Animation:
        return self.animate_param("hi", target_hi, run_time=run_time)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.0, 1.8, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Uniform PDF")
        self.add(grp)
        return grp

    @classmethod
    def unit(cls, axes: StatsAxes3D, **kwargs) -> "UniformContDist3D":
        return cls(axes, lo=0.0, hi=1.0, **kwargs)

    @classmethod
    def symmetric(cls, axes: StatsAxes3D,
                  half: float = 2.0, **kwargs) -> "UniformContDist3D":
        return cls(axes, lo=-half, hi=half, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  LOG-NORMAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class LogNormalDist3D(ContinuousDistribution3D):
    r"""
    Log-Normal distribution  LogN(μ, σ²).

    If X ~ LogN(μ, σ) then ln(X) ~ N(μ, σ²).

    Parameters
    ----------
    mu    : float — mean of the underlying normal (log-mean)
    sigma : float — std  of the underlying normal (log-std) > 0

    Unique features
    ---------------
    • ``show_log_transform()``   — annotate "ln(X) ~ N(μ,σ²)"
    • ``show_natural_params()``  — display actual mean = exp(μ+σ²/2)
    • ``animate_sigma_sweep()``
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{1}{x\sigma\sqrt{2\pi}}"
                r"\exp\!\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)")

    def __init__(
        self,
        axes:   StatsAxes3D,
        mu:     float = 0.0,
        sigma:  float = 0.5,
        color:  Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.log_normal(mu, sigma),
            params  = {"mu": mu, "sigma": sigma},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.14),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.log_normal(
            mu=self.params.get("mu"),
            sigma=self.params.get("sigma"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"sigma": lambda v: v > 0}

    def show_natural_params(self) -> VGroup:
        mu    = self.params.get("mu")
        sigma = self.params.get("sigma")
        mean  = math.exp(mu + sigma**2 / 2)
        var   = (math.exp(sigma**2) - 1) * math.exp(2*mu + sigma**2)
        t     = self._palette
        grp   = VGroup()
        lbl1  = MathTex(
            rf"E[X] = e^{{\mu+\sigma^2/2}} = {mean:.3g}",
            font_size=20, color=t.accent)
        lbl2  = MathTex(
            rf"\text{{Var}}(X) = {var:.3g}",
            font_size=20, color=t.secondary)
        lbl1.move_to(np.array([3.2, 2.0, 0.0]))
        lbl2.next_to(lbl1, DOWN, buff=0.12)
        grp.add(lbl1, lbl2)
        self.add(grp)
        return grp

    def show_log_transform(self) -> VGroup:
        t   = self._palette
        lbl = MathTex(r"\ln(X) \sim \mathcal{N}(\mu,\,\sigma^2)",
                      font_size=24, color=t.positive)
        lbl.move_to(np.array([3.0, -1.5, 0.0]))
        self.add(lbl)
        return VGroup(lbl)

    def animate_sigma_sweep(
        self, target: float = 1.2, run_time: float = 3.0
    ) -> Animation:
        return self.animate_param("sigma", target, run_time=run_time)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Log-Normal PDF", font_size=21)
        self.add(grp)
        return grp


# ─────────────────────────────────────────────────────────────────────────────
# 10.  WEIBULL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class WeibullDist3D(ContinuousDistribution3D):
    r"""
    Weibull distribution  Weibull(c, scale).

    Parameters
    ----------
    c     : float — shape parameter c > 0
    scale : float — scale parameter > 0

    Hazard rate interpretation
    --------------------------
        c < 1 → decreasing hazard (infant mortality / burn-in)
        c = 1 → constant hazard   (Exponential, random failures)
        c > 1 → increasing hazard (wear-out / ageing)

    Unique features
    ---------------
    • ``hazard_rate_demo()``    — switch to HAZARD mode + annotate bathtub
    • ``animate_shape_sweep()`` — morph c from <1 through 1 to >1
    • ``show_bathtub_label()``  — annotate the three failure-rate phases
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{c}{\lambda}\left(\frac{x}{\lambda}\right)^{c-1}"
                r"e^{-(x/\lambda)^c}")

    def __init__(
        self,
        axes:   StatsAxes3D,
        c:      float = 1.5,
        scale:  float = 1.0,
        color:  Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.weibull(c, scale),
            params  = {"c": c, "scale": scale},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.13),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.weibull(
            c=self.params.get("c"),
            scale=self.params.get("scale"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"c": lambda v: v > 0, "scale": lambda v: v > 0}

    def hazard_rate_demo(self) -> Animation:
        """
        Switch to HAZARD representation mode and annotate the
        monotone nature of h(x) for current c value.
        """
        return self.animate_mode_switch(RepresentationMode.HAZARD)

    def animate_shape_sweep(
        self,
        targets: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.5),
        run_time_each: float = 1.5,
    ) -> Succession:
        """
        Sequentially morph through the given c values,
        demonstrating decreasing → constant → increasing hazard.
        """
        anims = [
            self.animate_param("c", t, run_time=run_time_each)
            for t in targets
        ]
        return Succession(*anims)

    def show_bathtub_label(self) -> VGroup:
        """Annotate the current c value's failure-rate regime."""
        c   = self.params.get("c")
        t   = self._palette
        if c < 0.999:
            regime = "c<1: Decreasing hazard (burn-in)"
        elif abs(c - 1.0) < 0.05:
            regime = "c=1: Constant hazard (Exponential)"
        else:
            regime = "c>1: Increasing hazard (wear-out)"
        lbl = Text(regime, font_size=22, color=t.accent)
        lbl.move_to(np.array([0.0, -2.5, 0.0]))
        self.add(lbl)
        return VGroup(lbl)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Weibull PDF", font_size=21)
        self.add(grp)
        return grp

    @classmethod
    def exponential_special(cls, axes: StatsAxes3D,
                             **kwargs) -> "WeibullDist3D":
        """Weibull(1, 1) = Exponential(1)."""
        return cls(axes, c=1.0, scale=1.0, **kwargs)

    @classmethod
    def rayleigh(cls, axes: StatsAxes3D,
                 scale: float = 1.0, **kwargs) -> "WeibullDist3D":
        """Weibull(2, scale) = Rayleigh distribution."""
        return cls(axes, c=2.0, scale=scale, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CAUCHY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class CauchyDist3D(ContinuousDistribution3D):
    r"""
    Cauchy distribution  Cauchy(x₀, γ).

    Parameters
    ----------
    x0    : float — location parameter (median, mode)
    gamma : float — scale (half-width at half-maximum) > 0

    Notable properties
    ------------------
    • Mean, variance, and all higher moments are undefined (infinite)
    • Heavy tails — the CLT does NOT apply to Cauchy samples
    • Ratio of two independent N(0,1) → Cauchy(0,1)

    Unique features
    ---------------
    • ``show_undefined_moments()`` — annotate "E[X] = undefined"
    • ``compare_to_normal()``      — overlay N(0,1) to show tail difference
    • ``animate_scale_sweep()``    — vary γ
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{1}{\pi\gamma"
                r"\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}")

    def __init__(
        self,
        axes:   StatsAxes3D,
        x0:     float = 0.0,
        gamma:  float = 1.0,
        color:  Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.cauchy(x0, gamma),
            params  = {"x0": x0, "gamma": gamma},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.11),
            # No moment markers — mean/var undefined
            moment_cfg = MomentMarkerConfig(show_mean=False,
                                             show_sigma_bands=False,
                                             show_mode=True),
            stats_cfg  = StatsAnnotationConfig(
                show_mean=False, show_variance=False,
                show_std=False, show_skewness=False,
                show_kurtosis=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.cauchy(
            x0=self.params.get("x0"),
            gamma=self.params.get("gamma"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"gamma": lambda v: v > 0}

    def show_undefined_moments(self) -> VGroup:
        """Floating annotation: mean and variance are undefined."""
        t   = self._palette
        grp = VGroup()
        l1  = MathTex(r"E[X] = \text{undefined}",
                      font_size=24, color=t.negative)
        l2  = MathTex(r"\text{Var}(X) = \text{undefined}",
                      font_size=24, color=t.negative)
        l3  = Text("CLT does not apply", font_size=20, color=t.accent)
        l1.move_to(np.array([3.0,  2.0, 0.0]))
        l2.next_to(l1, DOWN, buff=0.14)
        l3.next_to(l2, DOWN, buff=0.14)
        grp.add(l1, l2, l3)
        self.add(grp)
        return grp

    def compare_to_normal(self) -> "NormalDist3D":
        """Return an N(0,1) overlay to show tail heaviness (add manually)."""
        return NormalDist3D(
            self._axes, mu=0.0, sigma=1.0,
            color=self._palette.secondary,
            curve_cfg=DistributionCurveConfig(
                stroke_color=self._palette.secondary,
                fill_opacity=0.0, show_fill=False,
                stroke_width=2.0,
            ),
            moment_cfg=MomentMarkerConfig(show_mean=False,
                                           show_sigma_bands=False),
        )

    def animate_scale_sweep(
        self, target_gamma: float = 2.5, run_time: float = 3.0
    ) -> Animation:
        return self.animate_param("gamma", target_gamma, run_time=run_time)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Cauchy PDF", font_size=22)
        self.add(grp)
        return grp

    @classmethod
    def standard(cls, axes: StatsAxes3D, **kwargs) -> "CauchyDist3D":
        return cls(axes, x0=0.0, gamma=1.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  PARETO DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class ParetoDist3D(ContinuousDistribution3D):
    r"""
    Pareto distribution  Pareto(α, xₘ).

    Parameters
    ----------
    alpha : float — shape (tail index) α > 0
    xm    : float — scale (minimum value) > 0

    Power-law properties
    --------------------
        α ≤ 1 → mean undefined
        α ≤ 2 → variance undefined
        α ≈ 1.161 → 80/20 rule (top 20% hold 80% of the mass)

    Unique features
    ---------------
    • ``show_power_law()``    — log-log linearity annotation + slope label
    • ``show_8020_rule()``    — shade top 20% and annotate 80% mass
    • ``animate_tail_sweep()`` — vary α from heavy to light tails
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}},\;"
                r"x \geq x_m")

    def __init__(
        self,
        axes:   StatsAxes3D,
        alpha:  float = 2.0,
        xm:     float = 1.0,
        color:  Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.pareto(alpha, xm),
            params  = {"alpha": alpha, "xm": xm},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.13),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=False),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.pareto(
            alpha=self.params.get("alpha"),
            xm=self.params.get("xm"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"alpha": lambda v: v > 0, "xm": lambda v: v > 0}

    def show_8020_rule(self) -> Optional[VGroup]:
        """
        For α ≈ 1.161, shade the top 20% of the population
        holding 80% of the mass, annotated.
        """
        alpha = self.params.get("alpha")
        xm    = self.params.get("xm")
        if alpha <= 0:
            return None

        # 80th percentile of Pareto
        p80 = float(self._dist_fn.ppf(0.80))
        r   = self.shade_tail_right("pareto_80",
                                     x=p80,
                                     color=self._palette.accent,
                                     opacity=0.40)
        axes  = self._axes
        y_lbl = float(self._dist_fn.pdf([p80])[0]) * 0.5
        pos   = axes.c2p(p80 * 1.3, y_lbl) + UP * 0.3
        lbl   = Text("Top 20% → 80% mass", font_size=20,
                     color=self._palette.accent)
        lbl.move_to(pos)
        self.add(lbl)
        return VGroup(r, lbl)

    def show_power_law(self) -> VGroup:
        """
        Annotate log-log linearity: in log-log space, slope = −(α+1).
        """
        alpha = self.params.get("alpha")
        t     = self._palette
        lbl   = MathTex(
            rf"\log f(x) = \text{{const}} - (\alpha+1)\log x",
            font_size=20, color=t.accent)
        slope = Text(f"slope = −({alpha:.2f}+1) = −{alpha+1:.2f}",
                     font_size=20, color=t.secondary)
        lbl.move_to(np.array([2.5, 1.5, 0.0]))
        slope.next_to(lbl, DOWN, buff=0.12)
        grp = VGroup(lbl, slope)
        self.add(grp)
        return grp

    def animate_tail_sweep(
        self,
        target_alpha: float = 4.0,
        run_time:     float = 3.5,
    ) -> Animation:
        return self.animate_param("alpha", target_alpha, run_time=run_time)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Pareto PDF")
        self.add(grp)
        return grp


# ─────────────────────────────────────────────────────────────────────────────
# 13.  LAPLACE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class LaplaceDist3D(ContinuousDistribution3D):
    r"""
    Laplace (double-exponential) distribution  Laplace(μ, b).

    Parameters
    ----------
    mu : float — location (mean, median, mode)
    b  : float — diversity (scale) > 0  (std = b√2)

    Unique features
    ---------------
    • Sharp peak at μ — visually distinctive vs Normal
    • ``compare_to_normal()``  — overlay N(μ, b²·2)
    • ``animate_b_sweep()``
    • ``show_formula()``
    """

    _FORMULA = (r"f(x) = \frac{1}{2b}\exp\!\left(-\frac{|x-\mu|}{b}\right)")

    def __init__(
        self,
        axes:  StatsAxes3D,
        mu:    float = 0.0,
        b:     float = 1.0,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.laplace(mu, b),
            params  = {"mu": mu, "b": b},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.14),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=True,
                                             n_sigma_bands=1),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.laplace(
            mu=self.params.get("mu"),
            b =self.params.get("b"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"b": lambda v: v > 0}

    def compare_to_normal(self) -> "NormalDist3D":
        """
        Return N(μ, b√2) with the same std for visual tail comparison.
        """
        b  = self.params.get("b")
        mu = self.params.get("mu")
        return NormalDist3D(
            self._axes, mu=mu, sigma=b * math.sqrt(2),
            color=self._palette.secondary,
            curve_cfg=DistributionCurveConfig(
                stroke_color=self._palette.secondary,
                fill_opacity=0.0, show_fill=False, stroke_width=2.0),
            moment_cfg=MomentMarkerConfig(show_mean=False,
                                           show_sigma_bands=False),
        )

    def animate_b_sweep(
        self, target_b: float = 2.0, run_time: float = 3.0
    ) -> Animation:
        return self.animate_param("b", target_b, run_time=run_time)

    def show_formula(self, position: Optional[np.ndarray] = None) -> VGroup:
        pos = position or np.array([3.0, 2.2, 0.0])
        grp = _formula_panel(self._FORMULA, self._palette, pos,
                              title="Laplace PDF")
        self.add(grp)
        return grp

    @classmethod
    def standard(cls, axes: StatsAxes3D, **kwargs) -> "LaplaceDist3D":
        return cls(axes, mu=0.0, b=1.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 14.  LOGISTIC DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class LogisticDist3D(ContinuousDistribution3D):
    r"""
    Logistic distribution  Logistic(μ, s).

    Parameters
    ----------
    mu : float — location (mean, median, mode)
    s  : float — scale > 0  (std = s·π/√3)

    Statistical significance
    ------------------------
    • The CDF is the sigmoid function — the core of logistic regression
    • Similar to Normal but with heavier tails
    • CDF: F(x) = 1 / (1 + exp(-(x-μ)/s))

    Unique features
    ---------------
    • ``show_sigmoid_cdf()``      — switch to CDF mode + annotate sigmoid
    • ``show_logit_transform()``  — annotate logit(F(x)) = (x-μ)/s
    • ``compare_to_normal()``     — overlay N(μ, s²π²/3)
    • ``animate_s_sweep()``
    • ``show_formula()``
    """

    _FORMULA_PDF = (r"f(x) = \frac{e^{-(x-\mu)/s}}{s\left(1+e^{-(x-\mu)/s}\right)^2}")
    _FORMULA_CDF = r"F(x) = \frac{1}{1+e^{-(x-\mu)/s}} \;(\text{sigmoid})"

    def __init__(
        self,
        axes:  StatsAxes3D,
        mu:    float = 0.0,
        s:     float = 1.0,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.logistic(mu, s),
            params  = {"mu": mu, "s": s},
            curve_cfg = DistributionCurveConfig(
                stroke_color=color, fill_opacity=0.14),
            moment_cfg = MomentMarkerConfig(show_mean=True,
                                             show_sigma_bands=True,
                                             n_sigma_bands=1),
            **kwargs,
        )

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.logistic(
            mu=self.params.get("mu"),
            s =self.params.get("s"),
        )

    def _param_constraints(self) -> Dict[str, Callable[[float], bool]]:
        return {"s": lambda v: v > 0}

    def show_sigmoid_cdf(self) -> Animation:
        """
        Switch to CDF mode to reveal the sigmoid shape,
        then annotate it.
        """
        return self.animate_mode_switch(RepresentationMode.CDF)

    def show_logit_transform(self) -> VGroup:
        """Annotate the logit link: logit(F(x)) = (x−μ)/s."""
        t   = self._palette
        lbl = MathTex(
            r"\text{logit}(F(x)) = \frac{x-\mu}{s}",
            font_size=24, color=t.accent)
        lbl.move_to(np.array([3.0, 2.0, 0.0]))
        self.add(lbl)
        return VGroup(lbl)

    def compare_to_normal(self) -> "NormalDist3D":
        """Return N(μ, (s·π/√3)²) for visual comparison."""
        s  = self.params.get("s")
        mu = self.params.get("mu")
        return NormalDist3D(
            self._axes, mu=mu, sigma=s * math.pi / math.sqrt(3),
            color=self._palette.secondary,
            curve_cfg=DistributionCurveConfig(
                stroke_color=self._palette.secondary,
                fill_opacity=0.0, show_fill=False, stroke_width=2.0),
            moment_cfg=MomentMarkerConfig(show_mean=False,
                                           show_sigma_bands=False),
        )

    def animate_s_sweep(
        self, target_s: float = 2.0, run_time: float = 3.0
    ) -> Animation:
        return self.animate_param("s", target_s, run_time=run_time)

    def show_formula(
        self, position: Optional[np.ndarray] = None, show_cdf: bool = False
    ) -> VGroup:
        pos = position or np.array([3.2, 2.2, 0.0])
        formula = self._FORMULA_CDF if show_cdf else self._FORMULA_PDF
        title   = "Logistic CDF (sigmoid)" if show_cdf else "Logistic PDF"
        grp     = _formula_panel(formula, self._palette, pos,
                                  title=title, font_size=21)
        self.add(grp)
        return grp

    @classmethod
    def standard(cls, axes: StatsAxes3D, **kwargs) -> "LogisticDist3D":
        return cls(axes, mu=0.0, s=1.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "NormalDist3D",
    "StudentTDist3D",
    "ChiSquaredDist3D",
    "FDist3D",
    "ExponentialDist3D",
    "GammaDist3D",
    "BetaDist3D",
    "UniformContDist3D",
    "LogNormalDist3D",
    "WeibullDist3D",
    "CauchyDist3D",
    "ParetoDist3D",
    "LaplaceDist3D",
    "LogisticDist3D",
]