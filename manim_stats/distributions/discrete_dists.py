"""
manim_stats/distributions/discrete_dists.py
=============================================
All discrete distribution assets for the Manim Statistics Extension.

Every class inherits from ``DiscreteDistribution3D`` and provides:

    • Full parameter tracker wiring  (live-updatable via ValueTracker)
    • _rebuild_dist_fn()             — reconstruct scipy backend from params
    • _param_constraints()           — validation predicates
    • DEFAULT_PARAMS / PARAM_LABELS / PARAM_BOUNDS / FORMULA_TEX
    • Formula panel                  — floating MathTex PDF/PMF formula
    • Distribution-specific annotations
    • Convenience shade_*() / animate_*() methods
    • prob_at(k)        — P(X = k) with annotation
    • cumulative_at(k)  — P(X ≤ k) with shading
    • tail_at(k)        — P(X ≥ k) with shading
    • animate_cdf_build() — CDF builds bar by bar left → right
    • from_data()        — MLE / moment fit from a dataset
    • from_moments()     — method-of-moments fit

Distributions implemented
--------------------------
 1.  BernoulliDistribution3D
 2.  BinomialDistribution3D
 3.  PoissonDistribution3D
 4.  GeometricDistribution3D
 5.  NegativeBinomialDistribution3D
 6.  HypergeometricDistribution3D
 7.  DiscreteUniformDistribution3D
 8.  MultinomialDistribution3D      (grouped bar variant)
"""

from __future__ import annotations

import math
import warnings
from typing import (
    Any, Callable, ClassVar, Dict, List,
    Optional, Sequence, Tuple, Union,
)

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as scipy_stats

from manim import (
    VGroup, VMobject,
    Line3D, Arrow3D, Dot3D,
    DashedLine, Polygon, Text, MathTex,
    RoundedRectangle, Rectangle,
    Animation, AnimationGroup, Succession,
    Create, FadeIn, FadeOut, Write, Transform,
    GrowFromCenter, GrowFromEdge,
    UpdateFromAlphaFunc,
    ORIGIN, UP, DOWN, LEFT, RIGHT, OUT,
    DEGREES, PI,
    smooth, there_and_back,
    ValueTracker,
    interpolate_color,
)

from ..core.base import (
    StatsTheme, StatsColorPalette,
    AnimationConfig, HighlightStyle,
    ThemeMode,
)
from ..core.math_utils import (
    DistributionFunction, DistributionResult,
    FloatArray,
)
from ..axes.axes3d import StatsAxes3D, AxisID
from .base_dist import (
    DiscreteDistribution3D,
    DistributionCurveConfig,
    ShadeRegionConfig, ShadeFillStyle,
    MomentMarkerConfig, MomentMarkerStyle,
    StatsAnnotationConfig,
    FillRegionSystem,
    RepresentationMode,
    BarConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _formula_panel(
    tex:       str,
    position:  np.ndarray,
    theme:     StatsColorPalette,
    font_size: float = 26,
) -> VGroup:
    lbl = MathTex(tex, font_size=font_size, color=theme.text_primary)
    bg  = RoundedRectangle(
        corner_radius=0.14,
        width=lbl.width  + 0.45,
        height=lbl.height + 0.30,
    )
    bg.set_fill(theme.surface, opacity=0.82)
    bg.set_stroke(theme.border, width=1.2, opacity=0.55)
    bg.move_to(lbl)
    grp = VGroup(bg, lbl)
    grp.move_to(position)
    return grp


def _stat_label(
    text:      str,
    position:  np.ndarray,
    theme:     StatsColorPalette,
    color:     Optional[str] = None,
    font_size: float = 20,
    is_math:   bool  = True,
) -> VMobject:
    c   = color or theme.accent
    cls = MathTex if is_math else Text
    mob = cls(text, font_size=font_size, color=c)
    mob.move_to(position)
    return mob


def _overlay_continuous(
    axes:   StatsAxes3D,
    dist_fn: DistributionFunction,
    x_vals:  np.ndarray,
    color:   str,
    opacity: float = 0.65,
    width:   float = 0.009,
    label:   str   = "",
) -> VGroup:
    """
    Draw a continuous PDF curve as a Line3D sequence overlay.
    Used for Normal / Poisson approximation overlays on discrete bars.
    """
    res = dist_fn.evaluate(x=x_vals)
    if res.pdf is None:
        return VGroup()
    grp = VGroup()
    pts = [axes.c2p(float(x), float(y))
           for x, y in zip(res.x, res.pdf)
           if math.isfinite(y)]
    for i in range(len(pts) - 1):
        seg = Line3D(pts[i], pts[i + 1], color=color, thickness=width)
        seg.set_opacity(opacity)
        grp.add(seg)
    if label and pts:
        lbl = MathTex(label, font_size=18, color=color)
        lbl.move_to(pts[len(pts) // 3] + UP * 0.35)
        grp.add(lbl)
    return grp


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BERNOULLI DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class BernoulliDistribution3D(DiscreteDistribution3D):
    """
    3-D Bernoulli distribution: single Bernoulli trial.

    Parameters
    ----------
    p : probability of success ∈ (0, 1)

    Two bars only — k=0 (failure, probability 1-p) and k=1 (success, p).

    Special features
    ----------------
    • annotate_success_failure()  — label bars "Failure" / "Success"
    • annotate_complement()       — arc showing P(X=0) = 1 - P(X=1)
    • animate_p_sweep(p0, p1)     — smoothly vary p
    • link_to_coin()              — floating annotation of coin-flip context
    • from_data(x)                — estimate p = mean(x)
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Bernoulli"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(X=k)=p^k(1-p)^{1-k},\;k\in\{0,1\}")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {"p": 0.5}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {"p": "p"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {"p": (0.001, 0.999)}

    def __init__(
        self,
        axes: StatsAxes3D,
        p:    float = 0.5,
        show_formula:          bool = True,
        show_success_labels:   bool = True,
        formula_pos:           Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._show_success_labels = show_success_labels
        self._formula_pos = formula_pos or np.array([3.5, 1.8, 0.0])

        # Use wide bars for only 2 values
        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.55,
            bar_depth=0.25,
            show_bar_labels=True,
            label_decimals=4,
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.bernoulli(p),
            params  = {"p": p},
            bar_cfg = bar_cfg,
            **kwargs,
        )
        self.params._bounds["p"] = (0.001, 0.999)

        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette))
        if show_success_labels:
            self.add(self._make_bar_labels())
        self.add(self._make_p_annotation())

    # ── abstract ──────────────────────────────────────────────────────────

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.bernoulli(
            p=self.params.get("p"))

    def _param_constraints(self) -> Dict[str, Callable]:
        return {"p": lambda v: 0 < v < 1}

    # ── annotations ───────────────────────────────────────────────────────

    def _make_bar_labels(self) -> VGroup:
        t    = self._palette
        axes = self._axes
        grp  = VGroup()
        p    = self.params.get("p")
        for k, lbl_text, color in [
            (0, "Failure", t.negative),
            (1, "Success", t.positive),
        ]:
            pos = axes.c2p(float(k), 0.0) + DOWN * 0.50
            lbl = Text(lbl_text, font_size=20, color=color)
            lbl.move_to(pos)
            grp.add(lbl)
        return grp

    def _make_p_annotation(self) -> VGroup:
        t    = self._palette
        axes = self._axes
        p    = self.params.get("p")
        grp  = VGroup()
        # P(X=1) = p label above success bar
        pos1 = axes.c2p(1.0, p + 0.04)
        l1   = MathTex(f"p = {p:.3f}", font_size=22, color=t.positive)
        l1.move_to(pos1)
        # P(X=0) = 1-p label above failure bar
        pos0 = axes.c2p(0.0, (1 - p) + 0.04)
        l0   = MathTex(f"1-p = {1-p:.3f}", font_size=22, color=t.negative)
        l0.move_to(pos0)
        grp.add(l0, l1)
        return grp

    # ── public API ────────────────────────────────────────────────────────

    def animate_p_sweep(
        self,
        p_start:  float = 0.1,
        p_end:    float = 0.9,
        run_time: float = 3.0,
    ) -> Animation:
        self.params.set("p", p_start)
        return self.animate_param("p", p_end, run_time=run_time)

    def prob_at(self, k: int) -> Tuple[float, VGroup]:
        """Return (P(X=k), annotation_VGroup)."""
        p   = self.params.get("p")
        val = p if k == 1 else (1 - p)
        grp = self.probe_at(f"bern_k{k}", float(k))
        return val, VGroup(grp)

    def cumulative_at(self, k: int) -> VGroup:
        return self.shade_region(
            f"bern_cdf_{k}", lo=-0.5, hi=float(k) + 0.5,
            color=self._palette.positive, opacity=0.35)

    def tail_at(self, k: int) -> VGroup:
        return self.shade_region(
            f"bern_tail_{k}", lo=float(k) - 0.5, hi=1.5,
            color=self._palette.secondary, opacity=0.35)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike, **kwargs
    ) -> "BernoulliDistribution3D":
        x = np.asarray(data, float)
        return cls(axes, p=float(x.mean()), **kwargs)

    @classmethod
    def from_moments(
        cls, axes: StatsAxes3D, mean: float,
        variance: float = None, **kwargs
    ) -> "BernoulliDistribution3D":
        return cls(axes, p=float(np.clip(mean, 0.001, 0.999)), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BINOMIAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class BinomialDistribution3D(DiscreteDistribution3D):
    """
    3-D Binomial distribution.

    Parameters
    ----------
    n : number of trials (positive integer)
    p : probability of success per trial ∈ (0, 1)

    Special features
    ----------------
    • overlay_normal_approx()     — N(np, np(1-p)) overlay
    • overlay_poisson_approx()    — Poisson(np) overlay (large n, small p)
    • highlight_mode()            — highlight bar at mode = floor((n+1)p)
    • annotate_mean_var()         — E[X]=np, Var[X]=np(1-p)
    • shade_at_least(k)           — P(X ≥ k)
    • shade_at_most(k)            — P(X ≤ k)
    • shade_exact(k)              — P(X = k) single bar highlight
    • animate_n_increase(n0, n1)  — watch bars multiply as n grows
    • animate_p_sweep(p0, p1)     — shift distribution shape
    • animate_to_normal()         — CLT: n→∞ morphing to bell
    • prob_at(k)                  — P(X=k) with readout
    • cumulative_at(k)            — P(X≤k) shaded
    • tail_at(k)                  — P(X≥k) shaded
    • from_data(x)                — MLE: p̂ = x̄/n
    • from_moments(mean, var)     — MOM
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Binomial"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {"n": 10.0, "p": 0.5}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {"n": "n", "p": "p"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {
        "n": (1.0, 200.0), "p": (0.001, 0.999)}

    def __init__(
        self,
        axes: StatsAxes3D,
        n:    int   = 10,
        p:    float = 0.5,
        show_formula:       bool = True,
        show_mean_var:      bool = True,
        show_mode_highlight: bool = True,
        formula_pos:        Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._show_mean_var      = show_mean_var
        self._show_mode_highlight = show_mode_highlight
        self._formula_pos = formula_pos or np.array([3.5, 2.5, 0.0])

        mode_k = int(math.floor((n + 1) * p))
        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.72,
            bar_depth=0.18,
            show_bar_labels=n <= 15,
            label_decimals=3,
            highlight_bar_idx=mode_k if show_mode_highlight else None,
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.binomial(n, p),
            params  = {"n": float(n), "p": p},
            bar_cfg = bar_cfg,
            **kwargs,
        )
        for name, (lo, hi) in self.PARAM_BOUNDS.items():
            self.params._bounds[name] = (lo, hi)

        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette,
                font_size=24))
        if show_mean_var:
            self.add(self._make_mean_var_labels())

    # ── abstract ──────────────────────────────────────────────────────────

    def _rebuild_dist_fn(self) -> None:
        n = max(1, int(round(self.params.get("n"))))
        p = self.params.get("p")
        self._dist_fn = DistributionFunction.binomial(n, p)

    def _param_constraints(self) -> Dict[str, Callable]:
        return {
            "n": lambda v: v >= 1,
            "p": lambda v: 0 < v < 1,
        }

    # ── stats labels ──────────────────────────────────────────────────────

    def _make_mean_var_labels(self) -> VGroup:
        t    = self._palette
        axes = self._axes
        n    = int(round(self.params.get("n")))
        p    = self.params.get("p")
        mean = n * p
        var  = n * p * (1 - p)
        grp  = VGroup()
        xhi, yhi = axes.x_range[1], axes.y_range[1]
        l1 = MathTex(
            rf"E[X]=np={mean:.2f}",
            font_size=20, color=t.accent)
        l2 = MathTex(
            rf"\mathrm{{Var}}[X]=np(1-p)={var:.2f}",
            font_size=20, color=t.secondary)
        l1.move_to(axes.c2p(xhi * 0.62, yhi * 0.88))
        l2.move_to(axes.c2p(xhi * 0.62, yhi * 0.74))
        grp.add(l1, l2)
        return grp

    # ── overlays ──────────────────────────────────────────────────────────

    def overlay_normal_approx(self) -> VGroup:
        """
        Overlay N(np, np(1-p)) bell curve.
        Valid approximation when n large and 0.1 < p < 0.9.
        """
        n   = int(round(self.params.get("n")))
        p   = self.params.get("p")
        mu  = n * p
        sig = math.sqrt(n * p * (1 - p))
        t   = self._palette
        x_vals = np.linspace(self._axes.x_range[0],
                              self._axes.x_range[1], 400)
        grp = _overlay_continuous(
            self._axes,
            DistributionFunction.normal(mu, sig),
            x_vals, t.secondary, opacity=0.70,
            label=r"N(np,\,np(1-p))")
        self.add(grp)
        return grp

    def overlay_poisson_approx(self) -> VGroup:
        """
        Overlay Poisson(np) — valid when n large, p small (np ≤ 10).
        """
        n   = int(round(self.params.get("n")))
        p   = self.params.get("p")
        lam = n * p
        t   = self._palette
        x_vals = np.arange(0, min(int(lam * 4) + 1,
                                   int(self._axes.x_range[1]) + 1))
        pois_pmf = scipy_stats.poisson(mu=lam).pmf(x_vals)
        grp = VGroup()
        for xv, yv in zip(x_vals, pois_pmf):
            p1 = self._axes.c2p(float(xv) + 0.15, 0.0)
            p2 = self._axes.c2p(float(xv) + 0.15, float(yv))
            seg = Line3D(p1, p2, color=t.accent, thickness=0.010)
            seg.set_opacity(0.60)
            grp.add(seg)
        lbl = MathTex(r"\mathrm{Pois}(\lambda=np)",
                      font_size=18, color=t.accent)
        lbl.move_to(self._axes.c2p(
            int(lam) + 1.5, float(pois_pmf.max()) * 0.9))
        grp.add(lbl)
        self.add(grp)
        return grp

    def highlight_mode(self) -> Animation:
        """Pulse-highlight the bar at the distribution mode."""
        n     = int(round(self.params.get("n")))
        p     = self.params.get("p")
        mode  = int(math.floor((n + 1) * p))
        return self.highlight_bar(mode)

    # ── shading ───────────────────────────────────────────────────────────

    def shade_at_least(self, k: int, key: str = None) -> VGroup:
        key = key or f"binom_geq_{k}"
        return self.shade_tail_right(key, float(k) - 0.5,
                                     color=self._palette.positive,
                                     opacity=0.35)

    def shade_at_most(self, k: int, key: str = None) -> VGroup:
        key = key or f"binom_leq_{k}"
        return self.shade_tail_left(key, float(k) + 0.5,
                                    color=self._palette.positive,
                                    opacity=0.35)

    def shade_exact(self, k: int, key: str = None) -> VGroup:
        key = key or f"binom_eq_{k}"
        return self.shade_region(key,
                                  lo=float(k) - 0.5,
                                  hi=float(k) + 0.5,
                                  color=self._palette.accent,
                                  opacity=0.55)

    def prob_at(self, k: int) -> Tuple[float, VGroup]:
        n = int(round(self.params.get("n")))
        p = self.params.get("p")
        val = float(scipy_stats.binom(n, p).pmf(k))
        grp = self.shade_exact(k, key=f"binom_prob_{k}")
        return val, grp

    def cumulative_at(self, k: int) -> Tuple[float, VGroup]:
        n = int(round(self.params.get("n")))
        p = self.params.get("p")
        val = float(scipy_stats.binom(n, p).cdf(k))
        grp = self.shade_at_most(k, key=f"binom_cdf_{k}")
        return val, grp

    def tail_at(self, k: int) -> Tuple[float, VGroup]:
        n = int(round(self.params.get("n")))
        p = self.params.get("p")
        val = float(scipy_stats.binom(n, p).sf(k - 1))
        grp = self.shade_at_least(k, key=f"binom_tail_{k}")
        return val, grp

    # ── animations ────────────────────────────────────────────────────────

    def animate_n_increase(
        self,
        n_start:  int   = 5,
        n_end:    int   = 50,
        run_time: float = 5.0,
    ) -> Animation:
        """Watch bars multiply as n grows (keeping p fixed)."""
        self.params.set("n", float(n_start))
        return self.animate_param("n", float(n_end), run_time=run_time)

    def animate_p_sweep(
        self,
        p_start:  float = 0.1,
        p_end:    float = 0.9,
        run_time: float = 4.0,
    ) -> Animation:
        self.params.set("p", p_start)
        return self.animate_param("p", p_end, run_time=run_time)

    def animate_to_normal(self, run_time: float = 4.0) -> Animation:
        """CLT: animate n → 100, showing convergence to bell shape."""
        return self.animate_param("n", 100.0, run_time=run_time)

    def animate_cdf_build(
        self, run_time: float = 3.0
    ) -> Animation:
        """
        CDF builds bar by bar left to right:
        shade grows one integer at a time.
        """
        n    = int(round(self.params.get("n")))
        p    = self.params.get("p")
        rv   = scipy_stats.binom(n, p)
        anims = []
        for k in range(n + 1):
            shade = self.shade_region(
                f"cdf_k{k}",
                lo=float(k) - 0.5, hi=float(k) + 0.5,
                color=self._palette.primary,
                opacity=0.40, label=False)
            anims.append(FadeIn(shade, run_time=run_time / (n + 1)))
        return Succession(*anims)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike,
        n: Optional[int] = None, **kwargs
    ) -> "BinomialDistribution3D":
        x    = np.asarray(data, float)
        n    = n or int(x.max())
        p    = float(x.mean() / n)
        return cls(axes, n=n, p=np.clip(p, 0.001, 0.999), **kwargs)

    @classmethod
    def from_moments(
        cls, axes: StatsAxes3D, mean: float, variance: float, **kwargs
    ) -> "BinomialDistribution3D":
        p = max(0.001, min(0.999, 1.0 - variance / mean))
        n = max(1, int(round(mean / p)))
        return cls(axes, n=n, p=p, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  POISSON DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class PoissonDistribution3D(DiscreteDistribution3D):
    """
    3-D Poisson distribution.

    Parameters
    ----------
    lam : rate parameter λ > 0  (mean = variance = λ)

    Special features
    ----------------
    • annotate_mean_equals_variance()  — label E[X]=Var[X]=λ
    • highlight_mode()                  — bar at k = floor(λ)
    • shade_at_least(k)                — P(X ≥ k)
    • shade_at_most(k)                 — P(X ≤ k)
    • shade_exact(k)                   — single bar highlight
    • overlay_normal_approx()          — N(λ,λ) overlay for large λ
    • annotate_rare_event()            — annotation for small λ (λ≤1)
    • animate_lambda_sweep(l0, l1)     — watch distribution shift right
    • animate_cdf_build()              — CDF fills bar by bar
    • prob_at(k)                       — P(X=k)
    • cumulative_at(k)                 — P(X≤k)
    • tail_at(k)                       — P(X≥k)
    • from_data(x)                     — p̂ = x̄
    • from_moments(mean, _)            — λ = mean
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Poisson"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!},\;k=0,1,2,\ldots")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {"lam": 3.0}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {"lam": r"\lambda"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {"lam": (0.01, 50.0)}

    def __init__(
        self,
        axes: StatsAxes3D,
        lam:  float = 3.0,
        show_formula:        bool = True,
        show_mean_var_label: bool = True,
        annotate_rare:       bool = False,
        formula_pos:         Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._formula_pos = formula_pos or np.array([3.5, 2.5, 0.0])
        mode_k = int(math.floor(lam))

        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.68,
            bar_depth=0.18,
            show_bar_labels=lam <= 12,
            label_decimals=3,
            highlight_bar_idx=mode_k,
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.poisson(lam),
            params  = {"lam": lam},
            bar_cfg = bar_cfg,
            **kwargs,
        )
        self.params._bounds["lam"] = (0.01, 50.0)

        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette,
                font_size=22))
        if show_mean_var_label:
            self.add(self._make_mean_var_label())
        if annotate_rare and lam <= 1.0:
            self.add(self._make_rare_event_annotation())

    # ── abstract ──────────────────────────────────────────────────────────

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.poisson(
            lam=self.params.get("lam"))

    def _param_constraints(self) -> Dict[str, Callable]:
        return {"lam": lambda v: v > 0}

    # ── labels ────────────────────────────────────────────────────────────

    def _make_mean_var_label(self) -> VGroup:
        t    = self._palette
        axes = self._axes
        lam  = self.params.get("lam")
        xhi, yhi = axes.x_range[1], axes.y_range[1]
        grp  = VGroup()
        lbl  = MathTex(
            rf"E[X]=\mathrm{{Var}}[X]=\lambda={lam:.2f}",
            font_size=20, color=t.accent)
        lbl.move_to(axes.c2p(xhi * 0.55, yhi * 0.88))
        grp.add(lbl)
        return grp

    def _make_rare_event_annotation(self) -> VGroup:
        t   = self._palette
        pos = self._axes.c2p(
            self._axes.x_range[1] * 0.5,
            self._axes.y_range[1] * 0.65)
        lbl = Text("Rare event regime (λ ≤ 1)",
                   font_size=18, color=t.text_secondary)
        lbl.move_to(pos)
        return VGroup(lbl)

    # ── overlays ──────────────────────────────────────────────────────────

    def overlay_normal_approx(self) -> VGroup:
        lam  = self.params.get("lam")
        t    = self._palette
        x_vals = np.linspace(max(0, lam - 5 * math.sqrt(lam)),
                              lam + 5 * math.sqrt(lam), 400)
        grp = _overlay_continuous(
            self._axes,
            DistributionFunction.normal(lam, math.sqrt(lam)),
            x_vals, t.secondary, opacity=0.65,
            label=r"N(\lambda,\lambda)")
        self.add(grp)
        return grp

    # ── shading ───────────────────────────────────────────────────────────

    def shade_at_least(self, k: int, key: str = None) -> VGroup:
        key = key or f"pois_geq_{k}"
        return self.shade_tail_right(key, float(k) - 0.5,
                                     color=self._palette.positive,
                                     opacity=0.35)

    def shade_at_most(self, k: int, key: str = None) -> VGroup:
        key = key or f"pois_leq_{k}"
        return self.shade_tail_left(key, float(k) + 0.5,
                                    color=self._palette.positive,
                                    opacity=0.35)

    def shade_exact(self, k: int, key: str = None) -> VGroup:
        key = key or f"pois_eq_{k}"
        return self.shade_region(key,
                                  lo=float(k) - 0.5,
                                  hi=float(k) + 0.5,
                                  color=self._palette.accent,
                                  opacity=0.55)

    def prob_at(self, k: int) -> Tuple[float, VGroup]:
        lam = self.params.get("lam")
        val = float(scipy_stats.poisson(lam).pmf(k))
        grp = self.shade_exact(k, key=f"pois_prob_{k}")
        return val, grp

    def cumulative_at(self, k: int) -> Tuple[float, VGroup]:
        lam = self.params.get("lam")
        val = float(scipy_stats.poisson(lam).cdf(k))
        grp = self.shade_at_most(k, key=f"pois_cdf_{k}")
        return val, grp

    def tail_at(self, k: int) -> Tuple[float, VGroup]:
        lam = self.params.get("lam")
        val = float(scipy_stats.poisson(lam).sf(k - 1))
        grp = self.shade_at_least(k, key=f"pois_tail_{k}")
        return val, grp

    def highlight_mode(self) -> Animation:
        lam  = self.params.get("lam")
        mode = int(math.floor(lam))
        return self.highlight_bar(mode)

    # ── animations ────────────────────────────────────────────────────────

    def animate_lambda_sweep(
        self,
        lam_start: float = 1.0,
        lam_end:   float = 15.0,
        run_time:  float = 5.0,
    ) -> Animation:
        self.params.set("lam", lam_start)
        return self.animate_param("lam", lam_end, run_time=run_time)

    def animate_cdf_build(self, run_time: float = 3.5) -> Animation:
        lam   = self.params.get("lam")
        k_max = int(scipy_stats.poisson(lam).ppf(0.999))
        rv    = scipy_stats.poisson(lam)
        anims = []
        for k in range(k_max + 1):
            shade = self.shade_region(
                f"pois_cdf_k{k}",
                lo=float(k) - 0.5, hi=float(k) + 0.5,
                color=self._palette.primary, opacity=0.38, label=False)
            anims.append(FadeIn(shade, run_time=run_time / (k_max + 1)))
        return Succession(*anims)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike, **kwargs
    ) -> "PoissonDistribution3D":
        x = np.asarray(data, float)
        return cls(axes, lam=max(0.01, float(x.mean())), **kwargs)

    @classmethod
    def from_moments(
        cls, axes: StatsAxes3D, mean: float,
        variance: float = None, **kwargs
    ) -> "PoissonDistribution3D":
        return cls(axes, lam=max(0.01, float(mean)), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GEOMETRIC DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class GeometricDistribution3D(DiscreteDistribution3D):
    """
    3-D Geometric distribution.

    Parameterisation: k = number of trials until first success ∈ {1,2,3,...}
    P(X=k) = (1-p)^(k-1) * p

    Parameters
    ----------
    p : probability of success per trial ∈ (0, 1)

    Special features
    ----------------
    • annotate_memoryless()    — P(X>m+n|X>m) = P(X>n)
    • mark_mean()              — vertical line at 1/p
    • shade_first_success(k)   — P(X=k) single bar highlight
    • shade_within(k)          — P(X ≤ k) = 1-(1-p)^k
    • animate_p_sweep(p0, p1)  — watch geometric decay change
    • animate_cdf_build()      — bars appear one by one
    • from_data(x)             — p̂ = 1/x̄
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Geometric"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(X=k)=(1-p)^{k-1}p,\;k=1,2,\ldots")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {"p": 0.3}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {"p": "p"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {"p": (0.001, 0.999)}

    def __init__(
        self,
        axes: StatsAxes3D,
        p:    float = 0.3,
        show_formula:       bool = True,
        annotate_memoryless: bool = True,
        formula_pos:        Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._formula_pos = formula_pos or np.array([3.5, 2.2, 0.0])

        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.65,
            bar_depth=0.18,
            show_bar_labels=False,
            highlight_bar_idx=0,      # first success = k=1 (index 0)
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.geometric(p),
            params  = {"p": p},
            bar_cfg = bar_cfg,
            **kwargs,
        )
        self.params._bounds["p"] = (0.001, 0.999)

        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette,
                font_size=24))
        if annotate_memoryless:
            self.add(self._make_memoryless_annotation())
        self.add(self._make_mean_label())

    def _rebuild_dist_fn(self) -> None:
        self._dist_fn = DistributionFunction.geometric(
            p=self.params.get("p"))

    def _param_constraints(self) -> Dict[str, Callable]:
        return {"p": lambda v: 0 < v < 1}

    def _make_memoryless_annotation(self) -> VGroup:
        t   = self._palette
        pos = self._axes.c2p(
            self._axes.x_range[1] * 0.50,
            self._axes.y_range[1] * 0.65)
        lbl = MathTex(
            r"P(X>m+n\mid X>m)=P(X>n)",
            font_size=18, color=t.text_secondary)
        lbl.move_to(pos)
        bg = RoundedRectangle(
            corner_radius=0.10, width=lbl.width+0.3,
            height=lbl.height+0.20)
        bg.set_fill(t.surface, opacity=0.75)
        bg.set_stroke(t.border, width=0.9)
        bg.move_to(lbl)
        return VGroup(bg, lbl)

    def _make_mean_label(self) -> VMobject:
        p    = self.params.get("p")
        t    = self._palette
        mean = 1.0 / p
        pos  = self._axes.c2p(mean, self._axes.y_range[1] * 0.15)
        lbl  = MathTex(rf"E[X]=1/p={mean:.2f}",
                       font_size=20, color=t.accent)
        lbl.move_to(pos + UP * 0.3)
        return lbl

    def shade_first_success(self, k: int = 1) -> VGroup:
        return self.shade_region(
            f"geom_eq_{k}",
            lo=float(k) - 0.5, hi=float(k) + 0.5,
            color=self._palette.positive, opacity=0.55)

    def shade_within(self, k: int) -> Tuple[float, VGroup]:
        p   = self.params.get("p")
        val = 1.0 - (1 - p) ** k
        grp = self.shade_region(
            f"geom_leq_{k}", lo=0.5, hi=float(k) + 0.5,
            color=self._palette.accent, opacity=0.35)
        return val, grp

    def prob_at(self, k: int) -> Tuple[float, VGroup]:
        p   = self.params.get("p")
        val = float(scipy_stats.geom(p).pmf(k))
        grp = self.shade_first_success(k)
        return val, grp

    def cumulative_at(self, k: int) -> Tuple[float, VGroup]:
        p   = self.params.get("p")
        val = float(scipy_stats.geom(p).cdf(k))
        grp = self.shade_within(k)[1]
        return val, grp

    def tail_at(self, k: int) -> Tuple[float, VGroup]:
        p   = self.params.get("p")
        val = float(scipy_stats.geom(p).sf(k - 1))
        grp = self.shade_tail_right(
            f"geom_tail_{k}", float(k) - 0.5,
            color=self._palette.negative, opacity=0.32)
        return val, grp

    def animate_p_sweep(
        self,
        p_start:  float = 0.1,
        p_end:    float = 0.8,
        run_time: float = 4.0,
    ) -> Animation:
        self.params.set("p", p_start)
        return self.animate_param("p", p_end, run_time=run_time)

    def animate_cdf_build(self, run_time: float = 4.0) -> Animation:
        p     = self.params.get("p")
        k_max = int(scipy_stats.geom(p).ppf(0.999))
        anims = []
        for k in range(1, k_max + 1):
            shade = self.shade_region(
                f"geom_cdf_k{k}",
                lo=float(k) - 0.5, hi=float(k) + 0.5,
                color=self._palette.primary, opacity=0.38, label=False)
            anims.append(FadeIn(shade, run_time=run_time / k_max))
        return Succession(*anims)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike, **kwargs
    ) -> "GeometricDistribution3D":
        x = np.asarray(data, float)
        p = float(np.clip(1.0 / x.mean(), 0.001, 0.999))
        return cls(axes, p=p, **kwargs)

    @classmethod
    def from_moments(
        cls, axes: StatsAxes3D, mean: float,
        variance: float = None, **kwargs
    ) -> "GeometricDistribution3D":
        return cls(axes, p=float(np.clip(1.0 / mean, 0.001, 0.999)), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NEGATIVE BINOMIAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class NegativeBinomialDistribution3D(DiscreteDistribution3D):
    """
    3-D Negative Binomial distribution.

    P(X=k) = C(k+r-1, k) * p^r * (1-p)^k  for k = 0,1,2,...
    (number of failures before r-th success)

    Parameters
    ----------
    r : number of successes (positive integer)
    p : probability of success per trial ∈ (0,1)

    Special features
    ----------------
    • annotate_overdispersion()   — label Var[X] > E[X]
    • show_geometric_link()       — annotation: r=1 → Geometric
    • shade_at_most(k)            — P(X ≤ k)
    • shade_at_least(k)           — P(X ≥ k)
    • animate_r_sweep(r0, r1)     — vary r watching shape shift
    • animate_p_sweep(p0, p1)     — vary p
    • from_data(x)                — MOM fit
    • from_moments(mean, var)     — solve for r, p
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Negative Binomial"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(X=k)=\binom{k+r-1}{k}p^r(1-p)^k")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {"r": 3.0, "p": 0.5}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {"r": "r", "p": "p"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {
        "r": (1.0, 100.0), "p": (0.001, 0.999)}

    def __init__(
        self,
        axes: StatsAxes3D,
        r:    int   = 3,
        p:    float = 0.5,
        show_formula:         bool = True,
        show_overdispersion:  bool = True,
        formula_pos:          Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._formula_pos = formula_pos or np.array([3.5, 2.5, 0.0])

        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.68,
            bar_depth=0.18,
            show_bar_labels=False,
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.negative_binomial(r, p),
            params  = {"r": float(r), "p": p},
            bar_cfg = bar_cfg,
            **kwargs,
        )
        for name, (lo, hi) in self.PARAM_BOUNDS.items():
            self.params._bounds[name] = (lo, hi)

        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette,
                font_size=22))
        if show_overdispersion:
            self.add(self._make_overdispersion_label())

    def _rebuild_dist_fn(self) -> None:
        r = max(1, int(round(self.params.get("r"))))
        p = self.params.get("p")
        self._dist_fn = DistributionFunction.negative_binomial(r, p)

    def _param_constraints(self) -> Dict[str, Callable]:
        return {"r": lambda v: v >= 1, "p": lambda v: 0 < v < 1}

    def _make_overdispersion_label(self) -> VGroup:
        t    = self._palette
        axes = self._axes
        r    = int(round(self.params.get("r")))
        p    = self.params.get("p")
        mean = r * (1 - p) / p
        var  = r * (1 - p) / p**2
        grp  = VGroup()
        xhi, yhi = axes.x_range[1], axes.y_range[1]
        l1 = MathTex(
            rf"E[X]=\frac{{r(1-p)}}{{p}}={mean:.2f}",
            font_size=18, color=t.accent)
        l2 = MathTex(
            rf"\mathrm{{Var}}[X]=\frac{{r(1-p)}}{{p^2}}={var:.2f}",
            font_size=18, color=t.secondary)
        l3 = MathTex(
            r"\mathrm{Var}[X]>\mathrm{E}[X]\Rightarrow\mathrm{overdispersed}",
            font_size=16, color=t.text_secondary)
        l1.move_to(axes.c2p(xhi * 0.55, yhi * 0.90))
        l2.move_to(axes.c2p(xhi * 0.55, yhi * 0.76))
        l3.move_to(axes.c2p(xhi * 0.55, yhi * 0.62))
        grp.add(l1, l2, l3)
        return grp

    def shade_at_most(self, k: int, key: str = None) -> VGroup:
        key = key or f"nb_leq_{k}"
        return self.shade_tail_left(key, float(k) + 0.5,
                                    color=self._palette.positive, opacity=0.35)

    def shade_at_least(self, k: int, key: str = None) -> VGroup:
        key = key or f"nb_geq_{k}"
        return self.shade_tail_right(key, float(k) - 0.5,
                                     color=self._palette.positive, opacity=0.35)

    def prob_at(self, k: int) -> Tuple[float, VGroup]:
        r = int(round(self.params.get("r")))
        p = self.params.get("p")
        val = float(scipy_stats.nbinom(r, p).pmf(k))
        grp = self.shade_region(
            f"nb_eq_{k}", lo=float(k)-0.5, hi=float(k)+0.5,
            color=self._palette.accent, opacity=0.55)
        return val, grp

    def cumulative_at(self, k: int) -> Tuple[float, VGroup]:
        r = int(round(self.params.get("r")))
        p = self.params.get("p")
        val = float(scipy_stats.nbinom(r, p).cdf(k))
        return val, self.shade_at_most(k, f"nb_cdf_{k}")

    def tail_at(self, k: int) -> Tuple[float, VGroup]:
        r = int(round(self.params.get("r")))
        p = self.params.get("p")
        val = float(scipy_stats.nbinom(r, p).sf(k - 1))
        return val, self.shade_at_least(k, f"nb_tail_{k}")

    def animate_r_sweep(
        self,
        r_start:  int   = 1,
        r_end:    int   = 20,
        run_time: float = 5.0,
    ) -> Animation:
        self.params.set("r", float(r_start))
        return self.animate_param("r", float(r_end), run_time=run_time)

    def animate_p_sweep(
        self,
        p_start:  float = 0.2,
        p_end:    float = 0.8,
        run_time: float = 4.0,
    ) -> Animation:
        self.params.set("p", p_start)
        return self.animate_param("p", p_end, run_time=run_time)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike, **kwargs
    ) -> "NegativeBinomialDistribution3D":
        x    = np.asarray(data, float)
        mean = x.mean(); var = x.var(ddof=1)
        p    = float(np.clip(mean / var, 0.001, 0.999))
        r    = max(1, int(round(mean * p / (1 - p))))
        return cls(axes, r=r, p=p, **kwargs)

    @classmethod
    def from_moments(
        cls, axes: StatsAxes3D, mean: float, variance: float, **kwargs
    ) -> "NegativeBinomialDistribution3D":
        p = float(np.clip(mean / variance, 0.001, 0.999))
        r = max(1, int(round(mean**2 / (variance - mean))))
        return cls(axes, r=r, p=p, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HYPERGEOMETRIC DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class HypergeometricDistribution3D(DiscreteDistribution3D):
    """
    3-D Hypergeometric distribution.
    Sampling WITHOUT replacement from a finite population.

    Parameters
    ----------
    M : population size
    n : number of success states in population
    N : number of draws

    Special features
    ----------------
    • annotate_urn()            — floating urn diagram description
    • compare_to_binomial()     — overlay Binom(N, n/M) for comparison
    • shade_at_most(k)          — P(X ≤ k)
    • shade_at_least(k)         — P(X ≥ k)
    • animate_N_sweep(N0, N1)   — vary number of draws
    • animate_n_sweep(n0, n1)   — vary number of success states
    • from_data(x, M, N)        — estimate n from sample
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Hypergeometric"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(X=k)=\frac{\binom{n}{k}\binom{M-n}{N-k}}{\binom{M}{N}}")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {
        "M": 50.0, "n": 20.0, "N": 10.0}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {
        "M": "M", "n": "n", "N": "N"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {
        "M": (2.0, 500.0), "n": (1.0, 499.0), "N": (1.0, 499.0)}

    def __init__(
        self,
        axes: StatsAxes3D,
        M:    int = 50,
        n:    int = 20,
        N:    int = 10,
        show_formula:   bool = True,
        show_urn_note:  bool = True,
        formula_pos:    Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._formula_pos = formula_pos or np.array([3.5, 2.5, 0.0])

        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.68,
            bar_depth=0.20,
            show_bar_labels=True,
            label_decimals=3,
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.hypergeometric(M, n, N),
            params  = {"M": float(M), "n": float(n), "N": float(N)},
            bar_cfg = bar_cfg,
            **kwargs,
        )
        for name, (lo, hi) in self.PARAM_BOUNDS.items():
            self.params._bounds[name] = (lo, hi)

        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette,
                font_size=22))
        if show_urn_note:
            self.add(self._make_urn_annotation(M, n, N))
        self.add(self._make_mean_var_label(M, n, N))

    def _rebuild_dist_fn(self) -> None:
        M = max(2, int(round(self.params.get("M"))))
        n = max(1, min(int(round(self.params.get("n"))), M - 1))
        N = max(1, min(int(round(self.params.get("N"))), M))
        self._dist_fn = DistributionFunction.hypergeometric(M, n, N)

    def _param_constraints(self) -> Dict[str, Callable]:
        return {
            "M": lambda v: v >= 2,
            "n": lambda v: 1 <= v <= self.params.get("M") - 1,
            "N": lambda v: 1 <= v <= self.params.get("M"),
        }

    def _make_urn_annotation(self, M: int, n: int, N: int) -> VGroup:
        t   = self._palette
        pos = self._axes.c2p(
            self._axes.x_range[1] * 0.50,
            self._axes.y_range[1] * 0.72)
        lbl = Text(
            f"Urn: {M} balls, {n} red, {M-n} blue\nDraw {N} without replacement",
            font_size=18, color=t.text_secondary)
        lbl.move_to(pos)
        bg = RoundedRectangle(
            corner_radius=0.10,
            width=lbl.width + 0.35, height=lbl.height + 0.25)
        bg.set_fill(t.surface, opacity=0.78)
        bg.set_stroke(t.border, width=0.9)
        bg.move_to(lbl)
        return VGroup(bg, lbl)

    def _make_mean_var_label(self, M: int, n: int, N: int) -> VGroup:
        t    = self._palette
        axes = self._axes
        mean = N * n / M
        var  = N * (n/M) * (1 - n/M) * (M - N) / (M - 1)
        grp  = VGroup()
        xhi, yhi = axes.x_range[1], axes.y_range[1]
        l1 = MathTex(rf"E[X]=\frac{{Nn}}{{M}}={mean:.3f}",
                     font_size=18, color=t.accent)
        l2 = MathTex(rf"\mathrm{{Var}}[X]={var:.3f}",
                     font_size=18, color=t.secondary)
        l1.move_to(axes.c2p(xhi * 0.62, yhi * 0.90))
        l2.move_to(axes.c2p(xhi * 0.62, yhi * 0.76))
        grp.add(l1, l2)
        return grp

    def compare_to_binomial(self) -> VGroup:
        """Overlay Binom(N, n/M) to illustrate with-replacement comparison."""
        M = max(2, int(round(self.params.get("M"))))
        n = max(1, min(int(round(self.params.get("n"))), M - 1))
        N = max(1, min(int(round(self.params.get("N"))), M))
        p = n / M
        t = self._palette
        x_vals = np.arange(0, N + 1)
        binom_pmf = scipy_stats.binom(N, p).pmf(x_vals)
        grp = VGroup()
        for xv, yv in zip(x_vals, binom_pmf):
            p1 = self._axes.c2p(float(xv) + 0.18, 0.0)
            p2 = self._axes.c2p(float(xv) + 0.18, float(yv))
            seg = Line3D(p1, p2, color=t.secondary, thickness=0.009)
            seg.set_opacity(0.60)
            grp.add(seg)
        lbl = MathTex(
            r"\mathrm{Binom}(N,\,n/M)",
            font_size=16, color=t.secondary)
        lbl.move_to(self._axes.c2p(N // 2 + 1.5, binom_pmf.max() * 0.85))
        grp.add(lbl)
        self.add(grp)
        return grp

    def shade_at_most(self, k: int, key: str = None) -> VGroup:
        key = key or f"hyp_leq_{k}"
        return self.shade_tail_left(key, float(k) + 0.5,
                                    color=self._palette.positive, opacity=0.35)

    def shade_at_least(self, k: int, key: str = None) -> VGroup:
        key = key or f"hyp_geq_{k}"
        return self.shade_tail_right(key, float(k) - 0.5,
                                     color=self._palette.positive, opacity=0.35)

    def prob_at(self, k: int) -> Tuple[float, VGroup]:
        M = int(round(self.params.get("M")))
        n = int(round(self.params.get("n")))
        N = int(round(self.params.get("N")))
        val = float(scipy_stats.hypergeom(M, n, N).pmf(k))
        grp = self.shade_region(
            f"hyp_eq_{k}", lo=float(k)-0.5, hi=float(k)+0.5,
            color=self._palette.accent, opacity=0.55)
        return val, grp

    def cumulative_at(self, k: int) -> Tuple[float, VGroup]:
        M = int(round(self.params.get("M")))
        n = int(round(self.params.get("n")))
        N = int(round(self.params.get("N")))
        val = float(scipy_stats.hypergeom(M, n, N).cdf(k))
        return val, self.shade_at_most(k, f"hyp_cdf_{k}")

    def tail_at(self, k: int) -> Tuple[float, VGroup]:
        M = int(round(self.params.get("M")))
        n = int(round(self.params.get("n")))
        N = int(round(self.params.get("N")))
        val = float(scipy_stats.hypergeom(M, n, N).sf(k - 1))
        return val, self.shade_at_least(k, f"hyp_tail_{k}")

    def animate_N_sweep(
        self,
        N_start:  int   = 1,
        N_end:    int   = None,
        run_time: float = 4.0,
    ) -> Animation:
        M   = int(round(self.params.get("M")))
        end = N_end or M - 1
        self.params.set("N", float(N_start))
        return self.animate_param("N", float(end), run_time=run_time)

    def animate_n_sweep(
        self,
        n_start:  int   = 1,
        n_end:    int   = None,
        run_time: float = 4.0,
    ) -> Animation:
        M   = int(round(self.params.get("M")))
        end = n_end or M - 1
        self.params.set("n", float(n_start))
        return self.animate_param("n", float(end), run_time=run_time)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike,
        M: int = 100, N: int = 10, **kwargs
    ) -> "HypergeometricDistribution3D":
        x    = np.asarray(data, float)
        mean = float(x.mean())
        n    = max(1, min(int(round(mean * M / N)), M - 1))
        return cls(axes, M=M, n=n, N=N, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DISCRETE UNIFORM DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class DiscreteUniformDistribution3D(DiscreteDistribution3D):
    """
    3-D Discrete Uniform distribution on {lo, lo+1, ..., hi}.

    Parameters
    ----------
    lo : lower bound integer
    hi : upper bound integer (> lo)

    Special features
    ----------------
    • annotate_max_entropy()   — label "Maximum entropy discrete dist"
    • annotate_flat_height()   — label P(X=k) = 1/(hi-lo+1)
    • shade_range(a, b)        — shade P(a ≤ X ≤ b)
    • animate_range_expand()   — animate hi increasing
    • animate_range_shift()    — shift [lo,hi] keeping width fixed
    • from_data(x)             — fit from min/max of data
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Discrete Uniform"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(X=k)=\frac{1}{n},\;k=a,a+1,\ldots,b")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {"lo": 1.0, "hi": 6.0}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {"lo": "a", "hi": "b"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {
        "lo": (-50.0, 50.0), "hi": (-50.0, 50.0)}

    def __init__(
        self,
        axes: StatsAxes3D,
        lo:   int = 1,
        hi:   int = 6,
        show_formula:      bool = True,
        annotate_entropy:  bool = True,
        formula_pos:       Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._formula_pos = formula_pos or np.array([3.5, 1.8, 0.0])
        n  = hi - lo + 1
        p0 = 1.0 / n if n > 0 else 1.0

        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.70,
            bar_depth=0.20,
            show_bar_labels=n <= 12,
            label_decimals=4,
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.uniform_discrete(lo, hi),
            params  = {"lo": float(lo), "hi": float(hi)},
            bar_cfg = bar_cfg,
            **kwargs,
        )
        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette))
        if annotate_entropy:
            self.add(self._make_entropy_label(n))
        self.add(self._make_height_label(n))

    def _rebuild_dist_fn(self) -> None:
        lo = int(round(self.params.get("lo")))
        hi = int(round(self.params.get("hi")))
        hi = max(hi, lo + 1)
        self._dist_fn = DistributionFunction.uniform_discrete(lo, hi)

    def _param_constraints(self) -> Dict[str, Callable]:
        return {"lo": lambda v: v < self.params.get("hi"),
                "hi": lambda v: v > self.params.get("lo")}

    def _make_entropy_label(self, n: int) -> VMobject:
        import math
        t   = self._palette
        h   = math.log2(n) if n > 0 else 0.0
        pos = self._axes.c2p(
            (self._axes.x_range[0] + self._axes.x_range[1]) / 2,
            self._axes.y_range[1] * 0.88)
        lbl = MathTex(
            rf"H(X)=\log_2 n = {h:.3f}\;\mathrm{{bits}}",
            font_size=20, color=self._palette.accent)
        lbl.move_to(pos)
        return lbl

    def _make_height_label(self, n: int) -> VMobject:
        p   = 1.0 / n if n > 0 else 1.0
        t   = self._palette
        lo  = int(round(self.params.get("lo")))
        hi  = int(round(self.params.get("hi")))
        pos = self._axes.c2p(
            (lo + hi) / 2.0,
            p + self._axes.y_range[1] * 0.06)
        lbl = MathTex(
            rf"P(X=k)=\frac{{1}}{{{n}}}={p:.4f}",
            font_size=20, color=t.text_secondary)
        lbl.move_to(pos)
        return lbl

    def shade_range(self, key: str, a: int, b: int) -> VGroup:
        return self.shade_region(key,
                                  lo=float(a) - 0.5,
                                  hi=float(b) + 0.5,
                                  color=self._palette.positive,
                                  opacity=0.40)

    def prob_at(self, k: int) -> Tuple[float, VGroup]:
        lo = int(round(self.params.get("lo")))
        hi = int(round(self.params.get("hi")))
        val = 1.0 / (hi - lo + 1) if lo <= k <= hi else 0.0
        grp = self.shade_range(f"unif_eq_{k}", k, k)
        return val, grp

    def cumulative_at(self, k: int) -> Tuple[float, VGroup]:
        lo = int(round(self.params.get("lo")))
        hi = int(round(self.params.get("hi")))
        n  = hi - lo + 1
        val = (min(k, hi) - lo + 1) / n if k >= lo else 0.0
        grp = self.shade_range(f"unif_cdf_{k}", lo, min(k, hi))
        return val, grp

    def tail_at(self, k: int) -> Tuple[float, VGroup]:
        lo  = int(round(self.params.get("lo")))
        hi  = int(round(self.params.get("hi")))
        n   = hi - lo + 1
        val = (hi - max(k, lo) + 1) / n if k <= hi else 0.0
        grp = self.shade_range(f"unif_tail_{k}", max(k, lo), hi)
        return val, grp

    def animate_range_expand(
        self,
        new_hi:   int   = None,
        run_time: float = 3.0,
    ) -> Animation:
        hi  = int(round(self.params.get("hi")))
        end = float(new_hi or hi + 4)
        return self.animate_param("hi", end, run_time=run_time)

    def animate_range_shift(
        self,
        shift:    int   = 3,
        run_time: float = 3.0,
    ) -> AnimationGroup:
        lo = self.params.get("lo")
        hi = self.params.get("hi")
        return self.animate_params(
            {"lo": lo + shift, "hi": hi + shift},
            run_time=run_time)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike, **kwargs
    ) -> "DiscreteUniformDistribution3D":
        x = np.asarray(data, float)
        return cls(axes, lo=int(x.min()), hi=int(x.max()), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MULTINOMIAL DISTRIBUTION  (grouped bar variant)
# ─────────────────────────────────────────────────────────────────────────────

class MultinomialDistribution3D(DiscreteDistribution3D):
    """
    3-D Multinomial distribution visualisation.

    Unlike other discrete distributions, this shows a grouped bar chart
    where each group is one outcome category, and bars within a group
    show P(X_i = k) for each category i simultaneously.

    Alternatively in simplified mode it shows the marginal distributions
    (each marginal X_i ~ Binomial(n, p_i)) as side-by-side bar clusters.

    Parameters
    ----------
    n  : number of trials
    ps : probability vector  (sums to 1, length k ≥ 2)

    Special features
    ----------------
    • show_marginals()          — k separate Binomial bar charts
    • annotate_sum_constraint() — label Σ p_i = 1, Σ X_i = n
    • shade_category(i)         — highlight bars for category i
    • animate_p_shift(i, delta) — shift p_i and renormalise others
    • from_data(x)              — estimate p from category counts
    """

    DISTRIBUTION_NAME: ClassVar[str] = "Multinomial"
    FORMULA_TEX:       ClassVar[str] = (
        r"P(\mathbf{X}=\mathbf{k})=\frac{n!}{k_1!\cdots k_r!}"
        r"\prod_{i=1}^{r}p_i^{k_i}")
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {
        "n": 10.0, "p0": 0.4, "p1": 0.35, "p2": 0.25}
    PARAM_LABELS:   ClassVar[Dict[str, str]]   = {
        "n": "n", "p0": "p_1", "p1": "p_2", "p2": "p_3"}
    PARAM_BOUNDS:   ClassVar[Dict[str, Tuple]] = {
        "n": (1.0, 200.0), "p0": (0.001, 0.998),
        "p1": (0.001, 0.998), "p2": (0.001, 0.998)}

    def __init__(
        self,
        axes:      StatsAxes3D,
        n:         int                = 10,
        ps:        Optional[List[float]] = None,
        cat_names: Optional[List[str]] = None,
        show_formula:       bool = True,
        show_sum_constraint: bool = True,
        formula_pos:        Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._formula_pos = formula_pos or np.array([3.5, 2.5, 0.0])
        self._cat_names   = cat_names

        # Normalise ps
        ps = list(ps) if ps is not None else [0.4, 0.35, 0.25]
        ps = [max(1e-4, p) for p in ps]
        total = sum(ps)
        ps = [p / total for p in ps]
        self._ps = ps
        self._k  = len(ps)

        # Build params: n, p0, p1, ..., p_{k-1}
        param_dict = {"n": float(n)}
        for i, p in enumerate(ps):
            param_dict[f"p{i}"] = p

        # We build a Binomial for the first marginal as the base dist_fn
        # Full multinomial PMF visualisation uses grouped bars in _build_body
        bar_cfg = kwargs.pop("bar_cfg", BarConfig(
            bar_width_ratio=0.55,
            bar_depth=0.22,
            show_bar_labels=True,
            label_decimals=3,
        ))

        super().__init__(
            axes    = axes,
            dist_fn = DistributionFunction.binomial(n, ps[0]),
            params  = param_dict,
            bar_cfg = bar_cfg,
            **kwargs,
        )
        for name in list(param_dict.keys()):
            if name == "n":
                self.params._bounds[name] = (1.0, 200.0)
            else:
                self.params._bounds[name] = (0.001, 0.998)

        if show_formula:
            self.add(_formula_panel(
                self.FORMULA_TEX, self._formula_pos, self._palette,
                font_size=20))
        if show_sum_constraint:
            self.add(self._make_sum_constraint_label())

    def _rebuild_dist_fn(self) -> None:
        n    = max(1, int(round(self.params.get("n"))))
        ps   = [self.params.get(f"p{i}") for i in range(self._k)]
        total = sum(ps)
        ps   = [p / total for p in ps]
        self._ps = ps
        # Rebuild first marginal for base geometry
        self._dist_fn = DistributionFunction.binomial(n, ps[0])

    def _param_constraints(self) -> Dict[str, Callable]:
        return {"n": lambda v: v >= 1}

    def _make_sum_constraint_label(self) -> VGroup:
        t    = self._palette
        ps   = self._ps
        axes = self._axes
        grp  = VGroup()
        sum_str = "+".join([f"{p:.3f}" for p in ps])
        lbl = MathTex(
            rf"\sum p_i = {sum_str} = 1,\;\sum X_i = n",
            font_size=18, color=t.text_secondary)
        lbl.move_to(axes.c2p(
            (axes.x_range[0] + axes.x_range[1]) / 2,
            axes.y_range[1] * 0.88))
        grp.add(lbl)
        return grp

    def show_marginals(self) -> List[VGroup]:
        """
        Return a list of VGroups, each showing one marginal Binomial PMF
        as a set of stems at slightly offset x positions.
        """
        n    = max(1, int(round(self.params.get("n"))))
        ps   = self._ps
        t    = self._palette
        axes = self._axes
        marginals = []
        for i, p in enumerate(ps):
            c    = t.distribution_palette[i % len(t.distribution_palette)]
            rv   = scipy_stats.binom(n, p)
            ks   = np.arange(0, n + 1)
            pmfs = rv.pmf(ks)
            grp  = VGroup()
            x_offset = (i - len(ps) / 2) * 0.22
            for k, pmf in zip(ks, pmfs):
                p1 = axes.c2p(float(k) + x_offset, 0.0)
                p2 = axes.c2p(float(k) + x_offset, float(pmf))
                stem = Line3D(p1, p2, color=c, thickness=0.008)
                stem.set_opacity(0.75)
                grp.add(stem)
                dot = Dot3D(p2, radius=0.04, color=c)
                dot.set_opacity(0.85)
                grp.add(dot)
            cat_name = (self._cat_names[i]
                        if self._cat_names and i < len(self._cat_names)
                        else f"X_{{{i+1}}}")
            lbl = MathTex(cat_name, font_size=18, color=c)
            lbl.move_to(axes.c2p(n * p + x_offset,
                                  float(pmfs.max()) + 0.05))
            grp.add(lbl)
            self.add(grp)
            marginals.append(grp)
        return marginals

    def shade_category(self, i: int, key: str = None) -> VGroup:
        """Shade all bars for category i by highlighting the entire x-range."""
        key = key or f"multi_cat_{i}"
        n   = max(1, int(round(self.params.get("n"))))
        return self.shade_region(key, lo=-0.5, hi=float(n) + 0.5,
                                  color=self._palette.distribution_palette[
                                      i % len(self._palette.distribution_palette)],
                                  opacity=0.20)

    def animate_p_shift(
        self,
        i:        int,
        delta:    float = 0.1,
        run_time: float = 2.0,
    ) -> AnimationGroup:
        """Shift p_i by delta and renormalise remaining probabilities."""
        ps    = [self.params.get(f"p{j}") for j in range(self._k)]
        new_pi = float(np.clip(ps[i] + delta, 0.001, 0.998))
        remaining = 1.0 - new_pi
        others    = [j for j in range(self._k) if j != i]
        old_sum   = sum(ps[j] for j in others)
        if old_sum > 1e-6:
            new_ps = {f"p{j}": ps[j] / old_sum * remaining
                      for j in others}
        else:
            new_ps = {f"p{j}": remaining / len(others) for j in others}
        new_ps[f"p{i}"] = new_pi
        return self.animate_params(new_ps, run_time=run_time)

    @classmethod
    def from_data(
        cls, axes: StatsAxes3D, data: ArrayLike,
        n: int = None, **kwargs
    ) -> "MultinomialDistribution3D":
        """
        *data* should be a 2-D array (n_trials × k_categories) of 0/1 indicators
        or a 1-D array of category indices.
        """
        x = np.asarray(data)
        if x.ndim == 1:
            k     = int(x.max()) + 1
            counts = np.bincount(x.astype(int), minlength=k).astype(float)
            ps     = counts / counts.sum()
            n_est  = n or int(x.max())
        else:
            counts = x.sum(axis=0).astype(float)
            ps     = counts / counts.sum()
            n_est  = n or int(x.sum(axis=1).max())
        return cls(axes, n=n_est, ps=list(ps), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "BernoulliDistribution3D",
    "BinomialDistribution3D",
    "PoissonDistribution3D",
    "GeometricDistribution3D",
    "NegativeBinomialDistribution3D",
    "HypergeometricDistribution3D",
    "DiscreteUniformDistribution3D",
    "MultinomialDistribution3D",
]